"""Functional SGHMC PrefNet (fSGHMC, Wu et al. 2025).

Replaces the weight-space Gaussian prior gradient with a functional GP prior
gradient computed via a single vector-Jacobian product (VJP) at each step.
The rest of the infrastructure — AdaptiveSGHMC preconditioner, cyclical LR,
burn-in, parallel chain management — is inherited from PrefNet unchanged.

Gradient accounting
-------------------
Standard SGHMC accumulates:

    parameter.grad  =  -∂log_lik / ∂w / batch_size  +  ∂prior_energy / ∂w / N

AdaptiveSGHMC then scales by scale_grad = N, giving an effective gradient of:

    N/batch_size · (-∂log_lik/∂w)  +  ∂prior_energy/∂w   ≈  ∇U(w)

where U = -log p(w|D).

Here we drop the weight-space prior and add the functional GP prior instead:

    parameter.grad  +=  +J_w(X_M)ᵀ K_{X_M}⁻¹ (f(X_M;w) − m(X_M)) / N
                     =  -∇_w log p_GP(f(·;w)) / N     [negated for ∇U sign]

After N-scaling by AdaptiveSGHMC the contribution is O(1) — the same order as
the weight-space prior it replaces.

See Wu et al. (2025) "Functional Stochastic Gradient MCMC for Bayesian Neural
Networks", AISTATS 2025, for the full derivation.
"""

import math
import os
from itertools import islice

import numpy as np
import torch
import torch.utils.data as data_utils

from ..utils.util import ensure_dir, inf_loop
from .pref_net import PrefNet


# ---------------------------------------------------------------------------
# Module-level worker (must be picklable for mp.spawn)
# ---------------------------------------------------------------------------

def _fpref_chain_worker(
    rank,
    batch_start,
    base_ckpt_dir,
    net_args,
    ckpt_path,
    x_train,
    y_train,
    seed,
    train_kwargs,
    gp_prior_args,
    meas_kwargs,
    initial_weights=None,
):
    """Worker for one parallel FPrefNet chain, called by mp.spawn.

    Defined at module level so it is picklable under the ``spawn`` start
    method that CUDA requires.

    Args:
        rank: process rank within the current batch — the CUDA device index.
        batch_start: chain index of rank 0 in this batch.
        base_ckpt_dir: root directory; chain i writes to
            ``<base_ckpt_dir>/chain_<i>/``.
        net_args: dict forwarded to MLP(**net_args).
        ckpt_path: path to the OptimGaussianPrior checkpoint.
        x_train: numpy (N, 2, T, d_dim) training inputs.
        y_train: numpy (N,) training targets.
        seed: base random seed; chain i uses seed + i.
        train_kwargs: dict forwarded verbatim to FPrefNet.train().
        gp_prior_args: dict with keys:
            ``p_covariance`` (numpy array), ``function_vect`` (callable),
            ``p_mean`` (numpy array or None).
        meas_kwargs: dict with keys:
            ``x_meas`` (numpy, N_meas × obs_dim),
            ``aux_meas`` (numpy, N_meas × state_dim, or None),
            ``n_meas`` (int),
            ``meas_jitter`` (float).
        initial_weights: optional tuple of numpy arrays (one per parameter),
            shared starting point for all chains.
    """
    from optbnn.bnn.likelihoods import LikCE
    from optbnn.bnn.nets.mlp import MLP
    from optbnn.bnn.priors import OptimGaussianPrior
    from optbnn.gp.models.model import LCFModel
    from optbnn.sgmcmc_bayes_net.f_pref_net import FPrefNet
    from optbnn.utils.util import set_seed

    chain_idx = batch_start + rank
    torch.cuda.set_device(rank)
    set_seed(seed + chain_idx)

    import wandb as _wandb
    _wandb.init(mode="disabled")

    device = torch.device(f"cuda:{rank}")

    net = MLP(**net_args)
    if initial_weights is not None:
        with torch.no_grad():
            for param, w in zip(net.parameters(), initial_weights):
                param.copy_(torch.from_numpy(w))

    prior = OptimGaussianPrior(ckpt_path)
    likelihood = LikCE()

    # Reconstruct the LCFModel from plain-Python args (pickle-safe)
    gp_prior = LCFModel(
        p_covariance=gp_prior_args["p_covariance"],
        function_vect=gp_prior_args["function_vect"],
        device=device,
        p_mean=gp_prior_args.get("p_mean"),
    ).to(device)

    chain_dir = os.path.join(base_ckpt_dir, f"chain_{chain_idx}")
    os.makedirs(chain_dir, exist_ok=True)

    bayes_net = FPrefNet(
        net=net,
        likelihood=likelihood,
        prior=prior,
        ckpt_dir=chain_dir,
        gp_prior=gp_prior,
        x_meas=meas_kwargs["x_meas"],
        aux_meas=meas_kwargs.get("aux_meas"),
        n_meas=meas_kwargs.get("n_meas", 100),
        meas_jitter=meas_kwargs.get("meas_jitter", 1e-6),
        n_gpu=1,
        name=f"chain_{chain_idx}",
    )
    bayes_net.train(x_train, y_train, **train_kwargs)
    bayes_net._save_sampled_weights()


# ---------------------------------------------------------------------------
# FPrefNet
# ---------------------------------------------------------------------------

class FPrefNet(PrefNet):
    """PrefNet with a functional GP prior (fSGHMC).

    The weight-space Gaussian prior used in standard PrefNet is replaced by a
    functional GP prior defined by an LCFModel.  At every sampler step the
    functional prior gradient

        ∇_w log p_GP(f(·; w))  =  -J_w(X_M)ᵀ K_{X_M}⁻¹ (f(X_M; w) − m(X_M))

    is computed via a single VJP backward pass (cost: one extra forward +
    backward through the BNN at n_meas measurement points) and added to the
    gradient before the AdaptiveSGHMC update.

    Args:
        net: torch.nn.Module, the BNN.
        likelihood: LikelihoodModule.
        prior: PriorModule (its weight-space gradient is NOT used; the
            checkpoint is kept only to set the BNN architecture/dtype).
        ckpt_dir: str, checkpoint directory.
        gp_prior: LCFModel instance defining the functional GP prior.
        x_meas: numpy (N_meas, obs_dim) — pool of measurement observations.
            At each step ``n_meas`` rows are sampled uniformly without
            replacement and fed to the BNN + GP feature map.
        aux_meas: numpy (N_meas, aux_dim) or None — auxiliary inputs for the
            GP feature map (e.g. raw states when the feature function ignores
            actions).  Must satisfy ``aux_meas.shape[0] == x_meas.shape[0]``.
        n_meas: int, number of measurement points per step (default 100).
        meas_jitter: float, diagonal regularisation for the Cholesky solve in
            ``LCFModel.solve_prior`` (default 1e-6).
        (remaining kwargs forwarded to PrefNet.__init__)
    """

    def __init__(
        self,
        net,
        likelihood,
        prior,
        ckpt_dir,
        gp_prior,
        x_meas,
        aux_meas=None,
        n_meas=100,
        meas_jitter=1e-6,
        temperature=1.0,
        sampling_method="adaptive_sghmc",
        logger=None,
        n_gpu=0,
        name="fpref",
    ):
        super().__init__(
            net=net,
            likelihood=likelihood,
            prior=prior,
            ckpt_dir=ckpt_dir,
            temperature=temperature,
            sampling_method=sampling_method,
            logger=logger,
            n_gpu=n_gpu,
            name=name,
        )
        self._gp_prior = gp_prior.to(self.device)
        self._x_meas = x_meas        # numpy (N_meas, obs_dim)
        self._aux_meas = aux_meas    # numpy (N_meas, aux_dim) or None
        self._n_meas = int(n_meas)
        self._meas_jitter = float(meas_jitter)

    # ------------------------------------------------------------------
    # Training loop (overrides BayesNet.train)
    # ------------------------------------------------------------------

    def train(
        self,
        x_train=None,
        y_train=None,
        data_loader=None,
        num_samples=None,
        keep_every=100,
        n_discarded=0,
        num_burn_in_steps=3000,
        lr=1e-2,
        batch_size=32,
        epsilon=1e-10,
        mdecay=0.05,
        print_every_n_samples=10,
        continue_training=False,
        clear_sampled_weights=True,
        resample_prior_every=1000,
        resample_hyper_prior_burn_in=True,
        eval_map=False,
        use_cyclical_lr=False,
        lr_max=None,
        cycle_length=None,
        fraction_cool=0.25,
        max_param_step=None,
    ):
        """Train using fSGHMC: likelihood gradient + functional GP prior gradient.

        Signature is identical to BayesNet.train() so the same train_kwargs dict
        can be used for both PrefNet and FPrefNet runs.
        """
        # ---- Data loader setup (pref task only) -------------------------
        if data_loader is not None:
            num_datapoints = len(data_loader.sampler)
            train_loader = inf_loop(data_loader)
        else:
            num_datapoints = x_train.shape[0]
            x_train_ = torch.from_numpy(x_train.squeeze()).float()
            y_train_ = torch.from_numpy(y_train.squeeze()).float()
            train_loader = inf_loop(
                data_utils.DataLoader(
                    data_utils.TensorDataset(x_train_, y_train_),
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=(self.device.type == "cuda"),
                    num_workers=0,
                )
            )

        # ---- Cyclical LR parameters -------------------------------------
        _cycle_len = int(cycle_length) if cycle_length is not None else int(keep_every)
        _lr_max = float(lr_max) if lr_max is not None else float(lr) * 10.0
        _lr_min = float(lr)

        if use_cyclical_lr and num_samples is not None:
            num_steps = (num_samples + n_discarded) * _cycle_len
        else:
            num_steps = 0 if num_samples is None else (num_samples + 1) * keep_every

        # ---- Sampler initialisation -------------------------------------
        if not continue_training:
            if clear_sampled_weights:
                self.sampled_weights.clear()
            self.net = self.net.float()
            self._initialize_sampler(
                num_datapoints, lr, mdecay, num_burn_in_steps, epsilon,
                max_param_step=max_param_step,
            )
            num_steps += num_burn_in_steps

        # ---- Measurement-pool size check --------------------------------
        n_meas_actual = min(self._n_meas, len(self._x_meas))
        if n_meas_actual < self._n_meas:
            self.print_info(
                f"[fSGHMC] Measurement pool ({len(self._x_meas)}) smaller than "
                f"n_meas ({self._n_meas}); using all pool points every step."
            )

        # ---- Main training loop -----------------------------------------
        batch_generator = islice(enumerate(train_loader), num_steps)
        self.net.train()
        n_samples = 0

        for step, (x_batch, y_batch) in batch_generator:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # ---- Cyclical LR schedule -----------------------------------
            if use_cyclical_lr and step >= num_burn_in_steps:
                _post_burn = step - num_burn_in_steps
                _cycle_step = _post_burn % _cycle_len
                _cycle_lr = _lr_min + 0.5 * (_lr_max - _lr_min) * (
                    1.0 + math.cos(math.pi * _cycle_step / _cycle_len)
                )
                for _pg in self.sampler.param_groups:
                    _pg["lr"] = np.float32(_cycle_lr)
                if _cycle_step == 0:
                    # Zero momentum at the start of each hot phase
                    for _pg in self.sampler.param_groups:
                        for _p in _pg["params"]:
                            _s = self.sampler.state.get(_p)
                            if _s is not None and "momentum" in _s:
                                _s["momentum"].zero_()

            # ---- Preference task forward pass ---------------------------
            B, _, T, d_dim = x_batch.size()
            obs_dim = d_dim - 1
            am_1 = x_batch[:, 0, :, obs_dim]
            am_2 = x_batch[:, 1, :, obs_dim]
            x_batch_1 = x_batch[:, 0, :, :obs_dim].reshape(-1, obs_dim)
            x_batch_2 = x_batch[:, 1, :, :obs_dim].reshape(-1, obs_dim)

            pred_both = self.net(
                torch.cat([x_batch_1, x_batch_2], dim=0)
            ).view(2, B, T)
            pred_1 = pred_both[0] * am_1
            pred_2 = pred_both[1] * am_2
            sum_pred_1 = torch.nansum(pred_1, dim=1).view(-1, 1)
            sum_pred_2 = torch.nansum(pred_2, dim=1).view(-1, 1)
            fx_batch = torch.cat([sum_pred_1, sum_pred_2], dim=1)

            # ---- Gradient computation -----------------------------------
            self.sampler.zero_grad()

            # Likelihood-only loss — the functional GP prior handles the prior
            lik_loss = self.lik_module(fx_batch, y_batch) / y_batch.shape[0]
            lik_loss.backward()

            # ---- Functional GP prior gradient ---------------------------
            # Sample n_meas measurement points uniformly from the pool
            meas_idx = np.random.choice(
                len(self._x_meas), n_meas_actual, replace=False
            )
            x_meas_t = torch.from_numpy(
                self._x_meas[meas_idx]
            ).float().to(self.device)
            aux_meas_t = None
            if self._aux_meas is not None:
                aux_meas_t = torch.from_numpy(
                    self._aux_meas[meas_idx]
                ).to(self.device)

            # Returns ∇_w log p_GP = -J_w^T K^{-1}(f_M - m_M) per parameter.
            # This is a NEW forward+backward through _bare_net; it does NOT
            # touch parameter.grad (torch.autograd.grad is used, not .backward()).
            func_grads = self._gp_prior.functional_prior_grad(
                self._bare_net,
                x_meas_t,
                aux_X=aux_meas_t,
                jitter=self._meas_jitter,
            )

            # Add the functional prior's contribution to ∇U = -∇ log p(w|D).
            # ∇U_prior = -∇_w log p_GP = +J_w^T α.
            # Divide by num_datapoints so that after AdaptiveSGHMC's
            # scale_grad = num_datapoints, the effective contribution is O(1)
            # — the same scale as the weight-space prior it replaces.
            for param, fg in zip(self._bare_net.parameters(), func_grads):
                if param.grad is not None:
                    # fg = ∇_w log p_GP  →  -fg = ∇_w U_prior
                    param.grad.add_(-fg.to(param.grad.dtype) / num_datapoints)

            # ---- Clip and update ----------------------------------------
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 100.0)
            self.sampler.step()
            self.step += 1

            # ---- Resample prior hyper-parameters (if any) ---------------
            if self.prior_module.hyperprior:
                if step % resample_prior_every == 0:
                    if resample_hyper_prior_burn_in:
                        self.prior_module.resample(self._bare_net)
                    elif step > num_burn_in_steps:
                        self.prior_module.resample(self._bare_net)

            # ---- Sample collection --------------------------------------
            if use_cyclical_lr and step >= num_burn_in_steps:
                _post_burn = step - num_burn_in_steps
                _cycle_step = _post_burn % _cycle_len
                if _cycle_step == _cycle_len - 1:
                    n_samples += 1
                    if n_samples > n_discarded:
                        self.sampled_weights.append(self.network_weights)
                        self.num_samples += 1
            elif (not use_cyclical_lr) and (step > num_burn_in_steps) and (
                (step - num_burn_in_steps) % keep_every == 0
            ):
                n_samples += 1
                if n_samples > n_discarded:
                    self.sampled_weights.append(self.network_weights)
                    self.num_samples += 1

    # ------------------------------------------------------------------
    # Parallel chain orchestration (overrides PrefNet.sample_multi_chains_parallel)
    # ------------------------------------------------------------------

    def sample_multi_chains_parallel(
        self,
        x_train,
        y_train,
        net_args,
        ckpt_path,
        gp_prior_args,
        meas_kwargs,
        num_samples=None,
        num_chains=1,
        keep_every=100,
        n_discarded=0,
        num_burn_in_steps=3000,
        lr=1e-2,
        batch_size=32,
        epsilon=1e-10,
        mdecay=0.05,
        print_every_n_samples=10,
        resample_prior_every=1000,
        eval_map=False,
        seed=1,
        initial_weights=None,
        use_cyclical_lr=False,
        lr_max=None,
        cycle_length=None,
        fraction_cool=0.25,
        max_param_step=None,
    ):
        """Run multiple fSGHMC chains in parallel, one process per GPU.

        Extends PrefNet.sample_multi_chains_parallel with two extra arguments:

        Args:
            gp_prior_args: dict with keys:
                ``p_covariance`` (numpy array, shape (d,) or (d, d)),
                ``function_vect`` (module-level callable, must be picklable),
                ``p_mean``        (numpy array shape (d,), or None → zeros).
            meas_kwargs: dict with keys:
                ``x_meas``    (numpy (N_meas, obs_dim)),
                ``aux_meas``  (numpy (N_meas, aux_dim) or None),
                ``n_meas``    (int, measurement points per step),
                ``meas_jitter`` (float, Cholesky jitter).
            (all other args are identical to PrefNet.sample_multi_chains_parallel)
        """
        import torch.multiprocessing as mp

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError(
                "sample_multi_chains_parallel requires at least one CUDA device."
            )

        train_kwargs = dict(
            num_samples=num_samples,
            keep_every=keep_every,
            n_discarded=n_discarded,
            num_burn_in_steps=num_burn_in_steps,
            lr=lr,
            batch_size=batch_size,
            epsilon=epsilon,
            mdecay=mdecay,
            print_every_n_samples=print_every_n_samples,
            continue_training=False,
            clear_sampled_weights=True,
            resample_prior_every=resample_prior_every,
            eval_map=eval_map,
            use_cyclical_lr=use_cyclical_lr,
            lr_max=lr_max,
            cycle_length=cycle_length,
            fraction_cool=fraction_cool,
            max_param_step=max_param_step,
        )

        for batch_start in range(0, num_chains, num_gpus):
            n_parallel = min(num_gpus, num_chains - batch_start)
            self.print_info(
                "Launching fSGHMC chains {:d}–{:d} in parallel on {:d} GPU(s)".format(
                    batch_start, batch_start + n_parallel - 1, n_parallel
                )
            )
            mp.spawn(
                _fpref_chain_worker,
                args=(
                    batch_start,
                    self.ckpt_dir,
                    net_args,
                    ckpt_path,
                    x_train,
                    y_train,
                    seed,
                    train_kwargs,
                    gp_prior_args,
                    meas_kwargs,
                    initial_weights,
                ),
                nprocs=n_parallel,
                join=True,
            )
