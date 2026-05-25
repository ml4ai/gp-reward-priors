"""Functional SGHMC sampler for preference learning (fSGHMC, Wu et al. 2025).

FPrefNet is a *standalone* class — it does not inherit from BayesNet or
PrefNet and requires no weight-space prior (OptimGaussianPrior).  The only
regularisation comes from the functional GP prior (LCFModel), whose gradient
is injected at every sampler step via a single VJP backward pass.

Gradient accounting
-------------------
Standard SGHMC accumulates:

    parameter.grad  =  -∂log_lik / ∂w / batch_size  +  ∂prior_energy / ∂w / N

AdaptiveSGHMC then scales by scale_grad = N, giving an effective gradient of:

    N/batch_size · (-∂log_lik/∂w)  +  ∂prior_energy/∂w   ≈  ∇U(w)

Here we use a likelihood-only loss and add the functional GP prior gradient:

    parameter.grad  +=  -∇_w log p_GP(f(·;w)) / N
                     =  +J_w(X_M)ᵀ K_{X_M}⁻¹(f(X_M;w) − m(X_M)) / N

After N-scaling the effective contribution is O(1) — matching the weight-space
prior it replaces.

See Wu et al. (2025) "Functional Stochastic Gradient MCMC for Bayesian Neural
Networks", AISTATS 2025.
"""

import math
import os
from itertools import islice

import numpy as np
import torch
import torch.utils.data as data_utils

from ..metrics.metrics_tensor import accuracy
from ..samplers.adaptive_sghmc import AdaptiveSGHMC
from ..samplers.sghmc import SGHMC
from ..utils.util import ensure_dir, inf_loop, prepare_device


# ---------------------------------------------------------------------------
# Module-level worker — must be defined at module scope for mp.spawn pickle
# ---------------------------------------------------------------------------

def _fpref_chain_worker(
    rank,
    batch_start,
    base_ckpt_dir,
    net_args,
    x_train,
    y_train,
    seed,
    train_kwargs,
    gp_prior_args,
    meas_kwargs,
    initial_weights=None,
):
    """Worker for one parallel FPrefNet chain, called by mp.spawn.

    All arguments are plain Python objects (numpy arrays, dicts, module-level
    callables) so they survive the pickle/unpickle round-trip intact.

    Args:
        rank: int, process rank in this batch — maps to CUDA device index.
        batch_start: int, chain index of rank 0 in this batch.
        base_ckpt_dir: str, root directory; chain i writes to
            ``<base_ckpt_dir>/chain_<i>/``.
        net_args: dict of keyword arguments forwarded to MLP(**net_args).
        x_train: numpy (N, 2, T, d_dim) preference training inputs.
        y_train: numpy (N,) training targets.
        seed: int, base random seed; chain i uses seed + i.
        train_kwargs: dict forwarded verbatim to FPrefNet.train().
        gp_prior_args: dict with keys:
            ``p_covariance`` — numpy (d,) or (d, d) weight prior covariance,
            ``function_vect`` — module-level callable (must be picklable),
            ``p_mean``        — numpy (d,) or None (→ zeros).
        meas_kwargs: dict with keys:
            ``x_meas``      — numpy (N_meas, obs_dim),
            ``aux_meas``    — numpy (N_meas, aux_dim) or None,
            ``n_meas``      — int, measurement points per step,
            ``meas_jitter`` — float, Cholesky diagonal regularisation.
        initial_weights: optional tuple of numpy arrays (one per parameter),
            giving the shared warm-up starting point for all chains.
    """
    from optbnn.bnn.likelihoods import LikCE
    from optbnn.bnn.nets.mlp import MLP
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

    likelihood = LikCE()

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
# FPrefNet — standalone functional SGHMC sampler
# ---------------------------------------------------------------------------

class FPrefNet:
    """Standalone functional SGHMC sampler for preference learning.

    Does **not** inherit from BayesNet or PrefNet and requires **no**
    weight-space prior (OptimGaussianPrior).  Regularisation comes entirely
    from the functional GP prior defined by ``gp_prior`` (an LCFModel).

    At every sampler step:

    1. Preference forward pass → fx_batch (twin-network sum of masked rewards).
    2. Likelihood-only loss backward → parameter.grad from data.
    3. Functional GP prior gradient computed via one VJP backward pass at
       ``n_meas`` randomly-sampled measurement points.
    4. Prior gradient added to parameter.grad (scaled 1/N so AdaptiveSGHMC's
       N-scaling keeps it O(1), matching the weight-space prior it replaces).
    5. Gradient clip + AdaptiveSGHMC step.

    All SGHMC infrastructure (AdaptiveSGHMC preconditioner, cyclical LR,
    burn-in, parallel chain dispatch via mp.spawn) is implemented directly,
    without delegating to BayesNet.

    Args:
        net: torch.nn.Module, the BNN (e.g. MLP).
        likelihood: LikelihoodModule (e.g. LikCE).
        ckpt_dir: str, directory for sampled-weight checkpoints.
        gp_prior: LCFModel, the functional GP prior.
        x_meas: numpy float32 (N_meas, obs_dim) — measurement-point pool.
            At each step ``n_meas`` rows are sampled uniformly without
            replacement and used for the VJP backward pass.
        aux_meas: numpy (N_meas, aux_dim) or None — auxiliary feature inputs
            passed as ``aux_X`` to the LCFModel feature map (e.g. raw states
            for ``bb_reward_prior`` which ignores action columns).
        n_meas: int, measurement points per step (default 100).
        meas_jitter: float, Cholesky diagonal jitter in ``solve_prior``
            (default 1e-6).
        temperature: float, posterior temperature (default 1.0).
        sampling_method: str, ``"adaptive_sghmc"`` (default) or ``"sghmc"``.
        logger: optional logging.Logger; falls back to print.
        n_gpu: int, number of GPUs (0 = CPU).
        name: str, label for logging.
    """

    def __init__(
        self,
        net,
        likelihood,
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
        self.net = net
        self.lik_module = likelihood
        self.ckpt_dir = ckpt_dir
        self.sampling_method = sampling_method
        self.temperature = temperature
        self.name = name
        self.n_gpu = n_gpu

        self.print_info = print if logger is None else logger.info

        # Sampler / sampling state
        self.step = 0
        self.sampler = None
        self.sampler_params = {}
        self.sampled_weights = []
        self.num_samples = 0
        self.num_saved_sets_weights = 0
        self.map = None

        # Checkpoint directory
        self.sampled_weights_dir = os.path.join(ckpt_dir, "sampled_weights")
        ensure_dir(self.sampled_weights_dir)

        # Device setup
        self.device, device_ids = prepare_device(n_gpu)
        self.net = self.net.to(self.device)
        if len(device_ids) > 1:
            self.net = torch.nn.DataParallel(net, device_ids=device_ids)

        # Functional GP prior
        self._gp_prior = gp_prior.to(self.device)
        self._x_meas = x_meas        # numpy (N_meas, obs_dim)
        self._aux_meas = aux_meas    # numpy (N_meas, aux_dim) or None
        self._n_meas = int(n_meas)
        self._meas_jitter = float(meas_jitter)

    # ------------------------------------------------------------------
    # Network weight access
    # ------------------------------------------------------------------

    @property
    def network_weights(self):
        """Current network weights as a tuple of CPU numpy arrays."""
        return tuple(
            np.asarray(p.data.clone().detach().cpu().numpy())
            for p in self.net.parameters()
        )

    @network_weights.setter
    def network_weights(self, weights):
        """Load a tuple of numpy arrays into the network parameters."""
        for param, w in zip(self.net.parameters(), weights):
            param.copy_(torch.from_numpy(w))

    @property
    def _bare_net(self):
        """Underlying module, unwrapped from DataParallel if present."""
        return (
            self.net.module
            if isinstance(self.net, torch.nn.DataParallel)
            else self.net
        )

    # ------------------------------------------------------------------
    # Sampler initialisation
    # ------------------------------------------------------------------

    def _initialize_sampler(
        self,
        num_datapoints,
        lr=1e-2,
        mdecay=0.05,
        num_burn_in_steps=3000,
        epsilon=1e-10,
        max_param_step=None,
    ):
        """Instantiate AdaptiveSGHMC (or SGHMC) with scale_grad = N / T."""
        dtype = np.float32
        self.sampler_params = {}
        self.sampler_params["scale_grad"] = dtype(num_datapoints) / self.temperature
        self.sampler_params["lr"] = dtype(lr)
        self.sampler_params["mdecay"] = dtype(mdecay)

        if self.sampling_method == "adaptive_sghmc":
            self.sampler_params["num_burn_in_steps"] = num_burn_in_steps
            self.sampler_params["epsilon"] = dtype(epsilon)
            if max_param_step is not None:
                self.sampler_params["max_param_step"] = float(max_param_step)
            self.sampler = AdaptiveSGHMC(
                self.net.parameters(), **self.sampler_params
            )
        elif self.sampling_method == "sghmc":
            self.sampler = SGHMC(self.net.parameters(), **self.sampler_params)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_sampled_weights(self):
        """Save the current sampled_weights list to a numbered file."""
        file_path = os.path.join(
            self.sampled_weights_dir,
            "sampled_weights_{:07d}".format(self.num_saved_sets_weights),
        )
        torch.save({"sampled_weights": self.sampled_weights}, file_path)
        self.num_saved_sets_weights += 1

    def _load_sampled_weights(self, file_path):
        """Load a sampled_weights file and return the list."""
        checkpoint = torch.load(file_path, weights_only=False)
        return checkpoint["sampled_weights"]

    # ------------------------------------------------------------------
    # Prediction helpers (adapted from PrefNet)
    # ------------------------------------------------------------------

    def predict(self, x_test, use_map=False, map_only=False):
        """Posterior predictive mean and variance over sampled weights.

        Args:
            x_test: numpy (n, obs_dim) or tensor — single-timestep inputs.
            use_map: also return the MAP prediction alongside the mean/var.
            map_only: return only the MAP prediction (assert self.map is set).

        Returns:
            (pred_mean, pred_var) — or with map variants; see PrefNet.predict.
        """
        x_tensor = torch.from_numpy(np.asarray(x_test)).float().to(self.device)

        def _fwd(weights):
            with torch.no_grad():
                self.network_weights = weights
                return self.net(x_tensor).detach().cpu().numpy()

        if map_only:
            assert self.map is not None
            return _fwd(self.map)

        predictions = np.array([_fwd(w) for w in self.sampled_weights])
        pred_mean = np.mean(predictions, axis=0)
        pred_var = np.var(predictions, axis=0)

        if use_map:
            assert self.map is not None
            return pred_mean, pred_var, _fwd(self.map)
        return pred_mean, pred_var

    def _predict_pairs_batched(self, x_1, x_2, am_1, am_2, T,
                                use_map=False, batch_size=256):
        """Mini-batched preference-pair prediction (GPU-memory safe).

        Identical logic to PrefNet._predict_pairs_batched.
        """
        N = am_1.shape[0]
        parts_1, parts_2 = [], []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b = end - start
            x1_b = x_1[start * T : end * T]
            x2_b = x_2[start * T : end * T]
            x_both = np.concatenate([x1_b, x2_b], axis=0)

            if use_map:
                _, _, pred = self.predict(x_both, use_map=True)
            else:
                pred, _ = self.predict(x_both)

            pred_1 = pred[: b * T].reshape(b, T) * am_1[start:end]
            pred_2 = pred[b * T :].reshape(b, T) * am_2[start:end]
            parts_1.append(np.nansum(pred_1, axis=1))
            parts_2.append(np.nansum(pred_2, axis=1))

        return np.concatenate(parts_1), np.concatenate(parts_2)

    def _ce_and_acc(self, sum_pred_1, sum_pred_2, y):
        """Cross-entropy and accuracy from per-pair reward sums."""
        fx = np.stack([sum_pred_1, sum_pred_2], axis=1).astype(np.float32)
        fx_t = torch.from_numpy(fx).to(self.device)
        y_t = torch.from_numpy(y).float().to(self.device)
        ce = torch.nn.CrossEntropyLoss()(fx_t, y_t).detach().cpu().numpy()
        acc = accuracy(fx_t, y_t).detach().cpu().numpy()
        return ce, acc

    def find_map(self, x, y, max_map_samples=512):
        """Select the MAP weight set by minimum likelihood loss.

        Uses the likelihood (NLL) alone as the MAP criterion — appropriate
        since fSGHMC carries no weight-space prior.

        Args:
            x: numpy (N, 2, T, d_dim) preference-pair inputs.
            y: numpy (N,) or (N, 1) labels.
            max_map_samples: maximum pairs to evaluate (subset for speed).
        """
        assert self.sampled_weights, "No sampled weights to select MAP from."

        if x.shape[0] > max_map_samples:
            idx = np.random.choice(x.shape[0], max_map_samples, replace=False)
            x, y = x[idx], y[idx]

        x_t = torch.from_numpy(x.squeeze()).float().to(self.device)
        y_t = torch.from_numpy(y.squeeze()).float().to(self.device)

        def _nll(weights):
            with torch.no_grad():
                self.network_weights = weights
                B, _, T, d_dim = x_t.size()
                obs_dim = d_dim - 1
                am_1 = x_t[:, 0, :, obs_dim]
                am_2 = x_t[:, 1, :, obs_dim]
                x_1 = x_t[:, 0, :, :obs_dim].reshape(-1, obs_dim)
                x_2 = x_t[:, 1, :, :obs_dim].reshape(-1, obs_dim)
                pred_both = self.net(
                    torch.cat([x_1, x_2], dim=0)
                ).view(2, B, T)
                pred_1 = pred_both[0] * am_1
                pred_2 = pred_both[1] * am_2
                sum_1 = torch.nansum(pred_1, dim=1).view(-1, 1)
                sum_2 = torch.nansum(pred_2, dim=1).view(-1, 1)
                fx = torch.cat([sum_1, sum_2], dim=1)
                return float(self.lik_module(fx, y_t).detach().cpu())

        losses = np.array([_nll(w) for w in self.sampled_weights])
        self.map = self.sampled_weights[int(np.argmin(losses))]

    def eval_test_data(self, x, y, x_map=None, y_map=None, eval_batch_size=256):
        """Evaluate on test data; identical signature to PrefNet.eval_test_data.

        Args:
            x: numpy (N, 2, T, d_dim) test preference pairs.
            y: numpy (N,) labels.
            x_map, y_map: optional training data for MAP weight selection.
            eval_batch_size: mini-batch size for _predict_pairs_batched.

        Returns:
            (ce, acc): float cross-entropy and accuracy.
        """
        self.net.eval()
        B, _, T, d_dim = x.shape
        obs_dim = d_dim - 1
        am_1 = x[:, 0, :, obs_dim]
        am_2 = x[:, 1, :, obs_dim]
        x_1 = x[:, 0, :, :obs_dim].reshape(-1, obs_dim)
        x_2 = x[:, 1, :, :obs_dim].reshape(-1, obs_dim)

        use_map = (x_map is not None) and (y_map is not None)
        if use_map:
            self.find_map(x_map, y_map)

        sum_1, sum_2 = self._predict_pairs_batched(
            x_1, x_2, am_1, am_2, T,
            use_map=use_map,
            batch_size=eval_batch_size,
        )
        ce, acc = self._ce_and_acc(sum_1, sum_2, y)

        if use_map:
            self.map = None
        self.net.train()
        return ce, acc

    # ------------------------------------------------------------------
    # Training loop — fSGHMC with functional GP prior
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
        # kept for API compatibility with train_kwargs from the standard sampler;
        # not used (no weight-space hyperprior to resample, no separate MAP eval)
        resample_prior_every=1000,
        eval_map=False,
        use_cyclical_lr=False,
        lr_max=None,
        cycle_length=None,
        fraction_cool=0.25,
        max_param_step=None,
    ):
        """Run the fSGHMC training loop.

        Signature is a superset of BayesNet.train() so the same train_kwargs
        dict works for both PrefNet and FPrefNet runs.  ``resample_prior_every``
        and ``eval_map`` are accepted but silently ignored (no weight-space
        hyperprior; MAP selection is available via eval_test_data).
        """
        # ---- Data loader ------------------------------------------------
        if data_loader is not None:
            num_datapoints = len(data_loader.sampler)
            train_loader = inf_loop(data_loader)
        else:
            num_datapoints = x_train.shape[0]
            x_t = torch.from_numpy(x_train.squeeze()).float()
            y_t = torch.from_numpy(y_train.squeeze()).float()
            train_loader = inf_loop(
                data_utils.DataLoader(
                    data_utils.TensorDataset(x_t, y_t),
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=(self.device.type == "cuda"),
                    num_workers=0,
                )
            )

        # ---- Cyclical LR schedule parameters ----------------------------
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

        # ---- Main loop --------------------------------------------------
        batch_generator = islice(enumerate(train_loader), num_steps)
        self.net.train()
        n_samples = 0

        for step, (x_batch, y_batch) in batch_generator:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # ---- Cyclical LR --------------------------------------------
            if use_cyclical_lr and step >= num_burn_in_steps:
                _post_burn = step - num_burn_in_steps
                _cycle_step = _post_burn % _cycle_len
                _cycle_lr = _lr_min + 0.5 * (_lr_max - _lr_min) * (
                    1.0 + math.cos(math.pi * _cycle_step / _cycle_len)
                )
                for _pg in self.sampler.param_groups:
                    _pg["lr"] = np.float32(_cycle_lr)
                if _cycle_step == 0:
                    # Zero momentum at the start of each new hot phase
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

            # ---- Likelihood gradient ------------------------------------
            self.sampler.zero_grad()
            lik_loss = self.lik_module(fx_batch, y_batch) / y_batch.shape[0]
            lik_loss.backward()

            # ---- Functional GP prior gradient ---------------------------
            # Sample n_meas points from the measurement pool
            meas_idx = np.random.choice(len(self._x_meas), n_meas_actual, replace=False)
            x_meas_t = torch.from_numpy(self._x_meas[meas_idx]).float().to(self.device)
            aux_meas_t = (
                torch.from_numpy(self._aux_meas[meas_idx]).to(self.device)
                if self._aux_meas is not None
                else None
            )

            # functional_prior_grad returns ∇_w log p_GP = -J_w^T K^{-1}(f-m).
            # This uses torch.autograd.grad (not .backward()), so it does NOT
            # touch parameter.grad — we add the result manually below.
            func_grads = self._gp_prior.functional_prior_grad(
                self._bare_net,
                x_meas_t,
                aux_X=aux_meas_t,
                jitter=self._meas_jitter,
            )

            # Add ∇U_prior = -∇_w log p_GP to parameter.grad, scaled by 1/N.
            # After AdaptiveSGHMC's scale_grad = N multiplication the effective
            # contribution is O(1) — the same order as the weight-space prior.
            for param, fg in zip(self._bare_net.parameters(), func_grads):
                if param.grad is not None:
                    param.grad.add_(-fg.to(param.grad.dtype) / num_datapoints)

            # ---- Clip and step ------------------------------------------
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 100.0)
            self.sampler.step()
            self.step += 1

            # ---- Sample collection (cyclical or fixed-interval) ----------
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
    # Parallel chain dispatch
    # ------------------------------------------------------------------

    def sample_multi_chains_parallel(
        self,
        x_train,
        y_train,
        net_args,
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

        Sampled weights for chain i are written to::

            <self.ckpt_dir>/chain_<i>/sampled_weights/sampled_weights_0000000

        Args:
            x_train: numpy (N, 2, T, d_dim) training inputs.
            y_train: numpy (N,) training targets.
            net_args: dict of kwargs for MLP(**net_args).
            gp_prior_args: dict with keys ``p_covariance``, ``function_vect``,
                ``p_mean`` (see _fpref_chain_worker docstring).
            meas_kwargs: dict with keys ``x_meas``, ``aux_meas``, ``n_meas``,
                ``meas_jitter`` (see _fpref_chain_worker docstring).
            num_chains: int, total number of chains.
            seed: int, base seed; chain i uses seed + i.
            initial_weights: optional tuple of numpy arrays — shared warm-up
                starting point to prevent chains from diverging at init.
            (remaining args forwarded verbatim to FPrefNet.train())
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
