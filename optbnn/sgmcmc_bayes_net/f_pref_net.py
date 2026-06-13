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
    chains_per_gpu=1,
):
    """Worker for one parallel FPrefNet chain, called by mp.spawn.

    All arguments are plain Python objects (numpy arrays, dicts, module-level
    callables) so they survive the pickle/unpickle round-trip intact.

    Args:
        rank: int, process rank in this batch.  The CUDA device is
            ``rank // chains_per_gpu`` so consecutive ranks pack onto the same
            GPU before spilling to the next one.
        batch_start: int, chain index of rank 0 in this batch.
        base_ckpt_dir: str, root directory; chain i writes to
            ``<base_ckpt_dir>/chain_<i>/``.
        net_args: dict of keyword arguments forwarded to MLP(**net_args).
        x_train: numpy (N, 2, T, d_dim) preference training inputs.
        y_train: numpy (N,) training targets.
        seed: int, base random seed; chain i uses seed + i.
        train_kwargs: dict forwarded verbatim to FPrefNet.train().
        gp_prior_args: dict describing the functional GP prior to reconstruct.
            The optional key ``prior_type`` selects the prior class
            (``"lcf"`` default, or ``"map_informed"``).
            For ``"lcf"`` (LCFModel) the keys are:
            ``p_covariance`` — numpy (d,) or (d, d) weight prior covariance,
            ``function_vect`` — module-level callable (must be picklable),
            ``p_mean``        — numpy (d,) or None (→ zeros).
            For ``"map_informed"`` (MapInformedGPPrior) the keys are those
            produced by ``MapInformedGPPrior.to_args()`` (free_mask, scaling,
            offset, eta, sig_c2, sig_g2, sig_n2, xy_cols, xy_source).
        meas_kwargs: dict with keys:
            ``x_meas``      — numpy (N_meas, obs_dim),
            ``aux_meas``    — numpy (N_meas, aux_dim) or None,
            ``n_meas``      — int, measurement points per step,
            ``meas_jitter`` — float, Cholesky diagonal regularisation.
        initial_weights: optional tuple of numpy arrays (one per parameter),
            giving the shared warm-up starting point for all chains.
        chains_per_gpu: int, how many chains share each GPU; the device index
            for this worker is ``rank // chains_per_gpu``.
    """
    from optbnn.bnn.likelihoods import LikCE
    from optbnn.bnn.nets.mlp import MLP
    from optbnn.sgmcmc_bayes_net.f_pref_net import FPrefNet
    from optbnn.utils.util import set_seed

    chain_idx = batch_start + rank
    device_idx = rank // chains_per_gpu
    torch.cuda.set_device(device_idx)
    set_seed(seed + chain_idx)

    import wandb as _wandb
    _wandb.init(mode="disabled")

    device = torch.device(f"cuda:{device_idx}")

    net = MLP(**net_args)
    if initial_weights is not None:
        with torch.no_grad():
            for param, w in zip(net.parameters(), initial_weights):
                param.copy_(torch.from_numpy(w))

    likelihood = LikCE()

    # Reconstruct the functional GP prior (LCFModel or MapInformedGPPrior).
    prior_type = gp_prior_args.get("prior_type", "lcf")
    if prior_type == "map_informed":
        from optbnn.gp.models.map_informed_prior import MapInformedGPPrior
        gp_prior = MapInformedGPPrior.from_args(gp_prior_args, device=device)
    else:
        from optbnn.gp.models.model import LCFModel
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

    def predict(self, x_test):
        """Posterior predictive mean and variance over sampled weights.

        Args:
            x_test: numpy (n, obs_dim) or tensor — single-timestep inputs.

        Returns:
            (pred_mean, pred_var): posterior predictive mean and variance.
        """
        x_tensor = torch.from_numpy(np.asarray(x_test)).float().to(self.device)

        def _fwd(weights):
            with torch.no_grad():
                self.network_weights = weights
                return self.net(x_tensor).detach().cpu().numpy()

        predictions = np.array([_fwd(w) for w in self.sampled_weights])
        pred_mean = np.mean(predictions, axis=0)
        pred_var = np.var(predictions, axis=0)
        return pred_mean, pred_var

    def _predict_pairs_batched(self, x_1, x_2, am_1, am_2, T, batch_size=256):
        """Mini-batched preference-pair prediction using posterior predictive mean."""
        N = am_1.shape[0]
        parts_1, parts_2 = [], []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b = end - start
            x1_b = x_1[start * T : end * T]
            x2_b = x_2[start * T : end * T]
            x_both = np.concatenate([x1_b, x2_b], axis=0)

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

    def _eval_current_weights(self, x, y, max_pairs=512):
        """Evaluate NLL and accuracy using the current (single) network weights.

        Called periodically during warm-up burn-in to give a live convergence
        signal.  Uses one forward pass at the current weight point, not the
        posterior predictive (no samples exist yet during burn-in).

        Args:
            x: numpy (N, 2, T, d_dim) preference pairs.
            y: numpy (N,) labels.
            max_pairs: subsample if N > max_pairs, for speed.

        Returns:
            (nll, acc): float NLL and accuracy.
        """
        if x.shape[0] > max_pairs:
            idx = np.random.choice(x.shape[0], max_pairs, replace=False)
            x, y = x[idx], y[idx]

        B, _, T, d_dim = x.shape
        obs_dim = d_dim - 1
        am_1 = x[:, 0, :, obs_dim].astype(np.float32)
        am_2 = x[:, 1, :, obs_dim].astype(np.float32)
        x_1 = x[:, 0, :, :obs_dim].reshape(-1, obs_dim).astype(np.float32)
        x_2 = x[:, 1, :, :obs_dim].reshape(-1, obs_dim).astype(np.float32)

        self.net.eval()
        with torch.no_grad():
            x1_t = torch.from_numpy(x_1).to(self.device)
            x2_t = torch.from_numpy(x_2).to(self.device)
            am1_t = torch.from_numpy(am_1).to(self.device)
            am2_t = torch.from_numpy(am_2).to(self.device)
            y_t = torch.from_numpy(y.squeeze().astype(np.float32)).to(self.device)

            pred_both = self.net(torch.cat([x1_t, x2_t], dim=0)).view(2, B, T)
            pred_1 = pred_both[0] * am1_t
            pred_2 = pred_both[1] * am2_t
            sum_1 = torch.nansum(pred_1, dim=1).view(-1, 1)
            sum_2 = torch.nansum(pred_2, dim=1).view(-1, 1)
            fx = torch.cat([sum_1, sum_2], dim=1)

            nll = torch.nn.CrossEntropyLoss()(fx, y_t).item()
            acc = float(accuracy(fx, y_t).detach().cpu())

        self.net.train()
        return nll, acc

    def eval_test_data(self, x, y, eval_batch_size=256):
        """Evaluate using the posterior predictive mean over sampled weights.

        Args:
            x: numpy (N, 2, T, d_dim) preference pairs.
            y: numpy (N,) labels.
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

        sum_1, sum_2 = self._predict_pairs_batched(
            x_1, x_2, am_1, am_2, T,
            batch_size=eval_batch_size,
        )
        ce, acc = self._ce_and_acc(sum_1, sum_2, y)
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
        use_cyclical_lr=False,
        lr_max=None,
        cycle_length=None,
        fraction_cool=0.25,
        max_param_step=None,
        log_every=0,
        eval_data=None,
    ):
        """Run the fSGHMC training loop.

        Args:
            x_train: numpy (N, 2, T, d_dim) training inputs (or None if
                ``data_loader`` is provided).
            y_train: numpy (N,) training labels.
            data_loader: optional DataLoader (used instead of x_train/y_train).
            num_samples: number of posterior weight samples to collect.
                Pass ``None`` for burn-in only (no weights collected).
            keep_every: collect one sample every this many post-burn-in steps
                (ignored when use_cyclical_lr=True).
            n_discarded: discard the first n_discarded samples after burn-in.
            num_burn_in_steps: number of AdaptiveSGHMC burn-in steps.
            lr: base learning rate (also lr_min when use_cyclical_lr=True).
            batch_size: mini-batch size.
            epsilon: AdaptiveSGHMC numerical stabiliser.
            mdecay: momentum decay coefficient.
            print_every_n_samples: (accepted, currently informational only).
            continue_training: if True, skip sampler re-initialisation.
            clear_sampled_weights: if True (default), clear sampled_weights
                before starting.
            use_cyclical_lr: enable cosine cyclical step-size schedule.
            lr_max: peak learning rate for the cyclical schedule.
            cycle_length: number of steps per cycle.
            fraction_cool: (unused; kept for signature compatibility).
            max_param_step: optional per-element momentum clamp.
            log_every: if > 0 and ``eval_data`` is provided, evaluate NLL and
                accuracy every this many steps and log to stdout + wandb.
                Intended for warm-up monitoring (uses current weights, not
                posterior predictive, since no samples exist during burn-in).
            eval_data: optional ``(X_eval, y_eval)`` numpy tuple used for
                periodic evaluation when ``log_every`` > 0.
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
            # n_meas_actual == 0 means no measurement points: the functional GP
            # prior is dropped entirely and this reduces to pure-likelihood
            # SGHMC.  Skip the solve (an empty measurement set has no prior
            # gradient, and the Woodbury nugget is undefined for n_M = 0).
            if n_meas_actual > 0:
                # Sample n_meas points from the measurement pool
                meas_idx = np.random.choice(
                    len(self._x_meas), n_meas_actual, replace=False
                )
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

            # ---- Periodic evaluation (warm-up monitoring) ---------------
            if log_every > 0 and eval_data is not None and (step + 1) % log_every == 0:
                _nll, _acc = self._eval_current_weights(eval_data[0], eval_data[1])
                self.print_info(
                    f"[{self.name}] step {step + 1:5d}  "
                    f"warmup/nll={_nll:.4f}  warmup/acc={_acc:.4f}"
                )
                try:
                    import wandb as _wandb
                    if _wandb.run is not None:
                        _wandb.log({
                            "warmup/nll": _nll,
                            "warmup/acc": _acc,
                            "warmup/step": step + 1,
                        })
                except Exception:
                    pass

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
        seed=1,
        initial_weights=None,
        use_cyclical_lr=False,
        lr_max=None,
        cycle_length=None,
        fraction_cool=0.25,
        max_param_step=None,
        chains_per_gpu=1,
    ):
        """Run multiple fSGHMC chains in parallel, packing chains onto GPUs.

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
            chains_per_gpu: int >= 1, how many chains to co-locate on each GPU.
                Chains pack greedily: the first ``chains_per_gpu`` chains share
                cuda:0, the next share cuda:1, and so on, so a run uses only
                ``ceil(num_chains / chains_per_gpu)`` GPUs (capped at the number
                available).  The chains' tiny MLP + n_meas×n_meas kernel leave an
                A6000 far from memory-bound; co-located chains stay statistically
                independent (separate processes, seeds, RNG, ckpt dirs) and only
                share compute.  Default 1 reproduces one-chain-per-GPU behaviour.
            (remaining args forwarded verbatim to FPrefNet.train())
        """
        import torch.multiprocessing as mp

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError(
                "sample_multi_chains_parallel requires at least one CUDA device."
            )
        if chains_per_gpu < 1:
            raise ValueError(
                f"chains_per_gpu must be >= 1, got {chains_per_gpu}."
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
            use_cyclical_lr=use_cyclical_lr,
            lr_max=lr_max,
            cycle_length=cycle_length,
            fraction_cool=fraction_cool,
            max_param_step=max_param_step,
        )

        # Up to this many chains run concurrently per wave: chains_per_gpu on
        # each of the available GPUs.  Within a wave, the worker maps its rank to
        # device rank // chains_per_gpu, so chains pack onto the lowest GPU
        # indices first (cuda:2+ stay idle when fewer are needed).
        max_concurrent = num_gpus * chains_per_gpu
        for batch_start in range(0, num_chains, max_concurrent):
            n_parallel = min(max_concurrent, num_chains - batch_start)
            n_gpus_used = math.ceil(n_parallel / chains_per_gpu)
            self.print_info(
                "Launching fSGHMC chains {:d}–{:d} ({:d} chains) on {:d} GPU(s), "
                "{:d} chain(s)/GPU".format(
                    batch_start,
                    batch_start + n_parallel - 1,
                    n_parallel,
                    n_gpus_used,
                    chains_per_gpu,
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
                    chains_per_gpu,
                ),
                nprocs=n_parallel,
                join=True,
            )
