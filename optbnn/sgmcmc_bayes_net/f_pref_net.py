"""Functional SGHMC sampler for preference learning (fSGHMC, Wu et al. 2025).

FPrefNet is a *standalone* class — it does not inherit from BayesNet or
PrefNet and requires no weight-space prior (OptimGaussianPrior).  The only
regularisation comes from the functional GP prior (any module exposing the
``functional_prior_grad`` contract, e.g. MapInformedGPPrior or the legacy
LCFModel), whose gradient is injected at every sampler step via a single VJP
backward pass.

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
from ..utils.util import (
    bt_pool_logit,
    bt_pool_logit_np,
    ensure_dir,
    inf_loop,
    prepare_device,
)


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
    fpref_kwargs=None,
    chain_init_jitter=0.0,
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
            The key ``prior_type`` selects the prior class
            (``"map_informed"``, or the legacy ``"lcf"``; dicts lacking the
            key are treated as ``"lcf"`` for backward compatibility).
            For ``"map_informed"`` (MapInformedGPPrior) the keys are those
            produced by ``MapInformedGPPrior.to_args()`` (free_mask, scaling,
            offset, eta, sig_c2, sig_g2, sig_n2, xy_cols, xy_source).
            For the legacy ``"lcf"`` (LCFModel) the keys are:
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
        # Chain-start diversification: overdisperse each chain around the shared
        # warm-up point.  Shared starts under-estimate R-hat (chains are
        # artificially similar); overdispersed starts make R-hat a valid
        # convergence check and let the pooled chains cover more posterior modes.
        # Per-tensor relative scale (chain_init_jitter * std(w)), seeded per chain
        # by set_seed(seed + chain_idx) above.  0.0 -> identical shared start.
        if chain_init_jitter and chain_init_jitter > 0.0:
            with torch.no_grad():
                for param in net.parameters():
                    _sd = float(param.detach().std())
                    if _sd > 0.0:
                        param.add_(
                            torch.randn_like(param) * (chain_init_jitter * _sd)
                        )

    likelihood = LikCE()

    # Reconstruct the functional GP prior (MapInformedGPPrior, or the legacy
    # LCFModel).  The "lcf" fallback keeps old-style gp_prior_args dicts
    # (which predate the prior_type key and are LCF-shaped) working.
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
        **(fpref_kwargs or {}),
    )
    bayes_net.train(x_train, y_train, **train_kwargs)
    bayes_net._save_sampled_weights()
    # Persist grad-norm instrumentation (Issue 3, Step 1) for the main process
    # to aggregate — workers have no wandb run of their own.
    _stats = getattr(bayes_net, "_grad_norm_stats", None)
    if _stats is not None:
        torch.save(_stats, os.path.join(chain_dir, "grad_norm_stats.pt"))
    # Each chain runs its OWN 20k burn-in with tau/g/v_hat re-initialised, so
    # each freezes a DIFFERENT preconditioner and runs the whole sampling phase
    # at a different effective step eta_i = eps^2 * minv_i.  Discretisation bias
    # scales with eta, so the chains sit at slightly different effective
    # temperatures -- a persistent, scale-only, BETWEEN-chain difference that no
    # amount of burn-in or extra draws removes (independent review 2026-09-04,
    # section 9.4).  That is a candidate mechanism for scale_z failing while
    # loc_z passes, and unlike a jitter transient it is not transient at all.
    # Persisted here so the parent can log the per-chain SPREAD; workers run
    # with wandb disabled, so it was previously only in each chain's text log.
    _pre = getattr(bayes_net, "_precond_at_freeze", None)
    if _pre is not None:
        torch.save(_pre, os.path.join(chain_dir, "precond_at_freeze.pt"))


# ---------------------------------------------------------------------------
# FPrefNet — standalone functional SGHMC sampler
# ---------------------------------------------------------------------------

class FPrefNet:
    """Standalone functional SGHMC sampler for preference learning.

    Does **not** inherit from BayesNet or PrefNet and requires **no**
    weight-space prior (OptimGaussianPrior).  Regularisation comes entirely
    from the functional GP prior defined by ``gp_prior`` (any module exposing
    ``functional_prior_grad``, e.g. MapInformedGPPrior or the legacy
    LCFModel).

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
        gp_prior: functional GP prior module exposing
            ``functional_prior_grad(net, X_M, aux_X, jitter)`` and
            ``.to(device)`` (e.g. MapInformedGPPrior, or the legacy LCFModel).
        x_meas: numpy float32 (N_meas, obs_dim) — measurement-point pool.
            At each step ``n_meas`` rows are sampled uniformly without
            replacement and used for the VJP backward pass.
        aux_meas: numpy (N_meas, aux_dim) or None — auxiliary feature inputs
            passed as ``aux_X`` to the prior (e.g. raw states when the prior
            reads coordinates from a source other than the network inputs).
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
        bt_pool="mean",
        clip_grad_norm_value=100.0,
        clip_during_sampling=False,
    ):
        self.net = net
        self.lik_module = likelihood
        self.ckpt_dir = ckpt_dir
        self.sampling_method = sampling_method
        self.temperature = temperature
        self.name = name
        self.n_gpu = n_gpu
        # Bradley-Terry pooling: "mean" (masked mean over valid timesteps, shared
        # with MR/PT) or "sum" (legacy).  Grad-clip scope (Issue 3): apply the
        # clip in burn-in always, in sampling only if clip_during_sampling.
        self._bt_pool = bt_pool
        self._clip_grad_norm_value = clip_grad_norm_value
        self._clip_during_sampling = clip_during_sampling

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
        v_hat_min=None,
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
            # Floor on the variance estimate v_hat, which caps the
            # preconditioner gain at minv_t <= 1/sqrt(v_hat_min).  Exposed
            # because that cap is a candidate throttle on the sampling step
            # (handoff 4.3.71): at the 1e-4 default it is exactly 100, and
            # large_play sampled with 50.8% of elements pinned there.
            if v_hat_min is not None:
                self.sampler_params["v_hat_min"] = float(v_hat_min)
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
            parts_1.append(bt_pool_logit_np(pred_1, am_1[start:end], self._bt_pool))
            parts_2.append(bt_pool_logit_np(pred_2, am_2[start:end], self._bt_pool))

        return np.concatenate(parts_1), np.concatenate(parts_2)

    def _predict_pairs_per_draw(self, x_1, x_2, am_1, am_2, T, batch_size=256):
        """Per-draw pooled Bradley-Terry logits, shape [n_draws, N] per member.

        Same masking and pooling as ``_predict_pairs_batched``, but WITHOUT the
        average over sampled weights -- each draw is kept separate so the
        likelihood can be averaged in probability space (Eq. 10) rather than the
        function being averaged first.
        """
        N = am_1.shape[0]
        S = len(self.sampled_weights)
        out_1 = np.empty((S, N), dtype=np.float64)
        out_2 = np.empty((S, N), dtype=np.float64)
        x1_t = torch.from_numpy(np.asarray(x_1)).float().to(self.device)
        x2_t = torch.from_numpy(np.asarray(x_2)).float().to(self.device)
        self.net.eval()
        with torch.no_grad():
            for s, weights in enumerate(self.sampled_weights):
                self.network_weights = weights
                for start in range(0, N, batch_size):
                    end = min(start + batch_size, N)
                    b = end - start
                    p1 = self.net(x1_t[start * T:end * T]).cpu().numpy().reshape(b, T)
                    p2 = self.net(x2_t[start * T:end * T]).cpu().numpy().reshape(b, T)
                    out_1[s, start:end] = bt_pool_logit_np(
                        p1 * am_1[start:end], am_1[start:end], self._bt_pool)
                    out_2[s, start:end] = bt_pool_logit_np(
                        p2 * am_2[start:end], am_2[start:end], self._bt_pool)
        return out_1, out_2

    def eval_test_data_predictive(self, x, y, eval_batch_size=256):
        """Posterior-predictive CE and accuracy, per Wu et al. (2025) Eq. (10).

        Eq. (10) defines the predictive as an average of the LIKELIHOOD over
        function draws, ``p(y*|x*,D) ~= (1/S) sum_j p(y*|f(x*; w_j))``, i.e.
        ``E[sigma(f)]``.  ``eval_test_data`` instead averages the reward over
        draws and squashes once, ``sigma(E[f])`` -- a plug-in estimate that is
        blind to posterior width, since two posteriors with the same mean reward
        and very different spread score identically.  That matters here because
        the downstream quantity is CVaR_0.05, a functional of exactly that
        spread.

        Both are reported; this one is the theory-aligned quantity.

        Returns:
            (ce, acc): float cross-entropy and accuracy of the predictive.
        """
        B, _, T, d_dim = x.shape
        obs_dim = d_dim - 1
        am_1 = x[:, 0, :, obs_dim].astype(np.float32)
        am_2 = x[:, 1, :, obs_dim].astype(np.float32)
        x_1 = x[:, 0, :, :obs_dim].reshape(-1, obs_dim)
        x_2 = x[:, 1, :, :obs_dim].reshape(-1, obs_dim)

        s1, s2 = self._predict_pairs_per_draw(
            x_1, x_2, am_1, am_2, T, batch_size=eval_batch_size)
        # Stable sigmoid of the per-draw logit difference, then average the
        # PROBABILITIES over draws.
        d = s1 - s2
        p1 = np.exp(-np.logaddexp(0.0, -d))
        pbar = p1.mean(axis=0)

        yv = np.asarray(y, dtype=np.float64)
        eps = 1e-12
        ce = float(-(yv[:, 0] * np.log(pbar + eps)
                     + yv[:, 1] * np.log(1.0 - pbar + eps)).mean())
        acc = float(((pbar > 0.5) == (yv[:, 0] > 0.5)).mean())
        self.net.train()
        return ce, acc

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
            sum_1 = bt_pool_logit(pred_1, am1_t, self._bt_pool).view(-1, 1)
            sum_2 = bt_pool_logit(pred_2, am2_t, self._bt_pool).view(-1, 1)
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
        burn_in_lr=None,
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
        samples_per_cycle=1,
        resample_momentum=True,
        fix_meas_set=False,
        max_param_step=None,
        v_hat_min=None,
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
            burn_in_lr: optional fixed step size for the burn-in phase only.
                When None (default) burn-in inherits ``lr`` (= lr_min under the
                cyclical schedule) — the legacy behaviour.  When the cool-phase
                lr_min is swept small (sampling-tier schedules), a fixed-step
                burn-in at lr_min under-fits, so warm-up accuracy collapses and
                the warm-up gate rejects otherwise-good configs; set burn_in_lr
                to a value that fits (independent of lr_min) to decouple the two.
                Only applied when use_cyclical_lr=True.
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

        # samples_per_cycle > 1 harvests multiple thinned samples from each cool
        # phase (Issue 2), so num_samples (a TOTAL count) is reached in fewer
        # cycles -> fewer steps.  samples_per_cycle == 1 is the legacy one-sample-
        # per-cycle behaviour (num_steps unchanged).
        _spc = max(1, int(samples_per_cycle))
        if use_cyclical_lr and num_samples is not None:
            _n_cycles = math.ceil((num_samples + n_discarded) / _spc)
            num_steps = _n_cycles * _cycle_len
        else:
            num_steps = 0 if num_samples is None else (num_samples + 1) * keep_every

        # ---- Sampler initialisation -------------------------------------
        if not continue_training:
            if clear_sampled_weights:
                self.sampled_weights.clear()
            self.net = self.net.float()
            self._initialize_sampler(
                num_datapoints, lr, mdecay, num_burn_in_steps, epsilon,
                max_param_step=max_param_step, v_hat_min=v_hat_min,
            )
            num_steps += num_burn_in_steps

        # ---- Measurement-pool size check --------------------------------
        n_meas_actual = min(self._n_meas, len(self._x_meas))
        if n_meas_actual < self._n_meas:
            self.print_info(
                f"[fSGHMC] Measurement pool ({len(self._x_meas)}) smaller than "
                f"n_meas ({self._n_meas}); using all pool points every step."
            )

        # ---- Measurement set: resampled per step, or drawn once ---------
        # Resampling every step makes the functional-prior gradient STOCHASTIC,
        # and section 4.3.21 identified gradient noise -- not the thermostat --
        # as the uncorrected heat source inflating the sampled variance
        # (adaptive_sghmc.py:147's -lr^4 correction is ~1e-15 against a main
        # term of ~1e-8..1e-6, i.e. numerically inert).  The noise comes from
        # RESAMPLING, not from the subset size, so drawing the set once makes
        # this gradient exact at any n_meas -- far cheaper than enlarging it,
        # and standard for a GP functional prior, where the fixed set plays the
        # role of inducing points.  It also drops a per-step choice and a
        # host-to-device copy.
        #
        # The draw is seeded per chain by set_seed(seed + chain_idx) upstream,
        # so chains still get DIFFERENT fixed sets: the pooled prior is not
        # collapsed onto a single subset, only each chain's own gradient is
        # made deterministic.
        _fixed_meas = None
        if fix_meas_set and n_meas_actual > 0:
            _fi = np.random.choice(len(self._x_meas), n_meas_actual,
                                   replace=False)
            _fixed_meas = (
                torch.from_numpy(self._x_meas[_fi]).float().to(self.device),
                (torch.from_numpy(self._aux_meas[_fi]).to(self.device)
                 if self._aux_meas is not None else None),
            )
            self.print_info(
                f"[fSGHMC] Measurement set FIXED: {n_meas_actual} of "
                f"{len(self._x_meas)} pool points drawn once (chain-seeded); "
                f"the functional-prior gradient is deterministic."
            )

        # ---- Main loop --------------------------------------------------
        batch_generator = islice(enumerate(train_loader), num_steps)
        self.net.train()
        n_samples = 0
        # Best-by-NLL burn-in state, populated only when warm-up monitoring is
        # on (log_every > 0).  None means "no evaluation ran", and callers fall
        # back to the final state -- reproducing the pre-2026-08-24 behaviour.
        self._best_warmup = None
        # Preconditioner state captured at the burn-in/sampling boundary, where
        # adaptive_sghmc freezes tau/g/v_hat for the rest of the run.
        self._precond_at_freeze = None

        # Grad-norm instrumentation (Issue 3, Step 1): accumulate the PRE-clip
        # total gradient norm, split by phase, so we can tell whether the clip
        # actually fires during sampling (where it would bias the CVaR tail) or
        # only during burn-in (harmless).  Zero overhead — clip_grad_norm_
        # already returns this norm.  Behaviour of the clip itself is unchanged.
        self._grad_norm_stats = {
            phase: {"count": 0, "sum": 0.0, "max": 0.0, "n_over_clip": 0,
                    "clamp_hits": 0, "clamp_elems": 0}
            for phase in ("burnin", "sampling")
        }

        for step, (x_batch, y_batch) in batch_generator:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # ---- Learning-rate schedule ---------------------------------
            # Burn-in at a fixed step size, decoupled from the cool-phase lr_min
            # (= lr).  Applies in EITHER mode: the standalone warm-up uses
            # use_cyclical_lr=False, and its warm-up accuracy is what the gate
            # (early_stop_acc_threshold) reads, so the override must fire there
            # too.  The sampler was initialised at lr_min; without this, the
            # fixed-length burn-in inherits lr_min, and a small (sampling-tier)
            # lr_min under-fits so the gate rejects good configs.
            if step < num_burn_in_steps and burn_in_lr is not None:
                for _pg in self.sampler.param_groups:
                    _pg["lr"] = np.float32(burn_in_lr)
            elif (
                (not use_cyclical_lr)
                and step == num_burn_in_steps
                and burn_in_lr is not None
            ):
                # Non-cyclical sampling: burn-in ran at burn_in_lr; restore lr_min
                # once for the fixed-interval sampling phase (the cyclical branch
                # below sets the LR itself every post-burn-in step, so this restore
                # is only needed when the schedule is off).
                for _pg in self.sampler.param_groups:
                    _pg["lr"] = np.float32(_lr_min)
            if use_cyclical_lr and step >= num_burn_in_steps:
                _post_burn = step - num_burn_in_steps
                _cycle_step = _post_burn % _cycle_len
                _cycle_lr = _lr_min + 0.5 * (_lr_max - _lr_min) * (
                    1.0 + math.cos(math.pi * _cycle_step / _cycle_len)
                )
                for _pg in self.sampler.param_groups:
                    _pg["lr"] = np.float32(_cycle_lr)
                if _cycle_step == 0:
                    # Cycle start (Issue 1): resample momentum from its stationary
                    # Gaussian (Wu et al. 2025, Alg. 2) rather than zeroing it —
                    # zeroing drops the chain into an atypical set that costs
                    # ~1/mdecay steps to re-equilibrate.  The AdaptiveSGHMC
                    # "momentum" buffer is the position increment v ≈ εM⁻¹z, whose
                    # OU stationary law is v ~ N(0, lr²·minv_t), minv_t=1/(√v_hat+ε),
                    # with lr = lr_max at the cycle start (= _cycle_lr).  Plain
                    # SGHMC has no v_hat: std = √lr.  resample_momentum=False keeps
                    # the legacy zeroing for reproducibility.
                    for _pg in self.sampler.param_groups:
                        _eps = _pg.get("epsilon", 1e-16)
                        for _p in _pg["params"]:
                            _s = self.sampler.state.get(_p)
                            if _s is None or "momentum" not in _s:
                                continue
                            if not resample_momentum:
                                _s["momentum"].zero_()
                                continue
                            if "v_hat" in _s:  # adaptive_sghmc
                                _std = (
                                    _s["v_hat"].sqrt().add(_eps).reciprocal()
                                    .sqrt_().mul_(_cycle_lr)
                                )
                            else:              # plain sghmc
                                _std = math.sqrt(float(_cycle_lr))
                            _s["momentum"].normal_().mul_(_std)

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
            sum_pred_1 = bt_pool_logit(pred_1, am_1, self._bt_pool).view(-1, 1)
            sum_pred_2 = bt_pool_logit(pred_2, am_2, self._bt_pool).view(-1, 1)
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
                if _fixed_meas is not None:
                    x_meas_t, aux_meas_t = _fixed_meas
                else:
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
            # Grad-clip scope (Issue 3, Step 2): applied in burn-in always, in
            # sampling only if clip_during_sampling.  The pre-clip norm is always
            # measured for the instrumentation (Step 1), even when not applied
            # (max_norm=inf leaves gradients untouched but still returns the norm).
            _clip_value = self._clip_grad_norm_value
            _apply_clip = _clip_value is not None and (
                step < num_burn_in_steps or self._clip_during_sampling
            )
            _gnorm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(),
                    _clip_value if _apply_clip else float("inf"),
                )
            )
            _thr = _clip_value if _clip_value is not None else 100.0
            _phase = "burnin" if step < num_burn_in_steps else "sampling"
            _st = self._grad_norm_stats[_phase]
            _st["count"] += 1
            _st["sum"] += _gnorm
            if _gnorm > _st["max"]:
                _st["max"] = _gnorm
            if _gnorm > _thr:
                _st["n_over_clip"] += 1
            self.sampler.step()
            # Momentum-clamp activations for this step.  max_param_step is a
            # hard nonlinearity that breaks measure preservation whenever it
            # binds, so a selected run must show ~0 during sampling.
            _st["clamp_hits"] += int(getattr(self.sampler, "_clamp_hits", 0))
            _st["clamp_elems"] += int(getattr(self.sampler, "_clamp_elems", 0))
            self.step += 1

            # ---- Preconditioner freeze point (section 4.3.37) ------------
            # adaptive_sghmc adapts tau/g/v_hat only while
            # iteration <= num_burn_in_steps, so this is the last step on which
            # they change.  Capture what gets frozen for the whole sampling
            # phase; the caller logs it.
            if (self._precond_at_freeze is None
                    and num_burn_in_steps > 0
                    and step + 1 >= num_burn_in_steps
                    and hasattr(self.sampler, "preconditioner_snapshot")):
                self._precond_at_freeze = self.sampler.preconditioner_snapshot()
                _p = self._precond_at_freeze
                if _p:
                    self.print_info(
                        f"[precond] FROZEN at step {step + 1:,}: "
                        f"tau median {_p['precond_tau_median']:,.0f} "
                        f"({_p['precond_tau_over_burnin']:.3f} x burn-in -- "
                        f"{'DEGENERATE, window never saturated' if _p['precond_tau_over_burnin'] > 0.5 else 'saturated'}), "
                        f"v_hat median {_p['precond_v_hat_median']:.3e}, "
                        f"{_p['precond_v_hat_at_floor'] * 100:.1f}% at floor, "
                        f"minv median {_p['precond_minv_median']:.2f} "
                        f"(max {_p['precond_minv_max']:.1f})"
                    )

            # ---- Periodic evaluation (warm-up monitoring) ---------------
            if log_every > 0 and eval_data is not None and (step + 1) % log_every == 0:
                _nll, _acc = self._eval_current_weights(eval_data[0], eval_data[1])
                # Section 4.3.36: the warm-up does not converge monotonically --
                # large_play VISITED NLL 0.172 and handed off 0.426, 2.5x worse.
                # Handing chains the LAST state throws that away for no reason
                # the spec gives.  Keep the best-by-NLL state instead; the
                # evaluation already runs, so this costs one state copy.
                # NLL, not accuracy: accuracy is quantized on 54-107 pairs
                # (section 4.3.34) and section 3.5 already rejected gating on it.
                if step < num_burn_in_steps and (
                    self._best_warmup is None or _nll < self._best_warmup[0]):
                    self._best_warmup = (float(_nll), float(_acc), step + 1,
                                         self.network_weights)
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
                # Cool-phase harvesting (Issue 2): collect _spc thinned samples
                # from the low-LR tail of the cycle, spaced to end at the coldest
                # step (_from_end == 0).  _spc == 1 -> only the coldest sample,
                # i.e. exactly the legacy one-sample-per-cycle behaviour.
                _cool_len = max(1, int(fraction_cool * _cycle_len))
                _thin = max(1, _cool_len // _spc)
                _from_end = (_cycle_len - 1) - _cycle_step
                if 0 <= _from_end < _spc * _thin and _from_end % _thin == 0:
                    n_samples += 1
                    if n_samples > n_discarded and self.num_samples < num_samples:
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
        burn_in_lr=None,
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
        samples_per_cycle=1,
        resample_momentum=True,
        fix_meas_set=False,
        max_param_step=None,
        v_hat_min=None,
        chains_per_gpu=1,
        bt_pool="mean",
        clip_grad_norm_value=100.0,
        clip_during_sampling=False,
        chain_init_jitter=0.0,
    ):
        """Run multiple fSGHMC chains in parallel, packing chains onto GPUs.

        Sampled weights for chain i are written to::

            <self.ckpt_dir>/chain_<i>/sampled_weights/sampled_weights_0000000

        Args:
            x_train: numpy (N, 2, T, d_dim) training inputs.
            y_train: numpy (N,) training targets.
            net_args: dict of kwargs for MLP(**net_args).
            gp_prior_args: dict describing the prior to reconstruct in each
                worker, keyed by ``prior_type`` (see _fpref_chain_worker
                docstring for the per-type keys).
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
            burn_in_lr=burn_in_lr,
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
            samples_per_cycle=samples_per_cycle,
            resample_momentum=resample_momentum,
            fix_meas_set=fix_meas_set,
            max_param_step=max_param_step,
            v_hat_min=v_hat_min,
        )

        # Up to this many chains run concurrently per wave: chains_per_gpu on
        # each of the available GPUs.  Within a wave, the worker maps its rank to
        # device rank // chains_per_gpu, so chains pack onto the lowest GPU
        # indices first (cuda:2+ stay idle when fewer are needed).
        # FPrefNet construction params must reach each worker (workers build
        # their own FPrefNet), else worker sampling would silently use defaults.
        _fpref_kwargs = dict(
            bt_pool=bt_pool,
            clip_grad_norm_value=clip_grad_norm_value,
            clip_during_sampling=clip_during_sampling,
        )
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
                    _fpref_kwargs,
                    chain_init_jitter,
                ),
                nprocs=n_parallel,
                join=True,
            )
