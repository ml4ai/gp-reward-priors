#!/usr/bin/env python
# coding: utf-8
"""run_bnn_full_training_f.py — domain-agnostic scale-adapted cyclical fSGHMC,
trained on the full preference dataset (no held-out split).

Identical to run_bnn_training_f.py except that the entire dataset is used for
posterior sampling.  There is no training_split config field.  Post-sampling
diagnostics (CE, accuracy, R-hat, ESS) are computed on the training data
itself; they measure fit and MCMC convergence, not held-out generalisation.

Key differences from run_bnn_training_f.py
-------------------------------------------
1. No ``training_split`` config field — the full dataset is always used.
2. ``load_pref_data`` is called with ``training_ratio=1.0``, which returns
   ``(X, y)`` rather than the usual ``(X_train, y_train, X_test, y_test)``.
3. Post-sampling CE / accuracy are labelled ``train_*`` in wandb to make
   clear they are in-sample metrics.
"""

import math
import os
import os.path as osp
import sys
import uuid
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import matplotlib as mpl
import pyrallis

mpl.use("Agg")
import arviz_stats as azs
import numpy as np
import torch
import wandb

warnings.simplefilter("ignore", UserWarning)

sys.path.insert(0, os.path.abspath(".."))
os.chdir("..")

import optbnn.gp.reward_functions as _reward_fns
from optbnn.bnn.likelihoods import LikCE
from optbnn.bnn.nets.mlp import MLP
from optbnn.gp.models.model import LCFModel
from optbnn.sgmcmc_bayes_net.f_pref_net import FPrefNet
from optbnn.utils import util
from optbnn.utils.util import load_measurement_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    # wandb params
    project: str = "BNN-training"
    group: str = "fSGHMC"
    name: str = "run"
    # model params
    width: int = 64
    depth: int = 3
    # SGHMC Hyper-parameters
    batch_size: int = 256
    num_samples: int = 50
    n_discarded: int = 10
    num_burn_in_steps: int = 3000
    keep_every: int = 2000
    sghmc_lr: float = 0.008
    num_chains: int = 4
    mdecay: float = 0.01
    print_every_n_samples: int = 5
    # Cyclical step-size schedule (Zhang et al. 2020)
    use_cyclical_lr: bool = True
    sghmc_lr_max: float = 0.03
    cycle_length: int = 1000
    fraction_cool: float = 0.25
    # Safety clamp on per-element momentum (see bb_optim_star.py for details)
    max_param_step: Optional[float] = 0.5
    # Full preference dataset (no train/test split)
    dataset: str = "data/pref.hdf5"
    dataset_id: str = "run"
    # Measurement dataset for the fSGHMC functional GP prior.
    # Must be an HDF5 file with keys:
    #   "obs"     — (N, obs_dim)  required.  BNN inputs (state + action concatenated).
    #   "aux_obs" — (N, K)        optional.  Auxiliary GP feature inputs.
    # The presence of "aux_obs" is detected automatically by load_measurement_data().
    measurement_dataset: str = "data/meas.hdf5"
    # Number of measurement points sampled per training step from the pool.
    # Wu et al. (2025) use M = 100.
    n_meas: int = 256
    # Diagonal jitter added to K_{X_M} before the Cholesky solve.
    meas_jitter: float = 1e-6
    # Name of a module-level function in optbnn/gp/reward_functions.py.
    # The function must have signature f(X, device) or f(X, aux_X, device)
    # and return a (n, n_concepts) double tensor.
    reward_function: str = "bb_reward_prior"
    # GP feature dimension.  When None (default), inferred automatically by
    # calling reward_function on a 1-row dummy input before training starts.
    n_concepts: Optional[int] = None
    # GP prior covariance = gp_cov_scale * I_{n_concepts}.
    # Controls prior variance on reward-function coefficients.
    gp_cov_scale: float = 1.0
    # Warm-up monitoring: log NLL and accuracy every this many steps.
    # 0 = disabled.  Set to e.g. 100 to get a live convergence curve during
    # burn-in.  Evaluation uses a random 512-pair subsample of the training set
    # (the only data available when there is no held-out split).
    warmup_log_every: int = 0
    # general params
    seed: int = 1
    OUT_DIR: Optional[str] = "./exp/reward_learning/bnn_full_training_f"

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.OUT_DIR is not None:
            self.OUT_DIR = os.path.join(osp.expanduser(self.OUT_DIR), self.name)
            util.ensure_dir(self.OUT_DIR)


@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=f"{config.name}_bnn_full_training_f",
        id=str(uuid.uuid4()),
        save_code=True,
    )

    if config.OUT_DIR is not None:
        with open(os.path.join(config.OUT_DIR, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    util.set_seed(config.seed)

    width = config.width
    depth = config.depth
    transfer_fn = "relu"

    # ------------------------------------------------------------------ #
    # Resolve reward source function
    # ------------------------------------------------------------------ #
    if not hasattr(_reward_fns, config.reward_function):
        raise ValueError(
            f"reward_function={config.reward_function!r} not found in "
            "optbnn/gp/reward_functions.py.  "
            f"Available: {[n for n in dir(_reward_fns) if not n.startswith('_')]}"
        )
    function_vect = getattr(_reward_fns, config.reward_function)
    print(f"[GP prior] reward_function = {config.reward_function!r}")

    # ------------------------------------------------------------------ #
    # Load full preference dataset (no train/test split)
    # ------------------------------------------------------------------ #
    # load_pref_data with training_ratio=1.0 returns (X, y) — no split.
    X_train, y_train = util.load_pref_data(config.dataset, training_ratio=1.0)

    _n_nan_X = int(np.isnan(X_train).sum())
    _n_inf_X = int(np.isinf(X_train).sum())
    _n_nan_y = int(np.isnan(y_train).sum())
    _n_inf_y = int(np.isinf(y_train).sum())
    if _n_nan_X or _n_inf_X:
        raise ValueError(
            f"X: {_n_nan_X} NaN and {_n_inf_X} Inf values detected.  "
            "All input features must be finite."
        )
    if _n_nan_y or _n_inf_y:
        raise ValueError(
            f"y: {_n_nan_y} NaN and {_n_inf_y} Inf values detected.  "
            "Labels must be finite class indices."
        )
    print(f"[data] full dataset: {X_train.shape[0]} pairs — all values finite ✓")

    # X_train has shape (N, 2, T, d_dim); the last column of d_dim is the
    # attention mask, so obs_dim = state_dim + action_dim = d_dim - 1.
    input_dim = X_train.shape[-1] - 1
    print(f"[model] inferred input_dim = {input_dim}")

    # ------------------------------------------------------------------ #
    # Load measurement dataset (separate HDF5, raw observations)
    # ------------------------------------------------------------------ #
    print(f"[fSGHMC] Loading measurement dataset: {config.measurement_dataset}")
    x_meas, aux_meas = load_measurement_data(config.measurement_dataset)
    _aux_dim_str = str(aux_meas.shape[1]) if aux_meas is not None else "none"
    print(
        f"[fSGHMC] Measurement pool: {x_meas.shape[0]} observations "
        f"(obs_dim={x_meas.shape[1]}, aux_dim={_aux_dim_str}, "
        f"n_meas per step={config.n_meas})"
    )
    if x_meas.shape[0] < config.n_meas:
        warnings.warn(
            f"Measurement pool ({x_meas.shape[0]}) is smaller than n_meas "
            f"({config.n_meas}).  All pool points will be used every step.",
            RuntimeWarning,
        )
    if np.isnan(x_meas).any() or np.isinf(x_meas).any():
        raise ValueError(
            "Measurement dataset contains NaN or Inf values.  "
            "Check the HDF5 file and the load_measurement_data keys."
        )

    # ------------------------------------------------------------------ #
    # Infer n_concepts from a dummy forward pass through the source function
    # ------------------------------------------------------------------ #
    if config.n_concepts is not None:
        n_concepts = config.n_concepts
        print(f"[GP prior] n_concepts = {n_concepts} (from config)")
    else:
        with torch.no_grad():
            _dummy_X = torch.zeros(1, input_dim, device=device, dtype=torch.float64)
            if aux_meas is not None:
                _dummy_aux = torch.zeros(
                    1, aux_meas.shape[1], device=device, dtype=torch.float64
                )
                _phi_dummy = function_vect(_dummy_X, _dummy_aux, device)
            else:
                _phi_dummy = function_vect(_dummy_X, device)
        n_concepts = int(_phi_dummy.shape[-1])
        print(f"[GP prior] inferred n_concepts = {n_concepts}")

    # ------------------------------------------------------------------ #
    # Build the GP functional prior (LCFModel + selected source function)
    # ------------------------------------------------------------------ #
    # p_covariance = gp_cov_scale * I_{n_concepts} — isotropic GP weight prior.
    # p_mean = ones(n_concepts) — unit prior mean on reward-function coefficients.
    # gp_prior_args must survive pickle across mp.spawn:
    #   numpy arrays are picklable; function_vect is a module-level fn.
    p_covariance = np.eye(n_concepts, dtype=np.float32) * config.gp_cov_scale
    p_mean = np.ones(n_concepts, dtype=np.float32)
    gp_prior_args = {
        "p_covariance": p_covariance,
        "function_vect": function_vect,
        "p_mean": p_mean,
    }

    meas_kwargs = {
        "x_meas": x_meas,
        "aux_meas": aux_meas,
        "n_meas": config.n_meas,
        "meas_jitter": config.meas_jitter,
    }

    # Parent-process LCFModel (used only during warm-up; workers reconstruct
    # their own from gp_prior_args)
    gp_prior = LCFModel(
        p_covariance=p_covariance,
        function_vect=function_vect,
        device=device,
        p_mean=p_mean,
    ).to(device)

    # ------------------------------------------------------------------ #
    # Build BNN and FPrefNet (no OptimGaussianPrior needed)
    # ------------------------------------------------------------------ #
    util.set_seed(config.seed)
    net_args = dict(
        input_dim=input_dim,
        output_dim=1,
        hidden_dims=[width] * depth,
        activation_fn=transfer_fn,
    )
    net = MLP(**net_args)
    likelihood = LikCE()

    saved_dir = os.path.abspath(os.path.join(config.OUT_DIR, "sampling_f"))
    util.ensure_dir(saved_dir)

    bayes_net_f = FPrefNet(
        net=net,
        likelihood=likelihood,
        ckpt_dir=saved_dir,
        gp_prior=gp_prior,
        x_meas=x_meas,
        aux_meas=aux_meas,
        n_meas=config.n_meas,
        meas_jitter=config.meas_jitter,
        n_gpu=1,
        name="bnn_f",
    )

    # ------------------------------------------------------------------ #
    # Warm-up burn-in — shared starting point for all chains
    # ------------------------------------------------------------------ #
    # Warm-up runs fSGHMC so the starting point already reflects the GP prior.
    # When warmup_log_every > 0, NLL and accuracy are evaluated every
    # warmup_log_every steps on a 512-pair subsample of the training set and
    # logged to stdout + wandb under the "warmup/" prefix.
    util.set_seed(config.seed)
    bayes_net_f.train(
        X_train,
        y_train,
        num_samples=None,  # burn-in only; no weights collected
        num_burn_in_steps=config.num_burn_in_steps,
        lr=config.sghmc_lr,
        mdecay=config.mdecay,
        batch_size=config.batch_size,
        max_param_step=config.max_param_step,
        log_every=config.warmup_log_every,
        eval_data=(X_train, y_train) if config.warmup_log_every > 0 else None,
    )

    # Sanity-check warm-up weight magnitudes
    _w_norms = np.array([float(p.norm()) for p in bayes_net_f.net.parameters()])
    _total_norm = float(np.sqrt(np.sum(_w_norms**2)))
    _n_params = sum(p.numel() for p in bayes_net_f.net.parameters())
    _avg_weight_mag = _total_norm / math.sqrt(_n_params)
    print(f"[warm-up] weight L2 norms per layer: {[f'{n:.4f}' for n in _w_norms]}")
    print(
        f"[warm-up] total weight L2 norm: {_total_norm:.4f}  "
        f"(avg |w| = {_avg_weight_mag:.4f} over {_n_params} params)"
    )
    if _total_norm < 0.1:
        warnings.warn(
            f"Warm-up weight norm is very small ({_total_norm:.4e}).  "
            "Consider increasing num_burn_in_steps or adjusting gp_cov_scale.",
            RuntimeWarning,
        )
    elif _avg_weight_mag > 5.0:
        warnings.warn(
            f"Warm-up average weight magnitude is large ({_avg_weight_mag:.2f}).  "
            f"For a {width}×{depth} MLP, network outputs scale as w^{depth + 1}; "
            f"at avg |w|={_avg_weight_mag:.1f} reward logits may reach "
            f"O({_avg_weight_mag ** (depth + 1):.1e}), causing astronomical CE.  "
            "Consider reducing sghmc_lr / sghmc_lr_max or increasing mdecay.",
            RuntimeWarning,
        )
    wandb.log(
        {
            "warmup_total_weight_norm": _total_norm,
            "warmup_avg_weight_mag": _avg_weight_mag,
        }
    )

    initial_weights = bayes_net_f.network_weights

    # ------------------------------------------------------------------ #
    # Parallel chain sampling (fSGHMC)
    # ------------------------------------------------------------------ #
    bayes_net_f.sample_multi_chains_parallel(
        X_train,
        y_train,
        net_args=net_args,
        gp_prior_args=gp_prior_args,
        meas_kwargs=meas_kwargs,
        num_chains=config.num_chains,
        seed=config.seed,
        batch_size=config.batch_size,
        num_samples=config.num_samples,
        n_discarded=config.n_discarded,
        num_burn_in_steps=config.num_burn_in_steps,
        keep_every=config.keep_every,
        lr=config.sghmc_lr,
        mdecay=config.mdecay,
        print_every_n_samples=config.print_every_n_samples,
        initial_weights=initial_weights,
        use_cyclical_lr=config.use_cyclical_lr,
        lr_max=config.sghmc_lr_max,
        cycle_length=config.cycle_length,
        fraction_cool=config.fraction_cool,
        max_param_step=config.max_param_step,
    )

    # ------------------------------------------------------------------ #
    # Evaluation — in-sample (no held-out test set)
    # ------------------------------------------------------------------ #
    # CE / accuracy are training-set metrics; R-hat and ESS measure MCMC
    # convergence and are independent of the train/test distinction.
    _B_rhat = min(64, X_train.shape[0])
    _obs_dim = X_train.shape[-1] - 1
    x_rhat = X_train[:_B_rhat, 0, :, :_obs_dim].reshape(-1, _obs_dim).astype(np.float32)
    x_rhat_t = torch.from_numpy(x_rhat).to(bayes_net_f.device)

    mean_ce = []
    mean_acc = []
    pred_chains = []
    params_chains = []

    for i in range(config.num_chains):
        chain_dir = os.path.join(saved_dir, f"chain_{i}")
        bayes_net_f.sampled_weights = bayes_net_f._load_sampled_weights(
            os.path.join(chain_dir, "sampled_weights", "sampled_weights_0000000")
        )
        n_loaded = len(bayes_net_f.sampled_weights)
        print(f"[chain {i}] loaded {n_loaded} samples (expected {config.num_samples})")
        if n_loaded < 2:
            warnings.warn(
                f"Chain {i} has only {n_loaded} sample(s) — R-hat and ESS will be NaN.  "
                "Check that the worker completed and num_samples > n_discarded.",
                RuntimeWarning,
            )

        if n_loaded >= 2:
            _diff = max(
                float(np.abs(a - b).max())
                for a, b in zip(
                    bayes_net_f.sampled_weights[0],
                    bayes_net_f.sampled_weights[1],
                )
            )
            print(f"[chain {i}] max |w[0] - w[1]| = {_diff:.3e}")
            if _diff < 1e-8:
                warnings.warn(
                    f"Chain {i}: first two samples are numerically identical "
                    f"(max diff {_diff:.2e}).  SGHMC may be stuck.  "
                    "Try increasing lr_max or gp_cov_scale.",
                    RuntimeWarning,
                )
            wandb.log({f"chain_{i}_sample_max_diff_w0_w1": _diff})

        # eval_test_data called on the full training set (in-sample)
        ce, acc = bayes_net_f.eval_test_data(X_train, y_train, eval_batch_size=4096)
        mean_ce.append(ce)
        mean_acc.append(acc)

        bayes_net_f.net.eval()
        with torch.no_grad():
            chain_preds = []
            for weights in bayes_net_f.sampled_weights:
                bayes_net_f.network_weights = weights
                pred = bayes_net_f.net(x_rhat_t).detach().cpu().numpy().ravel()
                chain_preds.append(pred)
        pred_chains.append(np.stack(chain_preds))

        params_chains.append(
            np.stack(
                [
                    np.hstack([arr.ravel() for arr in arrays])
                    for arrays in bayes_net_f.sampled_weights
                ]
            )
        )

    pred_chains = np.stack(pred_chains)
    params_chains = np.stack(params_chains)

    pred_within_chain_var = float(np.mean(pred_chains.var(axis=1)))
    param_within_chain_var = float(np.mean(params_chains.var(axis=1)))
    print(f"[diag] pred within-chain var  = {pred_within_chain_var:.4e}")
    print(f"[diag] param within-chain var = {param_within_chain_var:.4e}")

    rhats_pred = azs.rhat(pred_chains)
    rhats_param = azs.rhat(params_chains)

    total_samples = config.num_chains * config.num_samples
    ess_pred = azs.ess(pred_chains)
    ess_param = azs.ess(params_chains)

    def _pct_over(arr, threshold):
        arr = np.asarray(arr, dtype=float)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return float("nan")
        return float(np.mean(valid > threshold) * 100)

    summary = {
        "train_mean_cross_entropy": np.mean(mean_ce),
        "train_mean_accuracy": np.mean(mean_acc),
        "pred_within_chain_var": pred_within_chain_var,
        "param_within_chain_var": param_within_chain_var,
        "pred_rhat_max": float(np.nanmax(rhats_pred)),
        "pred_rhat_95th_pct": float(np.nanpercentile(rhats_pred, 95)),
        "pred_rhat_median": float(np.nanmedian(rhats_pred)),
        "pred_rhat_mean": float(np.nanmean(rhats_pred)),
        "pred_rhat_pct_over_1.01": _pct_over(rhats_pred, 1.01),
        "pred_ess_min": float(np.nanmin(ess_pred)),
        "pred_ess_median": float(np.nanmedian(ess_pred)),
        "pred_ess_mean": float(np.nanmean(ess_pred)),
        "pred_ess_min_norm": float(np.nanmin(ess_pred)) / total_samples,
        "pred_ess_median_norm": float(np.nanmedian(ess_pred)) / total_samples,
        "param_rhat_max": float(np.nanmax(rhats_param)),
        "param_rhat_95th_pct": float(np.nanpercentile(rhats_param, 95)),
        "param_rhat_median": float(np.nanmedian(rhats_param)),
        "param_rhat_mean": float(np.nanmean(rhats_param)),
        "param_rhat_pct_over_1.01": _pct_over(rhats_param, 1.01),
        "param_ess_min": float(np.nanmin(ess_param)),
        "param_ess_median": float(np.nanmedian(ess_param)),
        "param_ess_min_norm": float(np.nanmin(ess_param)) / total_samples,
    }
    wandb.log(summary)


if __name__ == "__main__":
    train()
