#!/usr/bin/env python
# coding: utf-8
"""bb_optim_f.py — scale-adapted cyclical fSGHMC for the BB preference task.

Identical to bb_optim_star.py in structure and evaluation, but replaces the
weight-space Gaussian prior with a functional GP prior built from LCFModel
and bb_reward_prior (Wu et al. 2025 fSGHMC).

Key differences from bb_optim_star.py
--------------------------------------
1. Imports FPrefNet (f_pref_net.py) instead of PrefNet.
2. Loads a separate measurement dataset (HDF5) whose raw observations serve as
   the inducing/measurement points for the GP prior gradient.
3. Constructs an LCFModel from bb_reward_prior + identity covariance.
4. Passes gp_prior_args + meas_kwargs to FPrefNet.sample_multi_chains_parallel.

All SGHMC hyper-parameters, warm-up logic, R-hat / ESS diagnostics, and
wandb logging are unchanged.
"""

import math
import os
import os.path as osp
import sys
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import pyrallis

mpl.use("Agg")
import arviz_stats as azs
import h5py
import matplotlib.pylab as plt
import numpy as np
import torch
import wandb

warnings.simplefilter("ignore", UserWarning)

sys.path.insert(0, os.path.abspath(".."))
os.chdir("..")

from optbnn.bnn.likelihoods import LikCE
from optbnn.bnn.nets.mlp import MLP
from optbnn.bnn.priors import OptimGaussianPrior
from optbnn.gp.models.model import LCFModel
from optbnn.gp.reward_functions import bb_reward_prior
from optbnn.sgmcmc_bayes_net.f_pref_net import FPrefNet
from optbnn.utils import util
from optbnn.utils.util import load_measurement_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    # wandb params
    project: str = "BB-training"
    group: str = "BB"
    name: str = "bb"
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
    # max_param_step: element-wise momentum clamp (safety net against
    # catastrophic updates).  See bb_optim_star.py for full comment.
    max_param_step: Optional[float] = 0.5
    # Preference training dataset
    dataset: str = "data/bb/t0012_pref.hdf5"
    dataset_id: str = "bb_t0012"
    training_split: float = 0.8
    # Measurement dataset for fSGHMC functional GP prior
    # Must be an HDF5 file with "states" (and optionally "actions") keys.
    # Raw observations are used as inducing/measurement points at every step.
    measurement_dataset: str = "data/bb/t0012_meas.hdf5"
    # Number of measurement points sampled per training step.  Larger values
    # give a more accurate functional prior gradient but increase GPU memory
    # and compute.  Wu et al. (2025) use M = 100.
    n_meas: int = 100
    # Diagonal jitter added to K_{X_M} before the Cholesky solve.
    meas_jitter: float = 1e-6
    # GP prior covariance = gp_cov_scale * I_{n_concepts}.
    # The GP prior weights each feature linearly; scaling by gp_cov_scale
    # controls the prior variance on the reward function.
    gp_cov_scale: float = 1.0
    # general params
    seed: int = 1
    OUT_DIR: Optional[str] = "./exp/reward_learning/bb_optim_f"
    prior_dir: str = "./exp/reward_learning/bb_tuning_star"

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.OUT_DIR is not None:
            self.OUT_DIR = os.path.join(osp.expanduser(self.OUT_DIR), self.name)
            util.ensure_dir(self.OUT_DIR)
        self.prior_dir = os.path.join(
            osp.expanduser(self.prior_dir),
            f"bb-{self.width}_{self.depth}",
        )


@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=f"{config.name}_optim_f_training",
        id=str(uuid.uuid4()),
        save_code=True,
    )
    util.set_seed(config.seed)

    width = config.width
    depth = config.depth
    transfer_fn = "relu"

    sampling_configs = {
        "batch_size": config.batch_size,
        "num_samples": config.num_samples,
        "n_discarded": config.n_discarded,
        "num_burn_in_steps": config.num_burn_in_steps,
        "keep_every": config.keep_every,
        "lr": config.sghmc_lr,
        "num_chains": config.num_chains,
        "mdecay": config.mdecay,
        "print_every_n_samples": config.print_every_n_samples,
    }

    # ------------------------------------------------------------------ #
    # Load preference training / test data
    # ------------------------------------------------------------------ #
    X_train, y_train, X_test, y_test = util.load_pref_data(
        config.dataset, config.training_split
    )

    for _split, _X, _y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        _n_nan_X = int(np.isnan(_X).sum())
        _n_inf_X = int(np.isinf(_X).sum())
        _n_nan_y = int(np.isnan(_y).sum())
        _n_inf_y = int(np.isinf(_y).sum())
        if _n_nan_X or _n_inf_X:
            raise ValueError(
                f"X_{_split}: {_n_nan_X} NaN and {_n_inf_X} Inf values detected.  "
                "All input features must be finite."
            )
        if _n_nan_y or _n_inf_y:
            raise ValueError(
                f"y_{_split}: {_n_nan_y} NaN and {_n_inf_y} Inf values detected.  "
                "Labels must be finite class indices."
            )
        print(f"[data] {_split}: {_X.shape[0]} pairs — all values finite ✓")

    # ------------------------------------------------------------------ #
    # Load measurement dataset (separate HDF5, raw observations)
    # ------------------------------------------------------------------ #
    print(f"[fSGHMC] Loading measurement dataset: {config.measurement_dataset}")
    x_meas, aux_meas = load_measurement_data(config.measurement_dataset)
    print(
        f"[fSGHMC] Measurement pool: {x_meas.shape[0]} observations  "
        f"(obs_dim={x_meas.shape[1]}, state_dim={aux_meas.shape[1]})  "
        f"n_meas per step={config.n_meas}"
    )
    if x_meas.shape[0] < config.n_meas:
        warnings.warn(
            f"Measurement pool ({x_meas.shape[0]}) is smaller than n_meas "
            f"({config.n_meas}).  All pool points will be used every step.",
            RuntimeWarning,
        )
    # Finite check on measurement data
    if np.isnan(x_meas).any() or np.isinf(x_meas).any():
        raise ValueError(
            "Measurement dataset contains NaN or Inf values.  "
            "Check the HDF5 file and the load_measurement_data keys."
        )

    # ------------------------------------------------------------------ #
    # Build the GP functional prior (LCFModel + bb_reward_prior)
    # ------------------------------------------------------------------ #
    # bb_reward_prior maps (X, aux_X, device) → (n, 3) feature matrix
    # [intercept, -goal_distance, min_obs_dist].
    # p_covariance = gp_cov_scale * I_3 gives an isotropic GP prior.
    n_concepts = 3  # number of features returned by bb_reward_prior
    p_covariance = np.eye(n_concepts, dtype=np.float32) * config.gp_cov_scale

    # gp_prior_args must be pickle-safe (all numpy arrays + module-level fn)
    gp_prior_args = {
        "p_covariance": p_covariance,   # (3, 3) numpy
        "function_vect": bb_reward_prior,  # module-level → picklable
        "p_mean": None,                 # zeros (default in LCFModel)
    }

    # LCFModel for the parent process (warm-up diagnostic, not for training)
    bb_prior = LCFModel(
        p_covariance=p_covariance,
        function_vect=bb_reward_prior,
        device=device,
    ).to(device)

    meas_kwargs = {
        "x_meas": x_meas,
        "aux_meas": aux_meas,
        "n_meas": config.n_meas,
        "meas_jitter": config.meas_jitter,
    }

    # ------------------------------------------------------------------ #
    # Initialize prior and BNN
    # ------------------------------------------------------------------ #
    util.set_seed(config.seed)
    ckpt_path = os.path.abspath(
        os.path.join(config.prior_dir, "ckpts", "best.ckpt")
    )
    prior = OptimGaussianPrior(ckpt_path)

    net_args = dict(
        input_dim=24,
        output_dim=1,
        hidden_dims=[width] * depth,
        activation_fn=transfer_fn,
    )
    net = MLP(**net_args)
    likelihood = LikCE()

    saved_dir = os.path.abspath(os.path.join(config.OUT_DIR, "sampling_f"))
    util.ensure_dir(saved_dir)

    # Use FPrefNet for orchestration and warm-up
    bayes_net_f = FPrefNet(
        net=net,
        likelihood=likelihood,
        prior=prior,
        ckpt_dir=saved_dir,
        gp_prior=bb_prior,
        x_meas=x_meas,
        aux_meas=aux_meas,
        n_meas=config.n_meas,
        meas_jitter=config.meas_jitter,
        n_gpu=1,
        name="optim_f",
    )

    # ------------------------------------------------------------------ #
    # Warm-up burn-in (shared starting point for all chains)
    # ------------------------------------------------------------------ #
    # The warm-up uses fSGHMC (functional prior) too, so the starting point
    # already reflects the GP prior.
    util.set_seed(config.seed)
    bayes_net_f.train(
        X_train,
        y_train,
        num_samples=None,       # burn-in only; no weights collected
        num_burn_in_steps=config.num_burn_in_steps,
        lr=config.sghmc_lr,
        mdecay=config.mdecay,
        batch_size=config.batch_size,
        max_param_step=config.max_param_step,
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
            "The prior may be dominating — check the prior checkpoint and "
            "consider reducing the prior weight or increasing num_burn_in_steps.",
            RuntimeWarning,
        )
    elif _avg_weight_mag > 5.0:
        warnings.warn(
            f"Warm-up average weight magnitude is large ({_avg_weight_mag:.2f}).  "
            f"For a {width}×{depth} MLP, network outputs scale as w^{depth+1}; "
            f"at avg |w|={_avg_weight_mag:.1f} reward logits may reach "
            f"O({_avg_weight_mag**(depth+1):.1e}), causing astronomical CE.  "
            "Consider: (1) verifying the prior checkpoint std is reasonable, "
            "(2) reducing sghmc_lr and sghmc_lr_max, "
            "(3) increasing mdecay to damp weight growth.",
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
        ckpt_path=ckpt_path,
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
    # Evaluation (identical to bb_optim_star.py)
    # ------------------------------------------------------------------ #
    _B_rhat = min(64, X_test.shape[0])
    _obs_dim = X_test.shape[-1] - 1
    x_rhat = X_test[:_B_rhat, 0, :, :_obs_dim].reshape(-1, _obs_dim).astype(np.float32)
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
                "Check that the worker completed successfully and that num_samples > n_discarded.",
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
                    f"(max diff {_diff:.2e}).  SGHMC may be stuck at a flat region.  "
                    "Try increasing lr_max or checking the prior strength.",
                    RuntimeWarning,
                )
            wandb.log({f"chain_{i}_sample_max_diff_w0_w1": _diff})

        ce, acc = bayes_net_f.eval_test_data(X_test, y_test, X_train, y_train, 4096)
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
        "test_mean_cross_entropy": np.mean(mean_ce),
        "test_mean_accuracy": np.mean(mean_acc),
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
