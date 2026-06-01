#!/usr/bin/env python
# coding: utf-8
"""run_bnn_training_f.py — domain-agnostic scale-adapted cyclical fSGHMC.

Trains a preference-BNN using FPrefNet (f_pref_net.py) with a functional GP
prior built from LCFModel and any source function in optbnn/gp/reward_functions.py.
The reward source function is chosen at runtime via the ``reward_function``
config field.

Key differences from bb_optim_star.py
--------------------------------------
1. No OptimGaussianPrior — no prior tuning checkpoint required.
2. No prior_dir config field.
3. Uses FPrefNet (standalone, no weight-space prior) instead of PrefNet.
4. Loads a separate measurement dataset (HDF5) for the GP prior gradient.
5. ``reward_function`` selects any function from reward_functions.py by name.
6. ``input_dim`` is inferred automatically from the training data.
7. ``n_concepts`` (GP feature dimension) is inferred by probing the source
   function on a dummy input, or can be set explicitly in the config.

Train/test split vs. full-dataset training
------------------------------------------
``training_split`` controls how the dataset is used:

* ``0 < training_split < 1`` — the data is split; the model is trained on the
  training partition and all post-sampling metrics (CE, accuracy) are computed
  on the held-out test partition and logged as ``test_*``.
* ``training_split == 1.0`` — the *entire* dataset is used for posterior
  sampling (no held-out split).  Warm-up monitoring, the early-stop check, and
  post-sampling CE / accuracy are then computed in-sample on the training data
  and logged as ``train_*``.  R-hat and ESS measure MCMC convergence and are
  unaffected by the absence of a split.

All SGHMC hyper-parameters, warm-up logic, R-hat / ESS diagnostics, and
wandb logging are otherwise identical to bb_optim_star.py.
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
    # Preference training dataset
    dataset: str = "data/pref.hdf5"
    dataset_id: str = "run"
    # Fraction of the dataset used for training.  0 < training_split < 1 holds
    # out the remainder as a test set (metrics logged as test_*).  Set to 1.0 to
    # train on the FULL dataset with no held-out split — metrics are then
    # computed in-sample on the training data and logged as train_*.
    training_split: float = 0.8
    # Fraction of TRAINING labels to flip (0 = none, 1 = all).  Applied only to
    # the training partition so any held-out test labels stay clean.  Flipping a
    # preference label swaps the two trajectories' win/loss assignment.
    label_flip: float = 0.0
    # Fraction of TRAINING pairs to randomly discard after the split (0 = none,
    # 1 = all).  Test data is unaffected.  data_reduction=1.0 leaves no training
    # data and the run exits cleanly.
    data_reduction: float = 0.0
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
    # burn-in.  Evaluation uses a random 512-pair subsample of the test set.
    warmup_log_every: int = 100
    # Early-stop threshold on warm-up preference accuracy.  After warm-up,
    # accuracy is evaluated on the test set; if it is below this threshold,
    # parallel chain sampling is skipped and the run finishes cleanly (no
    # exception raised, so a wandb sweep records a completed run rather than a
    # crash).  Accuracy is used rather than NLL because the trajectory-sum
    # Bradley-Terry logit saturates the softmax, inflating NLL well above ln(2)
    # even for an accurate model — accuracy is the directly meaningful signal.
    # 0.5 is random chance for binary preferences.  Set to None to disable.
    early_stop_acc_threshold: Optional[float] = 0.6
    # general params
    seed: int = 1
    OUT_DIR: Optional[str] = "./exp/reward_learning/bnn_training_f"

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
        name=f"{config.name}_bnn_training_f",
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
    # Load preference data (optionally split into train / test)
    # ------------------------------------------------------------------ #
    # training_split == 1.0 -> load_pref_data returns (X, y) (no split); the full
    # dataset is used for sampling and all metrics are computed in-sample.
    full_dataset = config.training_split >= 1.0
    if full_dataset:
        X_train, y_train = util.load_pref_data(config.dataset, training_ratio=1.0)
        # Evaluate in-sample: eval set is the training set itself.
        X_eval, y_eval = X_train, y_train
        eval_label = "train"
        _splits = [("train", X_train, y_train)]
        print("[data] training_split=1.0 — using FULL dataset (no held-out test set)")
    else:
        X_train, y_train, X_test, y_test = util.load_pref_data(
            config.dataset, config.training_split
        )
        X_eval, y_eval = X_test, y_test
        eval_label = "test"
        _splits = [("train", X_train, y_train), ("test", X_test, y_test)]

    for _split, _X, _y in _splits:
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
    # Label flipping (training data only — any held-out test labels stay clean)
    # ------------------------------------------------------------------ #
    # y_train is one-hot (N, 2); flipping a label is 1.0 - y (swaps the columns).
    if config.label_flip > 0.0:
        n_train = X_train.shape[0]
        if config.label_flip >= 1.0:
            y_train = 1.0 - y_train
            n_flipped = n_train
        else:
            n_flipped = int(n_train * config.label_flip)
            flip_idx = np.random.choice(n_train, n_flipped, replace=False)
            y_train[flip_idx] = 1.0 - y_train[flip_idx]
        print(
            f"[data] label_flip={config.label_flip}: flipped {n_flipped}/{n_train} "
            "training labels"
        )

    # ------------------------------------------------------------------ #
    # Data reduction (training data only — test data unaffected)
    # ------------------------------------------------------------------ #
    if config.data_reduction > 0.0:
        if config.data_reduction >= 1.0:
            print("[data] data_reduction=1.0: no training data remains.  Exiting.")
            wandb.finish()
            return
        n_train = X_train.shape[0]
        n_keep = int(n_train * (1.0 - config.data_reduction))
        keep_idx = np.random.choice(n_train, n_keep, replace=False)
        X_train = X_train[keep_idx]
        y_train = y_train[keep_idx]
        print(
            f"[data] data_reduction={config.data_reduction}: kept {n_keep}/{n_train} "
            "training pairs"
        )

    # In full-dataset mode the eval set IS the training set, so re-sync it after
    # any label flipping / reduction (these may rebind or subset y_train/X_train).
    if full_dataset:
        X_eval, y_eval = X_train, y_train

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
    # First feature is the intercept (when present) — no prior bias on its sign.
    # All other coefficients default to 1 (positive weights on reward-relevant features).
    p_mean = np.ones(n_concepts, dtype=np.float32)
    p_mean[0] = 0.0
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
    # warmup_log_every steps on a 512-pair subsample of the eval set (test set,
    # or the training set when training_split=1.0) and logged to stdout + wandb
    # under the "warmup/" prefix.
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
        eval_data=(X_eval, y_eval) if config.warmup_log_every > 0 else None,
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

    # ------------------------------------------------------------------ #
    # Early-stop check — skip chain sampling if warm-up accuracy is too low
    # ------------------------------------------------------------------ #
    warmup_final_nll, warmup_final_acc = bayes_net_f._eval_current_weights(
        X_eval, y_eval
    )
    print(
        f"[warm-up] final NLL = {warmup_final_nll:.4f}, "
        f"acc = {warmup_final_acc:.4f}  (random-chance acc = 0.5)"
    )
    wandb.log(
        {
            "warmup_final_nll": warmup_final_nll,
            "warmup_final_acc": warmup_final_acc,
        }
    )
    if (
        config.early_stop_acc_threshold is not None
        and warmup_final_acc < config.early_stop_acc_threshold
    ):
        print(
            f"[early-stop] warm-up accuracy {warmup_final_acc:.4f} is below "
            f"threshold {config.early_stop_acc_threshold:.4f}.  "
            "Skipping parallel chain sampling and finishing run cleanly."
        )
        wandb.log({"early_stopped": 1})
        wandb.finish()
        return
    wandb.log({"early_stopped": 0})

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
    # Evaluation — on the eval set (held-out test set, or the training set
    # in-sample when training_split=1.0).  R-hat and ESS measure MCMC
    # convergence and are independent of the train/test distinction.
    # ------------------------------------------------------------------ #
    _B_rhat = min(64, X_eval.shape[0])
    _obs_dim = X_eval.shape[-1] - 1
    x_rhat = X_eval[:_B_rhat, 0, :, :_obs_dim].reshape(-1, _obs_dim).astype(np.float32)
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

        ce, acc = bayes_net_f.eval_test_data(X_eval, y_eval, eval_batch_size=4096)
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
        f"{eval_label}_mean_cross_entropy": np.mean(mean_ce),
        f"{eval_label}_mean_accuracy": np.mean(mean_acc),
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