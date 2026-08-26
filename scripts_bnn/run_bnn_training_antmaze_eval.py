#!/usr/bin/env python
# coding: utf-8
"""run_bnn_training_antmaze_eval.py — antmaze evaluation variant of
run_bnn_training.py (scale-adapted cyclical fSGHMC).

Trains a preference-BNN using FPrefNet (f_pref_net.py) with a functional GP
prior built from LCFModel and any source function in optbnn/gp/reward_functions.py.
The reward source function is chosen at runtime via the ``reward_function``
config field.

This variant is specialised for the antmaze evaluation layout: instead of taking
explicit ``train_dataset`` / ``val_dataset`` paths, it builds them from
``antmaze_variant`` and ``seed`` (see below) so that the model seed and the
loaded data split always match.

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

Antmaze eval train / validation / test data
--------------------------------------------
The training, validation and test sets are loaded from the per-seed antmaze eval
directory, with paths derived from ``antmaze_variant`` and ``seed``::

    {data_root}/{antmaze_variant}/eval/seed_{seed}/{antmaze_variant}_pref_train_{seed}.hdf5
    {data_root}/{antmaze_variant}/eval/seed_{seed}/{antmaze_variant}_pref_val_{seed}.hdf5
    {data_root}/{antmaze_variant}/eval/seed_{seed}/{antmaze_variant}_pref_test_{seed}.hdf5

Because the same ``seed`` drives both posterior sampling and the file selection,
the model seed and the loaded data splits are always consistent.  No splitting is
done in this script.  Build the partitions upstream (split_pref_nt_seeds.py) so
that any data-reduction or label-noise variants are just different files —
``train_dataset`` / ``val_dataset`` / ``test_dataset`` may be set explicitly to
override the derived paths (e.g. to point at a reduction/ or noise/ subdirectory
file).  The model is trained on ``train_dataset``.  Warm-up monitoring and the
early-stop check use the validation set only; the end-of-training metrics (CE,
accuracy, R-hat / ESS / tail / CVaR diagnostics) are computed on BOTH the
validation and test sets and logged as ``val_*`` and ``test_*``.  R-hat and ESS
measure MCMC convergence and are independent of the eval split.

Note this is distinct from the measurement dataset (``measurement_dataset``),
which supplies the raw observations for the functional GP prior gradient and is
loaded separately from the train / validation / test sets.

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

# --- CPU thread caps (see HANDOFF_HP_SELECTION.md section 10.7) --------------
# Set HERE, before torch/numpy are imported, because the intra-op thread pools
# are sized at import time.  Both launchers export these as well, but a run
# started any other way — a hand-typed diagnostic re-run, say — would otherwise
# inherit torch's default of one thread per core (255 on leviathan).  That
# oversubscribes the box, and more importantly thread count changes
# floating-point reduction order, so such a run would not be numerically
# comparable to the sweep trials it is meant to be compared against.
#
# setdefault, so an explicit environment value still wins — e.g. deliberately
# running uncapped to reproduce the section 10.7 A/B.
for _thr_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thr_var, "8")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import matplotlib as mpl
import pyrallis

mpl.use("Agg")
import arviz_stats as azs
import numpy as np
import torch
import wandb

# Enforce the cap on torch's own intra-op pool as well.  The environment
# variables above are read by OpenMP/MKL/OpenBLAS, but not every torch build
# honours them (a macOS build without OpenMP ignores OMP_NUM_THREADS entirely),
# so setting it explicitly makes the cap hold wherever this runs.  Reads the
# value back from the environment so an explicit override still wins.
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

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
    width: int = 6  # log2 exponent; actual width = 2**width (e.g. 6 → 64)
    depth: int = 3
    # SGHMC Hyper-parameters
    batch_size: int = 256
    num_samples: int = 50
    n_discarded: int = 10
    num_burn_in_steps: int = 3000
    keep_every: int = 2000
    sghmc_lr: float = 0.008
    # Fixed step size for the burn-in phase only.  None = inherit sghmc_lr
    # (= cool-phase lr_min under the cyclical schedule) — the legacy behaviour.
    # When lr_min is swept small (sampling-tier schedules), a fixed-length
    # burn-in at lr_min under-fits: warm-up accuracy collapses and the warm-up
    # gate (early_stop_acc_threshold) rejects otherwise-good configs.  Set this
    # to a value that fits (e.g. ~2e-3, near the warm-up tier's winning sghmc_lr)
    # so burn-in quality is decoupled from the swept cool-phase lr_min.
    burn_in_lr: Optional[float] = None
    num_chains: int = 4
    # How many chains to co-locate on each GPU during parallel sampling.  Chains
    # pack greedily onto the lowest GPU indices: with chains_per_gpu=2, chains
    # 0-1 share cuda:0, chains 2-3 share cuda:1, etc., so a run uses only
    # ceil(num_chains / chains_per_gpu) GPUs.  The MLP + n_meas×n_meas kernel are
    # tiny relative to an A6000's memory, and co-located chains stay independent
    # (separate processes/seeds/RNG/ckpt dirs), only sharing compute.  Default 1
    # preserves the original one-chain-per-GPU behaviour.
    chains_per_gpu: int = 1
    mdecay: float = 0.01
    print_every_n_samples: int = 5
    # Cyclical step-size schedule (Zhang et al. 2020)
    use_cyclical_lr: bool = True
    sghmc_lr_max: float = 0.03
    cycle_length: int = 1000
    fraction_cool: float = 0.25
    # Cool-phase harvesting (Issue 2): samples collected per cycle from the cool
    # tail.  num_samples is the TOTAL sample count, reached in
    # ceil(num_samples / samples_per_cycle) cycles -> the wall-clock lever.
    # 1 = legacy one-sample-per-cycle (num_samples cycles).
    samples_per_cycle: int = 1
    # Resample momentum from its stationary Gaussian at each cycle start (Issue 1)
    # instead of zeroing it (Wu et al. 2025).  False = legacy zeroing.
    resample_momentum: bool = True
    # Chain-start diversification: overdisperse each chain's initial weights around
    # the shared warm-up point by chain_init_jitter * per-tensor std.  0.0 = legacy
    # identical starts (which under-estimate R-hat).  A convergence-diagnostic /
    # mode-coverage lever, best tuned in the HP sweep alongside lr_max/fraction_cool.
    chain_init_jitter: float = 0.0
    # Safety clamp on per-element momentum (see bb_optim_star.py for details)
    max_param_step: Optional[float] = 0.5
    # Bradley-Terry trajectory pooling, shared across BNN/MR/PT: "mean" (masked
    # mean over valid timesteps, trajectory-length-independent) or "sum" (legacy).
    bt_pool: str = "mean"
    # Gradient-clip scope (Issue 3): clip in burn-in always, in sampling only if
    # clip_during_sampling.  With bt_pool="mean" the logits are bounded, so the
    # sampling-phase clip should be unnecessary (default off).
    clip_grad_norm_value: Optional[float] = 100.0
    clip_during_sampling: bool = False
    # Antmaze evaluation data.  Train / validation / test sets are loaded from
    # the per-seed eval directory:
    #   {data_root}/{antmaze_variant}/eval/seed_{seed}/{antmaze_variant}_pref_{train,val,test}_{seed}.hdf5
    # config.seed selects the data files AND, by default, drives sampling, so
    # the model seed and the loaded data splits always match.  `sampling_seed`
    # breaks that tie for REPLICATES only: it re-seeds the warm-up, the chain
    # RNG streams and the jitter draws while leaving the data split fixed, which
    # is what section 4.3.34 needs to measure run-to-run variance at a fixed
    # configuration.  Changing config.seed alone cannot do this -- it would load
    # a different dataset (and seeds 1-10 are reserved by section 1 anyway).
    # None = use config.seed, reproducing every existing run bit-identically.  All files are
    # read whole — no splitting is done in this script.  Warm-up monitoring runs
    # on the validation set only; the end-of-training metrics are computed on
    # both the validation and the test sets (val_* / test_*).
    sampling_seed: Optional[int] = None
    antmaze_variant: str = "antmaze-medium-play-v2"
    data_root: str = "data/antmaze"
    # train_dataset / val_dataset / test_dataset are derived from antmaze_variant
    # + seed in __post_init__ when left unset.  Set them explicitly only to
    # override the convention — e.g. a reduction/ or noise/ subdirectory file.
    train_dataset: Optional[str] = None
    val_dataset: Optional[str] = None
    test_dataset: Optional[str] = None
    dataset_id: str = "run"
    # Measurement dataset for the fSGHMC functional GP prior.
    # Must be an HDF5 file with keys:
    #   "obs"     — (N, obs_dim)  required.  BNN inputs (state + action concatenated).
    #   "aux_obs" — (N, K)        optional.  Auxiliary GP feature inputs.
    # The presence of "aux_obs" is detected automatically by load_measurement_data().
    measurement_dataset: str = "data/meas.hdf5"
    # Number of measurement points sampled per training step from the pool.
    # Wu et al. (2025) use M = 100.
    # Draw the n_meas measurement subset ONCE per chain instead of resampling
    # it every step.  Resampling makes the functional-prior gradient stochastic,
    # and section 4.3.21 found gradient noise -- not the thermostat -- to be the
    # uncorrected heat source inflating the sampled variance: adaptive_sghmc's
    # -lr^4 gradient-noise correction is ~1e-15 against a ~1e-8..1e-6 main term,
    # so it corrects nothing.  The noise comes from RESAMPLING rather than from
    # the subset size, so fixing the set makes the gradient exact at any n_meas.
    # Each chain draws its own set (seeded by seed + chain_idx), so the pooled
    # prior still covers the pool.  False = legacy per-step resampling.
    fix_meas_set: bool = False
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
    # ----------------------------------------------------------------------- #
    # Functional GP prior selection
    # ----------------------------------------------------------------------- #
    # Which functional GP prior to use for the fSGHMC prior gradient:
    #   "map_informed" — the map-informed, wall-respecting heat-kernel prior
    #                    (MapInformedGPPrior) for D4RL Antmaze (default).  When
    #                    selected, `reward_function` / `n_concepts` /
    #                    `gp_cov_scale` are ignored and the map_* fields below
    #                    take effect.
    #   "lcf"          — legacy linear-combination-of-features prior (LCFModel)
    #                    built from `reward_function`.  Obsolete: no current
    #                    test environment uses it.
    gp_prior_type: str = "map_informed"
    # --- map_informed prior settings (only used when gp_prior_type="map_informed") ---
    # Maze size selecting the hardcoded fallback layout: "medium" or "large".
    # medium-{play,diverse} share one layout; large-{play,diverse} share another.
    map_size: str = "medium"
    # Optional D4RL gym id (e.g. "antmaze-medium-play-v2").  When set, the maze
    # layout / scaling / offset are extracted from the live env (authoritative)
    # and `map_size` is ignored.  When None, the hardcoded `map_size` layout is
    # used.  Set this on the run machine (which has D4RL) for an exact layout.
    map_env_name: Optional[str] = None
    # Heat-kernel diffusion time: sets the spatial correlation length (~2-4
    # cells).  Fixed per map size; verify via prior-sample heatmaps, NOT reward.
    map_eta: float = 1.0
    # Constant-offset variance (Bradley-Terry additive constant freedom).
    map_sig_c2: float = 1.0
    # Map-informed signal variance (prior reward scale).
    map_sig_g2: float = 1.0
    # Nugget / diagonal jitter (mandatory for K invertibility).
    map_sig_n2: float = 1e-3
    # Overall variance-scale multiplier on the map kernel, K -> map_amp2 * K
    # (prior sd scales as sqrt(map_amp2)).  Leaves the correlation structure —
    # the map-informed part — exactly unchanged; only recalibrates the prior's
    # reward amplitude to the mean-pooled (bt_pool="mean") BT logit scale,
    # where per-point amplitude must match logit magnitude.  Unlike
    # map_eta/map_sig_* (design-fixed from geometry), this is a prior-STRENGTH
    # knob and is swept on validation loss alongside n_meas.  1.0 = legacy.
    map_amp2: float = 1.0
    # Where to read the torso (x, y) from: "obs" (X_M[:, :2], the antmaze
    # convention) or "aux" (aux_X).  Never read the goal columns.
    map_xy_source: str = "obs"
    # Warm-up monitoring: log NLL and accuracy every this many steps.
    # 0 = disabled.  Set to e.g. 100 to get a live convergence curve during
    # burn-in.  Evaluation uses a random 512-pair subsample of the validation set.
    warmup_log_every: int = 100
    # Early-stop threshold on warm-up preference accuracy.  After warm-up,
    # accuracy is evaluated on the validation set; if it is below this threshold,
    # parallel chain sampling is skipped and the run finishes cleanly (no
    # exception raised, so a wandb sweep records a completed run rather than a
    # crash).  Accuracy is used rather than NLL because the trajectory-sum
    # Bradley-Terry logit saturates the softmax, inflating NLL well above ln(2)
    # even for an accurate model — accuracy is the directly meaningful signal.
    # 0.5 is random chance for binary preferences.  Set to None to disable.
    early_stop_acc_threshold: Optional[float] = 0.6
    # general params
    seed: int = 1
    OUT_DIR: Optional[str] = "./exp/reward_learning/bnn_training"

    def __post_init__(self):
        # width is given as a log2 exponent so WandB Bayesian sweeps can range
        # over a small contiguous integer space; convert to the actual width.
        self.width = 2 ** self.width
        # Derive the pre-split train / validation paths from the antmaze variant
        # and seed so the loaded data files always match the run's seed.  An
        # explicit train_dataset / val_dataset (if given) takes precedence.
        eval_seed_dir = os.path.join(
            self.data_root, self.antmaze_variant, "eval", f"seed_{self.seed}"
        )
        prefix = f"{self.antmaze_variant}_pref"
        if self.train_dataset is None:
            self.train_dataset = os.path.join(
                eval_seed_dir, f"{prefix}_train_{self.seed}.hdf5"
            )
        if self.val_dataset is None:
            self.val_dataset = os.path.join(
                eval_seed_dir, f"{prefix}_val_{self.seed}.hdf5"
            )
        if self.test_dataset is None:
            self.test_dataset = os.path.join(
                eval_seed_dir, f"{prefix}_test_{self.seed}.hdf5"
            )
        # The wandb run name keeps a uuid for uniqueness across launches, but the
        # on-disk output directory is deterministic: {OUT_DIR}_{seed}, with no
        # uuid leaf.  This lets the IQL eval stage (iql_eval.py) derive the exact
        # reward-model directory from the seed alone.  Exactly one training run
        # per (variant, seed) writes here; a re-run overwrites in place.
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.OUT_DIR is not None:
            self.OUT_DIR = f"{osp.expanduser(self.OUT_DIR)}_{self.seed}"
            util.ensure_dir(self.OUT_DIR)


@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=f"{config.name}_bnn_training",
        id=str(uuid.uuid4()),
        save_code=True,
    )

    if config.OUT_DIR is not None:
        with open(os.path.join(config.OUT_DIR, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Everything stochastic below keys off _samp_seed; only the data paths use
    # config.seed.  See the sampling_seed comment above.
    _samp_seed = config.seed if config.sampling_seed is None else config.sampling_seed
    if _samp_seed != config.seed:
        print(f"[seed] data split = seed {config.seed}; sampling RNG = "
              f"{_samp_seed} (REPLICATE -- same data, independent chains)")
    util.set_seed(_samp_seed)

    width = config.width
    depth = config.depth
    transfer_fn = "relu"

    # ------------------------------------------------------------------ #
    # Resolve reward source function (only needed for the LCF prior)
    # ------------------------------------------------------------------ #
    map_informed = config.gp_prior_type == "map_informed"
    if config.gp_prior_type not in ("lcf", "map_informed"):
        raise ValueError(
            f"gp_prior_type={config.gp_prior_type!r} unknown; "
            "expected 'lcf' or 'map_informed'."
        )
    if map_informed:
        function_vect = None
        print(
            f"[GP prior] map_informed prior (map_size={config.map_size!r}, "
            f"env={config.map_env_name!r})"
        )
    else:
        if not hasattr(_reward_fns, config.reward_function):
            raise ValueError(
                f"reward_function={config.reward_function!r} not found in "
                "optbnn/gp/reward_functions.py.  "
                f"Available: {[n for n in dir(_reward_fns) if not n.startswith('_')]}"
            )
        function_vect = getattr(_reward_fns, config.reward_function)
        print(f"[GP prior] reward_function = {config.reward_function!r}")

    # ------------------------------------------------------------------ #
    # Load preference data (separate pre-split train / validation / test files)
    # ------------------------------------------------------------------ #
    # Each file is read whole (training_ratio=1.0 = no in-script split).  The
    # model trains on train_dataset; warm-up monitoring uses the validation set
    # (X_eval), and end-of-training metrics are computed on both the validation
    # and test sets (logged as val_* / test_*).
    X_train, y_train = util.load_pref_data(config.train_dataset, training_ratio=1.0)
    X_val, y_val = util.load_pref_data(config.val_dataset, training_ratio=1.0)
    X_test, y_test = util.load_pref_data(config.test_dataset, training_ratio=1.0)
    # Warm-up / early-stop evaluate on the validation set only.
    X_eval, y_eval = X_val, y_val
    # End-of-training metrics run over each of these (label, X, y) eval sets.
    eval_sets = [("val", X_val, y_val), ("test", X_test, y_test)]

    for _split, _X, _y in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
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
    # Build the functional GP prior (LCFModel or MapInformedGPPrior)
    # ------------------------------------------------------------------ #
    # gp_prior_args must survive pickle across mp.spawn (numpy arrays + a
    # module-level function reference); workers reconstruct their own prior from
    # it.  The parent-process prior is used only during warm-up.
    meas_kwargs = {
        "x_meas": x_meas,
        "aux_meas": aux_meas,
        "n_meas": config.n_meas,
        "meas_jitter": config.meas_jitter,
    }

    if map_informed:
        from optbnn.gp.maze_layouts import get_antmaze_layout
        from optbnn.gp.models.map_informed_prior import MapInformedGPPrior

        free_mask, scaling, offset = get_antmaze_layout(
            config.map_size, env_name=config.map_env_name
        )
        print(
            f"[GP prior] maze layout: {free_mask.shape} grid, "
            f"{int(free_mask.sum())} free cells, scaling={scaling}, "
            f"offset={offset}"
        )
        gp_prior = MapInformedGPPrior(
            free_mask=free_mask,
            scaling=scaling,
            offset=offset,
            eta=config.map_eta,
            sig_c2=config.map_sig_c2,
            sig_g2=config.map_sig_g2,
            sig_n2=config.map_sig_n2,
            amp2=config.map_amp2,
            xy_source=config.map_xy_source,
            device=device,
        )
        gp_prior_args = gp_prior.to_args()
    else:
        # ---- Infer n_concepts from a dummy forward pass through the source fn ----
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

        # p_covariance = gp_cov_scale * I_{n_concepts} — isotropic GP weight prior.
        # p_mean = ones(n_concepts); intercept (feature 0) gets 0 (no sign bias).
        p_covariance = np.eye(n_concepts, dtype=np.float32) * config.gp_cov_scale
        p_mean = np.ones(n_concepts, dtype=np.float32)
        p_mean[0] = 0.0
        gp_prior_args = {
            "prior_type": "lcf",
            "p_covariance": p_covariance,
            "function_vect": function_vect,
            "p_mean": p_mean,
        }

        # Parent-process LCFModel (used only during warm-up; workers reconstruct
        # their own from gp_prior_args).
        gp_prior = LCFModel(
            p_covariance=p_covariance,
            function_vect=function_vect,
            device=device,
            p_mean=p_mean,
        ).to(device)

    # ------------------------------------------------------------------ #
    # Build BNN and FPrefNet (no OptimGaussianPrior needed)
    # ------------------------------------------------------------------ #
    util.set_seed(_samp_seed)
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
        bt_pool=config.bt_pool,
        clip_grad_norm_value=config.clip_grad_norm_value,
        clip_during_sampling=config.clip_during_sampling,
    )

    # ------------------------------------------------------------------ #
    # Warm-up burn-in — shared starting point for all chains
    # ------------------------------------------------------------------ #
    # Warm-up runs fSGHMC so the starting point already reflects the GP prior.
    # When warmup_log_every > 0, NLL and accuracy are evaluated every
    # warmup_log_every steps on a 512-pair subsample of the validation set and
    # logged to stdout + wandb under the "warmup/" prefix.
    util.set_seed(_samp_seed)
    bayes_net_f.train(
        X_train,
        y_train,
        num_samples=None,  # burn-in only; no weights collected
        num_burn_in_steps=config.num_burn_in_steps,
        lr=config.sghmc_lr,
        burn_in_lr=config.burn_in_lr,
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
        seed=_samp_seed,
        batch_size=config.batch_size,
        num_samples=config.num_samples,
        n_discarded=config.n_discarded,
        num_burn_in_steps=config.num_burn_in_steps,
        keep_every=config.keep_every,
        lr=config.sghmc_lr,
        burn_in_lr=config.burn_in_lr,
        mdecay=config.mdecay,
        print_every_n_samples=config.print_every_n_samples,
        initial_weights=initial_weights,
        use_cyclical_lr=config.use_cyclical_lr,
        lr_max=config.sghmc_lr_max,
        cycle_length=config.cycle_length,
        fraction_cool=config.fraction_cool,
        samples_per_cycle=config.samples_per_cycle,
        resample_momentum=config.resample_momentum,
        fix_meas_set=config.fix_meas_set,
        max_param_step=config.max_param_step,
        chains_per_gpu=config.chains_per_gpu,
        bt_pool=config.bt_pool,
        clip_grad_norm_value=config.clip_grad_norm_value,
        clip_during_sampling=config.clip_during_sampling,
        chain_init_jitter=config.chain_init_jitter,
    )

    # ------------------------------------------------------------------ #
    # End-of-training evaluation.
    #
    # Predictive metrics — CE, accuracy, and the pred_* R-hat / ESS / tail /
    # CVaR diagnostics (evaluated at input points drawn from the eval set) —
    # are computed on BOTH the validation and test sets and logged with val_* /
    # test_* prefixes.  Parameter-space diagnostics (param_*) measure MCMC
    # convergence in weight space, do not depend on any eval set, and are logged
    # once (unprefixed).  R-hat and ESS are independent of the eval split.
    # ------------------------------------------------------------------ #
    def _pct_over(arr, threshold):
        arr = np.asarray(arr, dtype=float)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return float("nan")
        return float(np.mean(valid > threshold) * 100)

    total_samples = config.num_chains * config.num_samples

    # ---- Load every chain's sampled weights once (eval-set independent) ----
    chains_weights = []
    for i in range(config.num_chains):
        chain_dir = os.path.join(saved_dir, f"chain_{i}")
        sampled_weights = bayes_net_f._load_sampled_weights(
            os.path.join(chain_dir, "sampled_weights", "sampled_weights_0000000")
        )
        n_loaded = len(sampled_weights)
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
                for a, b in zip(sampled_weights[0], sampled_weights[1])
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
        chains_weights.append(sampled_weights)

    # ---- Weight-space diagnostics: DELIBERATELY ABSENT ----------------
    # Under Wu et al. (2025) the stationary measure of these dynamics is the
    # function-space posterior P_{f|D}; the weight-space measure is not the
    # object of inference and is not even confined -- U(w) depends on w only
    # through f, so the chain diffuses freely along f-preserving directions and
    # ||w|| grows without that indicating anything.  param_rhat / param_ess /
    # param_within_chain_var and the sampling weight-norm statistics were
    # therefore measuring nothing about convergence, and have been removed
    # rather than relabelled.  Stationarity is assessed in function space by
    # util.function_space_drift, logged per eval split below.
    summary = {}

    # ---- Predictive diagnostics, computed per eval set (val_* and test_*) ----
    def evaluate_eval_set(label, X_e, y_e):
        _B_rhat = min(64, X_e.shape[0])
        _obs_dim = X_e.shape[-1] - 1
        # Drop attention-masked (padded) timesteps: the last feature column is
        # the attn_mask (see util.load_pref_data), and the net produces garbage
        # at padded steps.  No-op for full-length trajectories.
        _block = X_e[:_B_rhat, 0, :, :]                       # [B, T, obs_dim+1]
        _valid = (_block[..., _obs_dim].reshape(-1) > 0.5)     # attn_mask column
        x_rhat = _block[..., :_obs_dim].reshape(-1, _obs_dim).astype(np.float32)
        x_rhat = x_rhat[_valid]
        print(
            f"[diag/{label}] x_rhat: {int(_valid.sum())}/{_valid.size} "
            "valid (non-padded) points"
        )
        x_rhat_t = torch.from_numpy(x_rhat).to(bayes_net_f.device)

        mean_ce = []
        mean_acc = []
        pred_ce = []
        pred_acc = []
        pred_chains = []
        for i in range(config.num_chains):
            bayes_net_f.sampled_weights = chains_weights[i]
            ce, acc = bayes_net_f.eval_test_data(X_e, y_e, eval_batch_size=4096)
            mean_ce.append(ce)
            mean_acc.append(acc)
            # Posterior-predictive per Wu et al. (2025) Eq. (10): average the
            # likelihood over draws, E[sigma(f)], rather than the reward,
            # sigma(E[f]).  The plug-in above is blind to posterior width; the
            # downstream CVaR is a functional of exactly that width.  Wrapped
            # defensively so a failure here cannot discard an expensive run.
            try:
                _pce, _pacc = bayes_net_f.eval_test_data_predictive(
                    X_e, y_e, eval_batch_size=4096
                )
            except Exception as e:  # noqa: BLE001 — keep the run, surface the cause
                warnings.warn(
                    f"[{label}] predictive CE failed on chain {i} "
                    f"({type(e).__name__}: {e}); logging NaN.",
                    RuntimeWarning,
                )
                _pce = _pacc = float("nan")
            pred_ce.append(_pce)
            pred_acc.append(_pacc)
            bayes_net_f.net.eval()
            with torch.no_grad():
                chain_preds = []
                for weights in chains_weights[i]:
                    bayes_net_f.network_weights = weights
                    pred = bayes_net_f.net(x_rhat_t).detach().cpu().numpy().ravel()
                    chain_preds.append(pred)
            pred_chains.append(np.stack(chain_preds))
        pred_chains = np.stack(pred_chains)

        pred_within_chain_var = float(np.mean(pred_chains.var(axis=1)))
        print(f"[diag/{label}] pred within-chain var = {pred_within_chain_var:.4e}")

        rhats_pred = azs.rhat(pred_chains)
        ess_pred = azs.ess(pred_chains)

        # ---- Lower-tail (5% quantile) convergence.  The defaults above are
        # *bulk* diagnostics (centred on the median) and do NOT certify a
        # quantile.  Downstream we take a 95% lower confidence bound on the
        # reward (the 5th percentile of the per-point posterior predictive), so
        # we need tail-ESS / tail-R-hat and the MC error of that quantile.
        # Wrapped defensively so a signature mismatch cannot discard the run. ----
        _nan_pred = np.full(pred_chains.shape[-1], np.nan)
        try:
            ess_pred_q05 = np.asarray(azs.ess(pred_chains, method="quantile", prob=0.05))
            rhat_pred_folded = np.asarray(azs.rhat(pred_chains, method="folded"))
            mcse_pred_q05 = np.asarray(azs.mcse(pred_chains, method="quantile", prob=0.05))
        except Exception as e:  # noqa: BLE001 — keep the run, surface the cause
            warnings.warn(
                f"[{label}] Tail diagnostics failed ({type(e).__name__}: {e}); "
                "logging NaN.  Check the arviz_stats ess/rhat/mcse signatures.",
                RuntimeWarning,
            )
            ess_pred_q05 = rhat_pred_folded = mcse_pred_q05 = _nan_pred

        # Absolute MCSE is scale-confounded: reward magnitude varies by orders of
        # magnitude across states, so normalise by each point's posterior sd to
        # get a scale-free "fraction of the spread" — the trustworthy _max.
        _pred_sd = pred_chains.reshape(-1, pred_chains.shape[-1]).std(axis=0)
        mcse_pred_q05_rel = mcse_pred_q05 / (_pred_sd + 1e-8)

        # ---- CVaR (mean of the lowest 5%) convergence — the downstream
        # quantity.  Rockafellar–Uryasev: CVaR = VaR + (1/a)·E[min(X-VaR,0)] is
        # exact, so CVaR's MC error is the mean ESS/MCSE of u=(1/a)·min(X-VaR,0). ----
        _alpha = 0.05
        try:
            _var = np.quantile(pred_chains.reshape(-1, pred_chains.shape[-1]),
                               _alpha, axis=0)
            _u = np.minimum(pred_chains - _var[None, None, :], 0.0) / _alpha
            ess_pred_cvar = np.asarray(azs.ess(_u, method="mean"))
            mcse_pred_cvar = np.asarray(azs.mcse(_u, method="mean"))
            rhat_pred_cvar = np.asarray(azs.rhat(_u, method="folded"))
            mcse_pred_cvar_rel = mcse_pred_cvar / (_pred_sd + 1e-8)
        except Exception as e:  # noqa: BLE001 — keep the run, surface the cause
            warnings.warn(
                f"[{label}] CVaR diagnostics failed ({type(e).__name__}: {e}); logging NaN.",
                RuntimeWarning,
            )
            ess_pred_cvar = mcse_pred_cvar = rhat_pred_cvar = mcse_pred_cvar_rel = _nan_pred

        # Stationarity of the FUNCTION-space measure — the object of inference.
        # Also tests the cyclical step-size schedule, which is an addition to
        # Wu et al.: early and late cycles must be samples from one measure.
        _drift = {f"{label}_{k}": v
                  for k, v in util.function_space_drift(pred_chains).items()}

        return {
            **_drift,
            # Theory-aligned predictive (Wu et al. Eq. 10) — the SELECTION
            # metric.  Averages the likelihood over draws, so it is sensitive to
            # posterior width, which the plug-in below is not.
            f"{label}_predictive_cross_entropy": float(np.mean(pred_ce)),
            f"{label}_predictive_accuracy": float(np.mean(pred_acc)),
            # Plug-in sigma(E[f]) — retained for continuity with round-1/round-2
            # numbers and as the paired comparison, NOT for selection.
            f"{label}_mean_cross_entropy": np.mean(mean_ce),
            f"{label}_mean_accuracy": np.mean(mean_acc),
            f"{label}_pred_within_chain_var": pred_within_chain_var,
            f"{label}_pred_rhat_max": float(np.nanmax(rhats_pred)),
            f"{label}_pred_rhat_95th_pct": float(np.nanpercentile(rhats_pred, 95)),
            f"{label}_pred_rhat_median": float(np.nanmedian(rhats_pred)),
            f"{label}_pred_rhat_mean": float(np.nanmean(rhats_pred)),
            f"{label}_pred_rhat_pct_over_1.01": _pct_over(rhats_pred, 1.01),
            f"{label}_pred_ess_min": float(np.nanmin(ess_pred)),
            f"{label}_pred_ess_median": float(np.nanmedian(ess_pred)),
            f"{label}_pred_ess_mean": float(np.nanmean(ess_pred)),
            f"{label}_pred_ess_min_norm": float(np.nanmin(ess_pred)) / total_samples,
            f"{label}_pred_ess_median_norm": float(np.nanmedian(ess_pred)) / total_samples,
            # ---- Lower-tail (5% quantile) diagnostics: certify the 95% bound ----
            f"{label}_pred_q05_ess_min": float(np.nanmin(ess_pred_q05)),
            f"{label}_pred_q05_ess_median": float(np.nanmedian(ess_pred_q05)),
            f"{label}_pred_q05_ess_min_norm": float(np.nanmin(ess_pred_q05)) / total_samples,
            f"{label}_pred_folded_rhat_max": float(np.nanmax(rhat_pred_folded)),
            f"{label}_pred_folded_rhat_95th_pct": float(np.nanpercentile(rhat_pred_folded, 95)),
            f"{label}_pred_folded_rhat_median": float(np.nanmedian(rhat_pred_folded)),
            f"{label}_pred_folded_rhat_pct_over_1.01": _pct_over(rhat_pred_folded, 1.01),
            f"{label}_pred_q05_mcse_max": float(np.nanmax(mcse_pred_q05)),
            f"{label}_pred_q05_mcse_median": float(np.nanmedian(mcse_pred_q05)),
            f"{label}_pred_q05_mcse_rel_max": float(np.nanmax(mcse_pred_q05_rel)),
            f"{label}_pred_q05_mcse_rel_median": float(np.nanmedian(mcse_pred_q05_rel)),
            # ---- CVaR (mean of lowest 5%): the downstream quantity ----
            f"{label}_pred_cvar_ess_min": float(np.nanmin(ess_pred_cvar)),
            f"{label}_pred_cvar_ess_median": float(np.nanmedian(ess_pred_cvar)),
            f"{label}_pred_cvar_rhat_max": float(np.nanmax(rhat_pred_cvar)),
            f"{label}_pred_cvar_rhat_median": float(np.nanmedian(rhat_pred_cvar)),
            f"{label}_pred_cvar_rhat_pct_over_1.01": _pct_over(rhat_pred_cvar, 1.01),
            f"{label}_pred_cvar_mcse_rel_max": float(np.nanmax(mcse_pred_cvar_rel)),
            f"{label}_pred_cvar_mcse_rel_median": float(np.nanmedian(mcse_pred_cvar_rel)),
        }

    for _label, _Xe, _ye in eval_sets:
        summary.update(evaluate_eval_set(_label, _Xe, _ye))

    # ------------------------------------------------------------------ #
    # Gradient-clip instrumentation (Issue 3, Step 1) — aggregate the per-chain
    # pre-clip grad-norm stats the workers wrote to disk.  Split-independent (a
    # property of sampling), so logged once, unprefixed.  The decisive number is
    # gradnorm_sampling_pct_over_clip: ~0 => the clip is inert during sampling
    # and the CVaR tail is clip-unbiased; >0 => it fires => real tail bias.
    # ------------------------------------------------------------------ #
    for _phase in ("burnin", "sampling"):
        _cnt = _nover = 0
        _sm = _mx = 0.0
        _chits = _celems = 0
        for _i in range(config.num_chains):
            _p = os.path.join(saved_dir, f"chain_{_i}", "grad_norm_stats.pt")
            if not os.path.exists(_p):
                continue
            _s = torch.load(_p, weights_only=False).get(_phase, {})
            _cnt += _s.get("count", 0)
            _sm += _s.get("sum", 0.0)
            _nover += _s.get("n_over_clip", 0)
            _mx = max(_mx, _s.get("max", 0.0))
            _chits += _s.get("clamp_hits", 0)
            _celems += _s.get("clamp_elems", 0)
        if _cnt > 0:
            summary[f"gradnorm_{_phase}_max"] = _mx
            summary[f"gradnorm_{_phase}_mean"] = _sm / _cnt
            summary[f"gradnorm_{_phase}_pct_over_clip"] = 100.0 * _nover / _cnt
        # max_param_step is a hard momentum clamp applied every step, sampling
        # included.  It is not measure-preserving, so it is only defensible if
        # it never binds on a selected run.  ~0 here => inert => no distortion.
        if _celems > 0:
            summary[f"param_clamp_{_phase}_pct"] = 100.0 * _chits / _celems

    wandb.log(summary)


if __name__ == "__main__":
    train()