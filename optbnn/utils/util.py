import argparse
import json
import os
import pickle
import warnings
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_uci_data(uci_dir, split_id, name, version="original"):
    """Load a split of a UCI dataset.

    Args:
        data_dir: str, path to the directory containing the UCI datasets.
        split_id: int, the index of the split to be loaded.
        name: str, the name of the dataset.
        version: str, the version of the uci dataset, must be either
            `original` or `gap`.

    Returns:
        x_train: numpy array, the training data points.
        y_train: numpy array, the training labels.
        x_test: numpy array, the test data points.
        y_test: numpy array, the test labels.
    """
    datasets = [
        "boston",
        "concrete",
        "energy",
        "kin8nm",
        "naval",
        "power",
        "protein",
        "wine",
        "yacht",
    ]
    if not (name in datasets):
        raise ValueError("Invalid dataset name.")
    assert version in ["original", "gap"]

    uci_dir = os.path.join(uci_dir, name)
    data_file = os.path.join(uci_dir, "data.txt")
    idx_train_file = os.path.join(uci_dir, "{}/index_train_{}.txt").format(
        version, split_id
    )
    idx_test_file = os.path.join(uci_dir, "{}/index_test_{}.txt").format(
        version, split_id
    )

    data = np.loadtxt(data_file)
    idx_train = np.loadtxt(idx_train_file).astype(np.int32)
    idx_test = np.loadtxt(idx_test_file).astype(np.int32)

    x, y = data[:, :-1], data[:, -1]
    x_train, y_train = x[idx_train, :], y[idx_train]
    x_test, y_test = x[idx_test, :], y[idx_test]

    return x_train, y_train, x_test, y_test


# The training_ratio argument must be > 0 and <= 1. int(sample_size*training_ratio) becomes training data
# and (sample_size - training data) becomes test data
# if training_ratio == 1.0, then the data is returned without any splitting
# splitting is random according to a seed set by the set_seed function below
# X is shaped as (samples,2,seq_size,obs_dim+1) where the 2 is from combining preference pairs
# and the +1 is concatenating the attn_mask
# y is shaped (samples,)
def load_pref_data(pref_dir, training_ratio=0.8):
    assert training_ratio > 0.0
    assert training_ratio <= 1.0
    with h5py.File(pref_dir) as f:
        obs_1 = np.concatenate(
            [f["states"][:], f["actions"][:], f["attn_mask"][:].reshape(-1, 100, 1)],
            axis=-1,
        )
        obs_2 = np.concatenate(
            [
                f["states_2"][:],
                f["actions_2"][:],
                f["attn_mask_2"][:].reshape(-1, 100, 1),
            ],
            axis=-1,
        )
        if f["labels"].ndim > 1:
            y = f["labels"][:]
        else:
            y = np.eye(2)[f["labels"][:].astype(int)]
    X = np.stack([obs_1, obs_2], axis=1)

    if training_ratio == 1.0:
        return X, y

    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_point = int(num_samples * training_ratio)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    X_train = X[train_indices, ...]
    y_train = y[train_indices, ...]
    X_test = X[test_indices, ...]
    y_test = y[test_indices, ...]
    return X_train, y_train, X_test, y_test


def load_measurement_data(meas_path):
    """Load measurement data from an HDF5 file for fSGHMC functional prior gradient.

    Expected HDF5 layout:

        "obs"     — (N, obs_dim)  required.  BNN inputs (state + action
                    concatenated).  Passed as ``X`` to LCFModel._feature_matrix
                    and as the BNN forward-pass input.
        "aux_obs" — (N, K)        optional.  Auxiliary GP feature inputs.
                    Passed as ``aux_X`` to LCFModel._feature_matrix.  When
                    absent, ``aux_meas`` is returned as None and LCFModel will
                    call ``function_vect(X, device)`` without an aux argument.

    Args:
        meas_path: str, path to the HDF5 measurement file.

    Returns:
        x_meas:   numpy float32 (N, obs_dim).
        aux_meas: numpy float32 (N, K), or None if "aux_obs" is not present.
    """
    with h5py.File(meas_path, "r") as f:
        x_meas   = f["obs"][:].astype(np.float32)
        aux_meas = f["aux_obs"][:].astype(np.float32) if "aux_obs" in f else None
    return x_meas, aux_meas


class Pref_H5Dataset(Dataset):
    def __init__(self, datafile, max_episode_length=None):
        super(Pref_H5Dataset, self).__init__()
        with h5py.File(datafile, "r") as f:
            if max_episode_length is None:
                self._max_episode_length = np.max(
                    [np.max(f["timesteps"][:]), np.max(f["timesteps_2"][:])]
                )
            else:
                self._max_episode_length = max_episode_length

            self._sts_shape = f["states"].shape
            self._acts_shape = f["actions"].shape
            self.states = f["states"][:].astype(np.float32)
            self.actions = f["actions"][:].astype(np.float32)
            self.timesteps = f["timesteps"][:]
            self.attn_mask = f["attn_mask"][:].astype(np.float32)

            self.states_2 = f["states_2"][:].astype(np.float32)
            self.actions_2 = f["actions_2"][:].astype(np.float32)
            self.timesteps_2 = f["timesteps_2"][:]
            self.attn_mask_2 = f["attn_mask_2"][:].astype(np.float32)
            self.labels = f["labels"][:].astype(np.float32)

    def __getitem__(self, index):
        return (
            self.states[index, ...],
            self.actions[index, ...],
            self.timesteps[index, ...],
            self.attn_mask[index, ...],
            self.states_2[index, ...],
            self.actions_2[index, ...],
            self.timesteps_2[index, ...],
            self.attn_mask_2[index, ...],
            self.labels[index],
        )

    def __len__(self):
        return self._sts_shape[0]

    def shapes(self):
        return self._sts_shape, self._acts_shape

    def max_episode_length(self):
        return self._max_episode_length


def prepare_device(n_gpu_use):
    """Setup GPU device if available, move model into configured device.

    Args:
        n_gpu_use: number of used GPUs.
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU's configured to use"
            " is {}, but only {} are available "
            "on this machine.".format(n_gpu_use, n_gpu)
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))

    return device, list_ids


def to_one_hot(y, num_classes=10):
    """Convert labels to one-hot vectors.

    Args:
        y: numpy array, shape [num_classes], the true labels.

    Returns:
        one_hot: numpy array, size (?, num_classes),
            array containing the one-hot encoding of the true classes.
    """
    if isinstance(y, torch.Tensor):
        one_hot = torch.zeros((y.shape[0], num_classes), dtype=torch.float32)
        one_hot[torch.arange(y.shape[0]), y] = 1.0
    else:
        one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float)
        one_hot[np.arange(y.shape[0]), y] = 1.0

    return one_hot


def str2bool(v):
    """Convert string to boolean variable"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_pickle(data, file_path):
    """Wrapper for saving data to a pickle file.

    Args:
        data: a dictionary containing the data needs to be saved.
        file_path: string, path to the output file.
    """
    with open(file_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    """Wrapper for loading data from a pickle file.

    Args:
        file_path: string, path to the pickle file.

    Returns:
        A dictionary containing the loaded data.
    """
    with open(file_path, "rb") as handle:
        data = pickle.load(handle)
    return data


def ensure_dir(dirname):
    """Check whether a given directory was created; if not, create a new one.

    Args:
        dirname: string, path to the directory.
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(file_path):
    """Wrapper for reading a json file.

    Args:
        file_path: string, path to the json file.

    Returns:
        A dictionary containing the loaded data.
    """
    file_path = Path(file_path)
    with file_path.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, file_path):
    """Write data to a json file.

    Args:
        content: a dictionary containing the data needs to be saved.
        file_path: string, path to the output file.
    """
    file_path = Path(file_path)
    with file_path.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def set_seed(seed=99):
    """Set seed for reproducibility purpose."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_all_data(data_loader):
    """Get all data from a data loader."""
    x, y = [], []
    for x_batch, y_batch in data_loader:
        x.append(x_batch)
        y.append(y_batch.reshape([-1, 1]))

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0).reshape([-1])

    return x, y


def split_train_val(x_train, y_train, splitting_ratio=0.2):
    """Split the data into training and validation set."""
    num_samples = x_train.shape[0]
    num_train_samples = int(num_samples * (1 - splitting_ratio))

    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train_samples]

    val_idx = indices[num_train_samples:]
    x_val, y_val = x_train.copy()[val_idx, :], y_train.copy()[val_idx]
    x_train, y_train = x_train.copy()[train_idx, :], y_train.copy()[train_idx]

    return x_train, y_train, x_val, y_val

# ---------------------------------------------------------------------------
# Bradley-Terry trajectory pooling (shared by BNN / MR / PT so all three use an
# identical likelihood).  Given per-timestep rewards already multiplied by the
# 0/1 attention mask, pool them into one scalar per trajectory:
#   mode="sum"  -> Sum_t r_t              (legacy, trajectory-length-dependent)
#   mode="mean" -> Sum_t r_t / Sum_t m_t  (masked mean over valid timesteps)
# The masked mean is trajectory-length-independent and keeps the logit scale
# bounded, which avoids the w^(depth+1) gradient blow-up in deep reward nets.
# ---------------------------------------------------------------------------
def bt_pool_logit(pred_masked, mask, mode="mean"):
    """Torch pooling. pred_masked,(B,T) rewards*mask; mask,(B,T) 0/1. -> (B,)."""
    s = torch.nansum(pred_masked, dim=1)
    if mode == "sum":
        return s
    if mode != "mean":
        raise ValueError(f"bt_pool_logit: mode must be 'sum' or 'mean', got {mode!r}")
    n = torch.nansum(mask, dim=1).clamp(min=1.0)
    return s / n


def bt_pool_logit_np(pred_masked, mask, mode="mean"):
    """NumPy counterpart of bt_pool_logit (used in the eval helpers)."""
    s = np.nansum(pred_masked, axis=1)
    if mode == "sum":
        return s
    if mode != "mean":
        raise ValueError(f"bt_pool_logit_np: mode must be 'sum' or 'mean', got {mode!r}")
    n = np.clip(np.nansum(mask, axis=1), 1.0, None)
    return s / n


def function_space_drift(pred_chains, eps=1e-12):
    """Is the induced function-space measure stationary across draw index?

    `pred_chains` is [chain, draw, point] — the predictive f at the diagnostic
    inputs.  Under Wu et al. (2025) the stationary measure of these dynamics is
    the function-space posterior P_{f|D}, so f is the object of inference and
    the weights are not: weight-space statistics say nothing about convergence
    here, because U(w) depends on w only through f and the chain diffuses freely
    along f-preserving directions.  Stationarity is a claim about f.

    Compares the first half of each chain's draws against the second half, and
    -- this is the part that makes it readable -- reports the difference
    RELATIVE TO ITS OWN MONTE CARLO ERROR.  A raw shift in units of posterior sd
    cannot be interpreted without knowing the noise floor, which depends on the
    draw count and the autocorrelation; the z-scores below divide by an MCSE
    computed from the effective sample size, so they are on a fixed scale
    whatever the budget:

        z_loc   = |E2[f] - E1[f]| / sqrt(mcse1^2 + mcse2^2)
        z_scale = |log(sd2/sd1)| / sqrt(1/(2*ess1) + 1/(2*ess2))

    Under stationarity both behave like |N(0,1)|: **median ~0.67, 95th ~2**.
    Values well above that are drift, not noise.  The raw sd-unit shift and the
    scale ratio are also returned, because they say how *large* the drift is
    once the z-scores say it is real.

    This also tests the cyclical step-size schedule, which is an addition to
    Wu et al.: early and late cycles must be samples from one measure.

    Returns a dict of `fn_drift_*` metrics (empty if too few draws to split).
    """
    import arviz_stats as azs

    a = np.asarray(pred_chains, dtype=np.float64)
    C, D, P = a.shape
    h = D // 2
    if h < 4:
        return {}

    # Section 4.3.28: the BT/CE likelihood is exactly invariant to f -> f + c,
    # so raw f mixes the identified SHAPE with an unidentified OFFSET that
    # cancels in every preference prediction.  Gating on raw therefore rejects
    # samplers that are stationary in the part that matters, and admits ones
    # whose offset happens to be pinned by an over-tight prior -- the 4.3.14
    # pathology.  The centred metrics below are the ones section 3.6.3 gates on
    # (amended 2026-08-24); the offset metrics are REPORTED, never gated.
    _off = a.mean(axis=2)                       # [chain, draw]
    out = {}
    # Degeneracy guard (section 4.3.40).  A CONSTANT f passes every stationarity
    # criterion perfectly -- its centred component is identically zero, so
    # ratio = 1 and both z-scores are 0 -- while being maximally useless
    # (Phi1 = Phi2, so CE = log 2 exactly).  That is not a hypothetical: an
    # eps = 0.008 run collapsed this way and scored a flawless gate.  The
    # fraction of f's variance living in the IDENTIFIED component detects it,
    # is scale-free, and is near 0 only when f is essentially constant.
    _shape = a - _off[:, :, None]
    _vr = float(np.var(a))
    out["fn_drift_shape_var_frac"] = (
        float(np.var(_shape)) / _vr if _vr > 0.0 else 0.0)
    for _pfx, _arr in (("", a),
                       ("centred_", _shape),
                       ("offset_", np.broadcast_to(_off[:, :, None], a.shape))):
        out.update(_function_space_drift_core(_arr, eps, _pfx))
    return _function_space_drift_report(out)


def _function_space_drift_core(a, eps, prefix):
    """The section 4.2 drift statistics for one array.  See function_space_drift."""
    import arviz_stats as azs

    C, D, P = a.shape
    h = D // 2
    first, second = a[:, :h, :], a[:, h:2 * h, :]

    sd = a.reshape(-1, P).std(axis=0) + eps
    m1, m2 = first.mean(axis=(0, 1)), second.mean(axis=(0, 1))
    raw_loc = np.abs(m2 - m1) / sd

    sd1 = first.reshape(-1, P).std(axis=0) + eps
    sd2 = second.reshape(-1, P).std(axis=0) + eps
    ratio = sd2 / sd1

    try:
        mcse1 = np.asarray(azs.mcse(first, method="mean"), dtype=np.float64)
        mcse2 = np.asarray(azs.mcse(second, method="mean"), dtype=np.float64)
        ess1 = np.asarray(azs.ess(first, method="mean"), dtype=np.float64)
        ess2 = np.asarray(azs.ess(second, method="mean"), dtype=np.float64)
        z_loc = np.abs(m2 - m1) / (np.sqrt(mcse1 ** 2 + mcse2 ** 2) + eps)
        z_scale = np.abs(np.log(ratio)) / (
            np.sqrt(1.0 / (2.0 * np.maximum(ess1, 2.0))
                    + 1.0 / (2.0 * np.maximum(ess2, 2.0))) + eps
        )
    except Exception as e:  # noqa: BLE001 — diagnostics must not kill a run
        warnings.warn(
            f"function_space_drift: MCSE/ESS calibration failed "
            f"({type(e).__name__}: {e}); reporting raw shifts only.",
            RuntimeWarning,
        )
        z_loc = z_scale = np.full(P, np.nan)

    def fin(x):
        x = np.asarray(x, float)
        return x[np.isfinite(x)]

    out = {}
    for name, arr in (("loc_z", z_loc), ("scale_z", z_scale),
                      ("loc_sd", raw_loc), ("scale_ratio", ratio)):
        v = fin(arr)
        if v.size:
            out[f"fn_drift_{prefix}{name}_median"] = float(np.median(v))
            out[f"fn_drift_{prefix}{name}_95th"] = float(np.percentile(v, 95))
    return out


def _function_space_drift_report(out):
    """One log line covering raw and centred.  See function_space_drift."""
    _svf = out.get("fn_drift_shape_var_frac")
    if _svf is not None and _svf < 1e-4:
        print(
            f"[diag] *** DEGENERATE: only {_svf:.2e} of f's variance is in the "
            f"identified component -- f is essentially CONSTANT across inputs. "
            f"Every drift statistic below will look perfect (ratio 1, z 0) "
            f"because there is nothing left to drift.  Check predictive CE "
            f"against log 2 = 0.6931 before reading any of it. ***"
        )
    if "fn_drift_loc_z_median" in out:
        print(
            f"[diag] function-space drift (first vs second half): "
            f"z_loc {out['fn_drift_loc_z_median']:.2f} (95th "
            f"{out.get('fn_drift_loc_z_95th', float('nan')):.2f}), "
            f"z_scale {out.get('fn_drift_scale_z_median', float('nan')):.2f}  "
            f"[stationary ~0.67 / 95th ~2]; raw shift "
            f"{out.get('fn_drift_loc_sd_median', float('nan')):.3f} sd, "
            f"scale {out.get('fn_drift_scale_ratio_median', float('nan')):.3f}x"
        )
        print(
            f"[diag]   CENTRED (gated, section 3.6.3): "
            f"z_loc {out.get('fn_drift_centred_loc_z_median', float('nan')):.2f}, "
            f"z_scale {out.get('fn_drift_centred_scale_z_median', float('nan')):.2f}; "
            f"raw shift {out.get('fn_drift_centred_loc_sd_median', float('nan')):.3f} sd, "
            f"scale {out.get('fn_drift_centred_scale_ratio_median', float('nan')):.3f}x"
            f"  |  OFFSET (reported only): z_scale "
            f"{out.get('fn_drift_offset_scale_z_median', float('nan')):.2f}"
        )
    return out
