#!/usr/bin/env python
# coding: utf-8
"""diagnose_sampling_tail.py — lower-tail MCMC convergence for a saved BNN run.

Reuses the SAVED chain weights of a completed `run_bnn_training.py` run (no
re-sampling) and reports convergence of the quantities that actually matter for
a risk-averse reward prior:

  * BULK  — reproduces the `pred_*` ESS/R-hat logged to W&B (a sanity check that
    this script matches the training pipeline).
  * VaR   — the lower 5% quantile (95% lower confidence bound).
  * CVaR  — the MEAN of the lowest 5% (conditional value-at-risk); this is the
    downstream quantity.  CVaR is harder to estimate than VaR because it averages
    the extreme tail, so it gets its own ESS / MCSE / R-hat.

Why weight-space and bulk diagnostics are NOT enough: the BNN posterior is
non-identifiable, so `param_*` R-hat/ESS are meaningless, and bulk `pred_*`
convergence reflects the median, not the tail (see memory
`bnn-sampling-tail-diagnostics`).

Methods (arviz_stats 1.1.0):
  * VaR  : ESS/MCSE via method="quantile", prob=alpha; tail R-hat via "folded".
  * CVaR : Rockafellar–Uryasev identity  CVaR_a = VaR_a + (1/a) E[min(X-VaR_a,0)]
    is EXACT, so CVaR's MC error is the mean-ESS/MCSE of the integrand
    u = (1/a) min(X - VaR_a, 0).  A between-chain CVaR spread is reported as an
    assumption-light cross-check.
  * MCSE is reported both absolute (reward units) and scale-free (÷ per-point
    posterior-predictive sd).  Reward magnitude spans orders of magnitude here,
    so only the scale-free `_max` is trustworthy.

Examples
--------
    python scripts_bnn/diagnose_sampling_tail.py \
        --run-dir exp/reward_learning/antmaze_medium_play_bnn/bnn-D4RL_antmaze-medium-play-v2-fb642974

    # override the auto-read config (e.g. a relocated dataset) and use a GPU
    python scripts_bnn/diagnose_sampling_tail.py --run-dir <dir> \
        --dataset data/antmaze/antmaze-medium-play-v2/antmaze-medium-play-v2_pref_nt.hdf5 \
        --alpha 0.05 --device cuda
"""

import argparse
import contextlib
import io
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import arviz_stats as azs
import torch
import yaml

from optbnn.bnn.nets.mlp import MLP
from optbnn.utils import util


def _load_run_config(run_dir):
    """Read the pyrallis-dumped config.yaml that every run writes to its OUT_DIR."""
    with open(os.path.join(run_dir, "config.yaml")) as f:
        return yaml.safe_load(f)


def _resolve_dataset(cfg, split, explicit=None):
    """Pick the eval split this run's `<split>_pred_*` metrics were computed on.

    `run_bnn_training_antmaze_eval.py` derives `val_dataset` / `test_dataset`
    from `antmaze_variant` + `seed` and never defines a plain `dataset` key, so
    reading `cfg["dataset"]` fails on any antmaze-eval run dir.  `dataset` is
    accepted as a legacy fallback for older run dirs.
    """
    if explicit:
        return explicit, "--dataset"
    key = f"{split}_dataset"
    if cfg.get(key):
        return cfg[key], key
    if cfg.get("dataset"):
        return cfg["dataset"], "dataset (legacy)"
    have = sorted(k for k in cfg if "dataset" in k)
    sys.exit(
        f"Could not resolve a dataset: config.yaml has no '{key}'"
        + (f" (dataset-ish keys present: {have})" if have else " and no dataset keys")
        + ".\nPass --dataset /path/to/<variant>_pref_val_<seed>.hdf5 explicitly."
    )


def _to_numpy_weights(weights):
    return tuple(
        np.asarray(a.detach().cpu().numpy()) if torch.is_tensor(a) else np.asarray(a)
        for a in weights
    )


def _chain_label(chain_ids):
    """Describe a chain selection by its on-disk `chain_N` names.

    Always names the directories rather than a count, because the two natural
    ways to say "the second half of a 16-chain run" differ by one: section 4.3.2
    calls them chains 9-16 (1-indexed prose) and they are chain_8..chain_15 on
    disk.  Printing the directory names makes the selection unambiguous in the
    captured output.
    """
    ids = list(chain_ids)
    contiguous = len(ids) > 1 and ids == list(range(ids[0], ids[-1] + 1))
    if len(ids) == 1:
        span = f"chain_{ids[0]}"
    elif contiguous:
        span = f"chain_{ids[0]}..chain_{ids[-1]}"
    else:
        span = "chain_" + ",".join(str(i) for i in ids)
    return f"{len(ids)} chain{'' if len(ids) == 1 else 's'} ({span})"


def _resolve_chain_ids(run_dir, num_chains_arg, chain_range_arg, cfg_num_chains):
    """Turn --num-chains / --chain-range into an explicit list of chain indices.

    Indices are 0-based and half-open, matching both Python slices and the
    on-disk `chain_N` names: `--chain-range 8:16` is chain_8..chain_15, the
    upper half of a 16-chain run.  `--num-chains N` remains exactly
    `--chain-range 0:N`, so every previously captured output is reproducible.
    """
    total = cfg_num_chains
    if num_chains_arg is not None and chain_range_arg is not None:
        sys.exit("--num-chains and --chain-range are mutually exclusive; "
                 "--num-chains N is the same as --chain-range 0:N.")

    if chain_range_arg is not None:
        raw = chain_range_arg.strip()
        if ":" not in raw:
            sys.exit(f"--chain-range must be START:END (0-based, END exclusive), "
                     f"got {chain_range_arg!r}.  For the upper half of a "
                     f"16-chain run use 8:16.")
        lo_s, hi_s = raw.split(":", 1)
        try:
            lo = 0 if not lo_s.strip() else int(lo_s)
            hi = total if not hi_s.strip() else int(hi_s)
        except ValueError:
            sys.exit(f"--chain-range bounds must be integers, got "
                     f"{chain_range_arg!r}")
        if lo < 0 or hi < 0:
            sys.exit("--chain-range bounds must be non-negative; negative "
                     "(from-the-end) indices are not supported.")
        if lo >= hi:
            sys.exit(f"--chain-range {lo}:{hi} is empty (END is exclusive).")
        ids = list(range(lo, hi))
    else:
        n = num_chains_arg if num_chains_arg is not None else total
        if n < 1:
            sys.exit("--num-chains must be >= 1")
        ids = list(range(n))

    # The config's num_chains is what the run was launched with, but the
    # authority on what exists is the filesystem -- a crashed or still-running
    # job can leave fewer.  Check before loading so the failure names the
    # missing directory instead of surfacing as a torch.load traceback.
    missing = [i for i in ids
               if not os.path.isdir(os.path.join(run_dir, "sampling_f",
                                                 f"chain_{i}"))]
    if missing:
        present = sorted(
            int(d.split("_")[1])
            for d in os.listdir(os.path.join(run_dir, "sampling_f"))
            if d.startswith("chain_") and d.split("_")[1].isdigit()
        )
        sys.exit(f"No saved chains at {['chain_%d' % i for i in missing]} in "
                 f"{run_dir}/sampling_f (config says num_chains={total}; "
                 f"present: {present or 'none'}).")
    return ids


def build_pred_chains(run_dir, dataset, width, depth, chain_ids, b_rhat, device):
    """Load saved chains and evaluate each weight set at the diagnostic inputs.

    Mirrors the eval block of run_bnn_training.py exactly: the first `b_rhat`
    preference pairs, member 0, all non-padded timesteps, features only (the
    trailing attn_mask column is dropped).  Returns (pred_chains, x_rhat) with
    pred_chains shaped [chain, draw, point].

    `chain_ids` is an explicit list of on-disk chain indices (the N in
    `chain_N`), so a subset need not be a prefix -- see `_resolve_chain_ids`.
    """
    X, _ = util.load_pref_data(dataset, training_ratio=1.0)
    obs_dim = X.shape[-1] - 1
    n = min(b_rhat, X.shape[0])
    block = X[:n, 0, :, :]                                   # [B, T, obs_dim+1]
    valid = block[..., obs_dim].reshape(-1) > 0.5           # attn_mask column
    x_rhat = block[..., :obs_dim].reshape(-1, obs_dim).astype(np.float32)[valid]
    print(f"[data] {dataset}")
    print(f"[data] x_rhat {x_rhat.shape}  "
          f"({int(valid.sum())}/{valid.size} valid, input_dim={obs_dim})")

    net = MLP(input_dim=obs_dim, output_dim=1,
              hidden_dims=[width] * depth, activation_fn="relu").to(device)
    net.eval()
    x_t = torch.from_numpy(x_rhat).to(device)

    pred_chains, n_loaded = [], []
    for i in chain_ids:
        path = os.path.join(run_dir, "sampling_f", f"chain_{i}",
                            "sampled_weights", "sampled_weights_0000000")
        ckpt = torch.load(path, weights_only=False, map_location=device)
        weights = [_to_numpy_weights(w) for w in ckpt["sampled_weights"]]
        n_loaded.append(len(weights))
        preds = []
        with torch.no_grad():
            for w in weights:
                for p, a in zip(net.parameters(), w):
                    p.copy_(torch.from_numpy(a).to(device))
                preds.append(net(x_t).detach().cpu().numpy().ravel())
        pred_chains.append(np.stack(preds))

    m = min(n_loaded)
    if len(set(n_loaded)) > 1:
        print(f"[warn] uneven sample counts {n_loaded}; truncating to {m}")
    pred_chains = np.stack([p[:m] for p in pred_chains])     # [chain, draw, point]
    print(f"[chains] loaded {n_loaded} -> using {len(chain_ids)}x{m} = "
          f"{len(chain_ids) * m} draws")
    return pred_chains, x_rhat


def _summ(name, arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    print(f"  {name:26s} min {a.min():9.4f}  median {np.median(a):9.4f}  "
          f"max {a.max():9.4f}")


def _load_chain_weights(run_dir, i, device):
    path = os.path.join(run_dir, "sampling_f", f"chain_{i}",
                        "sampled_weights", "sampled_weights_0000000")
    ckpt = torch.load(path, weights_only=False, map_location=device)
    return [_to_numpy_weights(w) for w in ckpt["sampled_weights"]]


def offset_shape_split(pred_chains):
    """Split the drift into the UNIDENTIFIED offset and the identified shape.

    The BT/CE likelihood is exactly invariant to a global additive shift of f.
    `bt_pool_logit` with mode="mean" returns sum(f*mask)/n (util.py:356-364), so
    a segment's logit is the mean of f over its timesteps; `LikCE` is
    CrossEntropyLoss on [Phi1, Phi2], and softmax depends only on Phi1 - Phi2.
    Hence f -> f + c leaves every preference probability UNCHANGED, and the
    data carry no information about the offset at all.  Only the functional GP
    prior constrains it, weakly.

    So drift along the offset is not evidence that the sampler is broken -- it
    is the chain exploring a direction the likelihood does not pin down, and it
    cancels in every preference prediction.  Drift in the SHAPE (f minus its
    own mean) is the part that would actually corrupt inference.

    The section 4.2 gate is computed on raw f and so mixes the two.  This
    recomputes it three ways: raw, centred (shape only), and on the offset
    alone.  If the centred numbers pass while raw fails, the stationarity
    problem is confined to a direction that does not matter downstream -- a
    constant reward offset also leaves the IQL greedy policy unchanged.
    """
    a = np.asarray(pred_chains, dtype=np.float64)
    C, D, P = a.shape
    print("\n=== OFFSET vs SHAPE DECOMPOSITION ===")
    print("  The BT/CE likelihood is EXACTLY invariant to f -> f + c:")
    print("  Phi pools by masked mean (util.py:356-364) and CrossEntropy on")
    print("  [Phi1, Phi2] depends only on Phi1 - Phi2.  The offset is therefore")
    print("  unidentified by the data and constrained only by the GP prior.")
    print("  Drift along it cancels in every preference prediction.")

    off = a.mean(axis=2)                       # [chain, draw] global offset
    shape = a - off[:, :, None]                # identified part

    rows = []
    for label, arr in (("raw f", a),
                       ("centred (shape)", shape),
                       ("offset only", off[:, :, None])):
        d = util.function_space_drift(arr, quiet=True)
        rows.append((label,
                     d.get("fn_drift_loc_sd_median", float("nan")),
                     d.get("fn_drift_scale_ratio_median", float("nan")),
                     d.get("fn_drift_loc_z_median", float("nan")),
                     d.get("fn_drift_scale_z_median", float("nan"))))

    print(f"\n  {'':17} {'loc_sd':>9} {'ratio':>9} {'loc_z':>9} {'scale_z':>9}")
    print("  " + "-" * 57)
    for label, ls, sr, lz, sz in rows:
        print(f"  {label:<17} {ls:>9.4f} {sr:>9.4f} {lz:>9.4f} {sz:>9.4f}")

    raw_ls, cen_ls = rows[0][1], rows[1][1]
    raw_lz, cen_lz = rows[0][3], rows[1][3]
    raw_sz, cen_sz = rows[0][4], rows[1][4]
    if (np.isfinite(raw_lz) and raw_lz <= 2.0
            and np.isfinite(raw_sz) and raw_sz <= 2.0):
        # Nothing to attribute.  Without this guard a stationary chain gets
        # told it has "a genuine sampling failure", because the fraction
        # explained is meaningless when the numerator is noise.
        print(f"\n  raw f already PASSES both gates (loc_z {raw_lz:.4f}, "
              f"scale_z {raw_sz:.4f}),")
        print("  so there is no drift to decompose.  The split below")
        print("  is noise divided by noise -- do not read a verdict from it.")
        return rows
    if not (raw_ls > 0 and np.isfinite(cen_ls)):
        return rows

    # Verdict keys on the GATE OUTCOMES, not on the fraction removed.  An
    # earlier version branched on `frac > 0.5` and mis-verdicted the real c16
    # run, where frac is exactly 0.500 while the centred loc_z (1.2566) passes
    # a gate the raw loc_z (2.5152) fails -- the fraction is descriptive, the
    # gate is what decides.  Scale is judged separately because centring can
    # make it WORSE: removing a common offset leaves a shape whose spread ratio
    # is larger than raw f's, and a widening in the identified part is not
    # excused by any invariance.
    frac = 1.0 - (cen_ls / raw_ls)
    print(f"\n  centring removes {frac * 100:.1f}% of the location drift "
          f"({raw_ls:.4f} -> {cen_ls:.4f}); the fraction is descriptive, the")
    print("  gate outcomes below are what decide.")

    loc_fixed = np.isfinite(cen_lz) and cen_lz <= 2.0 < raw_lz
    loc_ok = np.isfinite(cen_lz) and cen_lz <= 2.0
    scale_fails = np.isfinite(cen_sz) and cen_sz > 2.0
    scale_worse = np.isfinite(cen_sz) and np.isfinite(raw_sz) and cen_sz > raw_sz

    if loc_fixed:
        print(f"  -> LOCATION: the centred shape PASSES (loc_z {cen_lz:.4f}) a")
        print(f"     gate raw f FAILS (loc_z {raw_lz:.4f}), and the offset")
        print(f"     alone carries loc_z {rows[2][3]:.4f}.  The location drift")
        print("     is largely in the direction the likelihood cannot see.")
        print("     Preference predictions are unaffected by construction, and")
        print("     a constant reward offset leaves the IQL greedy policy")
        print("     unchanged.")
    elif loc_ok:
        print(f"  -> LOCATION: raw f already passed; centred loc_z "
              f"{cen_lz:.4f}.  Nothing to attribute here.")
    else:
        print(f"  -> LOCATION: the centred shape STILL FAILS "
              f"(loc_z {cen_lz:.4f}).")
        print("     The location drift is in the part of f the data DOES")
        print("     identify.  The offset invariance excuses none of it.")

    if scale_fails:
        print(f"  -> SCALE: the centred shape FAILS (scale_z {cen_sz:.4f}, raw"
              f" ratio {rows[1][2]:.4f}).")
        if scale_worse:
            print(f"     Centring made it WORSE (raw scale_z {raw_sz:.4f}), so")
            print("     the offset drift was partly MASKING it: the identified")
            print("     part of f is widening faster than raw f suggested.")
        print("     Scale is NOT invariant -- f -> a*f changes every preference")
        print("     probability -- so this is a real non-stationarity in the")
        print("     identified component, and it is what remains to be fixed.")
    elif np.isfinite(cen_sz):
        print(f"  -> SCALE: the centred shape passes (scale_z {cen_sz:.4f}).")

    if loc_fixed and not scale_fails:
        print("\n  Both centred gates pass: the failure is confined to the")
        print("  unidentified offset.  NOTE the tail statistics are still")
        print("  offset-sensitive -- CVaR of f moves with c -- so recompute")
        print("  them on centred f before selecting on them.")
    elif loc_fixed and scale_fails:
        print("\n  MIXED: the location drift is largely the free offset, but a")
        print("  scale non-stationarity remains in the identified shape.  Do")
        print("  not read the offset invariance as a clean bill of health.")
    return rows


def weight_f_coupling(run_dir, chain_ids, pred_chains, device="cpu",
                      n_perm=20000, seed=0):
    """Does weight-space growth explain the function-space drift?

    NOT a convergence diagnostic, and not a reinstatement of the removed
    `--weight-trace`.  Section 3.6.2 is right that weight-space statistics say
    nothing about convergence on their own: U(w) depends on w only through f, so
    the chain diffuses freely along f-preserving directions and a growing ||w||
    is expected behaviour.  This asks a different question -- whether that
    diffusion is only APPROXIMATELY f-preserving, which is the last surviving
    explanation for the common drift (sections 4.3.6, 4.3.8).  ||w|| is used
    only as a REGRESSOR against an f-space quantity that was measured
    independently; no claim is read off ||w|| by itself.

    The test is ACROSS CHAINS: chains that grew their weights more should, under
    the leak hypothesis, have drifted more in f.  That comparison has no common
    trend to fake it.

    KNOWN LIMITATION, measured 2026-08-19 (section 4.3.9): on medium_play every
    chain grows ||w|| by 1.51x to within +-1%, so the regressor has almost no
    variation and the test has no leverage -- it returned r ~ 0.02-0.04 with
    p ~ 0.9 on both c16 and jit16.  That is NO POWER, not no leak: an
    across-chain correlation can only see a mechanism that VARIES across
    chains, and a common cause of a common effect is invisible to it.  Check
    the wGrowth range before reading any verdict here.  `offset_shape_split`
    is the more informative follow-up.

    A within-chain correlation is also printed, but it is CONFOUNDED and must
    not be read as evidence: if ||w|| and f are both monotone in draw index --
    which is exactly what a drifting chain looks like -- they correlate near 1
    whether or not one causes the other.  It is shown only because a near-zero
    within-chain r would be informative in the negative direction.
    """
    C, D, P = pred_chains.shape
    h = D // 2
    if h < 2:
        print("\n=== WEIGHT-SPACE / FUNCTION-SPACE COUPLING ===")
        print("  too few draws -- not computed.")
        return

    print("\n=== WEIGHT-SPACE / FUNCTION-SPACE COUPLING ===")
    print("  Tests whether the diffusion is only APPROXIMATELY f-preserving")
    print("  (3.6.2) -- the last hypothesis standing for the common drift.")
    rows = []
    for k, i in enumerate(chain_ids):
        ws = _load_chain_weights(run_dir, i, device)[:D]
        wn = np.array([
            math.sqrt(sum(float(np.square(np.asarray(a, dtype=np.float64)).sum())
                          for a in w))
            for w in ws
        ])
        f = np.asarray(pred_chains[k], dtype=np.float64).mean(axis=1)
        w_growth = float(wn[h:2 * h].mean() / max(wn[:h].mean(), 1e-12))
        f_shift = float(f[h:2 * h].mean() - f[:h].mean())
        a_, b_ = wn[:2 * h], f[:2 * h]
        r_in = (float(np.corrcoef(a_, b_)[0, 1])
                if a_.std() > 0 and b_.std() > 0 else float("nan"))
        rows.append((i, float(wn[0]), w_growth, f_shift, r_in))

    print(f"  {'chain':>6} {'||w||_0':>10} {'wGrowth':>9} {'fShift':>10} "
          f"{'r_within':>9}")
    print("  " + "-" * 49)
    for i, w0, g, s, r in rows:
        print(f"  {i:>6} {w0:>10.3f} {g:>9.4f} {s:>10.4f} {r:>9.4f}")

    g = np.array([r[2] for r in rows])
    s = np.array([r[3] for r in rows])
    r_in_med = float(np.nanmedian([r[4] for r in rows]))
    print(f"\n  ||w|| growth (2nd half / 1st half): median {np.median(g):.4f}, "
          f"range {g.min():.4f}-{g.max():.4f}")
    print(f"  within-chain r: median {r_in_med:.4f}  "
          f"(CONFOUNDED by common trend -- do not read as support)")

    if g.std() < 1e-9 or s.std() < 1e-9 or len(g) < 4:
        print("  across-chain test needs >=4 chains with variation -- skipped.")
        return

    def _r(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    rng = np.random.default_rng(seed)
    for label, y in (("signed fShift", s), ("|fShift|", np.abs(s))):
        r_obs = _r(g, y)
        null = np.array([_r(g, rng.permutation(y)) for _ in range(n_perm)])
        p = float((np.abs(null) >= abs(r_obs)).mean())
        print(f"  across-chain r(wGrowth, {label:>13}) = {r_obs:>7.4f}   "
              f"permutation p = {p:.4f}  (n = {len(g)})")

    r_main = _r(g, np.abs(s))
    null = np.array([_r(g, rng.permutation(np.abs(s))) for _ in range(n_perm)])
    p_main = float((np.abs(null) >= abs(r_main)).mean())
    if p_main < 0.05 and r_main > 0:
        print("  -> SUPPORTS the leak: chains that grew ||w|| more drifted more")
        print("     in f.  The diffusion is not exactly f-preserving.")
    elif p_main < 0.05 and r_main < 0:
        print("  -> Significant but NEGATIVE, which the leak hypothesis does")
        print("     not predict.  Treat as unexplained, not as support.")
    else:
        print("  -> NO across-chain association detected.  Either the leak is")
        print("     not the mechanism, or 16 chains cannot resolve it -- with")
        print("     this n only a large effect would show, so this is weak")
        print("     evidence of absence.  Check the wGrowth range above: if")
        print("     the chains barely differ in growth, the regressor has no")
        print("     variation to work with and the test is uninformative.")


def cvar_ce(run_dir, dataset, width, depth, chain_ids, device="cpu",
            bt_pool="mean", alpha=0.05, max_pairs=None, max_draws=None,
            chunk_pairs=64):
    """Validation CE computed from the CVaR reward -- a selection objective.

    Section 4.3.14 showed that selecting on the posterior-MEAN predictive CE
    drives the prior to improperness: mean CE averages over the posterior, so it
    is robust to sampler defects by construction and cannot see a badly sampled
    tail.  That is how a runaway `map_amp2` scored well while sampling badly.

    This computes the same *form* of metric -- validation cross-entropy, so the
    MR/PT comparison stays like-for-like -- from the quantity the BNN actually
    DEPLOYS.  CVaR is taken per state-action, because that is the conservative
    reward IQL consumes:

        r_cvar(s,a) = CVaR_alpha[f(s,a)]  = mean of the lowest alpha fraction
        Phi_i       = masked pool of r_cvar over the segment's timesteps
        CE          = BCE on sigma(Phi_1 - Phi_2)

    Taking CVaR per (s,a) and pooling afterwards is deliberate: CVaR of the
    pooled logit is a different (and undeployed) quantity, and the difference
    matters because segments visit different states and so carry different
    posterior widths.

    Reports the mean-based metrics alongside so the two objectives can be
    ranked against each other, and a JACKKNIFE-OVER-CHAINS standard error on
    the CVaR CE.  That SE is the point of the exercise as much as the value: a
    selection objective is only usable if its run-to-run error is smaller than
    the differences it must resolve, and at alpha=0.05 the tail holds only
    `alpha * n_draws` draws (30 at 8x75, 15 at 4x75).
    """
    from optbnn.utils.util import bt_pool_logit_np

    X, y = util.load_pref_data(dataset, training_ratio=1.0)
    if max_pairs is not None and X.shape[0] > max_pairs:
        X, y = X[:max_pairs], y[:max_pairs]
        print(f"[cvar-ce] subsampled to the first {max_pairs} pairs -- CE will "
              f"NOT match the logged value")
    B, _, T, d_dim = X.shape
    obs_dim = d_dim - 1
    am1 = X[:, 0, :, obs_dim].astype(np.float32)
    am2 = X[:, 1, :, obs_dim].astype(np.float32)
    x1 = X[:, 0, :, :obs_dim].reshape(-1, obs_dim).astype(np.float32)
    x2 = X[:, 1, :, :obs_dim].reshape(-1, obs_dim).astype(np.float32)

    net = MLP(input_dim=obs_dim, output_dim=1,
              hidden_dims=[width] * depth, activation_fn="relu").to(device)
    net.eval()
    x1_t, x2_t = torch.from_numpy(x1).to(device), torch.from_numpy(x2).to(device)

    C = len(chain_ids)
    n_draws = min(len(_load_chain_weights(run_dir, i, device)) for i in chain_ids)
    if max_draws is not None:
        n_draws = min(n_draws, max_draws)

    # Per-TIMESTEP predictions are required (CVaR is per state-action), so this
    # holds [chain, draw, pair, step] rather than ce_ladder's pooled logits.
    P1 = np.empty((C, n_draws, B, T), dtype=np.float32)
    P2 = np.empty((C, n_draws, B, T), dtype=np.float32)
    print(f"[cvar-ce] {B} pairs x {T} steps, {_chain_label(chain_ids)} x "
          f"{n_draws} draws -> {2 * B * T * n_draws * C:,} forward rows "
          f"({2 * P1.nbytes / 1e6:.0f} MB held; device={device})")
    for c, cid in enumerate(chain_ids):
        weights = _load_chain_weights(run_dir, cid, device)[:n_draws]
        for d, w in enumerate(weights):
            with torch.no_grad():
                for p, a in zip(net.parameters(), w):
                    p.copy_(torch.from_numpy(a).to(device))
                for s in range(0, B, chunk_pairs):
                    e = min(s + chunk_pairs, B)
                    P1[c, d, s:e] = net(x1_t[s * T:e * T]).cpu().numpy().reshape(e - s, T)
                    P2[c, d, s:e] = net(x2_t[s * T:e * T]).cpu().numpy().reshape(e - s, T)
        print(f"[cvar-ce] chain {cid} done")

    yv = np.asarray(y, dtype=np.float64)
    eps = 1e-12

    def _ce_acc(p1):
        ce = -(yv[:, 0] * np.log(p1 + eps)
               + yv[:, 1] * np.log(1.0 - p1 + eps)).mean()
        return float(ce), float(((p1 > 0.5) == (yv[:, 0] > 0.5)).mean())

    def _cvar_over(idx):
        """CVaR CE using only the chains in `idx`.  Pools their draws."""
        a1 = P1[idx].reshape(-1, B, T).astype(np.float64)
        a2 = P2[idx].reshape(-1, B, T).astype(np.float64)
        S = a1.shape[0]
        k = max(1, int(math.floor(alpha * S)))
        # mean of the k lowest draws, per (pair, step)
        r1 = np.sort(a1, axis=0)[:k].mean(axis=0)
        r2 = np.sort(a2, axis=0)[:k].mean(axis=0)
        f1 = bt_pool_logit_np(r1 * am1, am1, bt_pool)
        f2 = bt_pool_logit_np(r2 * am2, am2, bt_pool)
        p1 = 1.0 / (1.0 + np.exp(-(f1 - f2)))
        return _ce_acc(p1) + (S, k)

    allc = np.arange(C)
    cvar_ce_v, cvar_acc, S_tot, k_tail = _cvar_over(allc)

    # Mean-based comparators on the same draws.
    m1 = P1.reshape(-1, B, T).astype(np.float64)
    m2 = P2.reshape(-1, B, T).astype(np.float64)
    g1 = bt_pool_logit_np(m1.mean(axis=0) * am1, am1, bt_pool)
    g2 = bt_pool_logit_np(m2.mean(axis=0) * am2, am2, bt_pool)
    plug_ce, plug_acc = _ce_acc(1.0 / (1.0 + np.exp(-(g1 - g2))))
    l1 = np.stack([bt_pool_logit_np(m1[s] * am1, am1, bt_pool) for s in range(m1.shape[0])])
    l2 = np.stack([bt_pool_logit_np(m2[s] * am2, am2, bt_pool) for s in range(m2.shape[0])])
    pred_ce, pred_acc = _ce_acc((1.0 / (1.0 + np.exp(-(l1 - l2)))).mean(axis=0))

    # Jackknife over chains: leave one chain out, C refits.
    jk = [_cvar_over(np.delete(allc, i))[0] for i in range(C)] if C > 1 else []
    if jk:
        jb = float(np.mean(jk))
        se = float(math.sqrt((C - 1) / C * np.sum((np.asarray(jk) - jb) ** 2)))
    else:
        se = float("nan")

    print(f"\n=== CVaR CROSS-ENTROPY (alpha={alpha}) ===")
    print("  A selection objective computed from the DEPLOYED quantity.")
    print("  Mean CE averages over the posterior and so cannot see a badly")
    print("  sampled tail (4.3.14); this can.  Same FORM as the MR/PT metric,")
    print("  so the cross-family comparison stays like-for-like (4.3.14).")
    print(f"  {'metric':<28}{'CE':>10}{'acc':>10}")
    print("  " + "-" * 48)
    print(f"  {'plug-in  sigma(E[f])':<28}{plug_ce:>10.4f}{plug_acc:>10.4f}")
    print(f"  {'predictive E[sigma(f)]':<28}{pred_ce:>10.4f}{pred_acc:>10.4f}")
    print(f"  {'CVaR      sigma(Phi_cvar)':<28}{cvar_ce_v:>10.4f}{cvar_acc:>10.4f}")
    # ---- Is a large CVaR CE a SCALE blow-up or a REORDERING? ------------
    # Section 4.3.47: large_play scored CVaR CE 10.02 against log 2 = 0.6931.
    # CE is unbounded above, and two very different failures produce a big one:
    #   * SCALE   -- |Phi1 - Phi2| inflates, so even correct signs become
    #                over-confident and the wrong ones are punished enormously.
    #   * REORDER -- |Phi1 - Phi2| is normal but CVaR ranks segments differently
    #                from the mean, because CVaR = mean - k*sd and sd varies
    #                across states, so segments through wide-posterior regions
    #                are penalised more.
    # These call for different fixes, so the diagnostic should not leave the
    # reader to guess.  The mean-based logit is the reference: it is computed on
    # the same draws and is known to predict well whenever mean CE is good.
    d_cvar = f1 - f2
    d_mean = g1 - g2
    yv0 = yv[:, 0] > 0.5
    flip = float((np.sign(d_cvar) != np.sign(d_mean)).mean())
    wrong = np.sign(d_cvar) != np.where(yv0, 1.0, -1.0)
    print(f"\n  --- CVaR logit sanity check (section 4.3.47) ---")
    print(f"  {'':22}{'median |d|':>12}{'95th |d|':>11}{'max |d|':>10}")
    for lab, d in (("CVaR  Phi1-Phi2", d_cvar), ("mean  Phi1-Phi2", d_mean)):
        ad = np.abs(d)
        print(f"  {lab:<22}{np.median(ad):>12.4f}{np.percentile(ad, 95):>11.4f}"
              f"{ad.max():>10.4f}")
    _sc = float(np.median(np.abs(d_cvar)) / max(np.median(np.abs(d_mean)), 1e-12))
    print(f"  CVaR/mean magnitude ratio {_sc:.2f}x")
    print(f"  sign disagreement with the mean logit: {flip * 100:.1f}% of pairs")
    if wrong.any():
        print(f"  wrong-signed CVaR pairs: {wrong.mean() * 100:.1f}%, "
              f"median |d| there {np.median(np.abs(d_cvar[wrong])):.4f}")
    if _sc > 3.0 and flip < 0.2:
        print("  -> SCALE blow-up: the CVaR logit is inflated but ordered like")
        print("     the mean.  The reward is over-confident, not mis-ranked;")
        print("     look at what widened the posterior, not at CVaR itself.")
    elif flip >= 0.2:
        print("  -> REORDERING: CVaR ranks a large share of pairs differently")
        print("     from the mean, i.e. per-state posterior WIDTH is driving the")
        print("     comparison.  That is CVaR doing what it is defined to do, on")
        print("     a posterior whose widths are not trustworthy.")
    else:
        print("  -> Neither pathology is pronounced; a large CE here is ordinary")
        print("     mis-prediction rather than a structural artefact.")

    print(f"\n  jackknife-over-chains SE on the CVaR CE: {se:.4f}")
    print(f"  tail depth: {k_tail} of {S_tot} draws at alpha={alpha}")
    if not math.isnan(se):
        print(f"  -> this objective can only resolve differences >~ {2 * se:.4f}"
              f" in CVaR CE.")
        if k_tail < 30:
            print(f"  !! only {k_tail} draws in the tail.  The CVaR estimate is")
            print("     dominated by a handful of draws, and the SE above is")
            print("     the honest cost of that.  Raise num_chains or")
            print("     num_samples before selecting on this.")
    return {"cvar_ce": cvar_ce_v, "cvar_acc": cvar_acc, "cvar_ce_se": se,
            "plug_ce": plug_ce, "pred_ce": pred_ce, "tail_draws": k_tail,
            "total_draws": S_tot}


def ce_ladder(run_dir, dataset, width, depth, chain_ids, levels,
              device="cpu", bt_pool="mean", max_pairs=None, max_draws=None,
              chunk_pairs=64):
    """Posterior-predictive cross-entropy and accuracy vs per-chain draw count.

    The tail diagnostics measure *convergence*, not correctness, so they cannot
    detect a sampler that mixes well in the wrong region.  CE can, which is why
    it belongs on the ladder despite section 4 saying not to select on it: here
    it is a divergence detector, not the selection metric.

    Reproduces `f_pref_net.eval_test_data` exactly: the posterior predictive is
    the mean over a chain's draws taken in REWARD space, then masked, then
    mean-pooled to a Bradley-Terry logit; CE is `CrossEntropyLoss` over the
    two pooled logits, computed per chain and averaged across chains (which is
    what `<split>_mean_cross_entropy` logs).

    Masking and mean-pooling are linear, so they commute with the average over
    draws.  Each draw is therefore evaluated ONCE and the levels are cumulative
    means of the per-draw pooled logits -- the whole ladder costs what a single
    full-budget evaluation costs.
    """
    from optbnn.metrics.metrics_tensor import accuracy
    from optbnn.utils.util import bt_pool_logit_np

    X, y = util.load_pref_data(dataset, training_ratio=1.0)
    if max_pairs is not None and X.shape[0] > max_pairs:
        X, y = X[:max_pairs], y[:max_pairs]
        print(f"[ce] subsampled to the first {max_pairs} pairs -- CE will NOT "
              f"match the logged value (which uses all pairs)")
    B, _, T, d_dim = X.shape
    obs_dim = d_dim - 1
    am1 = X[:, 0, :, obs_dim].astype(np.float32)
    am2 = X[:, 1, :, obs_dim].astype(np.float32)
    x1 = X[:, 0, :, :obs_dim].reshape(-1, obs_dim).astype(np.float32)
    x2 = X[:, 1, :, :obs_dim].reshape(-1, obs_dim).astype(np.float32)

    net = MLP(input_dim=obs_dim, output_dim=1,
              hidden_dims=[width] * depth, activation_fn="relu").to(device)
    net.eval()
    x1_t = torch.from_numpy(x1).to(device)
    x2_t = torch.from_numpy(x2).to(device)

    C = len(chain_ids)
    n_draws = min(len(_load_chain_weights(run_dir, i, device))
                  for i in chain_ids)
    if max_draws is not None:
        n_draws = min(n_draws, max_draws)
    print(f"[ce] {B} pairs x {T} steps, {_chain_label(chain_ids)} x {n_draws} "
          f"draws -> {2 * B * T * n_draws * C:,} forward rows "
          f"(device={device}; use --device cuda and/or --ce-pairs if slow)")

    S1 = np.zeros((C, n_draws, B), dtype=np.float64)
    S2 = np.zeros((C, n_draws, B), dtype=np.float64)
    for c, cid in enumerate(chain_ids):
        weights = _load_chain_weights(run_dir, cid, device)[:n_draws]
        for d, w in enumerate(weights):
            p1 = np.empty((B, T), dtype=np.float32)
            p2 = np.empty((B, T), dtype=np.float32)
            with torch.no_grad():
                for p, a in zip(net.parameters(), w):
                    p.copy_(torch.from_numpy(a).to(device))
                for s in range(0, B, chunk_pairs):
                    e = min(s + chunk_pairs, B)
                    p1[s:e] = net(x1_t[s * T:e * T]).cpu().numpy().reshape(e - s, T)
                    p2[s:e] = net(x2_t[s * T:e * T]).cpu().numpy().reshape(e - s, T)
            S1[c, d] = bt_pool_logit_np(p1 * am1, am1, bt_pool)
            S2[c, d] = bt_pool_logit_np(p2 * am2, am2, bt_pool)
        print(f"[ce] chain {cid} done")

    y_t = torch.from_numpy(y).float().to(device)
    yv = np.asarray(y, dtype=np.float64)
    levels = sorted({n for n in levels if 0 < n <= n_draws}) or [n_draws]
    if levels[-1] != n_draws:
        levels.append(n_draws)

    def _ce_acc_from_probs(p1):
        """CE and accuracy from P(member 1 preferred), matching the training
        script's soft-label CrossEntropyLoss when p1 = softmax(plug-in logits)."""
        eps = 1e-12
        ce = -(yv[:, 0] * np.log(p1 + eps) + yv[:, 1] * np.log(1.0 - p1 + eps)).mean()
        acc = ((p1 > 0.5) == (yv[:, 0] > 0.5)).mean()
        return float(ce), float(acc)

    print(f"\n=== CE LADDER ({_chain_label(chain_ids)}) ===")
    print("  plug-in   sigma(E[f])  -- what val_mean_cross_entropy logs today")
    print("  predictive E[sigma(f)] -- Wu et al. Eq. (10), the paper's predictive")
    print(f"  {'draws/ch':>8} {'total':>7} | {'plugin CE':>9} {'pred CE':>9} "
          f"{'delta':>8} | {'plugin acc':>10} {'pred acc':>9}")
    print("  " + "-" * 68)
    for n in levels:
        ces, accs, pces, paccs = [], [], [], []
        for c in range(C):
            # --- plug-in: average f over draws, then squash ---
            m1, m2 = S1[c, :n].mean(0), S2[c, :n].mean(0)
            fx_t = torch.from_numpy(np.stack([m1, m2], 1).astype(np.float32)).to(device)
            ce_torch = float(torch.nn.CrossEntropyLoss()(fx_t, y_t).cpu())
            ces.append(ce_torch)
            accs.append(float(accuracy(fx_t, y_t).cpu()))
            # cross-check our manual formula against the training script's loss
            p1_plug = 1.0 / (1.0 + np.exp(-(m1 - m2)))
            ce_manual, _ = _ce_acc_from_probs(p1_plug)
            if abs(ce_manual - ce_torch) > 1e-4:
                print(f"  [warn] chain {chain_ids[c]}: manual CE {ce_manual:.6f} "
                      f"!= torch {ce_torch:.6f}; predictive column may not be "
                      f"comparable")
            # --- predictive: squash each draw, then average the probabilities ---
            d = S1[c, :n] - S2[c, :n]                    # [draw, pair]
            p1_pred = (1.0 / (1.0 + np.exp(-d))).mean(0)
            pce, pacc = _ce_acc_from_probs(p1_pred)
            pces.append(pce); paccs.append(pacc)
        plug, pred = np.mean(ces), np.mean(pces)
        print(f"  {n:>8} {n * C:>7} | {plug:>9.4f} {pred:>9.4f} "
              f"{pred - plug:>+8.4f} | {np.mean(accs):>10.4f} {np.mean(paccs):>9.4f}")
    print(f"\n  ln 2 = {np.log(2):.4f} is chance.  A CE that RISES with draws means")
    print("  the added draws are worse than the ones before them.")
    print("  delta > 0 means the plug-in is OPTIMISTIC: it scores the posterior")
    print("  better than the paper's predictive does.  The plug-in is blind to")
    print("  posterior WIDTH -- two posteriors with the same mean reward and very")
    print("  different spread score identically -- while the downstream CVaR is")
    print("  entirely a function of that spread.")


def compute_stats(pred_chains, alpha=0.05):
    """Key tail statistics as a dict, without printing.  Backs the draw ladder.

    Reports the median / 95th-pct / %-over-threshold variants alongside the
    `_max`/`_min` extremes.  The extremes saturate at estimator ceilings once any
    single point has fully separated chains, so at small draw counts they rank
    nothing; they are kept here precisely so the ladder shows WHETHER they
    de-saturate as draws increase (HANDOFF_HP_SELECTION.md section 4).
    """
    C, D, P = pred_chains.shape
    flat = pred_chains.reshape(-1, P)
    pred_sd = flat.std(axis=0)
    eps = 1e-8

    var = np.quantile(flat, alpha, axis=0)
    u = np.minimum(pred_chains - var[None, None, :], 0.0) / alpha
    ess_cvar = np.asarray(azs.ess(u, method="mean"))
    rel_cvar = np.asarray(azs.mcse(u, method="mean")) / (pred_sd + eps)
    rhat_cvar = np.asarray(azs.rhat(u, method="folded"))
    ess_var = np.asarray(azs.ess(pred_chains, method="quantile", prob=alpha))
    rhat_fold = np.asarray(azs.rhat(pred_chains, method="folded"))

    def fin(a):
        a = np.asarray(a, float)
        return a[np.isfinite(a)]

    return dict(
        chains=C, draws=D, total=C * D,
        cvar_ess_med=float(np.median(fin(ess_cvar))),
        cvar_ess_min=float(fin(ess_cvar).min()),
        cvar_rhat_med=float(np.median(fin(rhat_cvar))),
        cvar_rhat_max=float(fin(rhat_cvar).max()),
        cvar_rhat_pct=float(100 * np.mean(fin(rhat_cvar) > 1.01)),
        cvar_relmcse_med=float(np.median(fin(rel_cvar))),
        cvar_relmcse_max=float(fin(rel_cvar).max()),
        var_ess_med=float(np.median(fin(ess_var))),
        var_ess_min=float(fin(ess_var).min()),
        folded_rhat_95=float(np.percentile(fin(rhat_fold), 95)),
        folded_rhat_max=float(fin(rhat_fold).max()),
        unresolved_pct=float(100 * np.mean(fin(rel_cvar) > 1.0)),
    )


def draw_ladder(pred_chains, levels, alpha=0.05):
    """Recompute the tail statistics at several per-chain draw counts.

    Answers the stage-3 question -- how many draws are enough -- from ONE
    completed run, by truncating each chain to the first N draws.  Reading a
    ladder off a single production-budget run replaces one training run per
    candidate budget (HANDOFF_HP_SELECTION.md section 4).

    Truncation takes the FIRST N draws, so a level is exactly what that run
    would have produced had it stopped early: the schedule, burn-in and
    discarded cycles are identical, and only the draw count differs.
    """
    C, D, _ = pred_chains.shape
    levels = sorted({n for n in levels if 0 < n <= D})
    if not levels:
        print(f"\n[ladder] no valid levels (chains have {D} draws each)")
        return
    if levels[-1] != D:
        levels.append(D)

    print(f"\n=== DRAW LADDER ({C} chains; per-chain draws truncated to first N) ===")
    print("  Steer on the median / 95th-pct / pct columns.  The *_max / *_min")
    print("  extremes are SPARSE, not censored (section 4.5): only alpha*C*D draws")
    print("  fall below VaR, so cvar_rhat can take few distinct values and a")
    print("  repeated one means the tail mass sits in a single chain.")
    hdr = (f"  {'draws/ch':>8} {'total':>7} | {'cvarESSmed':>10} {'cvarESSmin':>10} "
           f"{'cvarRhatMed':>11} {'cvarRhat%>1.01':>14} {'cvarRhatMax':>11} | "
           f"{'relMCSEmed':>10} {'unres%':>7} | {'varESSmed':>9} {'fold95':>7} {'foldMax':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for n in levels:
        s = compute_stats(pred_chains[:, :n, :], alpha=alpha)
        print(f"  {n:>8} {s['total']:>7} | {s['cvar_ess_med']:>10.1f} "
              f"{s['cvar_ess_min']:>10.1f} {s['cvar_rhat_med']:>11.4f} "
              f"{s['cvar_rhat_pct']:>14.1f} {s['cvar_rhat_max']:>11.4f} | "
              f"{s['cvar_relmcse_med']:>10.3f} {s['unresolved_pct']:>7.2f} | "
              f"{s['var_ess_med']:>9.1f} {s['folded_rhat_95']:>7.3f} "
              f"{s['folded_rhat_max']:>7.3f}")
    print("\n  Budget is sufficient where the median columns have flattened and")
    print("  relMCSEmed is comfortably below 1.  If R-hat stays high while ESS")
    print("  grows, that is a MIXING problem, not a budget problem -- more draws")
    print("  will not fix it (section 4).")


def drift_diagnostics(pred_chains):
    """Recompute the section 4.2 stationarity gate from these chains.

    `util.function_space_drift` normally runs inside training and its
    `fn_drift_*` metrics reach wandb for the run as a WHOLE.  Recomputing it
    here is what makes a chain SUBSET measurable: wandb has no per-half figure,
    so comparing the chains a rung added against the ones it added them to
    (section 4.3.2) is only possible offline, from the saved chains.

    Prints the raw effect sizes beside the z-scores, because the z's are not
    comparable across selections of different chain count -- see section 4.2.1.
    """
    d = util.function_space_drift(pred_chains)
    if not d:
        print("\n=== SECTION 4.2 DRIFT GATE ===")
        print("  too few draws to split into halves -- not computed.")
        return d

    C, D = pred_chains.shape[0], pred_chains.shape[1]

    def g(name, stat):
        return d.get(f"fn_drift_{name}_{stat}", float("nan"))

    loc_z, scale_z = g("loc_z", "median"), g("scale_z", "median")
    loc_sd, ratio = g("loc_sd", "median"), g("scale_ratio", "median")

    print("\n=== SECTION 4.2 DRIFT GATE (first vs second half of each chain) ===")
    print(f"  {C} chains x {D} draws")
    print(f"  {'':8} {'median':>9} {'95th':>9}")
    for label, name, note in (
        ("loc_z", "loc_z", "<= 2.0 ; stationary ~0.67 median, ~2 at 95th"),
        ("scale_z", "scale_z", "<= 2.0 ; same reference"),
        ("loc_sd", "loc_sd", "RAW |E2-E1| in posterior-sd units"),
        ("ratio", "scale_ratio", "RAW sd2/sd1"),
    ):
        print(f"  {label:<8} {g(name, 'median'):>9.4f} {g(name, '95th'):>9.4f}"
              f"   {note}")
    verdict = ("PASS" if (loc_z <= 2.0 and scale_z <= 2.0) else
               "FAIL -- these chains are not sampling P_{f|D}; every tail "
               "number below is meaningless")
    print(f"  verdict  {verdict}")
    print( "  NOTE: both z-scores divide by an MCSE, so their POWER grows with")
    print(f"  the {C} chains selected here.  A PASS at a low chain count is NOT")
    print( "  evidence of stationarity, and these z's are NOT comparable to a")
    print( "  selection of a different size (4.2.1).  Two subsets of EQUAL size")
    print( "  ARE comparable -- that is what makes chains 0:8 vs 8:16 a fair")
    print( "  test.  Either way the raw loc_sd is the effect size; prefer it.")
    return d


def per_chain_drift(pred_chains, chain_ids=None):
    """Per-chain 4.2 gate, and the test that says WHAT KIND of drift it is.

    The pooled gate reports one number for a set of chains, so it cannot
    distinguish "every chain carries the same drift" from "two chains drift
    badly and the rest are fine".  Those call for opposite fixes, and the
    lower-half/upper-half split of 4.3.2 cannot separate them either -- with
    `chain_init_jitter = 0` every chain starts from the SAME warm-up point and
    differs only in its RNG stream (`set_seed(seed + chain_idx)`), so chains are
    exchangeable and any lower-vs-upper gap is chance or GPU placement, not a
    property of "later" chains.

    The discriminating statistic is the ALIGNMENT of the half-to-half shift
    across chains.  Let delta[c, p] = E2[f] - E1[f] for chain c at point p, in
    pooled-sd units.  Then per point compare

        |mean_c delta[c, p]|   (do the chains move TOGETHER?)
        mean_c |delta[c, p]|   (how far does a typical chain move?)

    Their ratio is ~1 when every chain follows one common trajectory, and
    ~1/sqrt(C) when the chains wander independently (for iid deltas the numerator
    is E|N(0, sigma^2/C)| = sigma*sqrt(2/(pi*C)) and the denominator is
    sigma*sqrt(2/pi)).  A common shift is the signature of a shared start that
    has not equilibrated: the transient is identical in every chain, so it does
    not average out over chains and adding chains cannot reduce it -- which is
    exactly the non-shrinking loc_sd of 4.3.2.  Independent wandering instead
    says each chain is exploring on its own and pooling more of them helps.
    """
    a = np.asarray(pred_chains, dtype=np.float64)
    C, D, P = a.shape
    h = D // 2
    print("\n=== PER-CHAIN DRIFT (section 4.2 gate, one chain at a time) ===")
    if h < 4 or C < 2:
        print("  needs >= 2 chains and >= 8 draws -- not computed.")
        return
    ids = list(chain_ids) if chain_ids is not None else list(range(C))

    eps = 1e-12
    sd = a.reshape(-1, P).std(axis=0) + eps
    delta = (a[:, h:2 * h, :].mean(axis=1) - a[:, :h, :].mean(axis=1)) / sd

    print(f"  {'chain':>6} {'loc_z':>9} {'loc_sd':>9} {'ratio':>9} "
          f"{'signedShift':>12}")
    print("  " + "-" * 50)
    rows = []
    for k, cid in enumerate(ids):
        with contextlib.redirect_stdout(io.StringIO()):   # mute its [diag] line
            d = util.function_space_drift(a[k:k + 1])
        loc_z = d.get("fn_drift_loc_z_median", float("nan"))
        loc_sd = d.get("fn_drift_loc_sd_median", float("nan"))
        ratio = d.get("fn_drift_scale_ratio_median", float("nan"))
        signed = float(np.median(delta[k]))
        rows.append((cid, loc_z, loc_sd, ratio, signed))
        print(f"  {cid:>6} {loc_z:>9.4f} {loc_sd:>9.4f} {ratio:>9.4f} "
              f"{signed:>12.4f}")

    locs = np.array([r[2] for r in rows], dtype=float)
    sgn = np.array([r[4] for r in rows], dtype=float)
    finite = locs[np.isfinite(locs)]
    spread = float("nan")
    if finite.size:
        med = max(float(np.median(finite)), eps)
        spread = float(finite.max()) / med
        print(f"\n  loc_sd across chains: min {finite.min():.4f}  "
              f"median {np.median(finite):.4f}  max {finite.max():.4f}  "
              f"(max/median {spread:.2f}x)")
        if spread >= 3.0:
            worst = [str(r[0]) for r in rows
                     if np.isfinite(r[2]) and r[2] >= 3.0 * med]
            print(f"  !! A FEW CHAINS DOMINATE: chain(s) {', '.join(worst)} "
                  f"drift >=3x the median.")
            print("     The pooled gate is then reporting those chains, not the")
            print("     sampler as a whole.  This is a THIRD case, distinct from")
            print("     both verdicts below: neither one common transient nor")
            print("     uniform independent wandering.  Look at what those")
            print("     chains have in common (GPU, index) before changing any")
            print("     schedule setting.")
    n_pos = int((sgn > 0).sum())
    n_tot = len(sgn)
    # Two-sided sign test.  The earlier "one direction / mixed" label only fired
    # at a unanimous split, which called 14/16 (p ~ 0.004) "mixed" while the
    # alignment statistic below called the same data a common drift.
    k = max(n_pos, n_tot - n_pos)
    p_sign = min(1.0, 2.0 * sum(math.comb(n_tot, i)
                                for i in range(k, n_tot + 1)) / 2.0 ** n_tot)
    if p_sign < 0.01:
        sign_verdict = "STRONGLY directional"
    elif p_sign < 0.05:
        sign_verdict = "directional"
    else:
        sign_verdict = "no clear direction"
    print(f"  signed shift: {n_pos}/{n_tot} chains positive -- {sign_verdict} "
          f"(sign test p = {p_sign:.4f})")

    num = np.abs(delta.mean(axis=0))
    den = np.abs(delta).mean(axis=0) + eps
    align = float(np.median(num / den))
    indep = 1.0 / np.sqrt(C)
    print(f"\n  ALIGNMENT  {align:.4f}   (~1.00 = one common drift in every "
          f"chain;")
    print(f"  {'':13}~{indep:.4f} = {C} chains wandering independently)")
    if align >= 0.5 * (1.0 + indep):
        print("  -> COMMON drift.  Every chain carries the same shift, so it")
        print("     does not average out over chains and MORE CHAINS CANNOT")
        print("     REDUCE IT (4.3.2).  Consistent with a shared start that")
        print("     has not equilibrated: with chain_init_jitter = 0 all chains")
        print("     begin at the identical warm-up point.  Attack it with")
        print("     burn-in / chain_init_jitter / the cyclical schedule.")
    else:
        print("  -> INDEPENDENT wandering.  The chains are not sharing one")
        print("     trajectory, so the shared-start transient does NOT explain")
        print("     the pooled drift, and pooling more chains does help.")
    print("  NOTE: single-chain loc_z has the fewest draws behind it of any")
    print("  reading here, so it is the LOWEST-power form of the gate (4.2.1).")
    print("  Rank chains on loc_sd; use loc_z only to compare like with like.")


def drift_blocks(pred_chains, n_blocks=5):
    """Is the drift DECAYING (a transient) or CONSTANT (an ongoing drive)?

    The 4.2 gate splits each chain in half, which gives one number and cannot
    tell those apart -- yet they call for opposite fixes:

      * A relaxation transient from a start that is not in the typical set
        decays as the chain approaches equilibrium.  Block-to-block shifts
        shrink.  MORE BURN-IN fixes it.
      * An ongoing systematic drive -- a step-size schedule that injects energy
        faster than it dissipates, or weight-space diffusion along directions
        that are only approximately f-preserving (section 3.6.2) -- displaces
        the chain by the same amount every cycle.  Block-to-block shifts stay
        flat, and NO amount of burn-in helps because the drive acts during
        sampling, after burn-in has ended.

    Splits the draws into consecutive blocks and reports, per block, the shift
    from the previous block and the cumulative shift from the first, both in
    pooled-sd units, plus each block's spread relative to the first.  The
    spread column matters independently: a chain being progressively HEATED
    widens without its mean necessarily moving, which is what `scale_ratio`
    in the 4.2 gate reports as a single number.

    Draws are collected one per cycle at the coldest step, so block index is
    proportional to cycle count and a flat shift column means a fixed
    displacement per cycle.
    """
    a = np.asarray(pred_chains, dtype=np.float64)
    C, D, P = a.shape
    print(f"\n=== DRIFT ACROSS DRAW BLOCKS ({n_blocks} blocks) ===")
    if n_blocks < 3 or D // n_blocks < 3:
        print(f"  need >= 3 blocks of >= 3 draws each; have {D} draws "
              f"-- not computed.")
        return
    w = D // n_blocks
    eps = 1e-12
    sd = a.reshape(-1, P).std(axis=0) + eps
    # Section 3.6.3 gates on the CENTRED component, and section 4.3.30's
    # contraction is a centred phenomenon that raw f can hide entirely -- on
    # medium_diverse raw f WIDENS (sd/first 1.41) while the centred shape
    # CONTRACTS (scale_ratio 0.456).  Reporting raw alone answers the wrong
    # question, so both trajectories are tracked and the verdict reads centred.
    cen = a - a.mean(axis=2, keepdims=True)
    means, sds, sds_c = [], [], []
    for b in range(n_blocks):
        blk = a[:, b * w:(b + 1) * w, :]
        means.append(blk.reshape(-1, P).mean(axis=0))
        sds.append(blk.reshape(-1, P).std(axis=0) + eps)
        sds_c.append(cen[:, b * w:(b + 1) * w, :].reshape(-1, P).std(axis=0) + eps)

    # Noise floor.  Without it a STATIONARY chain reads as a constant drive:
    # its block-to-block shifts are pure Monte Carlo scatter, which is flat by
    # construction.  For a block of `ess_b` effective draws the median |shift|
    # of a stationary chain is ~0.6745*sqrt(1/ess_a + 1/ess_b) in sd units.
    # ESS is taken on the full chains and scaled by the block fraction -- a
    # per-block ESS on w draws is too unstable to trust.
    ess_full = float(np.median(np.asarray(azs.ess(a), dtype=np.float64)))
    ess_blk = max(ess_full * (w / float(D)), 1.0)
    floor = 0.6745 * math.sqrt(2.0 / ess_blk)

    print(f"  {'block':>6} {'draws':>9} {'shift/prev':>11} {'x floor':>8} "
          f"{'cum/first':>10} {'sd/first':>9} {'sd/first(cen)':>14}")
    print("  " + "-" * 74)
    steps = []
    for b in range(n_blocks):
        lo, hi = b * w, (b + 1) * w
        cum = float(np.median(np.abs(means[b] - means[0]) / sd))
        sdr = float(np.median(sds[b] / sds[0]))
        sdc = float(np.median(sds_c[b] / sds_c[0]))
        if b == 0:
            print(f"  {b:>6} {f'{lo}-{hi - 1}':>9} {'--':>11} {'--':>8} "
                  f"{cum:>10.4f} {sdr:>9.4f} {sdc:>14.4f}")
        else:
            stp = float(np.median(np.abs(means[b] - means[b - 1]) / sd))
            steps.append(stp)
            print(f"  {b:>6} {f'{lo}-{hi - 1}':>9} {stp:>11.4f} "
                  f"{stp / floor:>8.2f} {cum:>10.4f} {sdr:>9.4f} {sdc:>14.4f}")

    print(f"\n  noise floor {floor:.4f} per block step "
          f"(ESS {ess_full:.1f} over {D} draws -> {ess_blk:.1f} per block).")
    print("  A STATIONARY chain sits AT the floor, so 'flat' only means")
    print("  'ongoing drive' when the shifts CLEAR it.")

    sd_last = float(np.median(sds[-1] / sds[0]))
    sdc_last = float(np.median(sds_c[-1] / sds_c[0]))
    sdc_traj = [float(np.median(sds_c[b] / sds_c[0])) for b in range(n_blocks)]
    _dirn = "CONTRACTING" if sdc_last < 0.95 else (
        "WIDENING" if sdc_last > 1.05 else "flat")
    print(f"\n  CENTRED spread trajectory (section 3.6.3 gates on this): "
          f"{' -> '.join(f'{v:.3f}' for v in sdc_traj)}")
    print(f"  -> the identified component is {_dirn} ({sdc_last:.4f} by the "
          f"last block).")
    if _dirn != "flat":
        _mono = all((sdc_traj[i + 1] - sdc_traj[i]) * (sdc_last - 1.0) >= -0.02
                    for i in range(n_blocks - 1))
        _step = [abs(sdc_traj[i + 1] - sdc_traj[i]) for i in range(n_blocks - 1)]
        if _step[0] > 2.5 * _step[-1]:
            print("     The per-block change is DECAYING (first step "
                  f"{_step[0]:.3f} vs last {_step[-1]:.3f}), i.e. a TRANSIENT")
            print("     from an atypical start -- initialisation explanations")
            print("     survive, and a longer burn-in is the lever.")
        elif _mono:
            print("     The per-block change is roughly CONSTANT (first step "
                  f"{_step[0]:.3f} vs last {_step[-1]:.3f}), i.e. an ONGOING")
            print("     DRIVE throughout sampling.  Initialisation is NOT the")
            print("     cause and burn-in cannot fix it (section 4.3.21).")
        else:
            print("     The per-block changes are non-monotone; neither a clean")
            print("     transient nor a constant drive.  Do not force a verdict.")
    if max(steps) < 2.0 * floor:
        print(f"  -> SHAPE NOT RESOLVED.  No block step reaches 2x the floor "
              f"(max {max(steps) / floor:.2f}x),")
        print("     so the mean shifts here are consistent with noise and the")
        print("     decaying-vs-flat question CANNOT be answered at this block")
        print("     count.  Re-run with fewer blocks, and read the two columns")
        print("     that are better determined: cum/first, and sd/first -- a")
        print("     variance ratio converges faster than a difference of means.")
        if abs(sdc_last - 1.0) > 0.2:
            print(f"     But the CENTRED spread reaches {sdc_last:.4f}, so the")
            print("     identified component IS moving even though the location")
            print("     shape is unresolved -- read the centred verdict above,")
            print(f"     not raw sd/first ({sd_last:.4f}), which mixes in the")
            print("     unidentified offset and can move the opposite way.")
        return

    if len(steps) < 2:
        return
    half = max(1, len(steps) // 2)
    early = float(np.mean(steps[:half]))
    late = float(np.mean(steps[-half:]))
    ratio = late / max(early, eps)
    print(f"  early block-to-block shift {early:.4f} -> late {late:.4f}  "
          f"({ratio:.2f}x)")
    if ratio < 0.5:
        print("  -> DECAYING.  The shift is dying out, consistent with a")
        print("     relaxation transient from the shared start.  MORE BURN-IN")
        print("     is the fix; the drive is not ongoing.")
    elif ratio > 1.5:
        print("  -> GROWING.  The chain is being driven harder as sampling")
        print("     proceeds, not settling.  Suspect an energy injection that")
        print("     accumulates -- the cyclical schedule is the first thing to")
        print("     rule out (run with use_cyclical_lr=False at matched steps).")
    else:
        print("  -> FLAT.  A constant displacement per block, i.e. per cycle.")
        print("     This is an ONGOING DRIVE acting during sampling, not a")
        print("     transient, so MORE BURN-IN CANNOT FIX IT -- burn-in ends")
        print("     before the drive starts.  Rule out the cyclical schedule")
        print("     first (use_cyclical_lr=False at matched total steps); if")
        print("     the drift survives that, suspect weight-space diffusion")
        print("     that is only approximately f-preserving (section 3.6.2).")
    if sds and float(np.median(sds[-1] / sds[0])) > 1.2:
        print("  NOTE: the spread is also growing (sd/first > 1.2 at the last")
        print("  block).  A widening chain is being heated, which points at")
        print("  energy injected per cycle rather than at a start transient.")


def tail_diagnostics(pred_chains, x_rhat=None, alpha=0.05, worst_k=0):
    """Print bulk, VaR(alpha), and CVaR(alpha) convergence diagnostics.

    If worst_k > 0 and x_rhat is given, also list the worst_k points by CVaR
    rel-MCSE with their torso (x, y) coordinates, so you can tell whether the
    unresolved points are real in-support maze states or degenerate ones.
    """
    C, D, P = pred_chains.shape
    total = C * D
    flat = pred_chains.reshape(-1, P)
    pred_sd = flat.std(axis=0)                               # per-point spread
    eps = 1e-8

    drift_diagnostics(pred_chains)

    print("\n=== BULK (should match logged pred_* ESS/R-hat) ===")
    _summ("ess_bulk", azs.ess(pred_chains))
    _summ("rhat_bulk (rank)", azs.rhat(pred_chains))

    print(f"\n=== VaR (lower {alpha:.0%} quantile = 95% lower bound) ===")
    ess_var = np.asarray(azs.ess(pred_chains, method="quantile", prob=alpha))
    mcse_var = np.asarray(azs.mcse(pred_chains, method="quantile", prob=alpha))
    _summ("VaR ESS", ess_var)
    print(f"  {'VaR ESS min / total':26s} {ess_var.min() / total:.4f}")
    _summ("VaR R-hat (folded)", azs.rhat(pred_chains, method="folded"))
    _summ("VaR MCSE / pred_sd", mcse_var / (pred_sd + eps))

    print(f"\n=== CVaR (mean of lowest {alpha:.0%} — the downstream quantity) ===")
    var = np.quantile(flat, alpha, axis=0)                   # VaR per point
    # Rockafellar-Uryasev integrand: CVaR = VaR + mean(u), u = (1/a)min(X-VaR,0)
    u = np.minimum(pred_chains - var[None, None, :], 0.0) / alpha
    ess_cvar = np.asarray(azs.ess(u, method="mean"))         # ESS for E[u]
    mcse_cvar = np.asarray(azs.mcse(u, method="mean"))       # sd(u)/sqrt(ESS)
    rhat_cvar = np.asarray(azs.rhat(u, method="folded"))
    # assumption-light cross-check: between-chain CVaR spread / sqrt(C)
    per_chain = np.stack([
        var + np.minimum(pred_chains[c] - var[None, :], 0.0).mean(axis=0) / alpha
        for c in range(C)
    ])
    mcse_bc = per_chain.std(axis=0, ddof=1) / np.sqrt(C)
    n_unresolved = int(np.sum(mcse_cvar / (pred_sd + eps) > 1.0))

    _summ("CVaR effective draws", ess_cvar)
    print(f"  {'raw draws below VaR':26s} {int(round(alpha * total))} of {total}")
    _summ("CVaR R-hat (folded)", rhat_cvar)
    _summ("CVaR MCSE / pred_sd", mcse_cvar / (pred_sd + eps))
    _summ("  cross-check (between-chain)", mcse_bc / (pred_sd + eps))
    print(f"  {'points MCSE>sd (unresolved)':26s} {n_unresolved} of {P} "
          f"({100 * n_unresolved / P:.2f}%)")

    if worst_k and x_rhat is not None:
        rel_cvar = mcse_cvar / (pred_sd + eps)
        rel_var = mcse_var / (pred_sd + eps)
        cvar_val = var + u.reshape(-1, P).mean(axis=0)       # actual CVaR reward
        pred_mean = flat.mean(axis=0)
        order = np.argsort(rel_cvar)[::-1][:worst_k]         # worst first
        # torso (x, y) is obs[:, :2] under the antmaze convention (map_xy_source=obs)
        xy = x_rhat[:, :2]
        print(f"\n=== WORST {worst_k} POINTS by CVaR rel-MCSE "
              f"(coords = torso x,y = obs[:, :2]) ===")
        print(f"  {'idx':>6} {'x':>8} {'y':>8} | {'cvar_relMCSE':>12} "
              f"{'cvar_Rhat':>9} {'cvar_ESS':>8} {'var_relMCSE':>11} "
              f"{'pred_mean':>10} {'pred_sd':>10} {'CVaR':>10}")
        for j in order:
            print(f"  {j:>6} {xy[j, 0]:>8.3f} {xy[j, 1]:>8.3f} | "
                  f"{rel_cvar[j]:>12.3f} {rhat_cvar[j]:>9.3f} {ess_cvar[j]:>8.1f} "
                  f"{rel_var[j]:>11.3f} {pred_mean[j]:>10.3f} {pred_sd[j]:>10.3f} "
                  f"{cvar_val[j]:>10.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="A run's OUT_DIR (contains config.yaml and sampling_f/).")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Lower-tail fraction for VaR/CVaR (default 0.05).")
    ap.add_argument("--b-rhat", type=int, default=64,
                    help="Number of preference pairs to evaluate (default 64, "
                         "matching run_bnn_training.py).")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (default cpu).")
    ap.add_argument("--worst-k", type=int, default=10,
                    help="List the K worst points by CVaR rel-MCSE with their "
                         "torso (x, y) coords (default 10; 0 to disable).")
    # The following default to the run's config.yaml; override only if needed.
    ap.add_argument("--split", choices=("val", "test"), default="val",
                    help="Which eval split to reproduce (default val). Picks "
                         "<split>_dataset from the run's config.yaml, so the "
                         "output is comparable to the logged <split>_pred_* "
                         "metrics.")
    ap.add_argument("--dataset", default=None,
                    help="Explicit dataset path, overriding --split.")
    ap.add_argument("--width", type=int, default=None,
                    help="Actual (already-expanded) layer width; default from config.")
    ap.add_argument("--depth", type=int, default=None)
    # Mutually exclusive so the conflict is reported before the run directory
    # is touched; _resolve_chain_ids repeats the check for programmatic callers.
    chain_sel = ap.add_mutually_exclusive_group()
    chain_sel.add_argument("--num-chains", type=int, default=None,
                    help="Use only the first N chains (default: all). Because "
                         "chains are deterministic in (seed, index), this "
                         "reproduces a lower rung of the section 4.3 ladder "
                         "exactly from a higher rung's saved chains. Same as "
                         "--chain-range 0:N.")
    chain_sel.add_argument("--chain-range", default=None,
                    help="Use chains START:END -- 0-based, END exclusive, like "
                         "a Python slice and like the on-disk chain_N names. "
                         "'8:16' is the upper half of a 16-chain run (the "
                         "chains section 4.3.2 calls 9-16 in 1-indexed prose). "
                         "Either bound may be omitted. Mutually exclusive with "
                         "--num-chains, which is the same as 0:N. Use this to "
                         "test whether the chains ADDED at a rung drift more "
                         "than the ones they were added to (section 4.3.3).")
    ap.add_argument("--per-chain-drift", action="store_true",
                    help="Also run the section 4.2 gate on each chain "
                         "separately, and report how ALIGNED the chains' "
                         "half-to-half shifts are. Separates one common drift "
                         "shared by every chain (a shared start that has not "
                         "equilibrated -- more chains cannot help) from "
                         "independent per-chain wandering (more chains do "
                         "help). Needs no extra sampling (section 4.3.4).")
    ap.add_argument("--offset-shape-split", action="store_true",
                    help="Split the section 4.2 drift into the UNIDENTIFIED "
                         "global offset (which the BT/CE likelihood is exactly "
                         "invariant to) and the identified shape. Recomputes "
                         "the gate on raw f, centred f, and the offset alone. "
                         "Needs no new sampling (section 4.3.9).")
    ap.add_argument("--weight-f-coupling", action="store_true",
                    help="Test whether weight-space growth explains the "
                         "function-space drift, by correlating per-chain ||w|| "
                         "growth against per-chain f drift ACROSS chains. Uses "
                         "||w|| only as a regressor, never as a convergence "
                         "diagnostic (section 3.6.2). Needs no new sampling "
                         "(section 4.3.8).")
    ap.add_argument("--drift-blocks", type=int, default=None, metavar="K",
                    help="Split the draws into K consecutive blocks and report "
                         "the block-to-block shift. Separates a DECAYING "
                         "transient (fix: more burn-in) from a FLAT per-cycle "
                         "drive (burn-in cannot fix it -- it acts during "
                         "sampling). Try K=5. Needs no extra sampling "
                         "(section 4.3.5).")
    ap.add_argument("--max-draws", type=int, default=None,
                    help="Use only the first N draws per chain (default: all). "
                         "Lets one completed run stand in for a smaller budget.")
    ap.add_argument("--draw-ladder", default=None,
                    help="Comma-separated per-chain draw counts, e.g. "
                         "'33,75,150,305'.  Recomputes the tail statistics at "
                         "each level from this one run, so the stage-3 draw "
                         "budget is read off a curve instead of costing one "
                         "training run per candidate.")
    # --weight-trace was removed 2026-08-18.  No statistic computed from w
    # belongs here: U depends on w only through f, so weight space carries no
    # information about convergence (section 3.6.2), and the mechanical
    # integrator check it was re-scoped to is already covered directly by
    # param_clamp_sampling_pct, which section 4.2 gates on.  Section 4.5 has the
    # full argument; section 3.6.2 keeps the one measurement it produced.
    ap.add_argument("--cvar-ce", action="store_true",
                    help="Validation CE computed from the CVaR reward -- the "
                         "quantity the BNN deploys -- alongside the mean-based "
                         "CE, plus a jackknife-over-chains SE. A candidate "
                         "SELECTION objective: mean CE averages over the "
                         "posterior and cannot see a badly sampled tail "
                         "(section 4.3.14), while keeping the same FORM as the "
                         "MR/PT metric so the comparison stays like-for-like.")
    ap.add_argument("--cvar-ce-alpha", type=float, default=0.05, metavar="A",
                    help="Tail fraction for --cvar-ce (default 0.05, matching "
                         "every CVaR diagnostic in section 4).")
    ap.add_argument("--ce-ladder", action="store_true",
                    help="Also compute posterior-predictive CE and accuracy at "
                         "each --draw-ladder level. Catches a sampler that "
                         "mixes well in the wrong region, which the tail "
                         "diagnostics cannot see. Needs a forward pass per "
                         "draw over the eval split -- use --device cuda.")
    ap.add_argument("--ce-pairs", type=int, default=None,
                    help="Use only the first N preference pairs for the CE "
                         "ladder (default: all). Faster, but then the "
                         "full-budget row no longer reproduces the logged "
                         "<split>_mean_cross_entropy.")
    args = ap.parse_args()

    cfg = _load_run_config(args.run_dir)
    dataset, src = _resolve_dataset(cfg, args.split, args.dataset)
    print(f"[split] {args.split}  (from {src}) -> compare against this run's "
          f"logged {args.split}_pred_* metrics")
    width = args.width or cfg["width"]
    depth = args.depth or cfg["depth"]
    chain_ids = _resolve_chain_ids(args.run_dir, args.num_chains,
                                   args.chain_range, cfg["num_chains"])
    print(f"[run] {args.run_dir}")
    print(f"[run] seed={cfg.get('seed')} width={width} depth={depth} "
          f"num_chains={cfg['num_chains']} num_samples={cfg.get('num_samples')}")
    print(f"[chains] using {_chain_label(chain_ids)} of the run's "
          f"{cfg['num_chains']}")
    if len(chain_ids) < cfg["num_chains"]:
        print("[chains] SUBSET -- every statistic below is for these chains "
              "only, not the full run.")

    pred_chains, x_rhat = build_pred_chains(
        args.run_dir, dataset, width, depth, chain_ids, args.b_rhat, args.device)

    if args.max_draws is not None:
        if args.max_draws < 1:
            sys.exit("--max-draws must be >= 1")
        avail = pred_chains.shape[1]
        if args.max_draws > avail:
            sys.exit(f"--max-draws {args.max_draws} exceeds the {avail} draws saved "
                     f"per chain; truncation can only go down.")
        pred_chains = pred_chains[:, :args.max_draws, :]
        print(f"[draws] truncated to first {args.max_draws} per chain -> "
              f"{pred_chains.shape[0] * args.max_draws} total")

    tail_diagnostics(pred_chains, x_rhat=x_rhat, alpha=args.alpha,
                     worst_k=args.worst_k)

    if args.per_chain_drift:
        per_chain_drift(pred_chains, chain_ids=chain_ids)

    if args.drift_blocks:
        drift_blocks(pred_chains, n_blocks=args.drift_blocks)

    if args.offset_shape_split:
        offset_shape_split(pred_chains)

    if args.weight_f_coupling:
        weight_f_coupling(args.run_dir, chain_ids, pred_chains,
                          device=args.device)

    if args.draw_ladder:
        try:
            levels = [int(t) for t in args.draw_ladder.split(",") if t.strip()]
        except ValueError:
            sys.exit(f"--draw-ladder must be comma-separated integers, got "
                     f"{args.draw_ladder!r}")
        draw_ladder(pred_chains, levels, alpha=args.alpha)

    if args.cvar_ce:
        cvar_ce(args.run_dir, dataset, width, depth, chain_ids,
                device=args.device, bt_pool=cfg.get("bt_pool", "mean"),
                alpha=args.cvar_ce_alpha, max_pairs=args.ce_pairs,
                max_draws=args.max_draws)

    if args.ce_ladder:
        levels = []
        if args.draw_ladder:
            levels = [int(t) for t in args.draw_ladder.split(",") if t.strip()]
        ce_ladder(args.run_dir, dataset, width, depth, chain_ids, levels,
                  device=args.device, bt_pool=cfg.get("bt_pool", "mean"),
                  max_pairs=args.ce_pairs, max_draws=args.max_draws)


if __name__ == "__main__":
    main()
