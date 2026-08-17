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


def build_pred_chains(run_dir, dataset, width, depth, num_chains, b_rhat, device):
    """Load saved chains and evaluate each weight set at the diagnostic inputs.

    Mirrors the eval block of run_bnn_training.py exactly: the first `b_rhat`
    preference pairs, member 0, all non-padded timesteps, features only (the
    trailing attn_mask column is dropped).  Returns (pred_chains, x_rhat) with
    pred_chains shaped [chain, draw, point].
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
    for i in range(num_chains):
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
    print(f"[chains] loaded {n_loaded} -> using {num_chains}x{m} = "
          f"{num_chains * m} draws")
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


def weight_trace(run_dir, num_chains, depth, levels, device="cpu"):
    """Free-diffusion check on ||w||.  NOT a convergence diagnostic.

    Under fSGHMC (Wu et al. 2025) the potential `U(f) = Phi(f) + I0(f)` depends
    on w only through f, so every f-preserving direction is flat and the chain
    performs an unconfined random walk along it.  **A growing weight norm is
    what this sampler is supposed to do**, and ||w||^2 should grow linearly in
    draw index -- see HANDOFF section 3.6.2, which removed `param_rhat`,
    `param_ess` and the sampling weight-norm statistics for this reason.

    An earlier version of this function reported a `growth > 1.5x` verdict of
    "DRIFTING -- chains are not stationary".  That was the round-1 weight-space
    reading and it was wrong: at 75 draws the free-diffusion law alone produces
    growth well past 1.5x, so the check fired on healthy runs.  It has been
    removed rather than re-thresholded, because no threshold on ||w|| carries
    information about the convergence of f.

    What this IS good for, and all it is good for: confirming the chains
    diffuse as theory predicts (||w||^2 linear in t) and at a COMMON rate, so
    that no chain has a broken step size or a binding `max_param_step` clamp.
    That is a mechanical check on the integrator, not a statement about f.
    For stationarity use the function-space drift z-scores (section 4.2); for
    mixing use the tail R-hat/ESS above.

    Costs no forward passes: the saved weights are read directly.
    """
    mags = []
    for c in range(num_chains):
        chain = []
        for w in _load_chain_weights(run_dir, c, device):
            sq = sum(float(np.sum(np.asarray(a, dtype=np.float64) ** 2)) for a in w)
            n = sum(np.asarray(a).size for a in w)
            chain.append(np.sqrt(sq / n))
        mags.append(chain)
    m = min(len(c) for c in mags)
    mags = np.array([c[:m] for c in mags])                    # [chain, draw]

    levels = sorted({n for n in levels if 0 < n <= m}) or [m]
    if levels[-1] != m:
        levels.append(m)
    if levels[0] != 1:
        levels.insert(0, 1)

    print(f"\n=== WEIGHT FREE-DIFFUSION CHECK ({num_chains} chains, {m} draws) ===")
    print("  NOT A CONVERGENCE DIAGNOSTIC.  U(w) depends on w only through f, so")
    print("  every f-preserving direction is flat and |w| random-walks along it:")
    print("  a rising |w| is CORRECT behaviour here (section 3.6.2).  Read this")
    print("  only as a check that the chains diffuse as theory predicts, at a")
    print("  common rate.  For stationarity of f use the drift z-scores (4.2).")
    print("  avg |w| = total L2 norm / sqrt(n_params), matching the training")
    print("  script's warm-up readout.")
    print(f"  {'at draw':>8} | {'avg|w| mean':>11} {'min':>9} {'max':>9} "
          f"{'x vs draw 1':>11} | {'||w||^2 mean':>13}")
    print("  " + "-" * 72)
    base = float(mags[:, 0].mean())
    for n in levels:
        col = mags[:, n - 1]
        mean = float(col.mean())
        print(f"  {n:>8} | {mean:>11.4f} {col.min():>9.4f} {col.max():>9.4f} "
              f"{mean / base:>11.2f} | {mean ** 2:>13.4f}")

    # Free-diffusion law: ||w||^2 = ||w0||^2 + c*t.  Fit c per chain and report
    # how well the law holds (R^2) and how uniform the rate is across chains.
    sq = mags ** 2                                            # [chain, draw]
    t = np.arange(1, m + 1, dtype=np.float64)
    slopes, r2s = [], []
    for c in range(sq.shape[0]):
        slope, intercept = np.polyfit(t, sq[c], 1)
        resid = sq[c] - (slope * t + intercept)
        ss_tot = float(((sq[c] - sq[c].mean()) ** 2).sum())
        slopes.append(float(slope))
        r2s.append(1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0)
    slopes, r2s = np.array(slopes), np.array(r2s)
    spread = slopes.max() / slopes.min() if slopes.min() > 0 else float("inf")

    print(f"\n  free-diffusion fit  ||w||^2 = ||w0||^2 + c*t")
    print(f"    c (per chain)   mean {slopes.mean():.4f}  "
          f"min {slopes.min():.4f}  max {slopes.max():.4f}  spread {spread:.2f}x")
    print(f"    linearity R^2   min {r2s.min():.4f}  median {np.median(r2s):.4f}")
    law = ("consistent with free diffusion (expected)" if r2s.min() > 0.98
           else "NOT linear in t -- integrator worth inspecting")
    rate = ("chains diffuse at a common rate" if spread < 1.5
            else f"ANOMALOUS: chain rates differ {spread:.2f}x -- check the "
                 f"step size and max_param_step clamp on the outlier")
    print(f"    {law}")
    print(f"    {rate}")


def ce_ladder(run_dir, dataset, width, depth, num_chains, levels,
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

    n_draws = min(len(_load_chain_weights(run_dir, i, device))
                  for i in range(num_chains))
    if max_draws is not None:
        n_draws = min(n_draws, max_draws)
    print(f"[ce] {B} pairs x {T} steps, {num_chains} chains x {n_draws} draws "
          f"-> {2 * B * T * n_draws * num_chains:,} forward rows "
          f"(device={device}; use --device cuda and/or --ce-pairs if slow)")

    S1 = np.zeros((num_chains, n_draws, B), dtype=np.float64)
    S2 = np.zeros((num_chains, n_draws, B), dtype=np.float64)
    for c in range(num_chains):
        weights = _load_chain_weights(run_dir, c, device)[:n_draws]
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
        print(f"[ce] chain {c} done")

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

    print(f"\n=== CE LADDER ({num_chains} chains) ===")
    print("  plug-in   sigma(E[f])  -- what val_mean_cross_entropy logs today")
    print("  predictive E[sigma(f)] -- Wu et al. Eq. (10), the paper's predictive")
    print(f"  {'draws/ch':>8} {'total':>7} | {'plugin CE':>9} {'pred CE':>9} "
          f"{'delta':>8} | {'plugin acc':>10} {'pred acc':>9}")
    print("  " + "-" * 68)
    for n in levels:
        ces, accs, pces, paccs = [], [], [], []
        for c in range(num_chains):
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
                print(f"  [warn] chain {c}: manual CE {ce_manual:.6f} != torch "
                      f"{ce_torch:.6f}; predictive column may not be comparable")
            # --- predictive: squash each draw, then average the probabilities ---
            d = S1[c, :n] - S2[c, :n]                    # [draw, pair]
            p1_pred = (1.0 / (1.0 + np.exp(-d))).mean(0)
            pce, pacc = _ce_acc_from_probs(p1_pred)
            pces.append(pce); paccs.append(pacc)
        plug, pred = np.mean(ces), np.mean(pces)
        print(f"  {n:>8} {n * num_chains:>7} | {plug:>9.4f} {pred:>9.4f} "
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
    ap.add_argument("--num-chains", type=int, default=None)
    ap.add_argument("--max-draws", type=int, default=None,
                    help="Use only the first N draws per chain (default: all). "
                         "Lets one completed run stand in for a smaller budget.")
    ap.add_argument("--draw-ladder", default=None,
                    help="Comma-separated per-chain draw counts, e.g. "
                         "'33,75,150,305'.  Recomputes the tail statistics at "
                         "each level from this one run, so the stage-3 draw "
                         "budget is read off a curve instead of costing one "
                         "training run per candidate.")
    ap.add_argument("--weight-trace", action="store_true",
                    help="Check ||w||^2 grows linearly in draw index (the "
                         "free-diffusion law) at a common rate across chains. "
                         "Free (no forward passes). NOT a convergence "
                         "diagnostic -- a rising |w| is correct under fSGHMC; "
                         "see section 3.6.2. Use the drift z-scores instead.")
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
    num_chains = args.num_chains or cfg["num_chains"]
    print(f"[run] {args.run_dir}")
    print(f"[run] seed={cfg.get('seed')} width={width} depth={depth} "
          f"num_chains={num_chains} num_samples={cfg.get('num_samples')}")

    pred_chains, x_rhat = build_pred_chains(
        args.run_dir, dataset, width, depth, num_chains, args.b_rhat, args.device)

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

    if args.draw_ladder:
        try:
            levels = [int(t) for t in args.draw_ladder.split(",") if t.strip()]
        except ValueError:
            sys.exit(f"--draw-ladder must be comma-separated integers, got "
                     f"{args.draw_ladder!r}")
        draw_ladder(pred_chains, levels, alpha=args.alpha)

    if args.weight_trace or args.ce_ladder:
        levels = []
        if args.draw_ladder:
            levels = [int(t) for t in args.draw_ladder.split(",") if t.strip()]

    if args.weight_trace:
        weight_trace(args.run_dir, num_chains, depth, levels,
                     device=args.device)

    if args.ce_ladder:
        ce_ladder(args.run_dir, dataset, width, depth, num_chains, levels,
                  device=args.device, bt_pool=cfg.get("bt_pool", "mean"),
                  max_pairs=args.ce_pairs, max_draws=args.max_draws)


if __name__ == "__main__":
    main()
