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
    print("  extremes are censored at estimator ceilings -- watch whether they")
    print("  move at all as draws increase (section 4).")
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


if __name__ == "__main__":
    main()
