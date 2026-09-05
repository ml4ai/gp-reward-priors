#!/usr/bin/env python
"""Does model CAPACITY predict the centred stationarity failure?

The question: these are small preference sets -- 254 to 514 training pairs --
and the round-3 search reaches 1.3M parameters.  Is the `scale_z` gate
rejecting overparameterised configurations?

Section 4.3.78 found `depth` to be the strongest swept predictor of centred
`scale_z` in three of four variants, and `width` the strongest in a fourth
(large_diverse, rho +0.956).  Neither was analysed as CAPACITY.  Under this
architecture `hidden_dims = [W] * depth`, so

    n_params = W(d_in + 1) + (depth - 1) W(W + 1) + (W + 1)

is dominated by `depth * W^2`.  Depth and width are therefore two views of one
underlying quantity, and correlating them separately understates it.

WHAT THIS CAN AND CANNOT SHOW
-----------------------------
Within a variant the training-set size N is CONSTANT, so `params_per_pair` is
a monotone transform of `n_params` and their rank correlations are IDENTICAL.
The two columns differ only when pooling across variants.  They are reported
side by side anyway, because the pooled row is the one that answers "is it
capacity, or capacity RELATIVE TO DATA", and reporting only one invites the
reader to assume the other was checked.

Nothing here is a controlled comparison: all five dimensions vary at once, and
the Bayes optimiser chose where to sample, so capacity is entangled with
whatever else it was chasing.  Treat a correlation as a flag for a controlled
run, not as an effect size.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/capacity_vs_drift.py
"""

import argparse
import itertools
import sys

import numpy as np

from precond_vs_drift import TARGET, col, fetch, spearman

# Measured from the seed-0 splits, 2026-09-04.  obs_dim is 37 and T is 100 for
# every antmaze variant; the pair count is what differs.
N_TRAIN = {"medium_play": 358, "medium_diverse": 498,
           "large_play": 254, "large_diverse": 514}
OBS_DIM = 37


def n_params(width_log2, depth, d_in=OBS_DIM):
    """Parameter count for MLP(input_dim=d_in, hidden_dims=[2**w]*depth, out=1).

    `width` in the sweep is the log2 exponent; run_bnn_training expands it.
    """
    if not (np.isfinite(width_log2) and np.isfinite(depth)):
        return float("nan")
    W = 2.0 ** int(width_log2)
    d = int(depth)
    return W * (d_in + 1) + (d - 1) * W * (W + 1) + (W + 1)


def exact_p(x, y, max_n=9):
    """Exact two-sided permutation p for a rank correlation.

    n here is a per-variant trial count of 6-8, where a normal approximation
    on Spearman's rho is worthless -- and where a PERFECT ordering is not as
    impressive as it looks: at n = 6 there are only 720 permutations, so the
    smallest attainable two-sided p is 2/720 = 0.0028.  Enumerating says so;
    a t-approximation would report ~1e-6.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 4 or n > max_n:
        return float("nan")
    r0 = abs(spearman(x, y)[0])
    hits = tot = 0
    for perm in itertools.permutations(range(n)):
        tot += 1
        if abs(spearman(x, y[list(perm)])[0]) >= r0 - 1e-12:
            hits += 1
    return hits / tot


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("WHAT THIS")[0].strip())
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--sweep-only", action="store_true", default=True,
                    help="use only round-3 sweep trials (the designed sample); "
                         "the ad-hoc 4.3.x runs are not a sample from the "
                         "search space")
    ap.add_argument("--all-runs", dest="sweep_only", action="store_false")
    args = ap.parse_args()

    rows = fetch(args.limit)
    if args.sweep_only:
        rows = [r for r in rows if r["kind"] == "sweep"]
    for r in rows:
        r["n_params"] = n_params(r.get("width"), r.get("depth"))
        n = N_TRAIN.get(r["variant"])
        r["params_per_pair"] = (r["n_params"] / n if n else float("nan"))
        r["n_train"] = n if n else float("nan")

    print(f"{len(rows)} runs "
          f"({'round-3 sweep trials only' if args.sweep_only else 'all runs'})")
    print(f"obs_dim {OBS_DIM}, T 100; training pairs per variant: "
          + ", ".join(f"{k} {v}" for k, v in N_TRAIN.items()))
    print()

    npm = col(rows, "n_params")
    fin = npm[np.isfinite(npm)]
    if fin.size:
        print(f"capacity spanned: {fin.min():,.0f} to {fin.max():,.0f} "
              f"parameters ({fin.max() / fin.min():.0f}x)")
        ppp = col(rows, "params_per_pair")
        pf = ppp[np.isfinite(ppp)]
        print(f"  = {pf.min():,.0f} to {pf.max():,.0f} parameters per "
              f"training pair")
        print("  Every configuration is heavily overparameterised, which for a "
              "BNN is\n  expected and not by itself a defect -- the prior, not "
              "the parameter\n  count, is what regularises.  The question is "
              "whether the RANGE matters.")
    print()

    # ---- Per variant.  N is constant within a variant, so n_params and
    # params_per_pair have identical rank correlations here by construction.
    print("PER VARIANT (N constant within each, so the two capacity columns "
          "have\nIDENTICAL rank correlations -- shown once)")
    print()
    print(f"  {'variant':>15} {'N':>4} {'n':>3} {'rho vs scale_z':>15} "
          f"{'exact p':>9} {'rho vs cvar_ce':>15}  {'%fail':>6}")
    print("  " + "-" * 74)
    for v in sorted(N_TRAIN, key=lambda k: N_TRAIN[k]):
        sub = [r for r in rows if r["variant"] == v]
        if len(sub) < 5:
            print(f"  {v:>15} {N_TRAIN[v]:>4} {len(sub):>3}   (n<5, skipped)")
            continue
        y, ce, p = col(sub, TARGET), col(sub, "val_cvar_ce"), col(sub, "n_params")
        r1, _ = spearman(p, y)
        r2, _ = spearman(p, ce)
        pv = exact_p(p, y)
        fail = np.mean(y[np.isfinite(y)] > 2.0)
        print(f"  {v:>15} {N_TRAIN[v]:>4} {len(sub):>3} {r1:>+15.3f} "
              f"{pv:>9.4f} {r2:>+15.3f}  {fail:>5.0%}")
    print("  Four variants are tested, so a Bonferroni-corrected threshold is "
          "0.0125.")

    # ---- The counterexample check.  If capacity drove the failure, the
    # SMALLEST model in each variant should be among the cleanest.
    print()
    print("SMALLEST vs LARGEST model in each variant (the direct check):")
    print(f"  {'variant':>15} {'smallest':>28} {'largest':>28}")
    print("  " + "-" * 74)
    for v in sorted(N_TRAIN, key=lambda k: N_TRAIN[k]):
        sub = [r for r in rows if r["variant"] == v
               and np.isfinite(r["n_params"]) and np.isfinite(r[TARGET])]
        if len(sub) < 5:
            continue
        sub.sort(key=lambda r: r["n_params"])
        lo, hi = sub[0], sub[-1]
        f = (lambda r: f"{r['n_params']:,.0f}p -> scale_z {r[TARGET]:.2f}")
        print(f"  {v:>15} {f(lo):>28} {f(hi):>28}")

    # ---- Pooled, where params_per_pair carries information n_params does not.
    y = col(rows, TARGET)
    print()
    print("POOLED across variants (here the two columns DIFFER):")
    for k in ("n_params", "params_per_pair", "width", "depth", "n_train"):
        rho, n = spearman(col(rows, k), y)
        if np.isfinite(rho):
            print(f"  rho({k:>16}, centred scale_z) = {rho:+.3f} (n={n})")

    # ---- The direct form of the user's question: does the variant with the
    # least data fail most?  Four points, so this is a look, not a test.
    print()
    print("DATA SIZE vs FAILURE RATE (4 variants -- a look, not a test):")
    print(f"  {'variant':>15} {'train pairs':>12} {'median scale_z':>15} "
          f"{'% over 2.0':>11}")
    print("  " + "-" * 57)
    pts = []
    for v in sorted(N_TRAIN, key=lambda k: N_TRAIN[k]):
        sub = [r for r in rows if r["variant"] == v]
        yy = col(sub, TARGET)
        yy = yy[np.isfinite(yy)]
        if not yy.size:
            continue
        pts.append((N_TRAIN[v], float(np.median(yy)), float(np.mean(yy > 2.0))))
        print(f"  {v:>15} {N_TRAIN[v]:>12} {np.median(yy):>15.2f} "
              f"{np.mean(yy > 2.0):>10.0%}")
    if len(pts) >= 3:
        a = np.array(pts, float)
        rho, _ = spearman(a[:, 0], a[:, 1])
        print(f"\n  rho(train pairs, median scale_z) = {rho:+.3f} over "
              f"{len(pts)} variants")
        print("  If small data drove the failure this would be strongly "
              "NEGATIVE.\n  With four points, only a near-perfect ordering "
              "would mean anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
