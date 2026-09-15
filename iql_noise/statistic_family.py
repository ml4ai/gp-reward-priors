#!/usr/bin/env python
"""Which end-of-training window, and do conclusions survive the choice?  (handoff 4.3.107)

Prompted by the user's question: why last-10 rather than last-n, the mean over
all evaluation points, or their median?  Last-10 entered Experiment 1 as a
conventional secondary statistic, not by principle, so choosing n (or mean /
median) by whichever resolves the most comparisons would be a garden of forking
paths.

PURPOSE AND READING RULES -- written before this script was first run
---------------------------------------------------------------------
1. OUTCOME-BLIND CHOICE OF n.  For every run, fit an OLS slope to the last n
   evaluation points and express the trend as the change across the window
   (slope x (n-1)), against the run's residual SD (evaluation noise).  Uses
   each run's own curve only -- never a between-method comparison.  Reported
   per n and variant: median signed change, median |change| / residual SD, and
   the fraction of runs whose slope is significant (|t| > 2).  The proposed
   rule: the LARGEST n at which the end-of-training window is flat, read as
   median |change| below one evaluation-noise SD AND at most ~10% of runs with
   a significant slope (about what chance gives at |t| > 2, plus a little) in
   every variant.  The rule is only a proposal for the user; this script
   does not adopt anything.

2. ROBUSTNESS, NOT SELECTION.  For the statistic family {max, final, last-5,
   last-10, last-20, last-50, last-100, mean of all, median of all}: the
   Experiment 1 separation (delta / pooled sigma, pure IQL noise), and for the
   24 seeds 1-10 method pairs the |t| > 2.1 count and which pairs change sign
   relative to last-10.  These are NOT to be used to pick a statistic; they show
   whether the headline comparisons depend on the choice within the family.

Reads the cache written by stage4_statistic_rescore.py (exp/iql_histories.pkl).

Usage:
    /opt/anaconda3/envs/irl/bin/python iql_noise/statistic_family.py
"""

import collections
import itertools
import math
import pickle
import sys

import numpy as np

CACHE = "exp/iql_histories.pkl"
N_EVALS = 200
NOISE_GROUP = "iql-noise-large-play-mr-best"
VARIANTS = ("medium-play", "medium-diverse", "large-play", "large-diverse")
METHODS = ("gt", "mr_best", "mr_ens_mean", "mr_ens_cvar", "pt")
WINDOWS = (5, 10, 20, 50, 100, 200)
FAMILY = ("max", "final", "last5", "last10", "last20", "last50", "last100", "mean_all", "median_all")


def stat(h, name):
    h = np.asarray(h)
    if name == "max":
        return float(h.max())
    if name == "final":
        return float(h[-1])
    if name.startswith("last"):
        return float(h[-int(name[4:]):].mean())
    if name == "mean_all":
        return float(h.mean())
    if name == "median_all":
        return float(np.median(h))
    raise ValueError(name)


def window_trend(h, n):
    y = np.asarray(h[-n:], float)
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    b = (xc * (y - y.mean())).sum() / (xc ** 2).sum()
    resid = y - y.mean() - b * xc
    s = math.sqrt((resid ** 2).sum() / (n - 2))
    se = s / math.sqrt((xc ** 2).sum()) if s > 0 else float("inf")
    return b * (n - 1), s, (b / se if se > 0 else 0.0)


def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else float("nan")


def main():
    rows = [r for r in pickle.load(open(CACHE, "rb"))
            if r["state"] == "finished" and r["h"] and len(r["h"]) == N_EVALS]
    print(f"{len(rows)} runs with {N_EVALS} evaluation points (cache {CACHE})")

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 100)
    print("1. IS THE END-OF-TRAINING WINDOW FLAT?  (outcome-blind: each run's own curve)")
    print("=" * 100)
    print("   change = OLS slope x (n-1) across the window; noise = residual SD; sig = |t_slope| > 2")
    print(f"   {'n':>4} {'steps':>8}   " + "   ".join(f"{v:^27}" for v in VARIANTS))
    print(f"   {'':>4} {'':>8}   " + "   ".join(f"{'chg':>7} {'|chg|/sd':>9} {'sig':>6}" for _ in VARIANTS))
    flat = {}
    for n in WINDOWS[:-1] + (199,):
        cells = []
        ok = True
        for v in VARIANTS:
            tr = [window_trend(r["h"], n) for r in rows if r["env"] == v]
            chg = np.array([t[0] for t in tr])
            rel = np.array([abs(t[0]) / t[1] if t[1] > 0 else 0.0 for t in tr])
            sig = np.mean([abs(t[2]) > 2 for t in tr])
            cells.append(f"{np.median(chg):>+7.3f} {np.median(rel):>9.2f} {sig:>6.0%}")
            ok &= np.median(rel) < 1.0 and sig <= 0.10
        flat[n] = ok
        print(f"   {n:>4} {n * 5000:>8,}   " + "   ".join(cells) + ("   <- flat in every variant" if ok else ""))
    flat_ns = [n for n, ok in flat.items() if ok]
    print(f"\n   Largest n flat in every variant under the proposed rule: {max(flat_ns) if flat_ns else 'none'}")

    # ---------------------------------------------------------------- 2a
    print("\n" + "=" * 100)
    print("2a. EXPERIMENT 1 SEPARATION PER STATISTIC (pure IQL noise, n = 6 per arm)")
    print("=" * 100)
    arm = collections.defaultdict(list)
    for r in rows:
        if r["env"] == "large-play" and r["method"] == "mr_best" and r["idx"] in (2, 3) and \
                (r["group"] == NOISE_GROUP or r["seed"] == 0):
            arm[r["idx"]].append(r)
    if len(arm[2]) == 6 and len(arm[3]) == 6:
        print(f"   {'statistic':<11}{'idx2':>7}{'idx3':>7}{'sigma':>8}{'delta':>8}{'delta/sigma':>12}")
        for s in FAMILY:
            a2 = [stat(r["h"], s) for r in arm[2]]
            a3 = [stat(r["h"], s) for r in arm[3]]
            sd = math.sqrt((np.var(a2, ddof=1) + np.var(a3, ddof=1)) / 2)
            d = np.mean(a3) - np.mean(a2)
            print(f"   {s:<11}{np.mean(a2):>7.3f}{np.mean(a3):>7.3f}{sd:>8.4f}{d:>+8.3f}{d / sd:>12.2f}")
    else:
        print(f"   arms incomplete: {len(arm[2])}, {len(arm[3])}")

    # ---------------------------------------------------------------- 2b
    print("\n" + "=" * 100)
    print("2b. SEEDS 1-10 METHOD PAIRS: robustness of conclusions across the family")
    print("=" * 100)
    ev = collections.defaultdict(dict)
    for r in sorted(rows, key=lambda r: r["created"]):
        if r["seed"] in range(1, 11) and r["group"] != NOISE_GROUP:
            ev[(r["env"], r["method"])][r["seed"]] = r
    cells = {k: list(v.values()) for k, v in ev.items() if len(v) == 10 and len({r["idx"] for r in v.values()}) == 1}
    pairs = []
    for env in VARIANTS:
        ms = sorted([k[1] for k in cells if k[0] == env], key=METHODS.index)
        pairs += [(env, a, b) for a, b in itertools.combinations(ms, 2)]
    T = {s: [welch_t([stat(r["h"], s) for r in cells[(e, a)]], [stat(r["h"], s) for r in cells[(e, b)]])
             for e, a, b in pairs] for s in FAMILY}
    print(f"   {len(pairs)} pairs")
    print(f"   {'statistic':<11}{'|t|>2.1':>9}{'median|t|':>11}   sign differs from last10 in")
    for s in FAMILY:
        flips = [f"{e} {a}-{b}" for (e, a, b), t, t10 in zip(pairs, T[s], T["last10"]) if np.sign(t) != np.sign(t10)]
        print(f"   {s:<11}{sum(abs(x) > 2.1 for x in T[s]):>9}{np.median(np.abs(T[s])):>11.2f}   {len(flips)}")
    print("\n   Pairs resolved (|t| > 2.1) under ANY statistic -- t across the family:")
    print(f"   {'pair':<38}" + "".join(f"{s:>9}" for s in FAMILY))
    for i, (e, a, b) in enumerate(pairs):
        if any(abs(T[s][i]) > 2.1 for s in FAMILY):
            print(f"   {e + ' ' + a + ' vs ' + b:<38}" + "".join(f"{T[s][i]:>+9.2f}" for s in FAMILY))
    return 0


if __name__ == "__main__":
    sys.exit(main())
