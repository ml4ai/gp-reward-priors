#!/usr/bin/env python
"""Fix the end-of-training statistic: which n in [10, 20], and mean or median?  (handoff 4.3.107)

The user adopted an end-of-training statistic, last-n with n in 10-20, and asked
what "set by curve shape" means and whether the window should be summarised by
its mean or its median.

RULES -- written before this script was first run.  Both use only the curves'
own properties (noise, drift, outliers), never a between-method comparison.
---------------------------------------------------------------------------
N.  Candidates n in {10, 15, 20}.  A longer window averages more evaluation
    noise but reaches back into training that may still drift, biasing the
    estimate of the end-of-training level.  So:
      choose n = 10 UNLESS a longer window cuts the between-run variance of the
      last-n mean by >= 10%, pooled over the seeds 1-10 cells (median ratio);
      if both 15 and 20 qualify, take the smaller.
    Reported alongside, for context: within-window evaluation-noise SD after
    detrending, lag-1 autocorrelation, the share of last-n variance attributable
    to evaluation noise (sigma_w^2 / n_eff, n_eff = n(1-rho)/(1+rho)), and each
    cell's systematic drift (mean signed change across the window, with SE).

LOCATION.  Mean UNLESS within-window outliers are common: median if >= 10% of
    runs have a point in their last-n window more than 3 robust SDs (1.4826 x
    MAD of the detrended window) from the window median.  Reported alongside,
    for context only: between-run SD of mean vs median (efficiency), and the
    seeds 1-10 comparison robustness (resolved pairs, sign differences) between
    last-n mean and last-n median.

Reads exp/iql_histories.pkl (written by stage4_statistic_rescore.py).

Usage:
    /opt/anaconda3/envs/irl/bin/python iql_noise/window_and_location.py
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
NS = (10, 15, 20)
VAR_CUT = 0.10
OUTLIER_SHARE = 0.10


def detrend(y):
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float) - (len(y) - 1) / 2
    b = (x * (y - y.mean())).sum() / (x ** 2).sum()
    return y - y.mean() - b * x, b * (len(y) - 1)


def lag1(r):
    r = r - r.mean()
    d = (r ** 2).sum()
    return float((r[1:] * r[:-1]).sum() / d) if d > 0 else 0.0


def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else float("nan")


def main():
    rows = [r for r in pickle.load(open(CACHE, "rb"))
            if r["state"] == "finished" and r["h"] and len(r["h"]) == N_EVALS]
    ev = collections.defaultdict(dict)
    for r in sorted(rows, key=lambda r: r["created"]):
        if r["seed"] in range(1, 11) and r["group"] != NOISE_GROUP:
            ev[(r["env"], r["method"])][r["seed"]] = r
    cells = {k: list(v.values()) for k, v in ev.items()
             if len(v) == 10 and len({r["idx"] for r in v.values()}) == 1}
    order = sorted(cells, key=lambda k: (VARIANTS.index(k[0]), METHODS.index(k[1])))
    print(f"{len(rows)} runs; {len(cells)} seeds 1-10 cells")

    # ------------------------------------------------------------ N
    print("\n" + "=" * 104)
    print("N. WINDOW LENGTH: noise structure and drift (seeds 1-10 cells)")
    print("=" * 104)
    print(f"  {'cell':<28}" + "".join(f"{'n=' + str(n) + ' sd':>10}" for n in NS)
          + f"{'var 15/10':>10}{'var 20/10':>10}{'sig_w':>8}{'rho1':>6}"
          + f"{'noise share n=10':>17}{'drift n=10 (se)':>17}{'drift n=20 (se)':>17}")
    r15, r20 = [], []
    for k in order:
        v = cells[k]
        sd = {n: float(np.std([np.mean(r["h"][-n:]) for r in v], ddof=1)) for n in NS}
        res = [detrend(r["h"][-20:]) for r in v]
        sig_w = float(np.sqrt(np.mean([np.var(x[0], ddof=2) for x in res])))
        rho = float(np.mean([lag1(x[0]) for x in res]))
        n_eff = 10 * (1 - rho) / (1 + rho)
        share = (sig_w ** 2 / max(n_eff, 1.0)) / sd[10] ** 2
        d10 = [detrend(r["h"][-10:])[1] for r in v]
        d20 = [x[1] for x in res]
        r15.append(sd[15] ** 2 / sd[10] ** 2)
        r20.append(sd[20] ** 2 / sd[10] ** 2)
        print(f"  {k[0] + ' ' + k[1]:<28}" + "".join(f"{sd[n]:>10.4f}" for n in NS)
              + f"{r15[-1]:>10.2f}{r20[-1]:>10.2f}{sig_w:>8.3f}{rho:>6.2f}{share:>17.0%}"
              + f"{np.mean(d10):>+10.3f} ({np.std(d10, ddof=1) / math.sqrt(10):.3f})"
              + f"{np.mean(d20):>+10.3f} ({np.std(d20, ddof=1) / math.sqrt(10):.3f})")
    m15, m20 = float(np.median(r15)), float(np.median(r20))
    print(f"\n  median variance ratio vs n=10: n=15 {m15:.3f} ({1 - m15:+.1%} reduction), "
          f"n=20 {m20:.3f} ({1 - m20:+.1%} reduction); cut needed >= {VAR_CUT:.0%}")
    choice = 10
    for n, m in ((15, m15), (20, m20)):
        if 1 - m >= VAR_CUT:
            choice = n
            break
    print(f"  RULE -> n = {choice}")

    # ------------------------------------------------------------ LOCATION
    print("\n" + "=" * 104)
    print(f"LOCATION. MEAN OR MEDIAN of the last-{choice} window")
    print("=" * 104)
    flags = []
    for r in rows:
        w = np.asarray(r["h"][-choice:], float)
        resid, _ = detrend(w)
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if mad == 0:
            flags.append(False)
            continue
        flags.append(bool(np.any(np.abs(resid - np.median(resid)) > 3 * mad)))
    share = float(np.mean(flags))
    by_v = {v: np.mean([f for f, r in zip(flags, rows) if r["env"] == v]) for v in VARIANTS}
    print(f"  runs with a >3 robust-SD point in the window: {share:.1%} of {len(rows)}  ("
          + ", ".join(f"{v} {x:.0%}" for v, x in by_v.items()) + f");  threshold {OUTLIER_SHARE:.0%}")
    gaps = [abs(np.mean(r["h"][-choice:]) - np.median(r["h"][-choice:])) for r in rows]
    print(f"  |mean - median| of the window: median {np.median(gaps):.4f}, 90th pct {np.percentile(gaps, 90):.4f}")
    loc = "median" if share >= OUTLIER_SHARE else "mean"
    print(f"  RULE -> {loc}")

    # Added after the first run (handoff 4.3.107): the rule's own null rate.  A
    # MAD from 10 points is noisy, so a 3-SD cutoff fires on outlier-free data.
    rng = np.random.default_rng(0)

    def fires(w):
        resid, _ = detrend(np.asarray(w, float))
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        return bool(mad > 0 and np.any(np.abs(resid - np.median(resid)) > 3 * mad))

    null_norm = np.mean([fires(rng.normal(size=choice)) for _ in range(20000)])
    levels = [float(np.clip(np.mean(r["h"][-choice:]), 0, 1)) for r in rows]
    null_binom = np.mean([fires(rng.binomial(100, p, size=choice) / 100) for _ in range(20) for p in levels])
    print(f"  CALIBRATION (added after first run): rule fires on outlier-free windows "
          f"{null_norm:.1%} (iid normal), {null_binom:.1%} (binomial at each run's level) "
          f"vs observed {share:.1%} -> {'NO excess outliers; verdict is a mis-calibration artefact' if share < null_binom + 0.05 else 'excess outliers'}")

    print("\n  context only -- efficiency and robustness:")
    eff = []
    for k in order:
        v = cells[k]
        a = np.std([np.mean(r["h"][-choice:]) for r in v], ddof=1)
        b = np.std([np.median(r["h"][-choice:]) for r in v], ddof=1)
        eff.append(b / a)
    print(f"  between-run SD, median/mean: median ratio {np.median(eff):.2f} over {len(eff)} cells")
    pairs = []
    for env in VARIANTS:
        ms = sorted([k[1] for k in cells if k[0] == env], key=METHODS.index)
        pairs += [(env, a, b) for a, b in itertools.combinations(ms, 2)]
    tm = [welch_t([np.mean(r["h"][-choice:]) for r in cells[(e, a)]],
                  [np.mean(r["h"][-choice:]) for r in cells[(e, b)]]) for e, a, b in pairs]
    td = [welch_t([np.median(r["h"][-choice:]) for r in cells[(e, a)]],
                  [np.median(r["h"][-choice:]) for r in cells[(e, b)]]) for e, a, b in pairs]
    flips = [f"{e} {a} vs {b} ({x:+.2f} / {y:+.2f})" for (e, a, b), x, y in zip(pairs, tm, td) if np.sign(x) != np.sign(y)]
    print(f"  seeds 1-10 pairs resolved (|t| > 2.1): mean {sum(abs(x) > 2.1 for x in tm)}, "
          f"median {sum(abs(x) > 2.1 for x in td)} of {len(pairs)}")
    print(f"  sign differences mean vs median: {len(flips)}" + ("".join(f"\n    {f}" for f in flips)))
    arm = collections.defaultdict(list)
    for r in rows:
        if r["env"] == "large-play" and r["method"] == "mr_best" and r["idx"] in (2, 3) and \
                (r["group"] == NOISE_GROUP or r["seed"] == 0):
            arm[r["idx"]].append(r)
    for name, f in (("mean", np.mean), ("median", np.median)):
        a2 = [f(r["h"][-choice:]) for r in arm[2]]
        a3 = [f(r["h"][-choice:]) for r in arm[3]]
        s = math.sqrt((np.var(a2, ddof=1) + np.var(a3, ddof=1)) / 2)
        print(f"  Experiment 1 last-{choice} {name}: sigma {s:.4f}, delta {np.mean(a3) - np.mean(a2):+.3f}, "
              f"delta/sigma {(np.mean(a3) - np.mean(a2)) / s:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
