#!/usr/bin/env python
"""Re-score every existing IQL run on three statistics (handoff 4.3.107, after Experiment 1).

Experiment 1 found, in ONE cell, that the stage-4 statistic (max over the 200
evaluation points) has the lowest run-to-run SD but the worst signal-to-noise,
while the final point and the mean of the last 10 points separated the two
normalization indices at 95%.  This asks, at zero compute, whether that holds
across all cells.

READING CRITERIA -- written before this script was first run
-------------------------------------------------------------
The stage-4 grids have one run per index, so on their own they cannot separate
signal from noise.  The seeds 1-10 evaluation runs are genuine replicates of each
reported method (10 each), so the primary test uses them:

  PRIMARY.  For every pair of reported methods within a variant (gt, mr_best,
  mr_ens_mean, mr_ens_cvar, pt -- whichever have 10 finished seeds), Welch t of
  the difference under each statistic.  The Experiment 1 finding GENERALISES if
  last-10 gives a larger |t| than max in at least 17 of 24 pairs (one-sided sign
  test p ~ 0.03 at n = 24; the threshold scales with the pair count actually
  available).  Pairs share methods within a variant, so they are not independent
  and the threshold is indicative.  Also reported for the final point.

  DESCRIPTIVE.
   A. Stage-4 selection per cell under max / final / last-10: agreement, top-2
      gaps, and what max-selection costs in late-statistic terms (in-sample,
      single runs -- noisy).
   B. Seeds 1-10 SD per statistic per cell (an upper bound on IQL noise), and
      whether max's lower SD is general.
   C. Optimism of max: max - last10 per run, and how often the max falls in the
      first 10% of training (evaluations 0-19).

CAVEAT.  Seeds 1-10 used the index max selected; the late statistics are scored
as if selection had not changed.

Histories are cached to exp/iql_histories.pkl (gitignored); --refresh refetches.

Usage:
    /opt/anaconda3/envs/irl/bin/python iql_noise/stage4_statistic_rescore.py [--refresh]
"""

import argparse
import collections
import itertools
import math
import os
import pickle
import sys
import time

import numpy as np
from scipy import stats

ENTITY, PROJECT = "champlin-university-of-arizona", "IQL-pref"
CACHE = "exp/iql_histories.pkl"
N_EVALS = 200
NOISE_GROUP = "iql-noise-large-play-mr-best"
VARIANTS = ("medium-play", "medium-diverse", "large-play", "large-diverse")
METHODS = ("gt", "mr_best", "mr_ens_mean", "mr_ens_cvar", "pt")
STATS = ("max", "final", "last10")


def method_of(c):
    rmp = str(c.get("reward_model_path"))
    if c.get("bnn_reward_model"):
        return "bnn_cvar" if c.get("bnn_alpha") else "bnn_mean"
    if rmp in ("None", "", "null"):
        return "gt"
    if "_pt_" in rmp:
        return "pt"
    if "_mr_" in rmp:
        if not c.get("mr_ensemble"):
            return "mr_best"
        return "mr_ens_mean" if c.get("mr_alpha") == 0 else "mr_ens_cvar"
    return "?"


def fetch(refresh):
    if os.path.exists(CACHE) and not refresh:
        return pickle.load(open(CACHE, "rb"))
    import wandb
    api = wandb.Api(timeout=300)
    rows = []
    for r in api.runs(f"{ENTITY}/{PROJECT}", per_page=300):
        c = r.config
        h = None
        for i in range(5):
            try:
                hh = r.history(keys=["mean_score"], samples=2000, pandas=False)
                hh = sorted((x for x in hh if x.get("mean_score") is not None), key=lambda x: x["_step"])
                h = [x["mean_score"] for x in hh]
                break
            except Exception:
                time.sleep(5 * (i + 1))
        rows.append(dict(id=r.id, state=r.state, created=str(r.created_at),
                         env=str(c.get("env", "")).replace("antmaze-", "").replace("-v2", ""),
                         method=method_of(c), seed=c.get("seed"), idx=c.get("normalize_reward"),
                         group=c.get("group"), h=h))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pickle.dump(rows, open(CACHE, "wb"))
    return rows


def score(h):
    return {"max": max(h), "final": h[-1], "last10": float(np.mean(h[-10:]))}


def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    rows = fetch(args.refresh)
    good = [r for r in rows if r["state"] == "finished" and r["h"] and len(r["h"]) == N_EVALS]
    print(f"{len(rows)} runs fetched; {len(good)} finished with {N_EVALS} evaluation points")
    bad = collections.Counter((r["state"], len(r["h"] or [])) for r in rows if r not in good)
    if bad:
        print(f"  excluded (state, n_evals): {dict(bad)}")
    print(f"  methods: {dict(collections.Counter(r['method'] for r in good))}")
    for r in good:
        r.update(score(r["h"]))

    # ------------------------------------------------------------------ A
    print("\n" + "=" * 100)
    print("A. STAGE-4 SELECTION UNDER EACH STATISTIC (seed 0, one run per index)")
    print("=" * 100)
    grid = collections.defaultdict(dict)
    for r in good:
        if r["seed"] == 0 and r["method"] != "gt" and r["group"] != NOISE_GROUP:
            key = (r["env"], r["method"])
            if r["idx"] in grid[key]:
                print(f"  DUPLICATE seed-0 run {key} idx {r['idx']}: keeping newest")
                if r["created"] < grid[key][r["idx"]]["created"]:
                    continue
            grid[key][r["idx"]] = r
    agree = collections.Counter()
    costs = {"final": [], "last10": []}
    print(f"  {'cell':<28}{'pick max':>9}{'pick final':>11}{'pick last10':>12}   top-2 gap max / final / last10"
          f"   cost of max-pick in final / last10")
    for key in sorted(grid, key=lambda k: (VARIANTS.index(k[0]), METHODS.index(k[1]) if k[1] in METHODS else 9)):
        g = grid[key]
        if sorted(g) != list(range(8)):
            print(f"  {key}: indices {sorted(g)} -- incomplete, skipped")
            continue
        pick, gap = {}, {}
        for s in STATS:
            order = sorted(g, key=lambda i: -g[i][s])
            pick[s] = order[0]
            gap[s] = g[order[0]][s] - g[order[1]][s]
        for s in ("final", "last10"):
            agree[s] += pick[s] == pick["max"]
            costs[s].append(g[pick[s]][s] - g[pick["max"]][s])
        print(f"  {key[0] + ' ' + key[1]:<28}{pick['max']:>9}{pick['final']:>11}{pick['last10']:>12}   "
              f"{gap['max']:.3f} / {gap['final']:.3f} / {gap['last10']:.3f}"
              f"          {costs['final'][-1]:.3f} / {costs['last10'][-1]:.3f}")
    n_cells = len(costs["final"])
    print(f"\n  {n_cells} complete cells.  Same pick as max: final {agree['final']}/{n_cells}, "
          f"last10 {agree['last10']}/{n_cells}")
    print(f"  in-sample cost of selecting by max, in late-statistic units: final median "
          f"{np.median(costs['final']):.3f} (max {np.max(costs['final']):.3f}); last10 median "
          f"{np.median(costs['last10']):.3f} (max {np.max(costs['last10']):.3f})")

    # ------------------------------------------------------------------ B + PRIMARY
    ev = collections.defaultdict(list)
    for r in good:
        if r["seed"] in range(1, 11) and r["group"] != NOISE_GROUP:
            ev[(r["env"], r["method"])].append(r)
    cells = {}
    for key, v in ev.items():
        by_seed = {}
        for r in sorted(v, key=lambda r: r["created"]):
            by_seed[r["seed"]] = r                       # newest per seed
        idxs = {r["idx"] for r in by_seed.values()}
        if len(by_seed) == 10 and len(idxs) == 1:
            cells[key] = list(by_seed.values())
        else:
            print(f"  seeds 1-10 {key}: {len(by_seed)} seeds, indices {idxs} -- skipped")

    print("\n" + "=" * 100)
    print("B. SEEDS 1-10: mean and SD of each statistic (SD = upper bound on IQL noise)")
    print("=" * 100)
    print(f"  {'cell':<28}{'idx':>4}   {'max mean (sd)':>15}{'final mean (sd)':>17}{'last10 mean (sd)':>18}   sd ratio final/max, last10/max")
    ratios = {"final": [], "last10": []}
    for key in sorted(cells, key=lambda k: (VARIANTS.index(k[0]), METHODS.index(k[1]) if k[1] in METHODS else 9)):
        v = cells[key]
        m = {s: (np.mean([r[s] for r in v]), np.std([r[s] for r in v], ddof=1)) for s in STATS}
        for s in ("final", "last10"):
            ratios[s].append(m[s][1] / m["max"][1])
        print(f"  {key[0] + ' ' + key[1]:<28}{v[0]['idx']:>4}   {m['max'][0]:.3f} ({m['max'][1]:.3f})"
              f"   {m['final'][0]:.3f} ({m['final'][1]:.3f})    {m['last10'][0]:.3f} ({m['last10'][1]:.3f})"
              f"      {ratios['final'][-1]:.2f}, {ratios['last10'][-1]:.2f}")
    print(f"\n  SD larger than max's in: final {sum(x > 1 for x in ratios['final'])}/{len(ratios['final'])} cells "
          f"(median ratio {np.median(ratios['final']):.2f}); last10 {sum(x > 1 for x in ratios['last10'])}/"
          f"{len(ratios['last10'])} (median {np.median(ratios['last10']):.2f})")

    print("\n" + "=" * 100)
    print("PRIMARY. RESOLVING POWER: Welch |t| for every within-variant pair of reported methods")
    print("=" * 100)
    tt = []
    for env in VARIANTS:
        ms = sorted([k[1] for k in cells if k[0] == env], key=lambda m: METHODS.index(m) if m in METHODS else 9)
        for a, b in itertools.combinations(ms, 2):
            t = {s: welch_t([r[s] for r in cells[(env, a)]], [r[s] for r in cells[(env, b)]]) for s in STATS}
            tt.append((env, a, b, t))
            print(f"  {env:<15}{a:>12} vs {b:<12}  t max {t['max']:+6.2f}   final {t['final']:+6.2f}   "
                  f"last10 {t['last10']:+6.2f}   sign flip vs max: "
                  f"{'final ' if np.sign(t['final']) != np.sign(t['max']) else ''}"
                  f"{'last10' if np.sign(t['last10']) != np.sign(t['max']) else ''}")
    n = len(tt)
    need = int(stats.binom.ppf(0.97, n, 0.5)) + 1 if n else 0
    for s in ("last10", "final"):
        wins = sum(abs(x[3][s]) > abs(x[3]["max"]) for x in tt)
        p = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue if n else float("nan")
        print(f"\n  |t| {s} > |t| max in {wins}/{n} pairs  (one-sided sign test p {p:.3f}; "
              f"criterion {17 if n == 24 else need}/{n})")
        print(f"  pairs with |t| > 2.1: max {sum(abs(x[3]['max']) > 2.1 for x in tt)}, "
              f"{s} {sum(abs(x[3][s]) > 2.1 for x in tt)};  median |t|: max "
              f"{np.median([abs(x[3]['max']) for x in tt]):.2f}, {s} {np.median([abs(x[3][s]) for x in tt]):.2f}")
    wins10 = sum(abs(x[3]["last10"]) > abs(x[3]["max"]) for x in tt)
    crit = 17 if n == 24 else need
    print(f"\n  VERDICT (last10 vs max): {'GENERALISES' if wins10 >= crit else 'does NOT generalise'} "
          f"({wins10}/{n} vs criterion {crit})")

    # ------------------------------------------------------------------ C
    print("\n" + "=" * 100)
    print("C. OPTIMISM OF MAX (all finished runs with 200 evaluations)")
    print("=" * 100)
    for env in VARIANTS:
        v = [r for r in good if r["env"] == env]
        gapv = [r["max"] - r["last10"] for r in v]
        early = [int(np.argmax(r["h"])) < 20 for r in v]
        print(f"  {env:<15} n={len(v):>3}  max - last10: median {np.median(gapv):.3f} "
              f"(IQR {np.percentile(gapv, 25):.3f}-{np.percentile(gapv, 75):.3f})   max in first 10% of training: "
              f"{np.mean(early):.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
