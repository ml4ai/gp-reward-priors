#!/usr/bin/env python
"""Capacity vs objective on the COMPLETED MR/PT baseline sweeps (handoff §4.3.115).

Supersedes §4.3.108's mid-sweep reading, which was taken at 19–143 trials while
the sweeps were still running and which that section flagged as unstable.

Three questions:

  1. Does capacity help, per family and variant, on the completed sweeps?
  2. How unstable is that estimate as a function of trial count?  §4.3.108 found
     medium_diverse reversing between 19 and 40 trials by accident; this measures
     the instability deliberately, across every sweep.
  3. Does the §3.2.16 width CEILING bind?  A bare min-over-trials favours widths
     the Bayes optimiser sampled more, so the ceiling comparison is controlled by
     a best-of-k resample at matched k.

Note the objective is `eval_loss_at_selected` and is MINIMISED, so a NEGATIVE
rank correlation means "bigger is better".

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/baseline_capacity_readout.py
"""

import sys

import numpy as np

ENTITY = "champlin-university-of-arizona"
MR = {"ymckz130": "medium_play", "oxac6roc": "medium_diverse",
      "p8yawcs2": "large_play", "n1d9qry8": "large_diverse"}
PT = {"xg6zk118": "medium_play", "beyi619f": "medium_diverse",
      "jj1or8i4": "large_play", "mtyctxxe": "large_diverse"}
D_IN = 37
KEY = "eval_loss_at_selected"
RESAMPLES = 20000


def mr_params(w, d):
    W = 2 ** w
    return W * (D_IN + 1) + (d - 1) * W * (W + 1) + (W + 1)


def pt_params(e, layers):
    E = 2 ** e
    return layers * 12 * E * E


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def fetch():
    import wandb
    api = wandb.Api(timeout=60)
    out = {}
    for fam, sweeps in (("MR", MR), ("PT", PT)):
        for sid, variant in sweeps.items():
            rows = []
            for r in api.sweep(f"{ENTITY}/{fam}-training/{sid}").runs:
                if r.state != "finished":
                    continue
                s, c = dict(r.summary), dict(r.config)
                loss = s.get(KEY)
                if loss is None:
                    continue
                if fam == "MR":
                    w, d = c.get("width"), c.get("depth")
                    n = mr_params(w, d) if w and d else None
                else:
                    w, d = c.get("embd_dim"), c.get("num_layers")
                    n = pt_params(w, d) if w and d else None
                if n is None:
                    continue
                rows.append((str(r.created_at), w, d, n, loss))
            rows.sort()                      # chronological — the sweep's own order
            out[(fam, variant)] = rows
    return out


def main():
    data = fetch()

    print("1. CAPACITY vs the objective, completed sweeps")
    print("   eval_loss_at_selected is MINIMISED -> negative rho = bigger is better\n")
    print(f"   {'fam':3s} {'variant':15s} {'n':>3s} {'rho(w)':>7s} {'rho(d)':>7s} "
          f"{'rho(np)':>8s} {'winner':>8s}")
    for (fam, v), rows in sorted(data.items()):
        w = [r[1] for r in rows]; d = [r[2] for r in rows]
        n = [r[3] for r in rows]; L = [r[4] for r in rows]
        best = min(rows, key=lambda r: r[4])
        print(f"   {fam:3s} {v:15s} {len(rows):3d} {spearman(w, L):+7.3f} "
              f"{spearman(d, L):+7.3f} {spearman(n, L):+8.3f} "
              f"{str(best[1]) + '/' + str(best[2]):>8s}")

    print("\n2. STABILITY of rho(n_params, loss) vs trial count")
    print("   (this is what 4.3.108's mid-sweep read got wrong)\n")
    print(f"   {'fam':3s} {'variant':15s} " +
          "".join(f"{f'k={k}':>8s}" for k in (10, 20, 30, 40)) + f"{'all':>8s}")
    moves, flips = [], 0
    for (fam, v), rows in sorted(data.items()):
        n = [r[3] for r in rows]; L = [r[4] for r in rows]
        cells = [f"{spearman(n[:k], L[:k]):+.3f}" if len(n) >= k else "   -"
                 for k in (10, 20, 30, 40)]
        full = spearman(n, L)
        print(f"   {fam:3s} {v:15s} " + "".join(f"{c:>8s}" for c in cells) +
              f"{full:>+8.3f}")
        if len(n) >= 20:
            early = spearman(n[:20], L[:20])
            moves.append(abs(early - full))
            flips += early * full < 0
    print(f"\n   k=20 -> all: median |move| {np.median(moves):.3f}, "
          f"max {max(moves):.3f}, SIGN FLIPS {flips} of {len(moves)}")

    print("\n3. Does the 3.2.16 width CEILING bind?  best-of-k at matched k")
    print("   (a bare min favours widths the optimiser sampled more)\n")
    rng = np.random.default_rng(0)
    print(f"   {'fam':3s} {'variant':15s} {'cmp':>4s} {'ceil':>5s} {'k':>3s} "
          f"{'cmp':>8s} {'ceil':>8s} {'P(ceil better)':>15s}")
    binds = 0
    for (fam, v), rows in sorted(data.items()):
        byw = {}
        for _, w, _, _, L in rows:
            byw.setdefault(w, []).append(L)
        ws = sorted(byw)
        if len(ws) < 2:
            continue
        top = ws[-1]
        cmp_ = max((w for w in ws if w != top), key=lambda w: len(byw[w]))
        k = min(len(byw[top]), len(byw[cmp_]))
        a, b = np.array(byw[cmp_]), np.array(byw[top])
        ba = np.array([rng.choice(a, k, replace=False).min() for _ in range(RESAMPLES)])
        bb = np.array([rng.choice(b, k, replace=False).min() for _ in range(RESAMPLES)])
        p = float((bb < ba).mean())
        binds += p >= 0.9
        print(f"   {fam:3s} {v:15s} {cmp_:>4d} {top:>5d} {k:>3d} "
              f"{ba.mean():>8.4f} {bb.mean():>8.4f} {100 * p:>14.0f}%")
    print(f"\n   ceiling genuinely better in {binds} of 8 sweeps "
          f"(P >= 90% at matched k)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
