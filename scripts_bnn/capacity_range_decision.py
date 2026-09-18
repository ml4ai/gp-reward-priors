#!/usr/bin/env python
"""Do the 3.2.16 capacity ranges need widening for the restart?  (handoff 4.3.111)

Reads the 25 finished round-4 trials from wandb, applies selection_gates, and
asks the three questions that decide it:

  1. Where does the ELIGIBLE region sit inside the searched range?  If it is
     pressed against a boundary, widening is indicated; if it is interior, it
     is not.
  2. Is gate-2 failure capacity-shaped?  If gate 2 fails mostly at small
     capacity, widening UP would buy passes.
  3. What do the designed capacity ladders (4.3.105) say happens ABOVE the
     ceiling?  Those are the only fixed-depth, single-axis evidence there is.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/capacity_range_decision.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selection_gates as G  # noqa: E402

ENTITY, PROJECT = "champlin-university-of-arizona", "BNN-training"
SWEEPS = {"2falo587": "medium_play", "vzd1zwim": "medium_diverse",
          "23ezwbbo": "large_play", "i6xhta53": "large_diverse"}
D_IN = 37
K_CE, K_MARGIN = "val_cvar_ce", "val_cvar_degeneracy_margin"

# 4.3.105's designed ladders, at FIXED depth -- the only single-axis evidence.
# (width, n_params, |log r|, val_cvar_ce)
LADDER = {
    "large_diverse (d4)": [(4, 1441, 0.3508, 0.4767), (5, 4417, 0.0202, 0.3907),
                           (6, 14977, 0.0500, 0.3911), (7, 54529, 0.1623, 0.3898),
                           (8, 207361, 0.2542, 0.4055), (9, 807937, 0.3620, 0.4740)],
    "large_play (d6)": [(4, 1985, 0.1082, 0.4381), (5, 6529, 0.2294, 0.4076),
                        (6, 23297, 0.2263, 0.4540), (7, 87553, 0.2583, 0.6663),
                        (8, 338945, 0.3224, 0.9427), (9, 1333249, 0.3459, 1.3238)],
}


def nparams(wexp, d):
    W = 2 ** wexp
    return W * (D_IN + 1) + (d - 1) * W * (W + 1) + (W + 1)


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(x, y):
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def fetch():
    import wandb
    api = wandb.Api(timeout=60)
    rows = []
    for sid, variant in SWEEPS.items():
        for r in api.sweep(f"{ENTITY}/{PROJECT}/{sid}").runs:
            if r.state != "finished":
                continue
            s, c = dict(r.summary), dict(r.config)
            w, d = c.get("width"), c.get("depth")
            if w is None or d is None:
                continue
            if w > 10:          # 4.3.104's trap: sweeps log the log2 exponent
                w = int(math.log2(w))
            rows.append(dict(id=r.id, v=variant, w=w, d=d, n=nparams(w, d),
                             lr=abs(math.log(s[G.K_RATIO])) if s.get(G.K_RATIO) else float("nan"),
                             marg=s.get(K_MARGIN, float("nan")),
                             ce=s.get(K_CE, float("nan")),
                             bad=G.gate_failures(s, gated=True)))
    return rows


def main():
    rows = fetch()
    lo, hi = nparams(4, 1), nparams(7, 4)
    print(f"searched range: w4 d1 = {lo} .. w7 d4 = {hi} params ({hi/lo:.0f}x)")
    print(f"{len(rows)} finished trials\n")

    elig = sorted(r["n"] for r in rows if not r["bad"])
    print("Q1  where is the ELIGIBLE region?")
    print(f"    eligible n_params: {elig}")
    if elig:
        print(f"    spans {elig[0]}-{elig[-1]}, i.e. {elig[0]/lo:.1f}x the floor "
              f"and {hi/elig[-1]:.1f}x below the ceiling")
        print("    => INTERIOR: neither boundary is pressed"
              if elig[0] > lo and elig[-1] < hi else
              "    => pressed against a boundary -- widening may be indicated")

    print("\nQ2  is gate-2 failure capacity-shaped?")
    srt = sorted(rows, key=lambda x: x["n"])
    k = len(srt) // 3
    print(f"    {'tercile':7s} {'n':>3s} {'params':>16s} {'elig':>5s} "
          f"{'g1 fail':>8s} {'g2 fail':>8s} {'med cvar_ce':>12s}")
    for name, grp in (("small", srt[:k]), ("mid", srt[k:2 * k]), ("large", srt[2 * k:])):
        g1 = sum(1 for r in grp if any(b.startswith(("scale", "loc")) for b in r["bad"]))
        g2 = sum(1 for r in grp if "degen" in r["bad"])
        print(f"    {name:7s} {len(grp):3d} {grp[0]['n']:7d}-{grp[-1]['n']:<8d} "
              f"{sum(1 for r in grp if not r['bad']):5d} {g1:8d} {g2:8d} "
              f"{np.median([r['ce'] for r in grp]):12.4f}")
    n = [r["n"] for r in rows]
    print(f"\n    rho(n_params, |log r|) = {spearman(n, [r['lr'] for r in rows]):+.3f} "
          f"(gate 1, lower better)")
    print(f"    rho(n_params, margin)  = {spearman(n, [r['marg'] for r in rows]):+.3f} "
          f"(gate 2, higher better)")
    print(f"    rho(n_params, cvar_ce) = {spearman(n, [r['ce'] for r in rows]):+.3f} "
          f"(objective, LOWER better)")

    print("\nQ3  what happens ABOVE the ceiling?  (4.3.105 ladders, fixed depth)")
    for name, rungs in LADDER.items():
        print(f"    {name}")
        for w, np_, lr, ce in rungs:
            mark = "  <- ceiling" if np_ <= hi and (w == 7) else ""
            over = "  ABOVE RANGE" if np_ > hi else ""
            print(f"      w{w} {np_:8d}p   |log r| {lr:.4f}   cvar_ce {ce:.4f}"
                  f"{mark}{over}")


if __name__ == "__main__":
    sys.exit(main())
