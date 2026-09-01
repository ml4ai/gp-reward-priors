#!/usr/bin/env python
"""Does selecting at alpha=0.25 pick the same config as alpha=0.05?

Handoff section 3.2.1 fixes the sweep's selection alpha against the compute
budget: at ~240k steps/trial a trial reaches centred ess ~35-40, so the tail
holds `alpha * ess` EFFECTIVE draws and alpha=0.05 would select on ~1.6 of
them.  alpha=0.25 is affordable; alpha=0.05 costs ~6x.

The risk in choosing 0.25 is that it ranks configurations differently from the
0.05 the model actually deploys at -- in which case the sweep would have to be
redone at higher budget.  That risk is measurable for free: the alpha sweep
reuses one sort, so every archived run already carries CE at both alphas.

This reads those files and reports whether the two alphas ORDER the configs the
same way.  High rank agreement => selecting at 0.25 is safe.  Low agreement =>
find out now, before spending 130 trials.

Usage:
    python scripts_bnn/alpha_rank_agreement.py exp/*_asweep2.txt
"""
import re
import sys

import numpy as np

ROW = re.compile(
    r"^\s*(?P<alpha>\d+\.\d+)\s+(?P<k>\d+)\s+(?P<ce>-?\d+\.\d+)\s+"
    r"(?P<acc>-?\d+\.\d+)\s+(?P<med>-?\d+\.\d+)\s+(?P<flip>-?\d+\.\d+)\s+"
    r"(?P<wrong>-?\d+\.\d+)\s*$"
)
SE = re.compile(r"jackknife-over-chains SE on the CVaR CE:\s*(-?\d+\.\d+)")
ESS = re.compile(r"ess_bulk\s+\(centred\)\s+min\s+\S+\s+median\s+(\d+\.\d+)")


def parse(path):
    """-> (name, {alpha: (ce, acc)}, se, centred_ess) or None if unusable."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    in_sweep = False
    rows = {}
    for line in txt.splitlines():
        if "ALPHA SWEEP" in line:
            in_sweep = True
            continue
        if in_sweep:
            m = ROW.match(line)
            if m:
                rows[float(m["alpha"])] = (float(m["ce"]), float(m["acc"]))
            elif rows and line.strip().startswith("reference:"):
                in_sweep = False
    if not rows:
        return None
    se = SE.search(txt)
    ess = ESS.search(txt)
    name = path.split("/")[-1].replace("_asweep2.txt", "").replace("stage3_", "")
    return (name, rows,
            float(se.group(1)) if se else float("nan"),
            float(ess.group(1)) if ess else float("nan"))


def spearman(a, b):
    """Rank correlation without scipy."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main(paths):
    runs = [r for r in (parse(p) for p in paths) if r]
    if len(runs) < 3:
        sys.exit(f"need >=3 parsed runs, got {len(runs)}")

    A_SEL, A_DEP = 0.25, 0.05
    runs = [r for r in runs if A_SEL in r[1] and A_DEP in r[1]]
    runs.sort(key=lambda r: r[1][A_SEL][0])          # best-at-0.25 first

    print(f"{'config':<34}{'CE@.25':>9}{'CE@.05':>9}{'±SE':>8}"
          f"{'ess_cen':>9}{'rank.25':>9}{'rank.05':>9}")
    print("-" * 87)
    ce25 = np.array([r[1][A_SEL][0] for r in runs])
    ce05 = np.array([r[1][A_DEP][0] for r in runs])
    rk25 = np.argsort(np.argsort(ce25)) + 1
    rk05 = np.argsort(np.argsort(ce05)) + 1
    for i, (name, rows, se, ess) in enumerate(runs):
        flag = "  <-- rank moves" if abs(rk25[i] - rk05[i]) >= 3 else ""
        print(f"{name[:34]:<34}{ce25[i]:>9.4f}{ce05[i]:>9.4f}{se:>8.4f}"
              f"{ess:>9.1f}{rk25[i]:>9d}{rk05[i]:>9d}{flag}")

    rho = spearman(ce25, ce05)
    print(f"\n  Spearman rank correlation, alpha 0.25 vs 0.05: {rho:.4f}"
          f"  (n={len(runs)})")

    # The decision does not hinge on the whole ordering -- it hinges on whether
    # the WINNER is the same, and on whether the config 0.25 picks is close to
    # optimal at 0.05.  Report that directly.
    w25, w05 = int(np.argmin(ce25)), int(np.argmin(ce05))
    print(f"  winner at 0.25: {runs[w25][0]}")
    print(f"  winner at 0.05: {runs[w05][0]}")
    if w25 == w05:
        print("  -> SAME WINNER: selecting at 0.25 picks what 0.05 would pick.")
    else:
        pen = ce05[w25] - ce05[w05]
        se_c = 2 * np.sqrt(runs[w25][2] ** 2 + runs[w05][2] ** 2)
        print(f"  -> DIFFERENT winner.  Cost of the 0.25 choice, measured at "
              f"0.05: {pen:+.4f}")
        print(f"     against a combined 2*SE of {se_c:.4f} -- "
              f"{'INSIDE noise, so the choice is harmless' if pen < se_c else 'RESOLVABLE, so 0.25 selection is genuinely worse'}.")
    if rho >= 0.8:
        print("  -> rank agreement is high; the alpha=0.25 budget choice is low risk.")
    elif rho >= 0.5:
        print("  -> partial agreement.  Log BOTH alphas per trial and re-check "
              "at the end before committing to stage 4.")
    else:
        print("  -> LOW agreement: 0.25 and 0.05 order configs differently. "
              "Budget for alpha=0.05 rather than redoing the sweep later.")


if __name__ == "__main__":
    main(sys.argv[1:])
