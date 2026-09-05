#!/usr/bin/env python
"""Per-chain frozen-preconditioner spread, from saved stdout logs.

Independent review 2026-09-04 section 9.4; handoff section 10.2 item 5.

THE MECHANISM UNDER TEST
------------------------
Every chain runs its OWN 20k burn-in with `tau/g/v_hat` re-initialised to
ones, so each freezes a DIFFERENT preconditioner and samples the whole run at
its own effective step `eta_i = eps^2 * minv_i`.  Discretisation bias scales
with `eta`, so the chains sit at slightly different effective temperatures.

That is a **persistent, scale-only, BETWEEN-chain** difference.  It is the
shape the observed failure has -- `loc_z` clean in 26 of 26 while `scale_z`
fails in 16 of 26 -- and unlike section 4.3.73's jitter transient it does not
decay, which section 4.3.74 showed is required, since the single-timescale
jitter model cannot produce `scale_z > 2`.

The review's own threshold: "If it is a few percent, forget it; if it is 2x,
it is a direct mechanism for scale_z failing while loc_z passes, and it is not
a transient."

WHERE THE DATA IS
-----------------
Not in wandb: chain workers run with wandb disabled AND are spawned via
mp.spawn, so their `[precond] FROZEN` lines never reach a run's output.log
(section 4.3.78).  They exist only in saved stdout from the ad-hoc diagnostic
runs of sections 4.3.x.

Line order within a log is load-bearing: the FIRST `[precond] FROZEN` is the
parent's warm-up, and every line after it is one chain.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/chain_precond_spread.py
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/chain_precond_spread.py --glob 'exp/stage3_medium_play_*'
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

# "... v_hat median 4.265e-03, 16.4% at floor, minv median 15.31 (max 100.0)"
PRECOND_RE = re.compile(
    r"\[precond\]\s+FROZEN at step ([0-9,]+):.*?v_hat median\s+([0-9.eE+-]+),\s+"
    r"([0-9.]+)%\s+at floor,\s+minv median\s+([0-9.]+)\s+\(max\s+([0-9.]+)\)")
# "[diag]   CENTRED (gated, section 3.6.3): z_loc 0.82, z_scale 1.49; raw
#  shift 0.219 sd, scale 1.326x  |  OFFSET ..."
CENTRED_RE = re.compile(
    r"CENTRED \(gated.*?z_loc\s+([0-9.eE+-]+),\s+z_scale\s+([0-9.eE+-]+);"
    r".*?scale\s+([0-9.eE+-]+)x")


def parse_log(path):
    """Return (warmup, chains, centred) from one saved stdout log."""
    warm, chains, centred = None, [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = PRECOND_RE.search(line)
            if m:
                # The freeze step IS num_burn_in_steps -- adaptive_sghmc stops
                # adapting there -- so the log labels its own burn-in for free.
                rec = {"burn": int(m.group(1).replace(",", "")),
                       "v_hat": float(m.group(2)),
                       "at_floor": float(m.group(3)),
                       "minv": float(m.group(4)),
                       "minv_max": float(m.group(5))}
                # First one is the parent's warm-up; the rest are chains.
                if warm is None:
                    warm = rec
                else:
                    chains.append(rec)
                continue
            m = CENTRED_RE.search(line)
            if m:
                centred.append({"loc_z": float(m.group(1)),
                                "scale_z": float(m.group(2)),
                                "scale_ratio": float(m.group(3))})
    return warm, chains, centred


def simulate(chains=16, draws=75, points=400, seed=0):
    """Can a between-chain scale difference produce `scale_z` at all?

    This is the structural question, and it is answerable without any run.
    `_function_space_drift_core` computes

        sd1 = first.reshape(-1, P).std(axis=0)      # pools CHAIN and draw
        sd2 = second.reshape(-1, P).std(axis=0)

    so each half is pooled over ALL chains.  A time-CONSTANT between-chain
    difference therefore inflates sd1 and sd2 by the same factor and cancels
    in the ratio.  The simulation below confirms it, and contrasts it with the
    same spread made time-VARYING.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from optbnn.utils.util import function_space_drift
    rng = np.random.default_rng(seed)

    def gate(a):
        o = function_space_drift(a, quiet=True)
        return (o["fn_drift_scale_z_median"],
                o["fn_drift_scale_ratio_median"],
                o["fn_drift_loc_z_median"])

    print("Can a between-chain scale difference produce centred `scale_z`?")
    print(f"{chains} chains x {draws} draws x {points} points; every chain is "
          "stationary.\n")
    print("A. PERSISTENT spread -- chain i keeps its own sd for the whole run.")
    print("   This is exactly what a frozen-at-burn-in preconditioner gives "
          "you.\n")
    print(f"   {'spread':>8} {'chain sd':>14} | {'scale_z':>8} "
          f"{'scale_ratio':>12} {'loc_z':>7} {'gate':>6}")
    print("   " + "-" * 62)
    for spread in (1.0, 1.5, 2.29, 3.65, 5.53, 10.0):
        sds = np.geomspace(1.0, spread, chains)
        a = rng.standard_normal((chains, draws, points)) * sds[:, None, None]
        sz, sr, lz = gate(a)
        print(f"   {spread:8.2f} {f'{sds.min():.2f}-{sds.max():.2f}':>14} | "
              f"{sz:8.3f} {sr:12.3f} {lz:7.3f} "
              f"{'PASS' if sz <= 2 else 'FAIL':>6}")

    print()
    print("B. The SAME spread, DECAYING -- chains converge on a common sd over")
    print("   the run.  A transient, not a standing difference.\n")
    print(f"   {'spread':>8} {'behaviour':>14} | {'scale_z':>8} "
          f"{'scale_ratio':>12} {'loc_z':>7} {'gate':>6}")
    print("   " + "-" * 62)
    for spread in (2.29, 3.65, 5.53):
        sds0 = np.geomspace(1.0, spread, chains)
        t = np.linspace(0, 1, draws)[None, :, None]
        a = (rng.standard_normal((chains, draws, points))
             * sds0[:, None, None] ** (1.0 - t))
        sz, sr, lz = gate(a)
        print(f"   {spread:8.2f} {'-> common sd':>14} | "
              f"{sz:8.3f} {sr:12.3f} {lz:7.3f} "
              f"{'PASS' if sz <= 2 else 'FAIL':>6}")

    print()
    print("READ: a persistent between-chain difference of even 10x passes the "
          "gate,\nbecause both halves pool over chains and it cancels.  The "
          "same spread while\nit is still DECAYING fails hard, with "
          "scale_ratio < 1 -- contraction, which\nis the observed signature "
          "(4.3.73: medium_play 0 of 8 expand, median 0.837).")
    print()
    print("So review 9.4's mechanism, AS STATED, cannot be the driver: it is\n"
          "explicitly non-transient, and non-transient is exactly what the "
          "gate cannot\nsee.  What it CAN do is make a transient "
          "HETEROGENEOUS across chains --\nchains at different effective steps "
          "equilibrate at different rates -- and\nheterogeneous decay is what "
          "panel B shows failing.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("THE MECH")[0].strip())
    ap.add_argument("--glob", default="exp/*",
                    help="files to scan (default exp/*)")
    ap.add_argument("--min-chains", type=int, default=4,
                    help="skip logs with fewer chain lines than this")
    ap.add_argument("--simulate", action="store_true",
                    help="run the structural test instead of reading logs: "
                         "can a between-chain scale difference produce "
                         "scale_z at all?")
    args = ap.parse_args()
    if args.simulate:
        return simulate()

    rows = []
    for p in sorted(glob.glob(args.glob)):
        if not os.path.isfile(p):
            continue
        try:
            warm, chains, centred = parse_log(p)
        except Exception:                      # noqa: BLE001
            continue
        if warm is None or len(chains) < args.min_chains:
            continue
        mv = np.array([c["minv"] for c in chains], float)
        fl = np.array([c["at_floor"] for c in chains], float)
        vh = np.array([c["v_hat"] for c in chains], float)
        # The centred block is printed once per eval split (val, then test);
        # take the first, which is val -- the split every gate reads.
        cz = centred[0] if centred else None
        rows.append({
            "name": os.path.basename(p),
            "n": len(chains),
            "burn": warm["burn"],
            "warm_minv": warm["minv"],
            "minv_min": mv.min(), "minv_max": mv.max(),
            "minv_med": float(np.median(mv)),
            "ratio": float(mv.max() / mv.min()) if mv.min() > 0 else np.inf,
            # eta = eps^2 * minv, so the spread in minv IS the spread in the
            # effective step; the sd/mean is the scale-free version of it.
            "cv": float(mv.std() / mv.mean()) if mv.mean() > 0 else np.nan,
            "floor_min": fl.min(), "floor_max": fl.max(),
            "vh_ratio": float(vh.max() / vh.min()) if vh.min() > 0 else np.inf,
            "scale_z": cz["scale_z"] if cz else np.nan,
            "loc_z": cz["loc_z"] if cz else np.nan,
            "scale_ratio": cz["scale_ratio"] if cz else np.nan,
        })

    if not rows:
        print(f"No logs under {args.glob!r} with >= {args.min_chains} chain "
              "[precond] lines.")
        return 1

    print(f"{len(rows)} run logs with per-chain preconditioner lines\n")
    print("Review 9.4 threshold: a few percent -> forget it; 2x -> a direct, "
          "NON-TRANSIENT\nmechanism for scale_z failing while loc_z passes.  "
          "minv sets the effective step\neta_i = eps^2 * minv_i, so this "
          "column IS the between-chain step-size spread.\n")
    hdr = (f"{'run':>42} {'ch':>3} {'burn':>7} {'warm':>7} "
           f"{'chain minv':>15} {'max/min':>8} | {'cen sc_z':>8} {'loc_z':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r["ratio"]):
        rng = f"{r['minv_min']:.2f}-{r['minv_max']:.2f}"
        sz = f"{r['scale_z']:.2f}" if np.isfinite(r["scale_z"]) else "-"
        lz = f"{r['loc_z']:.2f}" if np.isfinite(r["loc_z"]) else "-"
        print(f"{r['name'][:42]:>42} {r['n']:>3} {r['burn']:>7,} "
              f"{r['warm_minv']:>7.2f} {rng:>15} {r['ratio']:>8.2f} | "
              f"{sz:>8} {lz:>6}")

    rat = np.array([r["ratio"] for r in rows], float)
    rat = rat[np.isfinite(rat)]
    print()
    print(f"between-chain minv spread (max/min): median {np.median(rat):.2f}x, "
          f"range {rat.min():.2f}-{rat.max():.2f}x")
    print(f"  >= 2x in {np.mean(rat >= 2.0):.0%} of runs "
          f"({int(np.sum(rat >= 2.0))} of {rat.size})")
    print(f"  Every chain samples at eta_i = eps^2 * minv_i for the ENTIRE "
          f"run, so a {np.median(rat):.1f}x\n  spread is a "
          f"{np.median(rat):.1f}x spread in effective step size -- and it "
          f"never decays.")

    # Does the spread predict the gate that does the rejecting?
    sz = np.array([r["scale_z"] for r in rows], float)
    ok = np.isfinite(rat) & np.isfinite(sz)
    if ok.sum() >= 5:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from precond_vs_drift import spearman
        rho, n = spearman(rat[ok], sz[ok])
        rho_cv, _ = spearman(np.array([r["cv"] for r in rows])[ok], sz[ok])
        print()
        print(f"rho(minv spread, centred scale_z) = {rho:+.3f} (n={n})")
        print(f"rho(minv CV,     centred scale_z) = {rho_cv:+.3f}")
        print("  A spread that is large but does NOT track scale_z is a "
              "standing offset,\n  not the driver -- it would still bias every "
              "run, just not differentially.")
        print("  Run with --simulate for the structural reason this is the "
              "expected result.")

    # Burn-in: the freeze step labels it, so this grouping is free.  The
    # heterogeneous-transient reading predicts that absorbing the transient
    # kills scale_z while leaving the (persistent, invisible) spread alone.
    b = np.array([r["burn"] for r in rows], float)
    if len(set(b)) > 1:
        print()
        print("BY BURN-IN (the freeze step labels each log):")
        for bb in sorted(set(b)):
            m = (b == bb) & np.isfinite(sz)
            if not m.any():
                continue
            print(f"  {int(bb):>7,} steps: n={int(m.sum())}, centred scale_z "
                  f"median {np.median(sz[m]):.2f} "
                  f"(range {sz[m].min():.2f}-{sz[m].max():.2f}), "
                  f"minv spread median {np.median(rat[m]):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
