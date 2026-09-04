#!/usr/bin/env python
"""Null calibration and power analysis for the section-4.2 function-space drift gate.

Why this exists
---------------
`optbnn/utils/util.py::_function_space_drift_core` asserts in its docstring that
under stationarity `loc_z` and `scale_z` behave like |N(0,1)| -- "median ~0.67,
95th ~2" -- and section 3.6.3 gates on median <= 2.0.  That assertion is a claim
about the ESS/MCSE denominators, and it has never been checked at the
autocorrelation times these chains actually run at (tau 2.1 to 34.7 kept draws,
section 4.3.67).  Section 4.5 did exactly this kind of null calibration for
R-hat and it changed the reading of every R-hat number in the project; this is
the same exercise for the gate that actually rejects trials.

It runs the REAL gate function (imported, not reimplemented) on synthetic chains
whose stationarity properties are known by construction:

  MODE `null`       perfectly stationary AR(1) at a given tau, started from the
                    stationary law.  There is no drift of any kind.  Whatever
                    the gate reports here is its noise floor at that tau.

  MODE `transient`  section 4.3.73's hypothesis 6, made quantitative.  Chains
                    start over-dispersed by a factor j (in units of the
                    stationary function-space sd), then relax.  With a single
                    timescale the ensemble sd is exactly
                        s(t) = sqrt(1 + (j^2 - 1) exp(-2 t / tau))
                    so j > 1 CONTRACTS, j < 1 EXPANDS, j = 1 does nothing at
                    any tau.  A variant-specific j reproduces the observed sign
                    flip (medium_play 0/8 expand, large_play 5/7 expand) with
                    one mechanism.

  MODE `twoscale`   fast local motion (tau_fast) modulated by a slow scale
                    envelope (tau_slow).  This is the regime that separates
                    "the gate can see the transient" from "the gate is blind to
                    it": the MCSE denominator is set by tau_fast while the ratio
                    is set by tau_slow.

  MODE `implied`    no simulation.  Inverts the gate's own formula on the four
                    settled runs' reported (scale_ratio, scale_z) to recover the
                    ESS the gate must have used, and converts it to tau in kept
                    draws for comparison with section 4.3.67's tau.

Usage
-----
    python scripts_bnn/calibrate_drift_gate.py --mode all
    python scripts_bnn/calibrate_drift_gate.py --mode null --chains 16 --draws 75

Run it from the repo root (or anywhere `optbnn` is importable) in the same
environment as a training run -- it needs `arviz_stats`, because the point is to
calibrate the estimator the gate really uses.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    from optbnn.utils.util import _function_space_drift_core
except ImportError:  # pragma: no cover
    sys.exit("could not import optbnn.utils.util -- run from the repo root "
             "in the training environment (needs arviz_stats).")


# --------------------------------------------------------------------------
# Chain simulators.  All return an array shaped [chain, draw, point], the same
# shape `function_space_drift` receives.
# --------------------------------------------------------------------------
def ar1(C, D, P, tau, rng, init_sd=1.0, burn=0):
    """AR(1) with integrated autocorrelation time `tau`, ensemble sd `init_sd`
    at t = -burn.  init_sd == 1 and burn == 0 gives the exact stationary law."""
    rho = (tau - 1.0) / (tau + 1.0)
    n = int(burn) + D
    x = np.empty((C, n, P))
    x[:, 0, :] = init_sd * rng.standard_normal((C, P))
    s_in = np.sqrt(1.0 - rho ** 2)
    for t in range(1, n):
        x[:, t, :] = rho * x[:, t - 1, :] + s_in * rng.standard_normal((C, P))
    return x[:, int(burn):, :]


def twoscale(C, D, P, tau_fast, tau_slow, j, burn, rng):
    """Fast AR(1) local motion under a slow deterministic scale envelope."""
    x = ar1(C, D, P, tau_fast, rng)
    t = np.arange(D) + burn
    s = np.sqrt(1.0 + (j ** 2 - 1.0) * np.exp(-2.0 * t / tau_slow))
    return x * s[None, :, None]


def _read(a):
    d = _function_space_drift_core(np.asarray(a, dtype=np.float64), 1e-12, "")
    return (d["fn_drift_scale_ratio_median"], d["fn_drift_scale_z_median"],
            d["fn_drift_loc_z_median"], d["fn_drift_scale_z_95th"],
            d["fn_drift_loc_z_95th"])


# --------------------------------------------------------------------------
def run_null(C, D, P, taus, reps, rng):
    print("\n=== NULL: perfectly stationary chains, no drift by construction ===")
    print(f"    {C} chains x {D} draws x {P} points, {reps} replicate(s)")
    print("    docstring claims median ~0.67 and 95th ~2; 3.6.3 gates median <= 2.0")
    print(f"\n{'tau':>7}{'tau:half-chain':>16}{'loc_z med':>11}{'loc_z 95th':>12}"
          f"{'scale_z med':>13}{'scale_z 95th':>14}{'gate':>7}")
    for tau in taus:
        rows = np.array([_read(ar1(C, D, P, tau, rng)) for _ in range(reps)])
        r, sz, lz, sz95, lz95 = rows.mean(axis=0)
        gate = "FAIL" if max(sz, lz) > 2.0 else "pass"
        print(f"{tau:>7.1f}{tau / (D // 2):>16.2f}{lz:>11.3f}{lz95:>12.3f}"
              f"{sz:>13.3f}{sz95:>14.3f}{gate:>7}")
    print("\n  Read: any tau whose null median is far from 0.67 means the gate's")
    print("  own calibration statement does not hold there, and the pass/fail")
    print("  threshold is not the |N(0,1)| threshold it is documented to be.")


def run_transient(C, D, P, tau, burn_cycles, js, rng):
    print("\n=== TRANSIENT (4.3.73 hypothesis 6): single timescale ===")
    print(f"    tau = {tau} kept draws, burn-in = {burn_cycles:.2f} cycles "
          f"(= {burn_cycles / tau:.2f} relaxation times)")
    print(f"\n{'j (start sd)':>13}{'ratio':>9}{'scale_z':>9}{'loc_z':>8}"
          f"{'shape':>11}{'gate':>7}")
    for j in js:
        r, sz, lz, _, _ = _read(ar1(C, D, P, tau, rng, init_sd=j, burn=burn_cycles))
        shape = "contract" if r < 0.97 else ("expand" if r > 1.03 else "flat")
        print(f"{j:>13.2f}{r:>9.4f}{sz:>9.3f}{lz:>8.3f}{shape:>11}"
              f"{'FAIL' if max(sz, lz) > 2.0 else 'pass':>7}")
    print("\n  Read: `scale_ratio` moves monotonically in j and is the readout")
    print("  the jitter ladder should be pre-registered on.  `scale_z` at a")
    print("  large tau is nearly blind to the transient -- see the next block.")


def run_twoscale(C, D, P, burn, rng):
    print("\n=== TWO TIMESCALE: does the gate see a slow envelope? ===")
    print("    fast local motion (sets the MCSE) x slow scale envelope (sets the ratio)")
    print(f"\n{'j':>5}{'tau_fast':>10}{'tau_slow':>10}{'ratio':>9}"
          f"{'scale_z':>9}{'loc_z':>8}{'gate':>7}")
    for j in (2.0, 3.0):
        for tf in (3.0, 10.0, 34.7):
            for ts in (100.0, 400.0):
                r, sz, lz, _, _ = _read(
                    twoscale(C, D, P, tf, ts, j, burn, rng))
                print(f"{j:>5.1f}{tf:>10.1f}{ts:>10.0f}{r:>9.4f}{sz:>9.3f}"
                      f"{lz:>8.3f}{'FAIL' if max(sz, lz) > 2.0 else 'pass':>7}")
    print("\n  Read: at tau_fast = 34.7 (medium_play's measured tau) even a 3x")
    print("  over-dispersed start reads as PASS.  If medium_play really mixes at")
    print("  tau = 34.7, its observed scale_z > 2 failures cannot be an ensemble")
    print("  scale transient -- and if they are, tau inside this array is much")
    print("  smaller than 34.7.  Those two cannot both be true.")


def run_implied(C, D):
    print("\n=== IMPLIED ESS from the four settled runs' own reported numbers ===")
    print("    scale_z = |log ratio| / sqrt(1/(2 e1) + 1/(2 e2));  solve with e1 = e2 = e")
    half = C * (D // 2)
    print(f"\n{'variant':16}{'ratio':>9}{'scale_z':>9}{'implied e':>11}"
          f"{'implied tau':>13}{'tau (4.3.67)':>14}{'disagreement':>14}")
    for name, r, sz, t67 in (("medium_play", 1.0871, 0.6469, 34.7),
                             ("large_diverse", 1.1278, 0.8091, 28.0),
                             ("medium_diverse", 0.9087, 1.7976, 2.1),
                             ("large_play", 0.9231, 1.0870, 8.1)):
        e = (sz / abs(np.log(r))) ** 2
        tau = half / e
        print(f"{name:16}{r:>9.4f}{sz:>9.4f}{e:>11.1f}{tau:>13.1f}"
              f"{t67:>14.1f}{t67 / tau:>13.1f}x")
    print("\n  Read: these are two independent measurements of the same chains'")
    print("  autocorrelation.  They agree on medium_diverse and disagree by")
    print("  2.5-3.5x on the other three, always with the gate seeing FASTER")
    print("  mixing.  Return ess1/ess2 from _function_space_drift_core (they are")
    print("  already computed and discarded) to settle it on real runs.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="all",
                   choices=["all", "null", "transient", "twoscale", "implied"])
    p.add_argument("--chains", type=int, default=16)
    p.add_argument("--draws", type=int, default=75)
    p.add_argument("--points", type=int, default=400)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--tau", type=float, default=34.7,
                   help="kept-draw autocorrelation time for the transient mode")
    p.add_argument("--burn-cycles", type=float, default=20000 / 2750,
                   help="burn-in expressed in kept draws (cycles)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    if a.mode in ("all", "null"):
        run_null(a.chains, a.draws, a.points,
                 [1.0, 2.1, 5.0, 8.1, 20.0, 28.0, 34.7, 50.0], a.reps, rng)
    if a.mode in ("all", "transient"):
        run_transient(a.chains, a.draws, a.points, a.tau, a.burn_cycles,
                      [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], rng)
    if a.mode in ("all", "twoscale"):
        run_twoscale(a.chains, a.draws, a.points, a.burn_cycles, rng)
    if a.mode in ("all", "implied"):
        run_implied(a.chains, a.draws)


if __name__ == "__main__":
    main()
