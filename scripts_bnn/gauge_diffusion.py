#!/usr/bin/env python
"""Is the widening free diffusion along ReLU's exact rescaling symmetry?

HYPOTHESIS 8, and why it is different from its seven predecessors: it is a
PROVABLE STRUCTURAL PROPERTY OF THE TARGET, not a guess about the sampler.

THE SYMMETRY
------------
`transfer_fn` is hardcoded to "relu" (run_bnn_training_antmaze_eval.py:454) --
never a config field, never swept, used by every run in this investigation.
ReLU is positively homogeneous, `ReLU(a z) = a ReLU(z)` for `a > 0`, so
scaling layer i's weights AND bias by `a` while scaling layer i+1's weights by
`1/a` leaves `f` EXACTLY unchanged.  Verified numerically on this MLP: max
relative change in f is 2e-7 for relu and 0.55 for tanh.

That would be harmless if the potential saw `w`.  It does not.  There is no
weight-space prior term anywhere in this codebase: the gradient is the
Bradley-Terry likelihood plus Wu et al.'s functional prior, and BOTH depend on
`w` only through `f` -- which is exactly what section 3.6.2 already states
("U depends on w only through f, so weight space carries no information about
convergence") and what f_pref_net.py's own header documents.

    U(w) = V(f(w)),  and f is EXACTLY invariant along the rescaling orbits.

So U is exactly constant along a non-compact group orbit of dimension
(number of hidden layers) = DEPTH.  The posterior over `w` is IMPROPER along
those directions, and a sampler run on it does FREE DIFFUSION there forever.
No burn-in, no draw budget and no chain count can equilibrate a direction with
infinite mass -- which is section 4.3.13's "no budget fixes it", derived
rather than observed.

WHAT IT ALREADY EXPLAINS (all measured before the hypothesis existed)
--------------------------------------------------------------------
  * 4.3.12: the identified spread grows as `t^0.37-0.41`, "against 0.5 for
    FREE DIFFUSION", and that section already concluded "the weight-space
    diffusion is not purely f-preserving; a component of it leaks into the
    spread of the identified shape".  It did not identify the flat direction.
  * 4.3.9: ||w|| grows 1.51x per run in EVERY chain to +-1%.  Dismissed as
    "uninformative" only because a BETWEEN-chain test cannot see a common
    cause -- which that section says explicitly.  Gauge diffusion is common to
    every chain by construction.
  * Location clean in 26 of 26, scale failing in 16 of 26: the orbit
    coordinate diffuses symmetrically in +-log a, which inflates SCALE and
    leaves LOCATION alone.
  * 4.3.78/4.3.80: pooled rho(depth, centred scale_z) = +0.354 while
    rho(width, ...) = +0.038.  The symmetry group's dimension is the number of
    hidden layers -- DEPTH -- and width adds no flat directions of this kind.
  * The 2-5x per-chain `minv` spread (4.3.79): chains sitting at different
    points on the orbit have different gradient magnitudes, so `v_hat` adapts
    to different values and freezes differently.

THE TEST HERE, WHICH NEEDS NO NEW SAMPLING
------------------------------------------
Let `r_i(t) = log ||W_i(t)||_F` for layer i.  The rescaling group acts as
`r_i -> r_i + log a`, `r_{i+1} -> r_{i+1} - log a`, so in r-space:

    GAUGE (flat) subspace   = vectors summing to zero  = r - mean(r)
                              U is EXACTLY constant here
    INVARIANT direction     = mean(r)
                              U constrains this; f changes along it

    PREDICTION.  The gauge coordinate DIFFUSES -- mean squared displacement
    rising ~linearly in lag with no plateau -- while the invariant coordinate
    is CONFINED, its MSD flattening at a plateau.  If BOTH behave alike,
    hypothesis 8 is dead and the widening is not gauge diffusion.

READ AS A CONTRAST, NOT AGAINST A THRESHOLD.  A run here is 75 draws per
chain, so the lag range reaches only ~18 draws, and a confined process whose
relaxation time EXCEEDS that range is indistinguishable from a free one --
the same identification limit msd_probe.py documents.  Calibrated at this
exact geometry (128 chains x 75 draws, 12 replicates):

    truth                    gauge slope      invariant slope    contrast
    gauge walk + OU mean     +0.99            +0.70              +0.29
    everything OU            +0.695           +0.688             +0.007

So the ABSOLUTE slope of the invariant coordinate is uninformative here (0.70
whether or not it is confined), while the CONTRAST separates cleanly.  The
verdict is therefore taken on `slope(gauge) - slope(invariant)` with a
bootstrap over chains, which also cancels the shared lag-range limit.

This reuses the MSD estimator validated in msd_probe.py, so the readout is one
already checked against processes with known answers.  Note the lag unit here
is DRAWS (one cycle apart), not steps.

Usage, on the box where the saved chains live:
    python scripts_bnn/gauge_diffusion.py --run-dir exp/r3_trial1_medium_play_0
    python scripts_bnn/gauge_diffusion.py --self-test
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msd_probe import msd_curve, fit_ou_curve, fit_short_lag_slope  # noqa: E402


def layer_log_norms(weights):
    """log Frobenius norm of each WEIGHT MATRIX in one draw.

    Biases are excluded: the rescaling acts on `(W_i, b_i)` together but the
    gauge coordinate is defined by the multiplicative scale, and mixing a
    bias whose scale is set differently would blur it.  Matrices are picked by
    ndim == 2, so the loader's ordering does not have to be trusted.
    """
    return np.array([np.log(np.linalg.norm(np.asarray(a, float)) + 1e-30)
                     for a in weights if np.asarray(a).ndim == 2], float)


def gauge_split(R):
    """(draws, L) log-norms -> (gauge (draws, L), invariant (draws,)).

    gauge = R - mean over layers (the zero-sum subspace the group moves in);
    invariant = mean over layers (what the group cannot change).
    """
    m = R.mean(axis=1)
    return R - m[:, None], m


def loglog_slope(lags, msd, lo=None, hi=None):
    """d log MSD / d log lag over [lo, hi] -- the diffusion exponent.

    The standard free-vs-confined discriminator, and the one section 4.3.12
    already speaks in: it reports `sd ~ t^0.37-0.41` "against 0.5 for free
    diffusion".  MSD is sd squared, so that section's exponents correspond to
    an MSD slope of 0.74-0.82 against 1.0 for free diffusion.

        slope ~ 1     free diffusion, no restoring force
        slope ~ 0     confined, MSD has saturated at its plateau
    """
    lags = np.asarray(lags, float)
    msd = np.asarray(msd, float)
    ok = np.isfinite(lags) & np.isfinite(msd) & (lags > 0) & (msd > 0)
    if lo is not None:
        ok &= lags >= lo
    if hi is not None:
        ok &= lags <= hi
    if ok.sum() < 3:
        return float("nan")
    x, y = np.log(lags[ok]), np.log(msd[ok])
    return float(np.polyfit(x, y, 1)[0])


def pooled_msd(traces, lags):
    """MSD averaged ACROSS chains, weighted by pair count.

    Each element of `traces` is one chain's (draws, k) trace.  Chains must
    never be concatenated along the draw axis: consecutive chains are
    independent, so a lag spanning the join is not a lag at all, and pooling
    that way manufactures a plateau.  A run here is 75 draws per chain and
    many chains, so pooling across chains is also the only way to get enough
    pairs for a late-lag slope to mean anything.
    """
    num = np.zeros(len(lags))
    den = np.zeros(len(lags))
    for tr in traces:
        m, npair = msd_curve(tr, lags)
        ok = np.isfinite(m) & (npair > 0)
        num[ok] += m[ok] * npair[ok]
        den[ok] += npair[ok]
    out = np.divide(num, den, out=np.full(len(lags), np.nan), where=den > 0)
    return out, den


def msd_report(x, label, n_fit_frac=0.25):
    """MSD of a trace (or list of per-chain traces) against lag.

    `x` may be one (draws, k) array or a list of them, one per chain.
    """
    if isinstance(x, (list, tuple)):
        traces = [np.asarray(t, float) for t in x]
        traces = [t[:, None] if t.ndim == 1 else t for t in traces]
        n = min(t.shape[0] for t in traces)
        lag_max = max(4, int(n_fit_frac * n))
        lags = np.unique(np.round(np.geomspace(1, lag_max,
                                              num=min(30, lag_max))).astype(int))
        msd, npair = pooled_msd(traces, lags)
        var = float(np.mean([np.mean(t.var(axis=0)) for t in traces]))
        return _msd_print(label, lags, msd, var, n_chains=len(traces),
                          npair=npair)
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    lag_max = max(4, int(n_fit_frac * n))
    lags = np.unique(np.round(np.geomspace(1, lag_max,
                                           num=min(30, lag_max))).astype(int))
    msd, npair = msd_curve(x, lags)
    var = float(np.mean(x.var(axis=0)))
    return _msd_print(label, lags, msd, var, n_chains=1, npair=npair)


def _msd_print(label, lags, msd, var, n_chains, npair=None):
    a_all = loglog_slope(lags, msd)
    # The LATE slope is the discriminator: a confined process is linear at
    # short lag too (every process is), and only reveals itself by flattening.
    mid = lags[len(lags) // 2]
    a_late = loglog_slope(lags, msd, lo=mid)
    plateau_frac = float(msd[-1] / (2.0 * var)) if var > 0 else np.nan
    print(f"  {label}")
    print(f"    chains pooled                      {n_chains}")
    print(f"    variance within a chain            {var:.4e}")
    print(f"    MSD at lag {int(lags[-1]):>4} draws            {msd[-1]:.4e}  "
          f"({plateau_frac:.2f} x the plateau 2*var)")
    print(f"    log-log slope, all lags            {a_all:+.3f}")
    print(f"    log-log slope, lags >= {int(mid):>4}       {a_late:+.3f}"
          f"   <- the discriminator")
    print(f"       (1.0 = FREE diffusion; 0.0 = CONFINED.  4.3.12's measured "
          f"sd ~ t^0.37-0.41\n        is an MSD slope of 0.74-0.82.)")
    return {"var": var, "a_all": a_all, "a_late": a_late,
            "plateau_frac": plateau_frac, "lags": lags, "msd": msd}


def contrast_bootstrap(G, M, reps=400, seed=0):
    """slope(gauge) - slope(invariant), with a bootstrap over CHAINS.

    Chains are the independent unit here; draws within a chain are not.  The
    contrast also cancels the shared lag-range limit that makes the absolute
    invariant slope uninformative at 75 draws.
    """
    rng = np.random.default_rng(seed)
    C = len(G)
    n = min(min(g.shape[0] for g in G), min(m.shape[0] for m in M))
    lag_max = max(4, int(0.25 * n))
    lags = np.unique(np.round(np.geomspace(1, lag_max,
                                           num=min(30, lag_max))).astype(int))
    mid = lags[len(lags) // 2]

    def one(idx):
        mg, _ = pooled_msd([G[i] for i in idx], lags)
        mm, _ = pooled_msd([M[i] for i in idx], lags)
        return (loglog_slope(lags, mg, lo=mid)
                - loglog_slope(lags, mm, lo=mid))

    point = one(np.arange(C))
    boot = np.array([one(rng.integers(0, C, C)) for _ in range(reps)])
    boot = boot[np.isfinite(boot)]
    if boot.size < 20:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def analyse(run_dir, chain_ids=None, device="cpu"):
    import torch  # local: the self-test needs no torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from diagnose_sampling_tail import _load_chain_weights, _load_run_config

    cfg = _load_run_config(run_dir)
    nch = cfg["num_chains"]
    ids = list(chain_ids) if chain_ids else list(range(nch))
    print(f"[run] {run_dir}")
    print(f"[run] width={cfg.get('width')} depth={cfg.get('depth')} "
          f"num_chains={nch} num_samples={cfg.get('num_samples')}")
    print()

    G, M = [], []
    for i in ids:
        try:
            ws = _load_chain_weights(run_dir, i, device)
        except Exception as e:                    # noqa: BLE001
            print(f"  chain {i}: {type(e).__name__}: {e}")
            continue
        R = np.stack([layer_log_norms(w) for w in ws])     # (draws, L)
        g, m = gauge_split(R)
        G.append(g)
        M.append(m)
    if not G:
        print("No chains loaded.")
        return 1
    L = G[0].shape[1]
    print(f"{len(G)} chains x {G[0].shape[0]} draws x {L} weight matrices\n")

    print("=== GAUGE coordinate (r - mean r): U is EXACTLY flat here ===")
    gg = msd_report(G, "gauge, pooled across chains")
    print()
    print("=== INVARIANT coordinate (mean r): U constrains this ===")
    mm = msd_report([m[:, None] for m in M],
                    "invariant, pooled across chains")

    print()
    print("=== VERDICT (on the CONTRAST -- see the module docstring) ===")
    d, lo, hi = contrast_bootstrap(G, [m[:, None] for m in M])
    print(f"  slope(gauge) - slope(invariant) = {d:+.3f}  "
          f"[95% CI {lo:+.3f}, {hi:+.3f}] over chains")
    print(f"  calibration at this geometry: +0.29 if the gauge diffuses and "
          f"the\n  invariant is confined; +0.01 if both are confined.")
    print()
    if lo > 0.10:
        print("  GAUGE DIFFUSES FASTER THAN THE INVARIANT -- hypothesis 8")
        print("  SUPPORTED.  The chain is moving freely along a direction the")
        print("  potential cannot see, exactly as ReLU's exact rescaling")
        print("  symmetry with no weight-space prior predicts.  No budget can")
        print("  fix a direction with infinite mass (4.3.13, derived rather")
        print("  than observed).  The decisive follow-up is ONE run at")
        print("  transfer_fn='tanh', which has no such symmetry.")
    elif hi < 0.10:
        print("  NO CONTRAST -- hypothesis 8 REFUTED.  The gauge directions")
        print("  move no more freely than the direction the potential")
        print("  constrains, so something bounds the rescaling coordinate")
        print("  despite U being exactly flat along it -- the momentum clamp")
        print("  (max_param_step) and the frozen preconditioner are the")
        print("  candidates.  The widening is not gauge diffusion.")
    else:
        print("  INCONCLUSIVE.  The CI spans the decision point; more draws")
        print("  per chain (not more chains) would tighten it, since the lag")
        print("  range is set by draws.")
    print()
    print("  NOTE: lags are DRAWS (one cycle apart), not steps, so a 'free'")
    print("  reading here means free on the scale of the harvest interval.")
    return 0


def self_test(seed=0, C=128, n=75, L=4, sd=0.05, phi=0.95):
    """Validate the contrast on traces with known answers, at REAL geometry.

    C x n defaults to r3_trial1_medium_play's 128 chains x 75 draws, because
    the discriminator's power depends on the lag range and the lag range is
    set by the draw count.  Validating at 300 draws would certify the tool in
    a regime the data never reaches.
    """
    import contextlib
    import io as _io
    rng = np.random.default_rng(seed)
    print("=== gauge_diffusion SELF-TEST ===")
    print(f"{C} chains x {n} draws x {L} weight matrices "
          f"(r3_trial1_medium_play geometry)\n")

    def build(kind):
        G, M = [], []
        for _ in range(C):
            if kind == "walk":
                st = rng.standard_normal((n, L)) * sd
                st -= st.mean(axis=1, keepdims=True)   # motion in the gauge only
                R = np.cumsum(st, axis=0)
                m = np.zeros(n)
                for t in range(1, n):
                    m[t] = phi * m[t - 1] + rng.normal(0, sd)
                R += m[:, None]
            else:
                R = np.zeros((n, L))
                for t in range(1, n):
                    R[t] = phi * R[t - 1] + rng.normal(0, sd, L)
            g, mm = gauge_split(R)
            G.append(g)
            M.append(mm[:, None])
        return G, M

    ok = True
    for kind, expect_pos, desc in (
            ("walk", True,
             "gauge random-walks, invariant is OU  (hypothesis 8's picture)"),
            ("ou", False,
             "everything OU -- both confined       (hypothesis 8 refuted)")):
        G, M = build(kind)
        with contextlib.redirect_stdout(_io.StringIO()):
            gg = msd_report(G, "")
            mm = msd_report(M, "")
        d, lo, hi = contrast_bootstrap(G, M)
        good = (lo > 0.10) if expect_pos else (hi < 0.10)
        ok = ok and good
        print(f"  [{'ok ' if good else 'FAIL'}] {desc}")
        print(f"         gauge slope {gg['a_late']:+.3f}, invariant "
              f"{mm['a_late']:+.3f}")
        print(f"         contrast {d:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]"
              f"  -- expected {'> +0.10' if expect_pos else '< +0.10'}")
    print(f"\nSELF-TEST: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("Do NOT read --run-dir output until this passes.")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("THE SYMMETRY")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None,
                    help="A run's OUT_DIR (contains config.yaml and sampling_f/)")
    ap.add_argument("--num-chains", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the free-vs-confined readout and exit")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.run_dir:
        ap.error("--run-dir is required (except with --self-test)")
    ids = list(range(args.num_chains)) if args.num_chains else None
    return analyse(args.run_dir, ids, args.device)


if __name__ == "__main__":
    sys.exit(main())
