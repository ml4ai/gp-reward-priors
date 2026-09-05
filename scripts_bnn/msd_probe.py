#!/usr/bin/env python
"""MSD probe: estimate steps-per-independent-sample from ONE short chain.

Independent review 2026-09-04 section 8; handoff HANDOFF_HP_SELECTION.md
section 10.2 item 6.

WHY THIS EXISTS
---------------
Every mechanism test in handoff sections 4.3.6-4.3.73 cost a 220,000-step,
16-chain run and was then read on an ESS estimator that needs tau : chain
length >= 3:1 -- which none of those runs achieved.  Section 4.3.67 had to
withdraw a headline for exactly this reason, and five refuted mechanisms rest
on readings from the same regime.

The invariant section 4.3.67 named is `steps per independent sample`, and it
equals `sigma_f^2 / D_f` up to a constant this script MEASURES rather than
assumes.  `D_f` is the short-lag diffusion coefficient of f, and the mean
squared displacement

    MSD(lag) = E || f(t + lag) - f(t) ||^2

estimates it from a single chain over a few thousand steps, with no ESS
estimator anywhere: MSD is a mean over many overlapping lag pairs, so its
reliability needs `lag_max << n`, not `tau << n`.  The curve also separates
the two timescales section 4.3.74 showed are needed -- free diffusion is
linear in the lag, confinement plateaus at `2 sigma_f^2`, and the knee is the
relaxation time.

MODES
-----
    validate  Validate the estimator on an OU process with known relaxation
              time, against an arviz ESS reading taken exactly the way
              section 4.3.67 took its tau.  Run this first, and after any
              change to the estimator.
    analyse   Read chain_*/msd_trace.npz from a run's OUT_DIR and report
              D_f, sigma_f, tau and steps-per-independent-sample per chain.

    /opt/anaconda3/envs/irl/bin/python scripts_bnn/msd_probe.py validate
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/msd_probe.py analyse OUT_DIR

THE CONSTANT, AND WHY IT IS NOT ASSUMED
---------------------------------------
For a stationary process with exponential autocorrelation
`rho(lag) = exp(-lag / tau_exp)` in STEPS,

    MSD(lag) = 2 sigma^2 (1 - exp(-lag / tau_exp))
             -> 2 (sigma^2 / tau_exp) lag                for lag << tau_exp

so the short-lag slope is `2 D` with `D = sigma^2 / tau_exp`, giving
`tau_exp = 2 sigma^2 / slope`.

Section 4.3.67's quantity is different: it is `tau_int(draws) * cycle_length`,
where `tau_int` is the INTEGRATED autocorrelation time arviz reports through
ESS.  For draws spaced `c` steps apart, `tau_int = coth(c / 2 tau_exp)`, which
is `2 tau_exp / c` when `c << tau_exp`, so

    steps per independent sample  ~  2 tau_exp.

The review's section 8 writes the ratio as `sigma^2 / (2 D)`, which is
`tau_exp / 2` -- a factor of 4 from the above.  Rather than pick a convention,
`validate` measures the estimate against a real arviz ESS reading on a process
with a known answer.  If the printed ratio is not ~1.0, the estimator is wrong
and the number must not be used.

WHERE sigma_f MUST COME FROM -- the one thing the probe cannot do alone
----------------------------------------------------------------------
`D_f` is measurable from a window far shorter than tau: that is the whole
point of the probe.  `sigma_f` is NOT.  A window of length W samples a process
that has explored only `1 - exp(-W / tau)` of its stationary variance, so a
window with `W << tau` under-estimates `sigma_f^2` by exactly the factor that
would have told you tau in the first place.  The validation below demonstrates
this: at tau = 10,000 with a 5,000-step window, the window variance gives
tau = 1,463 against a truth of 10,000.

There is no way around it, and it is not a defect -- it is the same
information-theoretic fact that makes a chain shorter than tau uninformative
about tau.  So:

    D_f       <- the probe, from a short window          (cheap)
    sigma_f   <- the harvested draws POOLED ACROSS CHAINS (already computed)
    tau       <- sigma_f^2 / D_f

Pass the pooled sd with `--sigma-f`.  It is the same quantity the section 4.2
drift gate computes at `util.py:_function_space_drift_core` as
`a.reshape(-1, P).std(axis=0)`, on the same diagnostic points -- which is why
`diagnostic_points()` in run_bnn_training_antmaze_eval.py is shared by the
gate and the probe rather than written twice.

Without `--sigma-f`, `analyse` falls back to the window variance and labels
the result a LOWER BOUND, because that is all it is.
"""

import argparse
import glob
import os
import sys

import numpy as np


# ----------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------
def msd_curve(f, lags):
    """Mean squared displacement of f, averaged over points and lag pairs.

    Args:
        f: (n_samples, n_points) trace of f, sampled on a uniform grid.
        lags: 1-D int array of lags, in SAMPLE units (multiply by the
            recording stride to get steps).

    Returns:
        (msd, n_pairs) -- msd[i] is E||f(t+lags[i]) - f(t)||^2 / n_points,
        i.e. the per-point mean, and n_pairs[i] the number of (t, t+lag)
        pairs it averaged over.
    """
    f = np.asarray(f, dtype=np.float64)
    n = f.shape[0]
    msd = np.empty(len(lags), dtype=np.float64)
    npair = np.empty(len(lags), dtype=np.int64)
    for i, L in enumerate(lags):
        L = int(L)
        if L <= 0 or L >= n:
            msd[i] = np.nan
            npair[i] = 0
            continue
        d = f[L:] - f[:-L]
        msd[i] = float(np.mean(d * d))
        npair[i] = d.shape[0]
    return msd, npair


def fit_ou_curve(lags_steps, msd):
    """Least-squares fit of MSD(lag) = 2 A (1 - exp(-lag / tau)).

    Given tau the amplitude A is linear, so this is a 1-D search over log tau
    with A solved in closed form at each candidate -- no scipy, no starting
    point to get wrong, and no local minima to land in.

    Returns (A, tau, rel_residual).  When the observed lags are all << tau the
    curve is indistinguishable from a straight line and the fit is genuinely
    degenerate in (A, tau) at fixed A/tau; `tau` then runs to the top of the
    search range and only the RATIO is identified.  That degeneracy is the
    reason sigma_f has to come from the draws -- see the module docstring.
    """
    x = np.asarray(lags_steps, dtype=np.float64)
    y = np.asarray(msd, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")

    taus = np.geomspace(0.1 * x.min(), 1e4 * x.max(), 400)
    best = (np.inf, np.nan, np.nan)
    for tau in taus:
        b = 1.0 - np.exp(-x / tau)          # shape, up to the amplitude 2A
        d = float(np.dot(b, b))
        if d <= 0:
            continue
        A2 = float(np.dot(b, y)) / d        # = 2A
        r = float(np.sum((A2 * b - y) ** 2))
        if r < best[0]:
            best = (r, 0.5 * A2, tau)
    ss = float(np.sum((y - y.mean()) ** 2))
    return best[1], best[2], (best[0] / ss if ss > 0 else np.nan)


def fit_short_lag_slope(lags_steps, msd, max_lag_steps=None, min_pts=3):
    """Slope of MSD against lag in the linear regime, through the origin.

    Forced through the origin because MSD(0) = 0 exactly; a free intercept
    absorbs the very curvature the short-lag regime is being used to avoid,
    and on a noisy trace it can come back negative.

    `max_lag_steps` bounds the fit to the linear regime.  Including lags near
    the knee biases the slope DOWN and therefore tau UP -- the first
    validation run of this script over-estimated tau = 100 as 128 for exactly
    that reason.  The caller sets the bound from the fitted knee.
    """
    x = np.asarray(lags_steps, dtype=np.float64)
    y = np.asarray(msd, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if max_lag_steps is not None and np.isfinite(max_lag_steps):
        keep = x <= max_lag_steps
        if keep.sum() >= min_pts:
            x, y = x[keep], y[keep]
        else:
            x, y = x[:min_pts], y[:min_pts]
    if x.size < 2:
        return float("nan")
    return float(np.dot(x, y) / np.dot(x, x))


def msd_estimate(f, stride, sigma2=None, lag_max_frac=0.25, linear_frac=0.1):
    """Estimate D_f, and tau / steps-per-independent-sample from it.

    Args:
        f: (n_samples, n_points) trace.
        stride: steps between consecutive samples of f.
        sigma2: the POOLED per-point variance of f from the harvested draws.
            When None the window variance is used instead and the result is a
            LOWER BOUND on tau -- see the module docstring.
        lag_max_frac: longest lag reported, as a fraction of the trace, so
            every reported point averages over at least 1/lag_max_frac pairs.
        linear_frac: the slope is fitted over lags where MSD is below this
            fraction of the plateau 2*sigma^2, i.e. the linear regime.

    Returns a dict.  `steps_per_indep` is the section 4.3.67 quantity.
    """
    f = np.asarray(f, dtype=np.float64)
    n = f.shape[0]
    lag_max = max(2, int(lag_max_frac * n))
    lags = np.unique(np.round(
        np.geomspace(1, lag_max, num=min(40, lag_max))).astype(int))
    msd, npair = msd_curve(f, lags)
    lags_steps = lags * int(stride)

    # Window variance, pooled over points the same way MSD is, so the two are
    # the same quantity.  Only an estimate of sigma_f^2 when the window is
    # long relative to tau.
    sigma2_window = float(np.mean(f.var(axis=0)))
    A_fit, tau_fit, resid = fit_ou_curve(lags_steps, msd)

    # Fit the slope inside the linear regime.  The knee is bounded by whatever
    # is known: an externally supplied sigma^2 is authoritative, else the
    # curve fit's own knee, else (degenerate case) the shortest lags.
    _s2_for_knee = sigma2 if sigma2 is not None else A_fit
    knee = None
    if _s2_for_knee is not None and np.isfinite(_s2_for_knee) and _s2_for_knee > 0:
        _target = linear_frac * 2.0 * _s2_for_knee
        _below = lags_steps[np.asarray(msd) <= _target]
        knee = float(_below.max()) if _below.size else float(lags_steps[0])
    slope = fit_short_lag_slope(lags_steps, msd, max_lag_steps=knee)

    D = 0.5 * slope                       # MSD ~ 2 D lag
    s2 = sigma2_window if sigma2 is None else float(sigma2)
    tau_exp = s2 / D if (D == D and D > 0) else float("nan")

    # ---- Slow-component detector ------------------------------------------
    # Section 4.3.74 established that a SECOND, slower timescale is required:
    # the single-timescale jitter model reproduces the observed scale_ratio but
    # cannot produce scale_z > 2, and 2.67 / 3.71 / 848.90 are observed.  On a
    # two-component f the short-lag slope sees only the FAST mode, so
    # sigma_f^2 / D under-estimates steps-per-independent-sample -- measured
    # here at 8x to 100x on simulated two-scale processes.  That is not a
    # defect to hide: the DISCREPANCY is the measurement.  If the MSD curve
    # has reached its own fitted plateau (the fast mode is fully explored) at a
    # level well below the true 2*sigma_f^2, the shortfall is variance living
    # in modes slower than the window, and its share is reported here.
    # The detector rests on the fitted amplitude A_fit, which is identified
    # ONLY when the observed lags actually bracket the fitted knee.  Below
    # that the MSD curve is a straight line, A and tau trade off exactly along
    # A/tau = D, and A_fit wanders: measured at 0.14x to 125x the true
    # variance on single-timescale processes whose tau exceeded the lag range.
    # Reading slow_var_frac in that regime produces confident nonsense in both
    # directions -- an 86% "slow component" on a process that has none.  So it
    # is reported only when identified, and the detection limit is stated
    # rather than papered over.
    lag_max_steps = float(lags_steps[-1])
    identified = bool(np.isfinite(tau_fit) and tau_fit <= lag_max_steps)
    curve_flattened = bool(
        np.isfinite(A_fit) and A_fit > 0 and msd[-1] >= 0.8 * 2.0 * A_fit)
    slow_var_frac = float("nan")
    if (identified and sigma2 is not None
            and np.isfinite(A_fit) and sigma2 > 0):
        slow_var_frac = float(np.clip(1.0 - A_fit / float(sigma2), 0.0, 1.0))

    return {
        "lags_steps": lags_steps,
        "msd": msd,
        "n_pairs": npair,
        "slope": slope,
        "fit_max_lag_steps": knee,
        "D_f": D,
        "sigma2_f": s2,
        "sigma2_window": sigma2_window,
        "sigma2_external": sigma2 is not None,
        "sigma2_fit": A_fit,
        "tau_fit_steps": tau_fit,
        "fit_rel_resid": resid,
        "tau_exp_steps": tau_exp,
        "steps_per_indep": 2.0 * tau_exp,
        "curve_flattened": curve_flattened,
        "slow_var_frac": slow_var_frac,
        "knee_identified": identified,
        "lag_max_steps": lag_max_steps,
        "n_samples": n,
        "stride": int(stride),
        "window_steps": int(n * stride),
        # MSD at the longest reported lag, against the confinement plateau
        # 2 sigma^2.  Near 1.0 means the window covered the relaxation time,
        # so the WINDOW VARIANCE is a usable sigma_f and tau is interpolated.
        # Well below 1.0 means the window never saw the plateau: sigma_f must
        # come from the draws, and without it tau is a lower bound.
        "plateau_frac": float(msd[-1] / (2.0 * s2)) if s2 > 0 else np.nan,
    }


# ----------------------------------------------------------------------
# Reference reading: exactly how section 4.3.67 got its tau
# ----------------------------------------------------------------------
def ess_steps_per_indep(f_chains, cycle_length):
    """steps/indep sample from arviz ESS, the section 4.3.67 way.

    Args:
        f_chains: (chain, draw, point) -- draws spaced cycle_length steps.

    tau_int(draws) = draws / ess_per_chain, and steps/indep = tau_int * c.
    """
    import arviz_stats as azs
    a = np.asarray(f_chains, dtype=np.float64)
    C, D, _ = a.shape
    ess = np.asarray(azs.ess(a, method="mean"), dtype=np.float64)
    ess = ess[np.isfinite(ess) & (ess > 0)]
    if ess.size == 0:
        return float("nan"), float("nan")
    ess_per_chain = float(np.median(ess)) / C
    tau_draws = D / ess_per_chain
    return tau_draws * cycle_length, tau_draws


# ----------------------------------------------------------------------
# Validation on a process with a known answer
# ----------------------------------------------------------------------
def simulate_ou(n_steps, n_points, tau_exp, sigma, rng, n_chains=1):
    """Exact discrete OU with autocorrelation exp(-1/tau_exp) per step.

    Returns (n_chains, n_steps, n_points).  Started from the stationary
    distribution, so there is no transient to discard.
    """
    phi = float(np.exp(-1.0 / tau_exp))
    s = float(sigma) * np.sqrt(1.0 - phi ** 2)
    x = np.empty((n_chains, n_steps, n_points), dtype=np.float64)
    x[:, 0, :] = rng.normal(0.0, sigma, size=(n_chains, n_points))
    for t in range(1, n_steps):
        x[:, t, :] = phi * x[:, t - 1, :] + rng.normal(
            0.0, s, size=(n_chains, n_points))
    return x


def run_validate(args):
    """Two claims, tested separately, because they fail for different reasons.

    A. `D_f` is recoverable from a window MUCH shorter than tau.  This is the
       claim the whole probe rests on, and it is tested at windows down to
       0.5 tau.
    B. Given sigma_f, `steps_per_indep = sigma_f^2 / D_f * 2` reproduces the
       arviz ESS reading section 4.3.67 used.  Tested against a reference arm
       that reads long, well-conditioned chains the way the sweep does.

    The third row of each block is the control that must FAIL: the same probe
    with sigma_f taken from its own short window, which is how the first
    version of this script silently under-estimated tau by 7x.
    """
    rng = np.random.default_rng(args.seed)
    print("MSD probe validation -- OU process with a known relaxation time.")
    print("Probe arm: ONE chain, a short window, fine stride.")
    print("Reference: arviz ESS on long chains thinned to draws, as the sweep "
          "does.")
    print()

    hdr = (f"{'tau true':>9} {'sigma':>6} {'window/tau':>11} | "
           f"{'D est':>10} {'D true':>10} {'D err':>7} | "
           f"{'s/indep MSD':>12} {'s/indep ESS':>12} {'ratio':>7} | "
           f"{'no sigma':>9}")
    print(hdr)
    print("-" * len(hdr))

    d_err, ratios = [], []
    for tau_exp in args.tau:
        for sigma in args.sigma:
            sigma2_true = sigma ** 2
            D_true = sigma2_true / tau_exp

            # --- probe arm: one chain, a short window, fine stride ---------
            n_probe = args.probe_samples * args.stride
            xp = simulate_ou(n_probe, args.points, tau_exp, sigma, rng,
                             n_chains=1)[0]
            est = msd_estimate(xp[::args.stride], stride=args.stride,
                               sigma2=sigma2_true)
            # The control: same trace, sigma_f from the window itself.
            est_nw = msd_estimate(xp[::args.stride], stride=args.stride,
                                  sigma2=None)

            # --- reference arm: the section 4.3.67 reading ----------------
            # Chains long enough that the ESS estimator is well conditioned,
            # which is the regime the probe exists to make unnecessary.
            c = args.cycle_length
            n_draws = args.ref_draws
            xr = simulate_ou(n_draws * c, args.points, tau_exp, sigma, rng,
                             n_chains=args.ref_chains)
            ref, _ = ess_steps_per_indep(xr[:, c - 1::c, :][:, :n_draws, :], c)

            ratio = est["steps_per_indep"] / ref if ref == ref else np.nan
            de = est["D_f"] / D_true - 1.0
            d_err.append(de)
            ratios.append(ratio)
            print(f"{tau_exp:9.0f} {sigma:6.2f} "
                  f"{n_probe / tau_exp:11.2f} | "
                  f"{est['D_f']:10.4g} {D_true:10.4g} {de:+6.1%} | "
                  f"{est['steps_per_indep']:12.1f} {ref:12.1f} {ratio:7.3f} | "
                  f"{est_nw['steps_per_indep'] / ref:9.3f}")

    r = np.asarray(ratios, dtype=float)
    r = r[np.isfinite(r)]
    de = np.abs(np.asarray(d_err, dtype=float))
    de = de[np.isfinite(de)]
    print()
    print(f"A. D_f recovery:  max |error| {de.max():.1%} "
          f"(median {np.median(de):.1%}) over {de.size} cases")
    print(f"B. MSD / ESS:     median {np.median(r):.3f}, "
          f"range {r.min():.3f}-{r.max():.3f}")
    print("   'no sigma' column is the CONTROL: sigma_f taken from the probe's "
          "own\n   window, which under-estimates tau once the window is "
          "shorter than tau.")

    ok_d = de.size and de.max() <= args.d_tol
    ok_r = r.size and abs(np.median(r) - 1.0) <= args.tol

    # ---- C. two-timescale processes, which is the case 4.3.74 requires ----
    print()
    print("C. TWO-TIMESCALE control.  Section 4.3.74 showed a second, slower")
    print("   timescale is needed to explain scale_z > 2.  On such a process")
    print("   the short-lag slope sees only the FAST mode, so s/indep is an")
    print("   UNDER-estimate -- the detector below is what catches it.")
    print()
    print("   A slow component is only RESOLVABLE when the lag range brackets")
    print("   the fast knee; otherwise the honest answer is 'not identified',")
    print("   and the fix is a longer window, not a cleverer statistic.")
    print()
    hdr2 = (f"{'tau_fast':>9} {'tau_slow':>9} {'slow var':>9} "
            f"{'lag_max':>9} | {'s/indep MSD':>12} {'s/indep ESS':>12} "
            f"{'ratio':>7} | {'ident':>6} {'slow frac':>10} {'verdict':>16}")
    print(hdr2)
    print("-" * len(hdr2))

    ok_c = True
    c = args.cycle_length
    for tf, ts, wf, ws in args.two_scale:
        for w_mult in args.two_scale_windows:
            s2_true = wf ** 2 + ws ** 2
            n_probe = int(args.probe_samples * args.stride * w_mult)
            probe = (simulate_ou(n_probe, args.points, tf, wf, rng, 1)[0]
                     + simulate_ou(n_probe, args.points, ts, ws, rng, 1)[0])
            est = msd_estimate(probe[::args.stride], stride=args.stride,
                               sigma2=s2_true)
            xr = (simulate_ou(args.ref_draws * c, args.points, tf, wf, rng,
                              args.ref_chains)
                  + simulate_ou(args.ref_draws * c, args.points, ts, ws, rng,
                                args.ref_chains))
            ref, _ = ess_steps_per_indep(xr[:, c - 1::c, :], c)
            ratio = est["steps_per_indep"] / ref
            under = ratio < 0.5
            fired = bool(est["knee_identified"]
                         and est["slow_var_frac"] > args.slow_tol)
            if not est["knee_identified"]:
                # Not resolvable at this window.  Acceptable ONLY because the
                # estimator says so out loud instead of reporting a number.
                verdict, good = "not identified", True
            elif under:
                verdict, good = ("DETECTED" if fired else "MISSED"), fired
            else:
                verdict, good = ("false fire" if fired else "clean"), not fired
            ok_c = ok_c and good
            print(f"{tf:9.0f} {ts:9.0f} {ws ** 2 / s2_true:8.0%} "
                  f"{est['lag_max_steps']:9,.0f} | "
                  f"{est['steps_per_indep']:12,.0f} {ref:12,.0f} "
                  f"{ratio:7.2f} | {str(est['knee_identified']):>6} "
                  f"{est['slow_var_frac']:9.0%} {verdict:>16}"
                  f"{'' if good else '  <--'}")

    # A false-fire check on block A: single-timescale processes must never be
    # reported as having a slow component once the knee is identified.
    print()
    n_ff = 0
    for tau_exp in args.tau:
        for sigma in args.sigma:
            x = simulate_ou(args.probe_samples * args.stride, args.points,
                            tau_exp, sigma, rng, 1)[0]
            e = msd_estimate(x[::args.stride], stride=args.stride,
                             sigma2=sigma ** 2)
            if e["knee_identified"] and e["slow_var_frac"] > args.slow_tol:
                n_ff += 1
                print(f"   FALSE FIRE: single tau={tau_exp:g} sigma={sigma:g} "
                      f"reported slow_frac {e['slow_var_frac']:.0%}")
    print(f"D. False fires on single-timescale processes: {n_ff} of "
          f"{len(args.tau) * len(args.sigma)}")
    ok_c = ok_c and n_ff == 0

    ok = ok_d and ok_r and ok_c
    print()
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(D within {args.d_tol:.0%}; MSD/ESS median within {args.tol:.2f} "
          f"of 1.0; slow component detected wherever it bites)")
    if not ok:
        print("The estimator does not reproduce the section 4.3.67 quantity. "
              "Do NOT read any analyse output until this passes.")
    return 0 if ok else 1


# ----------------------------------------------------------------------
# Analysis of a real run
# ----------------------------------------------------------------------
def run_analyse(args):
    paths = sorted(glob.glob(os.path.join(args.out_dir, "chain_*",
                                          "msd_trace.npz")))
    if not paths:
        print(f"No chain_*/msd_trace.npz under {args.out_dir}.\n"
              "Was the run launched with msd_window > 0?")
        return 1

    print(f"{len(paths)} chain trace(s) under {args.out_dir}")
    _s2 = args.sigma_f ** 2 if args.sigma_f is not None else None
    print(f"component: {'RAW f' if args.raw else 'CENTRED f (gated, 3.6.3)'}"
          f"{'' if args.raw else ' -- f minus its mean over points, so the'}")
    if not args.raw:
        print("           likelihood-invariant offset direction (4.3.28) is "
              "excluded.\n           Pass --sigma-f the CENTRED sigma "
              "(val_pred_centred_sd_median).")
    print()
    hdr = (f"{'chain':>6} {'samples':>8} {'stride':>7} {'sigma_f':>10} "
           f"{'D_f':>11} {'tau (steps)':>12} {'s/indep':>10} {'plateau':>8}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        f = z["f"]
        if not args.raw:
            # The BT/CE likelihood is exactly invariant to f -> f + c, so the
            # offset is unidentified and diffuses freely; including it inflates
            # D_f with motion that changes no prediction.  Section 3.6.3 gates
            # on the centred component for the same reason, and sigma_f must
            # be the centred sigma to match.
            f = f - f.mean(axis=1, keepdims=True)
        stride = int(z["msd_every"])
        est = msd_estimate(f, stride=stride, sigma2=_s2)
        name = os.path.basename(os.path.dirname(p))
        rows.append((name, est, z))
        print(f"{name.replace('chain_', ''):>6} {est['n_samples']:8d} "
              f"{stride:7d} {np.sqrt(est['sigma2_f']):10.4g} "
              f"{est['D_f']:11.4g} {est['tau_exp_steps']:12.1f} "
              f"{est['steps_per_indep']:10.1f} {est['plateau_frac']:8.2f}")

    sp = np.array([r[1]["steps_per_indep"] for r in rows], dtype=float)
    sp = sp[np.isfinite(sp)]
    z0 = rows[0][2]
    print()
    if sp.size:
        _bound = "" if _s2 is not None else "  [LOWER BOUND -- no --sigma-f]"
        print(f"steps per independent sample: median {np.median(sp):,.0f}, "
              f"spread {sp.min():,.0f}-{sp.max():,.0f} "
              f"({sp.max() / max(sp.min(), 1e-12):.2f}x across chains)"
              f"{_bound}")
        c = int(z0["cycle_length"])
        _eff = 75 * c / np.median(sp)
        print(f"  = {np.median(sp) / c:.1f} draws at cycle_length {c:,} "
              f"-> {_eff:.1f} effective draws in a 75-draw chain"
              f"{'' if _s2 is not None else ' (an UPPER bound)'}")
    # Per-chain spread of D_f is the review section 9.4 mechanism read
    # directly: each chain freezes its own preconditioner and therefore
    # samples at its own effective step, which is a persistent, scale-only,
    # BETWEEN-chain difference -- the shape needed for scale_z to fail while
    # loc_z passes.  Compare against chainspread_precond_minv_*_ratio.
    _d = np.array([r[1]["D_f"] for r in rows], dtype=float)
    _d = _d[np.isfinite(_d) & (_d > 0)]
    if _d.size > 1:
        print(f"  per-chain D_f spread: {_d.max() / _d.min():.2f}x "
              f"({_d.min():.4g}-{_d.max():.4g}) -- review 9.4 predicts a "
              f"persistent between-chain\n    scale difference here if each "
              f"chain's frozen preconditioner differs")
    print(f"  schedule: lr {float(z0['lr_min']):.3e}-{float(z0['lr_max']):.3e}, "
          f"mdecay {float(z0['mdecay']):.4g}, "
          f"cycle_length {int(z0['cycle_length']):,}")

    _pf = np.array([r[1]["plateau_frac"] for r in rows], dtype=float)
    if np.nanmedian(_pf) < 0.5:
        print()
        print(f"NOTE: MSD reached only {np.nanmedian(_pf):.0%} of the "
              f"confinement plateau 2*sigma_f^2 at the longest lag, so tau is "
              f"an extrapolation from the linear fit rather than a measured "
              f"knee.  D_f is unaffected -- that is what the probe measures "
              f"directly -- but tau inherits all of sigma_f's error.")
    if _s2 is None:
        print()
        print("*** sigma_f NOT SUPPLIED: every tau above uses the probe's own "
              "window\n    variance, which under-estimates sigma_f whenever "
              "the window is shorter\n    than tau.  These are LOWER BOUNDS.  "
              "Re-run with --sigma-f set to the\n    pooled per-point sd from "
              "the run's harvested draws (the same quantity\n    the drift "
              "gate divides by).  D_f above is unaffected.")

    if args.curve:
        print()
        print("MSD curve, chain 0 (lag in steps; ratio is MSD / 2 sigma_f^2):")
        e = rows[0][1]
        for L, m, n in zip(e["lags_steps"], e["msd"], e["n_pairs"]):
            print(f"  {int(L):>8,}  {m:12.5g}  {m / (2 * e['sigma2_f']):6.3f}"
                  f"  ({int(n):,} pairs)")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("MODES")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    v = sub.add_parser("validate", help="OU process with a known answer")
    v.add_argument("--tau", type=float, nargs="+",
                   default=[100.0, 1000.0, 10000.0],
                   help="true relaxation times in steps")
    v.add_argument("--sigma", type=float, nargs="+", default=[1.0, 3.0])
    v.add_argument("--points", type=int, default=256)
    v.add_argument("--stride", type=int, default=10,
                   help="probe recording stride (msd_every)")
    v.add_argument("--probe-samples", type=int, default=500,
                   help="samples the probe records (window = this * stride)")
    v.add_argument("--ref-draws", type=int, default=2000)
    v.add_argument("--ref-chains", type=int, default=4)
    v.add_argument("--cycle-length", type=int, default=100)
    v.add_argument("--tol", type=float, default=0.15,
                   help="tolerance on the MSD/ESS ratio")
    v.add_argument("--d-tol", type=float, default=0.15,
                   help="tolerance on D_f recovery")
    v.add_argument("--two-scale", type=float, nargs=4, action="append",
                   metavar=("TAU_FAST", "TAU_SLOW", "SD_FAST", "SD_SLOW"),
                   default=None,
                   help="two-timescale control case; repeatable")
    v.add_argument("--two-scale-windows", type=float, nargs="+",
                   default=[1.0, 10.0],
                   help="window multipliers for the two-timescale block, so "
                        "the same case is seen both under- and adequately "
                        "resolved")
    v.add_argument("--slow-tol", type=float, default=0.2,
                   help="slow_var_frac above which a slow component is called")
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(func=run_validate)

    a = sub.add_parser("analyse", help="read a run's chain_*/msd_trace.npz")
    a.add_argument("out_dir")
    a.add_argument("--sigma-f", type=float, default=None,
                   help="pooled per-point sd of f from the harvested draws "
                        "(the drift gate's own sigma).  REQUIRED for a "
                        "trustworthy tau whenever the window is shorter than "
                        "tau; without it the result is a lower bound.")
    a.add_argument("--raw", action="store_true",
                   help="analyse RAW f instead of the centred component.  "
                        "Default is centred, matching the 3.6.3 gate: the "
                        "BT likelihood is invariant to f -> f + c, so the "
                        "offset direction is unidentified.")
    a.add_argument("--curve", action="store_true",
                   help="print the full MSD curve for chain 0")
    a.set_defaults(func=run_analyse)

    args = ap.parse_args()
    if getattr(args, "two_scale", None) is None and args.mode == "validate":
        # Defaults chosen around medium_play: tau_slow ~ 4.3.67's ~100,000
        # steps, tau_fast in the range 4.3.74's table needs to make scale_z
        # large, and a case where the slow mode holds most of the variance.
        args.two_scale = [(300.0, 100000.0, 1.0, 1.0),
                          (300.0, 100000.0, 1.0, 3.0),
                          (3000.0, 100000.0, 1.0, 1.0)]
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
