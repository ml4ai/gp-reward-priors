#!/usr/bin/env python
# coding: utf-8
"""check_winner_eligibility.py — apply the §3.6.3 winner acceptance criteria.

`check_sweep_convergence.py` answers "has the stopping rule fired, and which
trial has the best metric".  That is not the same as "which trial is the
winner": a configuration that samples something other than the target
function-space posterior is not a valid winner however good its score, so
selection is a CONSTRAINED minimisation (HANDOFF_HP_SELECTION.md §3.6.3):

    winner = lowest val_predictive_cross_entropy among ELIGIBLE trials
             up to the stopping trigger

Eligibility (all must hold; thresholds pre-registered before any sweep fired):

    val_fn_drift_loc_z_median    <= 2.0    stationary null is |N(0,1)|:
    val_fn_drift_scale_z_median  <= 2.0      median ~0.67, 95th ~2
    param_clamp_sampling_pct     <= 0.01%  0 is the exact null
    convergence diagnostics       not NaN/Inf

`gradnorm_sampling_pct_over_clip` is REPORTED but does NOT gate — the
sampling-phase gradient clip is disabled, so it measures a symptom rather than a
distortion of the measure (§3.6.1).

Paired diagnostic re-runs
-------------------------
Trials that predate the drift metric carry no `fn_drift_*`.  §3.6.3 says to
re-run that exact config once to populate diagnostics — but the re-run is a
SEPARATE wandb run, so the sweep trial's own summary still shows nothing and the
evidence is invisible at selection time.  This script closes that gap: for any
trial missing diagnostics it searches the project for a run whose swept
parameters match exactly and whose OUT_DIR marks it as a diagnostic re-run, and
reports the borrowed numbers with their source.

Ranking always uses the ORIGINAL trial's metric, never the re-run's (§3.6.3):
a re-run is a second draw, and taking whichever score came out better would be
selection bias.

Usage
-----
    python check_winner_eligibility.py --entity champlin-university-of-arizona \\
        BNN-training/ojk7k4vb BNN-training/9gifb8sa

    # thresholds are the pre-registered defaults; override only to explore
    python check_winner_eligibility.py --loc-z 2.0 --scale-z 2.0 --clamp-pct 0.01 ...
"""

import argparse
import math

import wandb

from check_sweep_convergence import better, diverged_reasons, parse_path, swept_keys

LOC_Z = "val_fn_drift_loc_z_median"
SCALE_Z = "val_fn_drift_scale_z_median"
CLAMP = "param_clamp_sampling_pct"
CLIP = "gradnorm_sampling_pct_over_clip"


def _num(summ, key):
    v = summ.get(key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")


def find_diag_reruns(api, entity, project, pattern):
    """Runs marked as diagnostic re-runs, keyed by their swept-parameter values."""
    out = []
    try:
        runs = api.runs(f"{entity}/{project}",
                        filters={"config.OUT_DIR": {"$regex": pattern}})
        out = [r for r in runs if r.state == "finished"]
    except Exception as e:  # noqa: BLE001 — a lookup failure must not hide results
        print(f"  [warn] diagnostic re-run lookup failed ({type(e).__name__}: {e})")
    return out


def _match(trial_cfg, run, keys):
    """Exact match on every swept parameter.

    `width` needs normalising.  A sweep trial's wandb config records the value
    the agent assigned — the log2 EXPONENT (8) — while a run launched by hand
    records what `__post_init__` produced after raising 2 to it (256).  The two
    describe the same network, so compare them modulo that transform, or every
    paired re-run silently fails to match (the §3.2 exponent trap again).
    """
    for k in keys:
        a, b = trial_cfg.get(k), run.config.get(k)
        if k == "width" and isinstance(a, int) and isinstance(b, int):
            if not (a == b or a == 2 ** b or b == 2 ** a):
                return False
        elif isinstance(a, float) and isinstance(b, float):
            if not math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0):
                return False
        elif a != b:
            return False
    return True


def eligibility(summ, loc_z, scale_z, clamp_pct):
    """(verdict, reasons) for one summary dict."""
    lz, sz, cl = _num(summ, LOC_Z), _num(summ, SCALE_Z), _num(summ, CLAMP)
    if lz != lz or sz != sz:
        return "NO DIAGS", []
    bad = []
    if lz > loc_z:
        bad.append(f"loc_z {lz:.2f} > {loc_z}")
    if sz > scale_z:
        bad.append(f"scale_z {sz:.2f} > {scale_z}")
    if cl == cl and cl > clamp_pct:
        bad.append(f"clamp {cl:.4f}% > {clamp_pct}%")
    if diverged_reasons(summ):
        bad.append("diverged: " + "; ".join(diverged_reasons(summ)))
    return ("REJECT" if bad else "ELIGIBLE"), bad


def report(entity, project, sweep_id, patience, loc_z, scale_z, clamp_pct, pattern):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    cfg = sweep.config or {}
    metric = (cfg.get("metric") or {}).get("name")
    goal = (cfg.get("metric") or {}).get("goal", "minimize")
    keys = swept_keys(cfg)

    runs = [r for r in sorted(sweep.runs, key=lambda r: r.created_at)
            if r.state != "running"]
    trials = [(r, _num(dict(r.summary or {}), metric)) for r in runs]
    trials = [(r, v) for r, v in trials if v == v]

    # Resolve eligibility once, in chronological order, borrowing diagnostics
    # from a paired re-run where the trial itself has none.
    diag_pool = find_diag_reruns(api, entity, project, pattern)
    resolved = []
    for r, v in trials:
        summ = dict(r.summary or {})
        verdict, bad = eligibility(summ, loc_z, scale_z, clamp_pct)
        src = ""
        if verdict == "NO DIAGS":
            for d in diag_pool:
                if _match({k: r.config.get(k) for k in keys}, d, keys):
                    summ = dict(d.summary or {})
                    verdict, bad = eligibility(summ, loc_z, scale_z, clamp_pct)
                    src = f"  [diagnostics from re-run {d.id[:8]}]"
                    break
        resolved.append((r, v, summ, verdict, bad, src))

    # Patience trigger on the RAW metric — mirrors
    # check_sweep_convergence.summarize(), which is the source of truth.  The
    # rule is deliberately NOT eligibility-aware: it can only ever extend a
    # sweep, and the §3.1 budget invariant is one-sided (the BNN must receive no
    # MORE tuning than the baselines).  See the frontier note below.
    best, since, trigger = None, 0, None
    for i, (_, v, *_ ) in enumerate(resolved, 1):
        if better(v, best, goal):
            best, since = v, 0
        else:
            since += 1
            if since >= patience and trigger is None:
                trigger = i

    # Eligible frontier: when did the best ELIGIBLE trial last improve?  Because
    # the stopping rule tracks the raw metric, a sweep can fire while the
    # eligible frontier is still moving — the search stops on progress that
    # cannot become a winner.  That is a known limitation of applying acceptance
    # as a filter over an unconstrained search, and it is disclosed rather than
    # fixed (§3.6.3).  This measures whether it actually bit.
    # Scan only the trials the rule KEPT: an eligible improvement after the
    # trigger is discarded like any other post-trigger trial, and counting it
    # would give a negative staleness.
    cut = trigger if trigger else len(resolved)
    ebest, elast = None, None
    for i, (_, v, _, verdict, *_ ) in enumerate(resolved[:cut], 1):
        if verdict == "ELIGIBLE" and better(v, ebest, goal):
            ebest, elast = v, i
    stale = (cut - elast) if elast else None

    print(f"\n=== {project}/{sweep_id} ===")
    print(f"  metric   : {metric} ({goal})")
    print(f"  trials   : {len(resolved)} scored"
          + (f"; STOP FIRED at {trigger} — ranking trials 1-{trigger} only"
             if trigger else "; stop NOT fired — ranking is provisional"))
    if elast is None:
        print("  frontier : no eligible trial yet")
    else:
        print(f"  frontier : best eligible last improved at trial {elast} "
              f"({stale} trial(s) before {'the trigger' if trigger else 'now'})")
        if trigger and stale < patience:
            print(f"             !! the eligible frontier was STILL IMPROVING when "
                  f"the rule fired ({stale} < patience {patience}).")
            print(f"             The stopping rule tracks the raw metric, so the "
                  f"search stopped on progress it could not use. DISCLOSE (§3.6.3).")

    ranked = sorted(resolved[:cut], key=lambda t: t[1], reverse=(goal != "minimize"))
    winner, n_reject, n_nodiag = None, 0, 0
    print(f"  {'rk':>2} {'run':10s} {'metric':>9} {'loc_z':>6} {'scl_z':>6} "
          f"{'clamp%':>7} {'clip%':>6}  verdict")
    for rk, (r, v, summ, verdict, bad, src) in enumerate(ranked, 1):
        if verdict == "NO DIAGS":
            n_nodiag += 1
        elif verdict == "REJECT":
            n_reject += 1
        if winner is None and verdict == "ELIGIBLE":
            winner = (r.id, v, rk)
        if rk <= 12 or verdict == "ELIGIBLE":
            print(f"  {rk:>2} {r.id:10s} {v:>9.4f} {_num(summ, LOC_Z):>6.2f} "
                  f"{_num(summ, SCALE_Z):>6.2f} {_num(summ, CLAMP):>7.4f} "
                  f"{_num(summ, CLIP):>6.2f}  {verdict}{src}"
                  + (f"  ({'; '.join(bad)})" if bad else ""))
        if winner and rk > 12:
            break

    print(f"  rejected : {n_reject} ineligible, {n_nodiag} still missing "
          f"diagnostics (re-run those configs — §3.6.3)")
    if winner:
        rid, v, rk = winner
        gap = v - ranked[0][1]
        print(f"  WINNER   : {rid} @ {v:.6f}"
              + (f"  (rank {rk}; +{gap:.4f} vs the lowest-metric trial "
                 f"{ranked[0][0].id} — DISCLOSE this gap, §3.6.3)"
                 if rk > 1 else "  (also the lowest-metric trial)"))
    else:
        print("  WINNER   : NONE eligible — §3.6.3 step 5: a finding about the "
              "sampler, not a licence to keep drawing")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweeps", nargs="+", metavar="[entity/]project/sweep_id")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--loc-z", type=float, default=2.0)
    ap.add_argument("--scale-z", type=float, default=2.0)
    ap.add_argument("--clamp-pct", type=float, default=0.01)
    ap.add_argument("--diag-pattern", default="diag_rerun",
                    help="OUT_DIR substring marking a diagnostic re-run.")
    a = ap.parse_args()
    for spec in a.sweeps:
        e, p, s = parse_path(spec, a.entity)
        report(e, p, s, a.patience, a.loc_z, a.scale_z, a.clamp_pct, a.diag_pattern)


if __name__ == "__main__":
    main()
