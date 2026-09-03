#!/usr/bin/env python
"""check_sweep_convergence.py — out-of-band convergence + sync check for wandb sweeps.

wandb has no built-in convergence stop for `method: bayes` sweeps, so the
project's uniform stopping rule ("stop when the best-so-far has not improved for
K consecutive trials") has to be evaluated manually.  This script does that from
the wandb API, for any mix of MR / PT / BNN sweeps.

For each sweep it reports:

  * progress (trials done vs run_cap) and the current best;
  * whether the K-patience criterion has FIRED, and at which trial;
  * the STOPPING-RULE winner (best among trials up to the trigger) and the
    BEST-OF-ALL winner — report both in the paper, per the pre-registered rule;
  * unsynced trials: finished/crashed but missing the sweep's metric, which
    means the run completed locally but its result never reached the server
    (recover with `wandb sync <run-dir>` on the box, else the Bayes optimizer
    never sees it and the trial slot is wasted);
  * diverged trials: scored normally, so NOT unsynced, but the chains blew up —
    NaN/Inf convergence diagnostics or a high fraction of sampling steps over
    the gradient-clip threshold.  These still count toward the search, but are
    unusable for the stage-3 draw-budget decision, which needs ESS and R-hat.

Usage (entity defaults to the sweep path or --entity):

    python check_sweep_convergence.py BNN-training/ppsecjoz BNN-training/xw4b5jmj
    python check_sweep_convergence.py --entity champlin-university-of-arizona \\
        MR-training/du38rwbg PT-training/r0gfuisx
    python check_sweep_convergence.py --patience 15 --emit-prior-runs MR-training/du38rwbg

`--emit-prior-runs` prints the `-R <run_id>` flags for `wandb sweep`, which
carry finished trials into a NEW sweep so their results still inform the
Bayesian search (used when raising run_cap on an already-finished sweep).

Environment: needs `wandb` and API auth (~/.netrc or WANDB_API_KEY).
On the analysis Mac use /opt/anaconda3/envs/irl/bin/python; on the GPU box the
`pt` env's `python` is fine.
"""

import argparse
import math
import sys

import wandb

DEFAULT_PATIENCE = 15


def parse_path(spec, default_entity):
    """'project/sweep' or 'entity/project/sweep' -> (entity, project, sweep)."""
    parts = spec.strip("/").split("/")
    if len(parts) == 3:
        return tuple(parts)
    if len(parts) == 2:
        if not default_entity:
            raise SystemExit(
                f"{spec!r} has no entity and --entity was not given."
            )
        return (default_entity, parts[0], parts[1])
    raise SystemExit(f"Cannot parse sweep path {spec!r}.")


def better(a, b, goal):
    """Is a strictly better than b under goal ('minimize'/'maximize')?"""
    if b is None:
        return a is not None
    if a is None:
        return False
    return a < b if goal == "minimize" else a > b


# Diagnostics that go NaN/Inf when the sampler diverges.  Checked only when
# present, so this is harmless for MR/PT sweeps (which log none of them).
_DIVERGENCE_KEYS = (
    "gradnorm_sampling_mean", "gradnorm_sampling_max",
    "val_pred_rhat_max", "val_pred_ess_min",
    "val_pred_cvar_ess_min", "val_pred_cvar_mcse_rel_max",
    "val_pred_within_chain_var", "param_within_chain_var",
)
# Fraction of sampling steps whose pre-clip gradient norm exceeded the clip
# threshold.  Healthy stage-2 trials sit at 0-0.05%; troubled ones run 0.5-60%.
_CLIP_PCT_WARN = 1.0


def diverged_reasons(summ):
    """Non-empty list of reasons if this trial's chains blew up numerically."""
    reasons = []
    bad = [k for k in _DIVERGENCE_KEYS
           if isinstance(summ.get(k), float)
           and (math.isnan(summ[k]) or math.isinf(summ[k]))]
    if bad:
        reasons.append("NaN/Inf in " + ", ".join(bad))
    pct = summ.get("gradnorm_sampling_pct_over_clip")
    if isinstance(pct, (int, float)) and not math.isnan(pct) and pct > _CLIP_PCT_WARN:
        reasons.append(f"gradnorm_sampling_pct_over_clip={pct:.1f}%")
    return reasons


# --- Round-3 eligibility gates (handoff 3.2.1) ---------------------------
# A wandb sweep cannot express a gate, so the optimiser ranks and the Bayes
# search explores INELIGIBLE trials freely.  Eligibility is applied here, when
# the winner is read off.  Each gate is checked only when its key is present,
# so MR/PT sweeps -- which log none of them -- are unaffected and behave exactly
# as before.
#
# The resolution gate is specified on CENTRED ess but the sweep logs only raw
# `val_pred_ess_median`; raw is a conservative proxy because centring removes a
# shared slowly-mixing component and can only raise ESS (centred > raw in all
# five cases measured, floor 1.42x).  See 3.2.1.
_GATES = (
    ("val_fn_drift_centred_loc_z_median",   "loc_z",   lambda v: v <= 2.0),
    ("val_fn_drift_centred_scale_z_median", "scale_z", lambda v: v <= 2.0),
    ("val_cvar_degeneracy_pass",            "degen",   lambda v: bool(v)),
    ("val_pred_ess_median",                 "ess",     lambda v: v >= 40.0),
)


def gate_failures(summ):
    """Names of the 3.2.1 gates this trial fails.  Absent keys are not checked."""
    bad = []
    for key, label, ok in _GATES:
        v = summ.get(key)
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            bad.append(label + "=NaN")
            continue
        if not ok(v):
            bad.append(label)
    return bad


def has_gate_keys(summ):
    """True if this sweep logs any gate key at all (i.e. it is a round-3 BNN sweep)."""
    return any(summ.get(k) is not None for k, _, _ in _GATES)


def frontier(trials, goal, patience, eligible_only):
    """Best-so-far and patience trigger, optionally over ELIGIBLE trials only.

    The patience counter runs over ALL trials in order either way: an
    ineligible trial consumes budget and counts as non-improving, which is the
    honest reading of "K consecutive trials without improvement".  Only the
    best-so-far is restricted.
    """
    best = best_i = trigger = None
    since = 0
    for i, t in enumerate(trials, 1):
        val, ok = t[1], t[3]
        improves = better(val, best, goal) and (ok or not eligible_only)
        if improves:
            best, best_i, since = val, i, 0
        else:
            since += 1
        if trigger is None and since >= patience:
            trigger = i
    return best, best_i, since, trigger


def swept_keys(cfg):
    """Names of the parameters this sweep actually searches (not fixed values)."""
    return sorted(
        k for k, v in (cfg.get("parameters") or {}).items()
        if isinstance(v, dict) and "value" not in v
    )


def summarize(entity, project, sweep_id, patience):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    cfg = sweep.config or {}
    metric = (cfg.get("metric") or {}).get("name")
    goal = (cfg.get("metric") or {}).get("goal", "minimize")
    run_cap = cfg.get("run_cap")
    keys = swept_keys(cfg)

    runs = sorted(sweep.runs, key=lambda r: r.created_at)
    trials, unsynced, diverged = [], [], []
    for r in runs:
        summ = dict(r.summary) if r.summary else {}
        val = summ.get(metric)
        if r.state == "running":
            continue
        # completed-but-unsynced fingerprint: metric absent on a run that was
        # NOT legitimately early-stopped (early_stopped==1 has no eval block)
        if val is None and summ.get("early_stopped") != 1:
            unsynced.append((r.id, r.state))
        fails = gate_failures(summ)
        trials.append((r.id, val, {k: r.config.get(k) for k in keys},
                       not fails, fails, summ.get("val_cvar_degeneracy_margin"),
                       has_gate_keys(summ)))

        # numerical-divergence fingerprint: the trial completed and reported the
        # metric, so it is NOT unsynced, but its chains blew up — the convergence
        # diagnostics are NaN/Inf and the run is useless for the stage-3 draw-budget
        # decision even though the optimiser scored it.
        why = diverged_reasons(summ)
        if why:
            diverged.append((r.id, val, why))

    # best-so-far + patience trigger, UNGATED (the historical behaviour)
    best, best_i, since, trigger = frontier(trials, goal, patience, False)

    n = len(trials)
    print(f"\n=== {project}/{sweep_id} ===")
    print(f"  metric      : {metric} ({goal})   swept: {', '.join(keys) or '-'}")
    print(f"  trials      : {n} done"
          + (f" / run_cap {run_cap}" if run_cap else "")
          + f"   (state: {sweep.state})")

    if best is None:
        print("  no trial has reported the metric yet")
        return

    stop_at = trigger if trigger is not None else n
    stop_best, stop_id, stop_cfg = None, None, None
    for rid, val, rcfg, _ok, _f, _m, _g in trials[:stop_at]:
        if better(val, stop_best, goal):
            stop_best, stop_id, stop_cfg = val, rid, rcfg
    all_id, all_cfg = trials[best_i - 1][0], trials[best_i - 1][2]

    print(f"  best-of-all : {best:.6g}  (trial {best_i}, run {all_id})")
    print(f"                {all_cfg}")
    if trigger is not None:
        print(f"  STOP FIRED  : yes — at trial {trigger} "
              f"({patience} non-improving trials since trial {best_i if best_i<=trigger else '?'})")
        agree = stop_id == all_id
        print(f"  rule winner : {stop_best:.6g}  (run {stop_id})"
              + ("   [same as best-of-all]" if agree else "   [DIFFERS from best-of-all]"))
        if not agree:
            print(f"                {stop_cfg}")
            gap = (best / stop_best - 1) * 100 if goal == "minimize" else (stop_best / best - 1) * 100
            print(f"                regret vs best-of-all: {abs(gap):.1f}%  -> disclose this")
        print(f"  ACTION      : stopping criterion met; trials after {trigger} are discardable")
    else:
        print(f"  STOP FIRED  : no — {since}/{patience} non-improving trials so far")
        print(f"  ACTION      : keep running ({patience - since} more non-improving trials would trigger)")

    # ---- ELIGIBLE frontier (handoff 3.2.1 / 3.2.7) ----------------------
    # Only meaningful for sweeps that log the gate keys; MR/PT skip this block.
    if any(t[6] for t in trials):
        e_best, e_i, e_since, e_trig = frontier(trials, goal, patience, True)
        n_el = sum(1 for t in trials if t[3])
        print(f"\n  --- ELIGIBLE frontier (3.2.1 gates applied) ---")
        print(f"  eligible    : {n_el} of {n} trials ({100.0*n_el/max(n,1):.0f}%)")
        from collections import Counter
        fc = Counter(f for t in trials for f in t[4])
        if fc:
            print(f"  rejected on : "
                  + ", ".join(f"{k} x{v}" for k, v in fc.most_common()))
        if e_best is None:
            print("  !! NO ELIGIBLE TRIAL HAS REPORTED THE METRIC.")
            print("     Per 3.2.9 that is itself a result -- the method produced no")
            print("     non-degenerate configuration at this budget -- and it is")
            print("     DISCLOSED rather than escalated around.")
        else:
            e_id = trials[e_i - 1][0]
            print(f"  best        : {e_best:.6g}  (trial {e_i}, run {e_id})")
            print(f"                {trials[e_i - 1][2]}")
            if e_trig is not None:
                print(f"  STOP FIRED  : yes -- at trial {e_trig}")
            else:
                print(f"  STOP FIRED  : no -- {e_since}/{patience} non-improving")
            # The disagreement 7.2 recorded for round 2: the UNGATED frontier can
            # keep improving while the eligible one stalls, so the rule can fire
            # on progress that is not selectable.
            if e_id != all_id:
                gap = ((e_best / best - 1) if goal == "minimize"
                       else (best / e_best - 1)) * 100
                print(f"  !! the ungated best ({best:.6g}, run {all_id}) is NOT eligible;")
                print(f"     selecting the eligible best costs {abs(gap):.1f}% "
                      f"on {metric} -> DISCLOSE (7.2)")
            if trigger is not None and e_trig is None:
                print(f"  !! the UNGATED rule fired at trial {trigger} but the eligible")
                print(f"     frontier is still improving -- stopping now would discard")
                print(f"     selectable progress.  This is 7.2's round-2 failure.")
            elif trigger is None and e_trig is not None:
                print(f"  !! the ELIGIBLE frontier stalled at trial {e_trig} while the")
                print(f"     ungated one keeps improving -- the search is chasing")
                print(f"     configurations the gates reject (3.2.7).")
            # 3.2.7: winners hugging the gate boundary mean the objective is
            # drifting toward CVaR ~ mean and the gate is doing all the work.
            mg = [t[5] for t in trials if t[3] and isinstance(t[5], (int, float))]
            if mg:
                mg_s = sorted(mg)
                print(f"  degeneracy margin among eligible: min {mg_s[0]:+.4f}  "
                      f"median {mg_s[len(mg_s)//2]:+.4f}  max {mg_s[-1]:+.4f}")
                wm = trials[e_i - 1][5]
                if isinstance(wm, (int, float)) and wm < 0.02:
                    print(f"  !! the eligible best sits at margin {wm:+.4f} -- hugging the")
                    print(f"     gate.  3.2.7: the objective is drifting toward CVaR ~ mean")
                    print(f"     and the gate is carrying the selection.  Disclose.")

    if unsynced:
        print(f"  !! UNSYNCED : {len(unsynced)} trial(s) completed but missing '{metric}':")
        for rid, state in unsynced:
            print(f"       {rid} (state={state})  -> find its dir on the box and `wandb sync` it")
    else:
        print("  unsynced    : none")

    if diverged:
        print(f"  !! DIVERGED : {len(diverged)} trial(s) scored but numerically blown up:")
        for rid, val, why in diverged:
            print(f"       {rid}  {metric}={val if val is None else f'{val:.6g}'}  ({'; '.join(why)})")
        print("       These count toward the search (the optimiser scored them) but are")
        print("       unusable for the stage-3 draw-budget decision, which needs ESS/R-hat.")
    else:
        print("  diverged    : none")


def emit_prior_runs(entity, project, sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    ids = [r.id for r in sweep.runs if r.state == "finished"]
    print(" ".join(f"-R {i}" for i in ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweeps", nargs="+", metavar="[entity/]project/sweep_id")
    ap.add_argument("--entity", default=None, help="default entity for 2-part paths")
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                    help=f"K: consecutive non-improving trials to trigger the stop (default {DEFAULT_PATIENCE})")
    ap.add_argument("--emit-prior-runs", action="store_true",
                    help="print '-R <run_id>' flags to carry finished trials into a new sweep")
    args = ap.parse_args()

    for spec in args.sweeps:
        entity, project, sweep_id = parse_path(spec, args.entity)
        try:
            if args.emit_prior_runs:
                emit_prior_runs(entity, project, sweep_id)
            else:
                summarize(entity, project, sweep_id, args.patience)
        except Exception as exc:  # keep going across sweeps
            print(f"\n=== {project}/{sweep_id} ===\n  ERROR: {exc}", file=sys.stderr)
    print()


if __name__ == "__main__":
    main()
