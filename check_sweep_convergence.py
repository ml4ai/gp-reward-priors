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
    never sees it and the trial slot is wasted).

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
    trials, unsynced = [], []
    for r in runs:
        summ = dict(r.summary) if r.summary else {}
        val = summ.get(metric)
        if r.state == "running":
            continue
        # completed-but-unsynced fingerprint: metric absent on a run that was
        # NOT legitimately early-stopped (early_stopped==1 has no eval block)
        if val is None and summ.get("early_stopped") != 1:
            unsynced.append((r.id, r.state))
        trials.append((r.id, val, {k: r.config.get(k) for k in keys}))

    # best-so-far + patience trigger
    best, best_i, since, trigger = None, None, 0, None
    for i, (_, val, _) in enumerate(trials, 1):
        if better(val, best, goal):
            best, best_i, since = val, i, 0
        else:
            since += 1
        if trigger is None and since >= patience:
            trigger = i

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
    for rid, val, rcfg in trials[:stop_at]:
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

    if unsynced:
        print(f"  !! UNSYNCED : {len(unsynced)} trial(s) completed but missing '{metric}':")
        for rid, state in unsynced:
            print(f"       {rid} (state={state})  -> find its dir on the box and `wandb sync` it")
    else:
        print("  unsynced    : none")


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
