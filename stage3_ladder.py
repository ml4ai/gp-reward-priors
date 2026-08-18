#!/usr/bin/env python
# coding: utf-8
"""stage3_ladder.py — assemble the stage-3 draw-budget ladder from wandb.

Stage 3 raises `num_chains` only, with `num_samples` pinned at the sweep's 75
draws per chain (HANDOFF_HP_SELECTION.md §4.1).  Each rung is a hand-launched
run, not a sweep trial, so there is no sweep to point a checker at; this script
collects the rungs by their `OUT_DIR` marker and prints them as one table.

It answers two questions, in the order §4 requires them to be asked:

  1. §4.2 — is the run still sampling the target measure at this chain count?
     `val_fn_drift_loc_z_median <= 2.0`, `val_fn_drift_scale_z_median <= 2.0`,
     `param_clamp_sampling_pct <= 0.01%`.  A run failing these is not sampling
     `P_{f|D}` and every tail number below it is meaningless however good it
     looks, so the gate is printed FIRST and a failing rung is marked.

  2. §4.3/§4.5 — the distributional tail statistics, and how they moved between
     rungs.  Expect `cvar_ess_median` and `q05_ess_median` to scale roughly
     linearly with total draws and `cvar_mcse_rel_median` to fall as
     1/sqrt(draws).  Expect R-hat to RISE: it measures between-chain
     disagreement, and more chains give more power to detect disagreement that
     was already there (§4.5).  That is correct behaviour, not a regression, so
     the script labels a rising R-hat "expected" rather than flagging it.

The `c4` rung
-------------
Three of the four variants have no dedicated `c4` run: the stage-1 sweep already
ran at 4 chains x 75 draws, so their `c4` row is the winning trial read back
(§4.3).  `--baseline` does that by run id, defaulting to the recorded stage-1
winner for the variant.  Note this covers the logged AGGREGATE statistics only —
each sweep trial wrote its chains to the same deterministic `{OUT_DIR}_{seed}`
path, so the winner's saved chains were overwritten by later trials in the same
sweep and `diagnose_sampling_tail.py --worst-k` cannot be run against them.

What this script does NOT cover
-------------------------------
The unresolved-point count (`points MCSE>sd`) and the per-point `--worst-k`
listing are not logged to wandb; they come from
`scripts_bnn/diagnose_sampling_tail.py`, which needs the run's saved chains and
so runs on the box.  §4.6 asks for both, so record the diagnostic output
alongside this table rather than in place of it.

Usage
-----
    python stage3_ladder.py medium_play
    python stage3_ladder.py --entity champlin-university-of-arizona large_diverse
    python stage3_ladder.py medium_play --no-baseline   # only the stage-3 runs
"""

import argparse
import math
import re

import wandb

# Stage-1 round-2 winners (§6/§10.5).  These ran at 4 chains x 75 draws, which
# is the c4 rung for every variant that has no dedicated c4 re-run.
WINNERS = {
    "medium_play": "0t5bqw02",
    "large_play": "oi7cqb9o",
    "medium_diverse": "n1ztawsx",
    "large_diverse": "yjezedlk",
}

# §4.2 gate: all three must hold before any tail number below is readable.
GATE = (
    ("val_fn_drift_loc_z_median", 2.0, "loc_z"),
    ("val_fn_drift_scale_z_median", 2.0, "scale_z"),
    ("param_clamp_sampling_pct", 0.01, "clamp%"),
)

# The z-scores above are NOT comparable across rungs.  From
# optbnn/utils/util.py:431, z_loc = |m2-m1| / sqrt(mcse1^2 + mcse2^2): the
# numerator is a first-half-vs-second-half shift in function space, the
# denominator an MCSE that falls as ~1/sqrt(C).  So the gate is a significance
# test whose POWER grows with chain count, and a fixed 2.0 threshold tightens
# mechanically as you climb the ladder.  A rung can fail purely by being
# measured better.  These raw effect sizes carry no such dependence and are
# what says whether the drift is actually larger (§4.2.1).
RAW = (
    ("val_fn_drift_loc_sd_median", "loc_sd", "{:8.4f}"),
    ("val_fn_drift_scale_ratio_median", "scaleRatio", "{:10.4f}"),
)

# §4.3 table, plus the extremes §4.6 asks to be recorded.  Steer on the
# median / 95th-pct / pct_over columns; the _max values are sparse rather than
# censored (§4.5) — a repeated value means the tail mass sits in one chain.
TAIL = (
    ("val_pred_cvar_ess_median", "cvarESSmed", "{:9.1f}"),
    ("val_pred_cvar_mcse_rel_median", "relMCSEmed", "{:10.3f}"),
    ("val_pred_cvar_rhat_median", "cvarRhatMed", "{:11.4f}"),
    ("val_pred_cvar_rhat_pct_over_1.01", "cvarRhat%>1.01", "{:14.1f}"),
    ("val_pred_q05_ess_median", "q05ESSmed", "{:9.1f}"),
    ("val_pred_folded_rhat_95th_pct", "fold95", "{:6.3f}"),
    ("val_pred_cvar_rhat_max", "cvarRhatMax", "{:11.4f}"),
)


def cfg(run, key, default=None):
    """Read a config value, unwrapping wandb's {'value': ...} boxing."""
    v = run.config.get(key, default)
    return v["value"] if isinstance(v, dict) and "value" in v else v


def rung_runs(api, entity, project, variant):
    """Stage-3 rungs for a variant, found by their OUT_DIR marker (§4.4).

    §4.4 requires a distinct OUT_DIR per candidate — the deterministic
    `{OUT_DIR}_{seed}` path has destroyed evidence three times (§10.3) — so the
    marker is also what makes the rungs findable after the fact.
    """
    pat = re.compile(rf"stage3_{re.escape(variant)}_c(\d+)")
    found = {}
    for run in api.runs(f"{entity}/{project}", order="+created_at"):
        m = pat.search(str(cfg(run, "OUT_DIR", "")))
        if not m:
            continue
        n = int(m.group(1))
        if cfg(run, "num_chains") != n:
            print(f"  !! {run.id}: OUT_DIR says c{n} but num_chains="
                  f"{cfg(run, 'num_chains')} — mislabelled, check before using")
        found[n] = run                      # later run wins a duplicate rung
    return found


def check_pinned(run):
    """§4.1: stage 3 moves num_chains only.  Anything else moving is a bug."""
    bad = []
    if cfg(run, "num_samples") != 75:
        bad.append(f"num_samples={cfg(run, 'num_samples')} (must be 75, §4.1)")
    if cfg(run, "num_burn_in_steps") != 20000:
        bad.append(f"num_burn_in_steps={cfg(run, 'num_burn_in_steps')}")
    if cfg(run, "seed") != 0:
        bad.append(f"seed={cfg(run, 'seed')} (selection lineage is seed 0, §1)")
    if cfg(run, "burn_in_lr") not in (None, "None"):
        bad.append(f"burn_in_lr={cfg(run, 'burn_in_lr')} (must be absent, §3.2)")
    return bad


def fmt(summ, key, spec):
    v = summ.get(key)
    if v is None:
        return " " * len(spec.format(0))
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return spec.format(0).replace("0", "?")[:len(spec.format(0))]
    return spec.format(v)


def draws(run):
    return cfg(run, "num_chains") * cfg(run, "num_samples")


def gate_verdict(rungs, gate_ok):
    """Separate the §4.2 gate's growing POWER from a real change in drift.

    Two things a bare `loc_z > 2.0` cannot distinguish, and which point at
    opposite conclusions:

      * the drift got worse, or
      * the drift is unchanged and you simply bought the resolution to see it.

    Because z ~ |m2-m1| / mcse and mcse ~ 1/sqrt(C), doubling the chains
    inflates z by ~sqrt(2) on power alone.  Dividing that out leaves the real
    component, which the raw effect sizes then confirm independently.

    The stronger test is the last block.  `loc_sd` is |m2-m1| in sd units, a
    difference of means over ALL draws, so under a stationary sampler it must
    fall as 1/sqrt(total draws).  A `loc_sd` that stays flat while the draws
    multiply is a real non-stationarity — and one that was already there in
    the rungs that PASSED, which simply lacked the power to resolve it.
    """
    ladder = sorted(rungs)
    drift_is_real = False
    if len(ladder) < 2:
        return drift_is_real

    print("\n  §4.2.1 power vs effect size — the 2.0 threshold is NOT "
          "chain-count invariant")
    for a, b in zip(ladder, ladder[1:]):
        sa, sb = rungs[a].summary, rungs[b].summary
        power = math.sqrt(draws(rungs[b]) / draws(rungs[a]))
        print(f"   c{a} -> c{b}  (z inflates {power:.3f}x on power alone)")
        for zk, rk, lbl in (
            ("val_fn_drift_loc_z_median", "val_fn_drift_loc_sd_median", "loc"),
            ("val_fn_drift_scale_z_median",
             "val_fn_drift_scale_ratio_median", "scale"),
        ):
            za, zb, ra, rb = sa.get(zk), sb.get(zk), sa.get(rk), sb.get(rk)
            if not all(isinstance(v, (int, float)) and v for v in (za, zb, ra, rb)):
                continue
            real = (zb / za) / power
            note = ("consistent with power alone — nothing degraded"
                    if real < 1.15 else
                    f"REAL {real:.2f}x rise beyond power")
            print(f"       {lbl:5} z {za:7.4f} -> {zb:7.4f}  {zb / za:5.3f}x "
                  f"= {power:.3f}x power x {real:5.3f}x real   {note}")
            print(f"       {'':5} raw {ra:7.4f} -> {rb:7.4f}  {rb / ra:5.3f}x "
                  f"  (independent check on the {real:.2f}x above)")

    print("\n  stationarity test — raw loc_sd must fall as 1/sqrt(total draws)")
    r0, d0 = rungs[ladder[0]].summary.get("val_fn_drift_loc_sd_median"), draws(rungs[ladder[0]])
    if not isinstance(r0, (int, float)) or not r0:
        return drift_is_real
    worst = 0.0
    print(f"   {'rung':>5} {'draws':>6} {'loc_sd':>9} {'required':>9} {'obs/req':>8}")
    for n in ladder:
        r, d = rungs[n].summary.get("val_fn_drift_loc_sd_median"), draws(rungs[n])
        if not isinstance(r, (int, float)):
            continue
        req = r0 * math.sqrt(d0 / d)
        worst = max(worst, r / req)
        print(f"   c{n:<4} {d:6d} {r:9.4f} {req:9.4f} {r / req:8.2f}")
    if worst > 1.5:
        drift_is_real = True
        print(f"\n   !! loc_sd is {worst:.1f}x what stationarity requires at the top rung.")
        print("      The drift is NOT shrinking with draws, so it is real and")
        print("      present at EVERY rung — the PASSes below the failing rung")
        print("      are low-power false negatives, not clean bills of health.")
        print("      Note this drift is measured WITHIN each chain (first vs")
        print("      second half of its draws).  Adding chains cannot shrink it;")
        print("      it only measures it better.  Raising num_chains (§4.1) is")
        print("      therefore orthogonal to this failure — the next rung has to")
        print("      move num_samples / num_burn_in_steps, or the cyclical")
        print("      step-size schedule that function_space_drift also tests.")
    elif not all(gate_ok.values()):
        print("\n   The failing rung's raw drift is in line with 1/sqrt(draws),")
        print("      so the failure is the gate's rising power, not worse drift.")
    return drift_is_real


def report(entity, project, variant, baseline_id):
    api = wandb.Api()
    rungs = rung_runs(api, entity, project, variant)

    if baseline_id and 4 not in rungs:
        run = api.run(f"{entity}/{project}/{baseline_id}")
        if cfg(run, "num_chains") == 4:
            rungs[4] = run
        else:
            print(f"  !! baseline {baseline_id} has num_chains="
                  f"{cfg(run, 'num_chains')}, not 4 — ignoring")

    if not rungs:
        print(f"  no rungs found for {variant} (looked for OUT_DIR ~ "
              f"stage3_{variant}_c<N>)")
        return

    print(f"\n=== {variant} — stage-3 ladder ({project}) ===")

    print("\n  §4.2 gate — is this still sampling P_{f|D}?  "
          "(loc_z<=2.0, scale_z<=2.0, clamp<=0.01%)")
    print("   rung  run        chains  draws |   loc_z  scale_z    clamp% | "
          "  loc_sd scaleRatio  verdict")
    gate_ok = {}
    for n in sorted(rungs):
        run, s = rungs[n], rungs[n].summary
        fails = [f"{lbl} {s.get(k):.2f} > {thr}" for k, thr, lbl in GATE
                 if s.get(k) is None or not s.get(k) <= thr]
        gate_ok[n] = not fails
        vals = "".join(f"{s.get(k, float('nan')):9.4f}" for k, _, _ in GATE)
        raws = "".join(fmt(s, k, spec) for k, _, spec in RAW)
        # a run synced from an offline directory carries a UUID, not the usual
        # 8-character id; truncate so the columns still line up
        print(f"   c{n:<4} {run.id[:10]:10s} {cfg(run, 'num_chains'):6d} "
              f"{cfg(run, 'num_chains') * cfg(run, 'num_samples'):6d} |{vals} |"
              f"{raws}  "
              + ("PASS" if not fails else "!! FAIL — " + "; ".join(fails)))
        for msg in check_pinned(run):
            print(f"          !! {msg}")

    drift_is_real = gate_verdict(rungs, gate_ok)

    print("\n  §4.3/§4.5 tail statistics — steer on the medians; R-hat rising "
          "with chains is EXPECTED (§4.5)")
    if drift_is_real:
        print("  !! EVERY row below is compromised, not just the failing rung:")
        print("     the stationarity test above puts a real drift at all of")
        print("     them.  These are Monte Carlo errors about an estimand that")
        print("     is still moving, so they understate the true uncertainty")
        print("     however tight they look.  Do not select on them.")
    head = "   rung  draws | " + " ".join(lbl for _, lbl, _ in TAIL)
    print(head)
    print("   " + "-" * (len(head) - 3))
    for n in sorted(rungs):
        s = rungs[n].summary
        row = " ".join(fmt(s, k, spec) for k, _, spec in TAIL)
        mark = "" if gate_ok[n] else "   << §4.2 FAILED"
        draws = cfg(rungs[n], "num_chains") * cfg(rungs[n], "num_samples")
        print(f"   c{n:<4} {draws:6d} | {row}{mark}")

    ladder = sorted(rungs)
    if len(ladder) < 2:
        print("\n  only one rung — run the next chain count (§4.4) to compare.")
        return

    print("\n  §4.6 stop test — ESS should scale ~linearly with total draws, "
          "relMCSE as 1/sqrt(draws)")
    for a, b in zip(ladder, ladder[1:]):
        ratio = b / a
        sa, sb = rungs[a].summary, rungs[b].summary
        print(f"   c{a} -> c{b}  ({ratio:.0f}x draws)")
        for key, lbl, ideal in (
            ("val_pred_cvar_ess_median", "cvar_ess_median", ratio),
            ("val_pred_q05_ess_median", "q05_ess_median", ratio),
            ("val_pred_cvar_mcse_rel_median", "cvar_mcse_rel_median",
             1 / math.sqrt(ratio)),
        ):
            va, vb = sa.get(key), sb.get(key)
            if va is None or vb is None or not va:
                continue
            got = vb / va
            print(f"       {lbl:22s} {va:9.3f} -> {vb:9.3f}   "
                  f"{got:5.2f}x  (ideal {ideal:.2f}x)")
        for key, lbl in (("val_pred_cvar_rhat_median", "cvar_rhat_median"),
                         ("val_pred_folded_rhat_95th_pct", "folded_rhat_95th")):
            va, vb = sa.get(key), sb.get(key)
            if va is None or vb is None:
                continue
            note = "expected to rise (§4.5)" if vb >= va else "fell"
            print(f"       {lbl:22s} {va:9.4f} -> {vb:9.4f}   {note}")

        # §4.6: ESS flattening only means "budget is enough" if relMCSE is also
        # falling.  ESS flat while relMCSE rises means sd(u) is growing -- the
        # sampler is still finding tail mass and the estimand is still moving,
        # so stopping here would pick the rung that found the LEAST tail.
        ess_a, ess_b = sa.get("val_pred_cvar_ess_median"), sb.get("val_pred_cvar_ess_median")
        mc_a, mc_b = (sa.get("val_pred_cvar_mcse_rel_median"),
                      sb.get("val_pred_cvar_mcse_rel_median"))
        # mcse = sd(u)/sqrt(ESS), so ESS up AND relMCSE up forces sd(u) up --
        # no threshold needed, the two moving together is the whole signal.
        if None not in (ess_a, ess_b, mc_a, mc_b) and ess_a and mc_a:
            if ess_b > ess_a and mc_b > mc_a:
                grew = (mc_b / mc_a) * math.sqrt(ess_b / ess_a)
                print(f"\n       !! ESS ROSE ({ess_b / ess_a:.2f}x) and relMCSE "
                      f"ROSE ({mc_a:.3f} -> {mc_b:.3f}) together.")
                print(f"          Since mcse = sd(u)/sqrt(ESS), sd(u)/pred_sd "
                      f"grew ~{grew:.2f}x: the sampler is")
                print("          still FINDING tail mass, not resolving it, so "
                      "the CVaR estimand is still")
                print("          moving.  §4.6's stop rule does not apply here "
                      "-- §4.1 does.")

    if drift_is_real:
        print("\n  §4.6's stop rule DOES NOT APPLY here.  It asks whether the "
              "budget has\n  resolved the estimand; the stationarity test says "
              "the sampler is not\n  holding still long enough for there to be "
              "a fixed estimand to resolve.\n  Climbing to the next chain count "
              "would buy more power to detect the\n  same within-chain drift, "
              "not less drift.  Fix stationarity first (§4.1).")
    else:
        print("\n  Stop when a doubling buys little on ESS *and* relMCSE is "
              "still falling, with\n  §4.2 passing (§4.6) — ESS alone will "
              "mislead if the tail is still being found.")
    print("  Record the unresolved-point count and --worst-k from "
          "diagnose_sampling_tail.py\n  alongside this table; they are not "
          "logged to wandb.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="+", choices=sorted(WINNERS),
                    help="antmaze variant(s) to report")
    ap.add_argument("--entity", default="champlin-university-of-arizona")
    ap.add_argument("--project", default="BNN-training")
    ap.add_argument("--baseline", default=None,
                    help="run id to use as the c4 rung (default: the recorded "
                         "stage-1 winner for the variant, §6)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="report only dedicated stage-3 runs")
    args = ap.parse_args()

    for variant in args.variants:
        base = None if args.no_baseline else (args.baseline or WINNERS[variant])
        report(args.entity, args.project, variant, base)


if __name__ == "__main__":
    main()
