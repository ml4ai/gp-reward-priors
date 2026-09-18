#!/usr/bin/env python
"""Read out the stratified-measurement diagnostics (handoff 4.3.109) against
their pre-registered rule.

Four arms, each matched to an EXISTING round-4 sweep trial that changes only
`meas_sampling` (or, for the control, only `n_meas`).  The pipeline is bitwise
deterministic at fixed seed (4.3.105), so the baselines need no re-run.

    arm                        baseline    tests
    strat_medium_play          wc4nkymc    does stratification fix DEGENERACY?
    strat_large_play           eeil9cq2    does it fix DRIFT at the top of the range?
    strat_large_diverse        o7g6texk    direction check only -- NOT decision-bearing
    n26_medium_play            wc4nkymc    control: is any effect just the batch size?

READ ON THE EFFECT SIZE, NOT THE VERDICT.  Every gate in this project is a
threshold on a noisy statistic, and 15 of the 25 finished round-4 trials sit
within one seed-sd of some threshold.  A pass/fail flip at n = 1 is not a
result; a change large against the measured run-to-run sd is.  The sds come
from 4.3.101's four pinned medium_play replicates (the only replicate set at
this budget):

    sd(|log centred scale_ratio|) = 0.0226   -> sd of a between-run DIFFERENCE 0.0320
    sd(degeneracy margin)         = 0.00358  -> sd of a between-run DIFFERENCE 0.00506

Those were measured at medium_play's settled sampler values, not at each arm's
architecture, so treating them as the noise floor here is an ASSUMPTION -- the
one 4.3.101 licenses ("use 0.0226 for any comparison between pinned runs"), and
it is stated rather than hidden.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/strat_readout.py
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/strat_readout.py --selftest
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selection_gates as G  # noqa: E402

ENTITY, PROJECT = "champlin-university-of-arizona", "BNN-training"

# 4.3.101's pinned-config replicate sds.  A difference of two independent runs
# carries sqrt(2) times these.
SD_LOGR = 0.0226
SD_MARGIN = 0.00358
SD_LOGR_DIFF = SD_LOGR * math.sqrt(2)
SD_MARGIN_DIFF = SD_MARGIN * math.sqrt(2)

TAU = math.log(G.TAU_SCALE)

# arm marker in OUT_DIR -> (label, baseline run id, variant, keys allowed to differ)
ARMS = {
    "strat_medium_play": ("medium_play stratified", "wc4nkymc", "medium_play",
                          {"meas_sampling", "n_meas"}),
    "strat_large_play": ("large_play stratified", "eeil9cq2", "large_play",
                         {"meas_sampling", "n_meas"}),
    "strat_large_diverse": ("large_diverse stratified", "o7g6texk", "large_diverse",
                            {"meas_sampling", "n_meas"}),
    "n26_medium_play": ("medium_play n_meas=26 CONTROL", "wc4nkymc", "medium_play",
                        {"meas_sampling", "n_meas"}),
}

# Always expected to differ between a hand-launched arm and a sweep trial.
ALWAYS_DIFF = {"OUT_DIR", "name", "group", "wandb_project", "checkpoints_path"}

K_RATIO = G.K_RATIO
K_LOCSD = G.K_LOC_SD
K_ESS = G.K_ESS
K_DEG = G.K_DEGEN
K_MARGIN = "val_cvar_degeneracy_margin"
K_GAP = "val_cvar_degeneracy_gap"
K_THR = "val_cvar_degeneracy_thr"
K_CE = "val_cvar_ce"
K_SE = "val_cvar_ce_se"
K_SD = "val_pred_centred_sd_median"
K_CLAMP = "param_clamp_sampling_pct"
K_CLIP = "gradnorm_sampling_pct_over_clip"

SUMMARY_KEYS = [K_RATIO, K_LOCSD, K_ESS, K_DEG, K_MARGIN, K_GAP, K_THR,
                K_CE, K_SE, K_SD, K_CLAMP, K_CLIP]


def logr(ratio):
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
        return float("nan")
    return abs(math.log(ratio))


def fmt(x, spec=".4f"):
    return format(x, spec) if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def fetch():
    """Baselines by run id; arms by OUT_DIR marker.  Newest arm wins."""
    import wandb
    api = wandb.Api(timeout=60)

    base = {}
    for rid in {b for _, b, _, _ in ARMS.values()}:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        base[rid] = {"id": rid, "state": r.state, "cfg": dict(r.config),
                     "s": dict(r.summary)}

    arms = {}
    for r in api.runs(f"{ENTITY}/{PROJECT}", order="-created_at", per_page=100):
        od = str(r.config.get("OUT_DIR", ""))
        tag = od.rstrip("/").split("/")[-1]
        body = tag[:-2] if tag.endswith("_0") else tag
        if body in ARMS and body not in arms:
            arms[body] = {"id": r.id, "state": r.state, "cfg": dict(r.config),
                          "s": dict(r.summary)}
    return base, arms


def audit(arm_cfg, base_cfg, allowed):
    """Every config key, arm vs baseline.  Returns the unexpected differences."""
    bad = []
    for k in sorted(set(arm_cfg) | set(base_cfg)):
        if k in allowed or k in ALWAYS_DIFF:
            continue
        a, b = arm_cfg.get(k, "<absent>"), base_cfg.get(k, "<absent>")
        if repr(a) != repr(b):
            bad.append((k, b, a))
    return bad


def gates(s):
    return G.gate_failures({k: s.get(k) for k in G.GATE_KEYS}, gated=True)


def row(label, s):
    return (f"{label:34s} {fmt(logr(s.get(K_RATIO))):>8s} "
            f"{fmt(s.get(K_LOCSD)):>8s} {fmt(s.get(K_ESS), '.1f'):>7s} "
            f"{fmt(s.get(K_MARGIN), '+.5f'):>9s} {fmt(s.get(K_CE)):>8s} "
            f"{fmt(s.get(K_SD), '.3f'):>7s}  {','.join(gates(s)) or 'ELIGIBLE'}")


def report(base, arms):
    print("=" * 108)
    print("STRATIFIED MEASUREMENT DIAGNOSTICS -- handoff 4.3.109")
    print(f"gates: |log r| <= {TAU:.4f} (ratio {G.TAU_SCALE})  loc_sd <= {G.TAU_LOC}  "
          f"ess >= {G.ESS_MIN:.0f}  degeneracy pass")
    print(f"noise: sd(|log r| diff) = {SD_LOGR_DIFF:.4f}   "
          f"sd(margin diff) = {SD_MARGIN_DIFF:.5f}   (4.3.101, assumed to transfer)")
    print("=" * 108)

    missing = [k for k in ARMS if k not in arms]
    if missing:
        print(f"\n!! ARMS NOT FOUND IN WANDB: {', '.join(missing)}")
        print("   (expected if they have not run yet -- the baselines below are still valid)")
    unfinished = [k for k, v in arms.items() if v["state"] != "finished"]
    if unfinished:
        print(f"\n!! ARMS NOT FINISHED: {', '.join(unfinished)} -- do not read these")

    # ---------------- config audit ----------------
    print("\n" + "-" * 108)
    print("CONFIG AUDIT -- every key, arm vs its baseline "
          "(only meas_sampling / n_meas / OUT_DIR / name may differ)")
    print("-" * 108)
    audit_ok = True
    for marker, (label, bid, _, allowed) in ARMS.items():
        if marker not in arms:
            continue
        bad = audit(arms[marker]["cfg"], base[bid]["cfg"], allowed)
        if bad:
            audit_ok = False
            print(f"  {label} vs {bid}: UNEXPECTED DIFFERENCES")
            for k, b, a in bad:
                print(f"      {k}: baseline={b!r}  arm={a!r}")
        else:
            print(f"  {label} vs {bid}: clean")
    if arms and audit_ok:
        print("\n  No unexpected config differences.  The arms are matched.")
    elif not audit_ok:
        print("\n  !! NOT MATCHED.  An unmatched comparison is how 4.3.90, 4.3.96 and")
        print("     4.3.98 each produced a wrong conclusion.  Fix before reading below.")

    # ---------------- validity ----------------
    print("\n" + "-" * 108)
    print("VALIDITY -- the momentum clamp is not measure-preserving (3.3, 4.3.71);")
    print("if it fires during sampling the tail numbers are invalid whatever they say")
    print("-" * 108)
    for marker, (label, _, _, _) in ARMS.items():
        if marker not in arms:
            continue
        s = arms[marker]["s"]
        clamp, clip = s.get(K_CLAMP), s.get(K_CLIP)
        flag = "" if (clamp or 0) <= 0.01 else "   !! CLAMP FIRED -- RUN INVALID"
        print(f"  {label:34s} clamp {fmt(clamp, '.4f'):>8s} %   "
              f"clip {fmt(clip, '.4f'):>8s} %{flag}")

    # ---------------- the table ----------------
    print("\n" + "-" * 108)
    print(f"{'run':34s} {'|log r|':>8s} {'loc_sd':>8s} {'ess':>7s} "
          f"{'margin':>9s} {'cvar_ce':>8s} {'pred sd':>7s}  gate failures")
    print("-" * 108)
    for marker, (label, bid, _, _) in ARMS.items():
        print(row(f"  BASE {bid} ({label.split()[0]})", base[bid]["s"]))
        if marker in arms:
            print(row(f"  ARM  {label}", arms[marker]["s"]))
        else:
            print(f"  ARM  {label:29s} -- not run --")
        print()

    # ---------------- paired deltas ----------------
    print("-" * 108)
    print("PAIRED DELTAS (arm - baseline), in units of the between-run sd")
    print("negative d|log r| = more stationary;  positive d margin = less degenerate")
    print("-" * 108)
    print(f"{'arm':34s} {'d|log r|':>9s} {'z':>6s}  {'d margin':>10s} {'z':>6s}  "
          f"{'d cvar_ce':>10s} {'2SE':>7s}  {'d ess':>7s}")
    for marker, (label, bid, _, _) in ARMS.items():
        if marker not in arms:
            continue
        a, b = arms[marker]["s"], base[bid]["s"]
        d_lr = logr(a.get(K_RATIO)) - logr(b.get(K_RATIO))
        d_mg = (a.get(K_MARGIN) or float("nan")) - (b.get(K_MARGIN) or float("nan"))
        d_ce = (a.get(K_CE) or float("nan")) - (b.get(K_CE) or float("nan"))
        se2 = 2 * math.hypot(a.get(K_SE) or float("nan"), b.get(K_SE) or float("nan"))
        d_ess = (a.get(K_ESS) or float("nan")) - (b.get(K_ESS) or float("nan"))
        print(f"{label:34s} {fmt(d_lr, '+.4f'):>9s} {fmt(d_lr / SD_LOGR_DIFF, '+.1f'):>6s}  "
              f"{fmt(d_mg, '+.5f'):>10s} {fmt(d_mg / SD_MARGIN_DIFF, '+.1f'):>6s}  "
              f"{fmt(d_ce, '+.4f'):>10s} {fmt(se2, '.4f'):>7s}  {fmt(d_ess, '+.1f'):>7s}")

    # ---------------- the pre-registered verdict ----------------
    print("\n" + "-" * 108)
    print("PRE-REGISTERED VERDICT (4.3.109, fixed before the runs)")
    print("-" * 108)
    if any(k not in arms for k in ARMS):
        print("  Arms missing -- no verdict.")
        return
    mp, ctl, lp = arms["strat_medium_play"]["s"], arms["n26_medium_play"]["s"], \
        arms["strat_large_play"]["s"]
    b_mp, b_lp = base["wc4nkymc"]["s"], base["eeil9cq2"]["s"]

    d_mg_mp = (mp.get(K_MARGIN) or float("nan")) - (b_mp.get(K_MARGIN) or float("nan"))
    d_mg_ctl = (ctl.get(K_MARGIN) or float("nan")) - (b_mp.get(K_MARGIN) or float("nan"))
    d_lr_lp = logr(lp.get(K_RATIO)) - logr(b_lp.get(K_RATIO))
    worst_lr = max(
        logr(arms[m]["s"].get(K_RATIO)) - logr(base[ARMS[m][1]]["s"].get(K_RATIO))
        for m in ("strat_medium_play", "strat_large_play", "strat_large_diverse"))

    g2_fixed = d_mg_mp >= 2 * SD_MARGIN_DIFF
    g1_held = worst_lr <= SD_LOGR_DIFF
    g1_worse = worst_lr >= 2 * SD_LOGR_DIFF
    ctl_same = abs(d_mg_mp - d_mg_ctl) <= SD_MARGIN_DIFF

    print(f"  medium_play d margin  {d_mg_mp:+.5f}  ({d_mg_mp / SD_MARGIN_DIFF:+.1f} sd) "
          f"-> gate 2 fixed: {g2_fixed}")
    print(f"  control     d margin  {d_mg_ctl:+.5f}  "
          f"-> control reproduces it: {ctl_same}")
    print(f"  worst d|log r|        {worst_lr:+.4f}  ({worst_lr / SD_LOGR_DIFF:+.1f} sd) "
          f"-> gate 1 held: {g1_held}, gate 1 worse: {g1_worse}")
    print(f"  large_play d|log r|   {d_lr_lp:+.4f}  ({d_lr_lp / SD_LOGR_DIFF:+.1f} sd)")
    print()
    if ctl_same and g2_fixed:
        print("  >> BATCH SIZE.  The n_meas=26 control reproduces the stratified effect,")
        print("     so this is measurement-batch size, not the cell structure.  Change")
        print("     n_meas -- a pinned value -- not the prior.  No restart on this basis.")
    elif g2_fixed and g1_held:
        print("  >> ADOPT-CANDIDATE.  Gate 2 improves resolvably, gate 1 does not")
        print("     degrade, and the control does not explain it.  Pre-register the")
        print("     change and restart (10.2 item 4); batch it with item 5.")
    elif g2_fixed and g1_worse:
        print("  >> TRADE.  Gate 2 improves and gate 1 degrades -- stratification moves")
        print("     along 4.3.101's stationarity/degeneracy frontier rather than off it.")
        print("     Do not adopt on this evidence.")
    else:
        print("  >> NULL.  No resolvable improvement.  Record it and resume the sweeps;")
        print("     do not restart.")
    print("\n  large_diverse is a DIRECTION CHECK ONLY.  Its baseline margin is")
    print(f"  {base['o7g6texk']['s'].get(K_MARGIN):+.5f} = "
          f"{abs(base['o7g6texk']['s'].get(K_MARGIN)) / SD_MARGIN_DIFF:.2f} sd from zero,")
    print("  so a pass/fail flip there is a coin toss and carries no evidence.")


def selftest():
    """Known-answer checks on the pure functions -- no wandb."""
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = (abs(got - want) < 1e-9) if isinstance(want, float) else got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got!r}")

    chk("logr(1.0) == 0", logr(1.0), 0.0)
    chk("logr is symmetric", logr(1.122) - logr(1 / 1.122), 0.0)
    chk("logr(0) is nan", math.isnan(logr(0)), True)
    chk("tau", round(TAU, 6), 0.115113)

    # gate wiring: the three real baselines, against the verdicts computed by hand
    # from wandb on 2026-09-17.
    cases = [
        ("wc4nkymc", {K_RATIO: 0.9565436765303784, K_LOCSD: 0.11856261787813815,
                      K_ESS: 66.08769946326152, K_DEG: 0}, ["degen"]),
        ("eeil9cq2", {K_RATIO: 0.8272307386633364, K_LOCSD: 0.09739438340101449,
                      K_ESS: 332.81843470598557, K_DEG: 1}, ["scale"]),
        ("o7g6texk", {K_RATIO: 0.8947660161485049, K_LOCSD: 0.08746627494334178,
                      K_ESS: 120.58161991782518, K_DEG: 0}, ["degen"]),
    ]
    for rid, s, want in cases:
        chk(f"gates({rid})", gates(s), want)

    # a missing gate key must make a trial ineligible, never waved through
    chk("missing ess is a failure", gates({K_RATIO: 1.0, K_LOCSD: 0.0, K_DEG: 1}),
        ["ess=missing"])

    # the audit must catch a swept-field mismatch and ignore the allowed ones
    chk("audit clean", audit({"width": 5, "meas_sampling": "stratified_cell",
                              "OUT_DIR": "a"},
                             {"width": 5, "OUT_DIR": "b"},
                             {"meas_sampling", "n_meas"}), [])
    chk("audit catches width", [k for k, _, _ in
                                audit({"width": 6}, {"width": 5}, set())], ["width"])

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run known-answer checks and exit (no wandb)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    base, arms = fetch()
    report(base, arms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
