#!/usr/bin/env python
"""Read out the n_meas ladder (handoff §4.3.112) against its pre-registered rule.

Rungs are multiples of the OCCUPIED-CELL count c (26 medium, 46 large — §4.3.110),
against the pinned 256.  Every rung matches its variant's baseline sweep trial on
every field except `n_meas`.

    variant         baseline(256)   rungs
    medium_play     wc4nkymc        26 (= n26_medium_play), 52, 104
    large_diverse   o7g6texk        46, 92
    large_play      eeil9cq2        46, 92

PRIMARY READOUT IS THE DEPLOYMENT LEVEL, `val_cvar_ce_c0p95` — not the selection
level.  §4.3.109 measured the two disagreeing in SIGN on medium_play (0.318 →
0.244 at conservatism 0.75 while 0.322 → 0.373 at 0.95), and 0.95 is what the
paper reports.

The rule FIRES only if ALL of:
  1. `val_cvar_ce_c0p95` improves monotonically as n_meas falls, in ALL THREE
     variants, endpoint improvement beyond the combined 2*SE;
  2. gate-1 LOCATION holds with >= 1 sd headroom at every rung, every variant:
     loc_sd <= 0.1433 (= 0.155 - 0.0117).  0.1530, where the existing control
     sits, does NOT count as holding;
  3. gate-1 SCALE is not BROKEN by the change: a rung must satisfy
     |log r| <= log(1.122) where that variant's 256 baseline already passes, and
     |log r| <= baseline + 1 sd where it already fails.  (large_play's baseline
     is 0.1897, i.e. already failing, so an absolute test there would make the
     rule unfireable for a failure n_meas did not cause -- caught in the dry run
     before any rung ran, §4.3.112.)
  4. CVaR accuracy falls no more than 0.03 below that variant's baseline.

If it fires: pin n_meas at the LARGEST rung satisfying (2)-(4) whose
val_cvar_ce_c0p95 is within 2*SE of the best rung's — largest, not best, because
§4.3.24-25 established coverage is a real gain.  If it does not fire, n_meas
stays at 256 and this is a null result.  No partial adoption.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/nmeas_ladder_readout.py
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/nmeas_ladder_readout.py --selftest
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selection_gates as G  # noqa: E402

ENTITY, PROJECT = "champlin-university-of-arizona", "BNN-training"

SD_LOC = 0.0117          # §3.2.12, from a range — the weakest input (§4.3.112)
SD_SCALE = 0.0226        # §4.3.101, four pinned replicates
LOC_HEADROOM = G.TAU_LOC - SD_LOC          # 0.1433
TAU = math.log(G.TAU_SCALE)
ACC_DROP_MAX = 0.03

CELLS = {"medium_play": 26, "large_diverse": 46, "large_play": 46}

# variant -> {n_meas: (kind, key)}; kind "run" = wandb run id, "dir" = OUT_DIR marker
LADDER = {
    "medium_play": {256: ("run", "wc4nkymc"), 26: ("dir", "n26_medium_play"),
                    52: ("dir", "nmeas_medium_play_n52"),
                    104: ("dir", "nmeas_medium_play_n104")},
    "large_diverse": {256: ("run", "o7g6texk"),
                      46: ("dir", "nmeas_large_diverse_n46"),
                      92: ("dir", "nmeas_large_diverse_n92")},
    "large_play": {256: ("run", "eeil9cq2"),
                   46: ("dir", "nmeas_large_play_n46"),
                   92: ("dir", "nmeas_large_play_n92")},
}

K_CE95 = "val_cvar_ce_c0p95"
K_CE75 = "val_cvar_ce"
K_SE = "val_cvar_ce_se"
K_ACC = "val_cvar_acc"
K_SD = "val_pred_centred_sd_median"
K_MARGIN = "val_cvar_degeneracy_margin"
K_CLAMP = "param_clamp_sampling_pct"

ALWAYS_DIFF = {"OUT_DIR", "name", "group", "wandb_project", "checkpoints_path",
               "config_path"}
ALLOWED_DIFF = {"n_meas", "meas_sampling"}


def _norm(key, val):
    """§4.3.104: a hand-launched run logs the expanded width, a sweep the exponent."""
    if key == "width" and isinstance(val, (int, float)) and val > 10:
        return int(math.log2(val))
    return val


def audit(cfg, base):
    bad = []
    for k in sorted(set(cfg) | set(base)):
        if k in ALLOWED_DIFF or k in ALWAYS_DIFF:
            continue
        a, b = _norm(k, cfg.get(k, "<absent>")), _norm(k, base.get(k, "<absent>"))
        if repr(a) != repr(b):
            bad.append((k, b, a))
    return bad


def logr(ratio):
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
        return float("nan")
    return abs(math.log(ratio))


def f(x, spec=".4f"):
    return format(x, spec) if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def fetch():
    import wandb
    api = wandb.Api(timeout=60)
    by_id, by_dir = {}, {}
    ids = {v for var in LADDER.values() for kind, v in var.values() if kind == "run"}
    for rid in ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        by_id[rid] = {"state": r.state, "cfg": dict(r.config), "s": dict(r.summary)}
    markers = {v for var in LADDER.values() for kind, v in var.values() if kind == "dir"}
    for r in api.runs(f"{ENTITY}/{PROJECT}", order="-created_at", per_page=100):
        tag = str(r.config.get("OUT_DIR", "")).rstrip("/").split("/")[-1]
        body = tag[:-2] if tag.endswith("_0") else tag
        if body in markers and body not in by_dir:
            by_dir[body] = {"state": r.state, "cfg": dict(r.config),
                            "s": dict(r.summary)}
    return by_id, by_dir


def collect(by_id, by_dir):
    out = {}
    for variant, rungs in LADDER.items():
        got = {}
        for n, (kind, key) in rungs.items():
            rec = (by_id if kind == "run" else by_dir).get(key)
            if rec is not None:
                got[n] = dict(rec, key=key, n=n)
        out[variant] = got
    return out


def report(data):
    print("=" * 112)
    print("n_meas LADDER — handoff §4.3.112")
    print(f"rule: loc_sd <= {LOC_HEADROOM:.4f} (1 sd headroom)   |log r| <= {TAU:.4f}   "
          f"CVaR acc drop <= {ACC_DROP_MAX}   primary readout = {K_CE95}")
    print("=" * 112)

    missing = [(v, n) for v, g in data.items() for n in LADDER[v] if n not in g]
    if missing:
        print(f"\n!! RUNGS NOT FOUND: {missing}")
    unfin = [(v, n) for v, g in data.items() for n, r in g.items()
             if r["state"] != "finished"]
    if unfin:
        print(f"!! RUNGS NOT FINISHED: {unfin} -- do not read these")

    print("\n" + "-" * 112)
    print("CONFIG AUDIT -- every rung vs its variant's 256 baseline (only n_meas may differ)")
    print("-" * 112)
    ok_all = True
    for v, g in data.items():
        base = g.get(256)
        if base is None:
            print(f"  {v}: no 256 baseline, cannot audit")
            continue
        for n, r in sorted(g.items()):
            if n == 256:
                continue
            bad = audit(r["cfg"], base["cfg"])
            if bad:
                ok_all = False
                print(f"  {v} n={n}: UNEXPECTED DIFFERENCES")
                for k, b, a in bad:
                    print(f"      {k}: baseline={b!r}  rung={a!r}")
            else:
                print(f"  {v} n={n}: clean")
    if ok_all:
        print("\n  All rungs matched.")
    else:
        print("\n  !! NOT MATCHED -- fix before reading anything below (§4.3.90/96/98).")

    print("\n" + "-" * 112)
    for v, g in data.items():
        c = CELLS[v]
        print(f"{v}   (c = {c} occupied cells)")
        print(f"  {'n_meas':>7s} {'xc':>5s} {'clamp%':>7s} {'|log r|':>8s} {'loc_sd':>8s} "
              f"{'ess':>7s} {'margin':>9s} {'ce@0.75':>8s} {'ce@0.95':>8s} {'2SE':>7s} "
              f"{'acc':>6s} {'pred sd':>8s}")
        for n, r in sorted(g.items()):
            s = r["s"]
            loc = s.get(G.K_LOC_SD)
            flag = ""
            if isinstance(loc, float) and loc > LOC_HEADROOM:
                flag = "  <- loc headroom" if loc <= G.TAU_LOC else "  <- LOC FAILS"
            print(f"  {n:7d} {n/c:5.1f} {f(s.get(K_CLAMP), '.4f'):>7s} "
                  f"{f(logr(s.get(G.K_RATIO))):>8s} {f(loc):>8s} "
                  f"{f(s.get(G.K_ESS), '.1f'):>7s} {f(s.get(K_MARGIN), '+.5f'):>9s} "
                  f"{f(s.get(K_CE75)):>8s} {f(s.get(K_CE95)):>8s} "
                  f"{f(2 * (s.get(K_SE) or float('nan')), '.4f'):>7s} "
                  f"{f(s.get(K_ACC), '.3f'):>6s} {f(s.get(K_SD), '.3f'):>8s}{flag}")
        print()

    # ---------------- the pre-registered rule ----------------
    print("-" * 112)
    print("PRE-REGISTERED RULE (§4.3.112, fixed before the runs)")
    print("-" * 112)
    if missing or unfin:
        print("  Rungs missing or unfinished -- no verdict.")
        return

    c1 = c2 = c3 = c4 = True
    for v, g in data.items():
        ns = sorted(g)
        ce = [g[n]["s"].get(K_CE95) for n in ns]
        base_acc = g[256]["s"].get(K_ACC)
        mono = all(ce[i] <= ce[i + 1] + 1e-12 for i in range(len(ce) - 1))
        se2 = 2 * math.hypot(g[ns[0]]["s"].get(K_SE) or 0, g[256]["s"].get(K_SE) or 0)
        endpoint = (ce[-1] - ce[0]) > se2
        loc_ok = all(g[n]["s"].get(G.K_LOC_SD, 9) <= LOC_HEADROOM for n in ns)
        # (3) is a NO-BREAKING test, not an absolute one: where the 256 baseline
        # already fails gate-1 scale (large_play), a rung only has to avoid making
        # it materially worse -- n_meas did not cause that failure.
        base_lr = logr(g[256]["s"].get(G.K_RATIO))
        lr_cap = TAU if base_lr <= TAU else base_lr + SD_SCALE
        scale_ok = all(logr(g[n]["s"].get(G.K_RATIO)) <= lr_cap for n in ns)
        acc_ok = all((g[n]["s"].get(K_ACC, 0) - base_acc) >= -ACC_DROP_MAX for n in ns)
        c1 &= mono and endpoint
        c2 &= loc_ok
        c3 &= scale_ok
        c4 &= acc_ok
        print(f"  {v:15s} monotone+endpoint {str(mono and endpoint):5s}  "
              f"loc headroom {str(loc_ok):5s}  scale {str(scale_ok):5s}  "
              f"acc {str(acc_ok):5s}")
    print(f"\n  (1) monotone improvement, all variants : {c1}")
    print(f"  (2) loc_sd <= {LOC_HEADROOM:.4f} everywhere     : {c2}")
    print(f"  (3) gate-1 scale not broken by the change : {c3}")
    print(f"  (4) CVaR acc drop <= {ACC_DROP_MAX}           : {c4}")

    if c1 and c2 and c3 and c4:
        print("\n  >> FIRES.  Licensed change: pin n_meas at the LARGEST rung "
              "satisfying (2)-(4)")
        print("     whose ce@0.95 is within 2*SE of the best rung's, expressed as a "
              "multiple of c.")
        for v, g in data.items():
            ns = sorted(g)
            best = min(ns, key=lambda n: g[n]["s"].get(K_CE95, 9e9))
            bse = 2 * (g[best]["s"].get(K_SE) or 0)
            cand = [n for n in ns
                    if g[n]["s"].get(K_CE95, 9e9) <= g[best]["s"].get(K_CE95, 9e9) + bse
                    and g[n]["s"].get(G.K_LOC_SD, 9) <= LOC_HEADROOM]
            print(f"     {v:15s} -> n_meas {max(cand) if cand else 'none'} "
                  f"({max(cand)/CELLS[v]:.1f}c)" if cand else f"     {v}: none")
    else:
        print("\n  >> DOES NOT FIRE.  n_meas stays pinned at 256; report as a null "
              "result.")
        print("     No partial adoption -- 'it worked on two of three' is the "
              "cross-variant")
        print("     story this document has refuted nine times (§4.3.64).")


def selftest():
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got!r}")

    chk("loc headroom threshold", round(LOC_HEADROOM, 4), 0.1433)
    chk("tau", round(TAU, 6), 0.115113)
    chk("logr symmetric", round(logr(1.122) - logr(1 / 1.122), 12), 0.0)
    chk("width normalised", _norm("width", 32), 5)
    chk("width exponent untouched", _norm("width", 5), 5)
    chk("audit ignores n_meas", audit({"n_meas": 52, "width": 32},
                                      {"n_meas": 256, "width": 5}), [])
    chk("audit catches mdecay",
        [k for k, _, _ in audit({"mdecay": 0.2}, {"mdecay": 0.1})], ["mdecay"])
    # the existing control must be flagged as NOT having loc headroom
    chk("control loc_sd 0.1530 fails the headroom test", 0.15298 <= LOC_HEADROOM,
        False)
    chk("cells", (CELLS["medium_play"], CELLS["large_play"]), (26, 46))
    # (3)'s no-breaking semantics
    chk("scale cap where the baseline PASSES is the gate",
        round(TAU if 0.0444 <= TAU else 0.0444 + SD_SCALE, 6), round(TAU, 6))
    chk("scale cap where the baseline FAILS is baseline + 1 sd",
        round(TAU if 0.1897 <= TAU else 0.1897 + SD_SCALE, 4), 0.2123)
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    report(collect(*fetch()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
