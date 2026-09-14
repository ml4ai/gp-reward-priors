#!/usr/bin/env python
"""Read out the capacity ladder (handoff 3.2.13, 3.2.15) against its pre-registered rule.

Rule (3.2.13, fixed before the ladder ran), applied to IN-RANGE rungs only
(width 6-9; w4/w5 are exploratory and license nothing, 3.2.15):

  * statistic: |log(centred scale_ratio)| and val_cvar_ce, per variant, vs n_params
  * FIRES if |log r| rises monotonically by >= 0.07 across the in-range widths
    (~3 sigma at the pinned-config sd 0.0226, 4.3.101) -- direction plus effect
    size, not a p-value (n = 4 gives an exact-p floor of 0.083)
  * if it fires: cap at the largest width whose |log r| is within tau (ratio
    1.122, i.e. |log r| <= log 1.122) AND whose val_cvar_ce is no worse than the
    smallest in-range width's by more than 2x the jackknife SE

Also: a config audit (every key, every rung, within variant), the round-3 gate
table for every rung, and the matched 8-vs-2-thread pair.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/capacity_ladder_readout.py
"""

import itertools
import math
import sys

import numpy as np

ENTITY, PROJECT = "champlin-university-of-arizona", "BNN-training"
TAU_RATIO = 1.122
TAU = math.log(TAU_RATIO)
SIGMA = 0.0226
FIRE_RISE = 0.07
LOC_SD_MAX = 0.155
ESS_MIN = 40
IN_RANGE = (6, 7, 8, 9)
DEPTH = {"large_diverse": 4, "large_play": 6}
N_TRAIN = {"large_play": 254, "large_diverse": 514}
D_IN = 37

K_RATIO = "val_fn_drift_centred_scale_ratio_median"
K_LOCSD = "val_fn_drift_centred_loc_sd_median"
K_ESS = "val_pred_centred_ess_median"
K_CE, K_CE75, K_SE = "val_cvar_ce", "val_cvar_ce_c0p75", "val_cvar_ce_se"
K_MEAN = "val_mean_cross_entropy"
K_DEG = "val_cvar_degeneracy_pass"
K_GAP, K_MARGIN = "val_cvar_degeneracy_gap", "val_cvar_degeneracy_margin"
K_SD = "val_pred_centred_sd_median"

# Keys expected to differ between rungs of one variant.
EXPECTED_DIFF = {"width", "OUT_DIR", "name"}


def n_params(W, d, d_in=D_IN):
    return W * (d_in + 1) + (d - 1) * W * (W + 1) + (W + 1)


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(x, y):
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def exact_p(x, y):
    r0 = abs(spearman(x, y))
    y = np.asarray(y)
    hits = tot = 0
    for perm in itertools.permutations(range(len(y))):
        tot += 1
        hits += abs(spearman(x, y[list(perm)])) >= r0 - 1e-12
    return hits / tot


def fetch():
    import wandb
    api = wandb.Api(timeout=60)
    out = {}
    for r in api.runs(f"{ENTITY}/{PROJECT}", order="-created_at", per_page=100):
        od = str(r.config.get("OUT_DIR", ""))
        if "cap_ladder" not in od:
            continue
        tag = od.rstrip("/").split("/")[-1]          # cap_ladder2_large_play_w6_0
        if tag in out:
            continue                                  # newest wins; flag below
        out[tag] = {"id": r.id, "state": r.state, "cfg": dict(r.config),
                    "s": dict(r.summary)}
    return out


def parse(tag):
    body = tag[:-2] if tag.endswith("_0") else tag
    gen = "2thr" if body.startswith("cap_ladder2_") else "8thr"
    rest = body.replace("cap_ladder2_", "").replace("cap_ladder_", "")
    variant, w = rest.rsplit("_w", 1)
    return gen, variant, int(w)


def f(x, spec):
    return format(x, spec) if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def main():
    runs = fetch()
    bad_state = [t for t, r in runs.items() if r["state"] != "finished"]
    if bad_state:
        print("NOT FINISHED:", bad_state)

    table = {}
    for tag, r in runs.items():
        gen, v, w = parse(tag)
        s, c = r["s"], r["cfg"]
        ratio = s.get(K_RATIO, float("nan"))
        row = dict(tag=tag, id=r["id"], gen=gen, variant=v, w=w, cfg=c, s=s,
                   depth=c.get("depth"), width_cfg=c.get("width"),
                   n=n_params(2 ** w, DEPTH[v]),
                   logr=abs(math.log(ratio)) if ratio and ratio > 0 else float("nan"),
                   ratio=ratio, locsd=s.get(K_LOCSD, float("nan")),
                   ess=s.get(K_ESS, float("nan")), ce=s.get(K_CE, float("nan")),
                   ce75=s.get(K_CE75, float("nan")), se=s.get(K_SE, float("nan")),
                   mean=s.get(K_MEAN, float("nan")), deg=s.get(K_DEG),
                   gap=s.get(K_GAP, float("nan")), margin=s.get(K_MARGIN, float("nan")),
                   sd=s.get(K_SD, float("nan")),
                   hours=(s.get("_runtime") or 0) / 3600)
        table.setdefault(gen, {}).setdefault(v, {})[w] = row

    # ---------------- config audit ----------------
    print("=" * 100)
    print("CONFIG AUDIT -- every key, every 2-thread rung, within variant "
          "(only width / OUT_DIR / name may differ)")
    print("=" * 100)
    audit_ok = True
    for v, rungs in sorted(table.get("2thr", {}).items()):
        ws = sorted(rungs)
        ref = rungs[ws[0]]["cfg"]
        keys = set().union(*(set(rungs[w]["cfg"]) for w in ws))
        diffs = []
        for k in sorted(keys):
            vals = {w: rungs[w]["cfg"].get(k, "<absent>") for w in ws}
            if len({repr(x) for x in vals.values()}) > 1 and k not in EXPECTED_DIFF:
                diffs.append((k, vals))
        for w in ws:
            c = rungs[w]["cfg"]
            if c.get("width") != 2 ** w or c.get("depth") != DEPTH[v]:
                diffs.append((f"width/depth for w{w}", {w: (c.get("width"), c.get("depth"))}))
        print(f"  {v:<14} rungs {ws}  keys {len(keys)}  unexpected differences: {len(diffs)}")
        for k, vals in diffs:
            audit_ok = False
            print(f"      {k}: {vals}")
    print(f"  pinned values: n_meas={ref.get('n_meas')} map_amp2={ref.get('map_amp2')} "
          f"jitter={ref.get('chain_init_jitter')} chains={ref.get('num_chains')} "
          f"samples={ref.get('num_samples')} cycle={ref.get('cycle_length')} "
          f"burn={ref.get('num_burn_in_steps')} n_discarded={ref.get('n_discarded')}")
    print("  NOTE: thread count is not logged by wandb and cannot be audited here.")

    # ---------------- per-variant ladder ----------------
    verdicts = {}
    for v in sorted(table.get("2thr", {})):
        rungs = table["2thr"][v]
        ws = sorted(rungs)
        print()
        print("=" * 100)
        print(f"{v.upper()}  (depth {DEPTH[v]}, {N_TRAIN[v]} training pairs)")
        print("=" * 100)
        print(f"  {'w':>2} {'n_params':>10} {'p/pair':>7} {'|log r|':>8} {'x tau':>6} "
              f"{'g1':>4} {'loc_sd':>7} {'ess':>6} {'g3':>4} {'cvar_ce':>8} {'SE':>7} "
              f"{'mean_ce':>8} {'gap':>7} {'margin':>8} {'g2':>4} {'pred_sd':>7} {'hours':>6}")
        print("  " + "-" * 118)
        for w in ws:
            r = rungs[w]
            g1 = r["logr"] <= TAU and r["locsd"] <= LOC_SD_MAX
            g3 = r["ess"] >= ESS_MIN
            g2 = bool(r["deg"]) if r["deg"] is not None else None
            mark = "" if w in IN_RANGE else "  (exploratory)"
            print(f"  {w:>2} {r['n']:>10,} {r['n'] / N_TRAIN[v]:>7,.0f} {f(r['logr'], '.4f'):>8} "
                  f"{f(r['logr'] / TAU, '.2f'):>5}x {'PASS' if g1 else 'FAIL':>4} "
                  f"{f(r['locsd'], '.4f'):>7} {f(r['ess'], '.1f'):>6} {'PASS' if g3 else 'FAIL':>4} "
                  f"{f(r['ce'], '.4f'):>8} {f(r['se'], '.4f'):>7} {f(r['mean'], '.4f'):>8} "
                  f"{f(r['gap'], '.4f'):>7} {f(r['margin'], '+.4f'):>8} "
                  f"{('PASS' if g2 else 'FAIL') if g2 is not None else 'n/a':>4} "
                  f"{f(r['sd'], '.3f'):>7} {r['hours']:>6.2f}{mark}")
            if math.isfinite(r["ce75"]) and math.isfinite(r["ce"]) and abs(r["ce75"] - r["ce"]) > 1e-9:
                print(f"       ! val_cvar_ce {r['ce']:.6f} != val_cvar_ce_c0p75 {r['ce75']:.6f}")

        for label, sel in (("ALL rungs (descriptive)", ws),
                           ("IN-RANGE rungs (the licensing sample)", [w for w in ws if w in IN_RANGE])):
            n = [rungs[w]["n"] for w in sel]
            lr = [rungs[w]["logr"] for w in sel]
            ce = [rungs[w]["ce"] for w in sel]
            print(f"\n  {label}, n={len(sel)}:")
            print(f"    rho(n_params, |log r|)  = {spearman(n, lr):+.3f}   exact p {exact_p(n, lr):.4f}")
            print(f"    rho(n_params, cvar_ce)  = {spearman(n, ce):+.3f}   exact p {exact_p(n, ce):.4f}")

        # ---- the pre-registered rule ----
        ir = [w for w in ws if w in IN_RANGE]
        lr = np.array([rungs[w]["logr"] for w in ir])
        rise = float(lr[-1] - lr[0])
        monotone = bool(np.all(np.diff(lr) > 0))
        fires = monotone and rise >= FIRE_RISE
        print(f"\n  RULE (3.2.13), in-range w{ir[0]}-w{ir[-1]}:")
        print(f"    |log r| sequence {np.round(lr, 4).tolist()}")
        print(f"    monotone rise: {monotone};  w{ir[-1]} - w{ir[0]} = {rise:+.4f} "
              f"= {rise / SIGMA:+.1f} sigma  (threshold +{FIRE_RISE}, ~3 sigma)")
        if not monotone:
            steps = np.diff(lr)
            print(f"    step changes {np.round(steps, 4).tolist()}  "
                  f"(largest reversal {steps.min():+.4f} = {steps.min() / SIGMA:+.1f} sigma)")
        print(f"    FIRES: {'YES' if fires else 'NO'}")

        ref = rungs[ir[0]]
        cap = None
        for w in ir:
            r = rungs[w]
            ok_tau = r["logr"] <= TAU
            ok_ce_ref = r["ce"] <= ref["ce"] + 2 * ref["se"]
            ok_ce_own = r["ce"] <= ref["ce"] + 2 * r["se"]
            print(f"    w{w}: |log r| within tau {ok_tau};  cvar_ce - w{ir[0]} = "
                  f"{r['ce'] - ref['ce']:+.4f}  vs 2*SE(w{ir[0]}) {2 * ref['se']:.4f} -> {ok_ce_ref};"
                  f"  vs 2*SE(w{w}) {2 * r['se']:.4f} -> {ok_ce_own}")
            if ok_tau and ok_ce_ref and ok_ce_own:
                cap = w
        print(f"    largest in-range width passing both conditions: "
              f"{'w' + str(cap) if cap else 'none'}")
        verdicts[v] = dict(fires=fires, cap=cap, rise=rise, monotone=monotone)

    # ---------------- 8 vs 2 threads ----------------
    old = table.get("8thr", {})
    if old:
        print()
        print("=" * 100)
        print("MATCHED 8-vs-2-THREAD PAIR (identical config otherwise; a reference, not rungs)")
        print("=" * 100)
        for v, rungs in sorted(old.items()):
            for w, r8 in sorted(rungs.items()):
                r2 = table.get("2thr", {}).get(v, {}).get(w)
                if not r2:
                    continue
                diffs = [k for k in set(r8["cfg"]) | set(r2["cfg"])
                         if repr(r8["cfg"].get(k)) != repr(r2["cfg"].get(k))
                         and k not in {"OUT_DIR", "name"}]
                print(f"  {v} w{w}: config differences {sorted(diffs) or 'none'}")
                for lab, k, spec in (("|log r|", "logr", ".4f"), ("loc_sd", "locsd", ".4f"),
                                     ("ess", "ess", ".1f"), ("cvar_ce", "ce", ".4f"),
                                     ("SE", "se", ".4f"), ("pred_sd", "sd", ".3f"),
                                     ("hours", "hours", ".2f")):
                    a, b = r8[k], r2[k]
                    extra = ""
                    if k == "logr":
                        extra = f"   diff {b - a:+.4f} = {(b - a) / SIGMA:+.2f} sigma"
                    if k == "ce":
                        extra = f"   diff {b - a:+.4f} = {(b - a) / max(r8['se'], 1e-12):+.2f} SE"
                    print(f"      {lab:<8} 8thr {f(a, spec):>8}   2thr {f(b, spec):>8}{extra}")

    print()
    print("SUMMARY:", {v: d for v, d in verdicts.items()}, "| audit clean:", audit_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
