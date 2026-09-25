#!/usr/bin/env python
"""Verify and read out a §3.2.9 escalation run (handoff §4.3.130–§4.3.134).

The escalation is the seed-0 production model (§4.3.129).  Three checks, the
same for every variant, first done by hand for large_diverse (§4.3.131, §4.3.133):

  1. CONFIG      the launched run's wandb config equals the winner's, apart
                 from the chain budget and known logging artefacts
                 (make_production_config.check_run).
  2. REPRODUCE   chains 0-31 of the escalation, read with
                 `diagnose_sampling_tail.py --num-chains 32 --cvar-ce
                 --centre-draws --conservatism 0.75`, reproduce the winner's
                 logged values on every printed statistic.  A different
                 sampling path would move them all.
  3. PRECISION   the 128-chain summary against the 32-chain winner: what the
                 extra chains bought (ESS, tail draws, relMCSE, SE) and what they
                 were pre-stated NOT to change (R-hat, the CE estimand, gates).

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/escalation_readout.py \\
        large_play q45qbz8h <escalation_run_id> exp/escalation_large_play_repro32.txt
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/escalation_readout.py --selftest
"""

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ENTITY, PROJECT = "champlin-university-of-arizona", "BNN-training"
TAU_LOGR, TAU_LOC = math.log(1.122), 0.155        # selection_gates, §3.2.12

# (label, logged wandb key, regex on the repro file capturing the printed value)
REPRO = [
    ("CVaR CE (centred, primary)", "val_cvar_ce",
     r"CVaR\s+sigma\(Phi_cvar\)\s+([\d.]+)"),
    ("jackknife SE", "val_cvar_ce_se",
     r"jackknife-over-chains SE on the CVaR CE:\s+([\d.]+)"),
    ("degeneracy margin", "val_cvar_degeneracy_margin",
     r"verdict\s+\w+\s+\[margin\s+([+\-\d.]+)\]"),
    ("plug-in CE", "val_cvar_plugin_ce", r"plug-in\s+sigma\(E\[f\]\)\s+([\d.]+)"),
    ("predictive CE", "val_cvar_predictive_ce",
     r"predictive E\[sigma\(f\)\]\s+([\d.]+)"),
    ("CVaR acc", "val_cvar_acc", r"CVaR\s+sigma\(Phi_cvar\)\s+[\d.]+\s+([\d.]+)"),
    ("ESS bulk (raw) median", "val_pred_ess_median",
     r"ess_bulk\s+min\s+[\d.]+\s+median\s+([\d.]+)"),
    ("ESS bulk (centred) median", "val_pred_centred_ess_median",
     r"ess_bulk\s+\(centred\)\s+min\s+[\d.]+\s+median\s+([\d.]+)"),
    ("R-hat (centred) median", "val_pred_centred_rhat_median",
     r"rhat_bulk \(centred\)\s+min\s+[\d.]+\s+median\s+([\d.]+)"),
    ("between-chain share", "val_pred_centred_between_frac",
     r"between / \(within\+between\)\s+([\d.]+)"),
    ("centred CVaR eff draws median", "val_pred_centred_cvar_ess_median",
     r"CVaR eff draws \(median\)\s+[\d.]+\s+([\d.]+)"),
    ("centred CVaR MCSE/sd median", "val_pred_centred_cvar_mcse_rel_median",
     r"CVaR MCSE/sd \(median\)\s+[\d.]+\s+([\d.]+)"),
    ("centred loc_z median", "val_fn_drift_centred_loc_z_median",
     r"loc_z\s+\(CEN\)\s+([\d.]+)"),
    ("centred scale_z median", "val_fn_drift_centred_scale_z_median",
     r"scale_z \(CEN\)\s+([\d.]+)"),
    ("raw loc_sd median", "val_fn_drift_loc_sd_median",
     r"loc_sd\s+\(raw\)\s+([\d.]+)"),
]


def parse_repro(text):
    """{label: (value, decimals printed)}; missing labels are absent."""
    out = {}
    for lab, _, pat in REPRO:
        m = re.search(pat, text)
        if m:
            s = m.group(1)
            out[lab] = (float(s), len(s.split(".")[1]) if "." in s else 0)
    return out


def check_repro(text, logged):
    """[(label, logged, printed, match)]; a statistic absent from either side
    is reported as a failure, never skipped."""
    got = parse_repro(text)
    rows = []
    for lab, key, _ in REPRO:
        v = logged.get(key)
        if lab not in got or v is None:
            rows.append((lab, v, None, False))
            continue
        p, dp = got[lab]
        rows.append((lab, v, p, round(v, dp) == round(p, dp)))
    return rows


def precision_rows(w, e):
    """(label, 32-ch, 128-ch, expectation) for the escalation table."""
    ess = lambda s: s["val_pred_centred_ess_median"]
    logr = lambda s: abs(math.log(s["val_fn_drift_centred_scale_ratio_median"]))
    k = lambda key: (w[key], e[key])
    return [
        ("ESS centred median", *k("val_pred_centred_ess_median"), "~4x"),
        ("eff. tail draws @0.95", 0.05 * ess(w), 0.05 * ess(e), ">= 10 (3.2.1)"),
        ("centred CVaR relMCSE median",
         *k("val_pred_centred_cvar_mcse_rel_median"), "falls, ideal x0.5"),
        ("centred CVaR relMCSE max", *k("val_pred_centred_cvar_mcse_rel_max"), ""),
        ("q05 ESS centred median", *k("val_pred_centred_q05_ess_median"), "~linear"),
        ("jackknife SE on CVaR CE", *k("val_cvar_ce_se"), "~x0.5"),
        ("R-hat centred median", *k("val_pred_centred_rhat_median"),
         "pre-stated to RISE (4.5)"),
        ("between-chain share", *k("val_pred_centred_between_frac"), "~flat"),
        ("val_cvar_ce @0.75", *k("val_cvar_ce"), "same estimand"),
        ("val_cvar_ce @0.95", *k("val_cvar_ce_c0p95"), ""),
        ("|log r| (gate 1)", logr(w), logr(e), f"<= {TAU_LOGR:.4f}"),
        ("loc_sd centred (gate 1)", *k("val_fn_drift_centred_loc_sd_median"),
         f"<= {TAU_LOC}"),
        ("degeneracy margin (gate 2)", *k("val_cvar_degeneracy_margin"), "> 0"),
        ("ESS centred (gate 3)", *k("val_pred_centred_ess_median"), ">= 40"),
    ]


def verdicts(w, e):
    """Pre-registered readings (4.3.129 s5, 4.3.133) as (name, ok, detail)."""
    se_w, ce_w, ce_e = w["val_cvar_ce_se"], w["val_cvar_ce"], e["val_cvar_ce"]
    ess_ratio = e["val_pred_centred_ess_median"] / w["val_pred_centred_ess_median"]
    logr = abs(math.log(e["val_fn_drift_centred_scale_ratio_median"]))
    return [
        ("tail resolution met (>= 10 eff. tail draws)",
         0.05 * e["val_pred_centred_ess_median"] >= 10,
         f"{0.05 * e['val_pred_centred_ess_median']:.1f}"),
        ("ESS scaled at least 86% of linear (3.2.5)", ess_ratio >= 0.86 * 4,
         f"x{ess_ratio:.2f}"),
        ("relMCSE fell (4.6)",
         e["val_pred_centred_cvar_mcse_rel_median"]
         < w["val_pred_centred_cvar_mcse_rel_median"], ""),
        ("CE estimand agrees within 2 SE of the trial",
         abs(ce_e - ce_w) <= 2 * se_w, f"|d| {abs(ce_e - ce_w):.4f} vs 2SE {2 * se_w:.4f}"),
        ("gate 1 still passes at 128 (else REPORT, never de-select)",
         logr <= TAU_LOGR and e["val_fn_drift_centred_loc_sd_median"] <= TAU_LOC,
         f"|log r| {logr:.4f}"),
        ("gate 2 still passes", e["val_cvar_degeneracy_margin"] > 0, ""),
    ]


def selftest():
    text = ("  ess_bulk                   min   45.8389  median   46.2206  max   46.8283\n"
            "  ess_bulk  (centred)        min   53.6536  median   81.7039  max  183.2866\n"
            "  CVaR      sigma(Phi_cvar)       0.3980    0.8273\n"
            "  verdict  PASS   [margin +0.0830]\n")
    got = parse_repro(text)
    assert got["ESS bulk (raw) median"] == (46.2206, 4), got
    assert got["ESS bulk (centred) median"] == (81.7039, 4), got   # not the raw row
    assert got["CVaR acc"] == (0.8273, 4) and got["degeneracy margin"] == (0.083, 4)
    rows = check_repro(text, {"val_cvar_ce": 0.39795145, "val_pred_ess_median": 46.22063,
                              "val_pred_centred_ess_median": 81.70386})
    d = {r[0]: r[3] for r in rows}
    assert d["CVaR CE (centred, primary)"] and d["ESS bulk (centred) median"]
    assert not d["jackknife SE"], "absent statistic must FAIL, not be skipped"
    assert not check_repro(text, {"val_cvar_ce": 0.3990})[0][3], "real difference"
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant", nargs="?")
    ap.add_argument("winner", nargs="?")
    ap.add_argument("escalation", nargs="?")
    ap.add_argument("repro_file", nargs="?")
    ap.add_argument("--num-chains", type=int, default=128)
    ap.add_argument("--chains-per-gpu", type=int, default=32)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not all((a.variant, a.winner, a.escalation, a.repro_file)):
        ap.error("variant, winner, escalation run id and repro file are required")

    import wandb
    from make_production_config import check_run
    api = wandb.Api(timeout=60)
    W, E = (api.run(f"{ENTITY}/{PROJECT}/{r}") for r in (a.winner, a.escalation))
    cfg = lambda r: {k: v for k, v in dict(r.config).items() if not k.startswith("_")}
    w, e = dict(W.summary), dict(E.summary)
    fails = []

    print(f"=== {a.variant}: escalation {a.escalation} ({E.state}, "
          f"{(e.get('_runtime') or 0) / 3600:.2f} h) vs winner {a.winner} "
          f"({(w.get('_runtime') or 0) / 3600:.2f} h)\n")

    print("1. CONFIG")
    rows = check_run(cfg(E), cfg(W), a.num_chains, a.chains_per_gpu)
    for k, x, y, v in rows:
        print(f"   {k:26s} winner={x!r:34.34s} run={y!r:34.34s} {v}")
    bad = [k for k, *_, v in rows if v == "UNEXPECTED"]
    print("   -> " + ("PASS" if not bad else f"FAIL on {bad}"))
    fails += [f"config:{k}" for k in bad]

    print("\n2. REPRODUCE (chains 0-31 vs the winner's logged values)")
    rep = check_repro(open(a.repro_file).read(), w)
    for lab, v, p, ok in rep:
        print(f"   {lab:32s} logged {v!r:22.22s} printed {p!s:<10} "
              f"{'MATCH' if ok else '** DIFFER / MISSING **'}")
    n = sum(r[3] for r in rep)
    print(f"   -> {n}/{len(rep)} match")
    fails += [f"repro:{r[0]}" for r in rep if not r[3]]

    print("\n3. PRECISION (32-chain winner -> 128-chain escalation)")
    print(f"   {'statistic':30s} {'32 ch':>10s} {'128 ch':>10s} {'ratio':>7s}  expectation")
    for lab, x, y, exp in precision_rows(w, e):
        print(f"   {lab:30s} {x:10.4f} {y:10.4f} {y / x if x else float('nan'):7.2f}  {exp}")
    print()
    for name, ok, det in verdicts(w, e):
        print(f"   [{'PASS' if ok else '** NO **'}] {name}  {det}")
        if not ok:
            fails.append(f"reading:{name}")

    print("\nVERDICT: " + ("VERIFIED -- the production model is the selected trial, "
                           "and the escalation delivered its precision"
                           if not fails else f"NOT VERIFIED -- {fails}"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
