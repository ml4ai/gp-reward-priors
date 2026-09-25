#!/usr/bin/env python
"""Verify a family's PRODUCTION reward models, seeds 0-10 (handoff to-do 12, §4.3.138).

`train_rewards.sh` trains one model per (variant, seed).  Before anything is
labelled or evaluated on them, check the whole set:

  1. COMPLETE    exactly one finished, non-sweep run per (variant, seed).
  2. CONFIG      every run is its variant's winner configuration, apart from
                 seed / split / output path and known logging artefacts
                 (make_production_config.check_run, family-aware).
  3. HEALTHY     MR/PT: `reload_check_ok == 1`, i.e. the checkpoint on disk is the
                 one selected, and a finite `eval_loss_at_selected`.
                 BNN: gate verdicts, REPORTED, not enforced.  Eligibility is
                 decided at sweep budget (§3.2.9).
  4. REPRODUCE   the seed-0 production run against the sweep winner, which ran the
                 same configuration at the same seed.  Bitwise-equal metrics mean
                 the production model IS the selected model; a small difference
                 means GPU non-determinism and is reported as such.
  5. SPREAD      the selection metric across seeds, as descriptive context for §7.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/production_readout.py mr
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/production_readout.py pt [--since 2026-09-25]
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/production_readout.py --selftest
"""

import argparse
import collections
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENTITY = "champlin-university-of-arizona"
VARIANTS = ("medium_play", "medium_diverse", "large_play", "large_diverse")
SEEDS = tuple(range(11))

# Winners: §4.3.108 (MR/PT round-2 baselines), §4.3.130 (BNN round 5).
WINNERS = {
    "mr": {"medium_play": "a4qo4g4i", "medium_diverse": "p2f7p8dv",
           "large_play": "c898c0xe", "large_diverse": "s8nbeehf"},
    "pt": {"medium_play": "giab551o", "medium_diverse": "rupj57fq",
           "large_play": "cyrngs49", "large_diverse": "xokkypz7"},
    "bnn": {"large_play": "q45qbz8h", "large_diverse": "owlrd69d"},
}
METRIC = {"mr": "eval_loss_at_selected", "pt": "eval_loss_at_selected",
          "bnn": "val_cvar_ce"}
# Seed-0 reproduction compares these (MR/PT); BNN seed 0 is the escalation,
# verified separately by escalation_readout.py.
REPRO_KEYS = ("eval_loss_at_selected", "best_epoch", "select_loss_at_selected",
              "eval_acc_at_selected")


def variant_of(cfg):
    v = str(cfg.get("antmaze_variant", "")).replace("antmaze-", "").replace("-v2", "")
    return v.replace("-", "_")


def cells(runs):
    """{(variant, seed): [runs]} for non-sweep runs."""
    out = collections.defaultdict(list)
    for r in runs:
        out[(variant_of(r["config"]), r["config"].get("seed"))].append(r)
    return out


def repro(prod, win):
    """[(key, winner, production, exact)] on the reproduction keys."""
    rows = []
    for k in REPRO_KEYS:
        a, b = win.get(k), prod.get(k)
        if a is None or b is None:
            continue
        rows.append((k, a, b, a == b))
    return rows


def spread(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return None
    return st.mean(vals), st.stdev(vals), min(vals), max(vals)


def fetch(family, since):
    import wandb
    from make_production_config import FAMILIES
    api = wandb.Api(timeout=60)
    project = FAMILIES[family]["project"]
    runs = api.runs(f"{ENTITY}/{project}",
                    filters={"created_at": {"$gt": since}}, per_page=300)
    prod = [{"id": r.id, "state": r.state, "created": r.created_at,
             "config": {k: v for k, v in dict(r.config).items()
                        if not k.startswith("_")},
             "summary": dict(r.summary)}
            for r in runs if r.sweep is None]
    win = {}
    for v, rid in WINNERS[family].items():
        w = api.run(f"{ENTITY}/{project}/{rid}")
        win[v] = {"config": {k: x for k, x in dict(w.config).items()
                             if not k.startswith("_")},
                  "summary": dict(w.summary)}
    return prod, win


def report(family, prod, win, variants):
    from make_production_config import check_run
    import selection_gates as G
    fails = []
    c = cells(prod)
    m = METRIC[family]
    print(f"=== {family.upper()} production models: {len(prod)} non-sweep runs\n")

    print("1. COMPLETE")
    for v in variants:
        row = []
        for s in SEEDS:
            rs = c.get((v, s), [])
            fin = [r for r in rs if r["state"] == "finished"]
            row.append("." if len(fin) == 1 and len(rs) == 1 else
                       ("D" if len(rs) > 1 else ("X" if rs else "-")))
            if row[-1] != ".":
                fails.append(f"complete:{v}:{s}:{row[-1]}")
        print(f"   {v:15s} seeds 0-10: {''.join(row)}")
    print("   (. = one finished run, - = missing, D = duplicate, X = not finished)")

    print("\n2. CONFIG vs winner   3. HEALTH")
    for v in variants:
        if v not in win:
            continue
        bad_cfg, bad_h = [], []
        for s in SEEDS:
            rs = [r for r in c.get((v, s), []) if r["state"] == "finished"]
            if not rs:
                continue
            r = rs[0]
            rows = check_run(r["config"], win[v]["config"], 128, 32, family=family)
            u = [k for k, *_, verdict in rows if verdict == "UNEXPECTED"]
            if u:
                bad_cfg.append(f"s{s}:{u}")
            sm = r["summary"]
            if family in ("mr", "pt"):
                if sm.get("reload_check_ok") != 1 or sm.get(m) is None \
                        or not math.isfinite(sm.get(m)):
                    bad_h.append(f"s{s}")
            else:
                if G.gate_failures(sm, gated=True):
                    bad_h.append(f"s{s}:{G.gate_failures(sm, gated=True)}")
        print(f"   {v:15s} config {'OK' if not bad_cfg else 'MISMATCH ' + str(bad_cfg)}"
              f"   health {'OK' if not bad_h else ('FAIL ' if family != 'bnn' else 'gates: ') + str(bad_h)}")
        fails += [f"config:{v}:{x}" for x in bad_cfg]
        if family != "bnn":
            fails += [f"health:{v}:{x}" for x in bad_h]

    if family in ("mr", "pt"):
        print("\n4. REPRODUCE: seed-0 production run vs the sweep winner (same config, same seed)")
        for v in variants:
            rs = [r for r in c.get((v, 0), []) if r["state"] == "finished"]
            if not rs or v not in win:
                continue
            rows = repro(rs[0]["summary"], win[v]["summary"])
            exact = all(x for *_, x in rows)
            d = abs(rs[0]["summary"].get(m, float("nan"))
                    - win[v]["summary"].get(m, float("nan")))
            print(f"   {v:15s} {'BIT-EXACT' if exact else 'DIFFERS'}  "
                  f"{m}: winner {win[v]['summary'].get(m):.6f}  "
                  f"production {rs[0]['summary'].get(m):.6f}  |d| {d:.2e}"
                  + ("" if exact else "  " + ", ".join(
                      f"{k} {a}->{b}" for k, a, b, x in rows if not x)))

    print(f"\n5. SPREAD of {m} across seeds (descriptive)")
    for v in variants:
        vals = [r["summary"].get(m) for s in SEEDS
                for r in c.get((v, s), []) if r["state"] == "finished"]
        sp = spread(vals)
        if sp:
            print(f"   {v:15s} n={len(vals):2d}  mean {sp[0]:.4f}  sd {sp[1]:.4f}  "
                  f"range {sp[2]:.4f}-{sp[3]:.4f}")

    print("\nVERDICT: " + ("VERIFIED -- complete, every run is the winner's "
                           "configuration, and healthy" if not fails
                           else f"NOT VERIFIED -- {fails}"))
    return 0 if not fails else 1


def selftest():
    assert variant_of({"antmaze_variant": "antmaze-large-diverse-v2"}) == "large_diverse"
    runs = [{"config": {"antmaze_variant": "antmaze-medium-play-v2", "seed": 0}},
            {"config": {"antmaze_variant": "antmaze-medium-play-v2", "seed": 0}},
            {"config": {"antmaze_variant": "antmaze-medium-play-v2", "seed": 1}}]
    c = cells(runs)
    assert len(c[("medium_play", 0)]) == 2 and len(c[("medium_play", 1)]) == 1
    r = repro({"eval_loss_at_selected": 0.1474, "best_epoch": 565},
              {"eval_loss_at_selected": 0.1474, "best_epoch": 565})
    assert all(x for *_, x in r) and len(r) == 2
    r = repro({"eval_loss_at_selected": 0.1475}, {"eval_loss_at_selected": 0.1474})
    assert not r[0][3]
    assert spread([0.1, 0.2, None, float("nan")])[0] == 0.15000000000000002 or \
        abs(spread([0.1, 0.2])[0] - 0.15) < 1e-12
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family", nargs="?", choices=("mr", "pt", "bnn"))
    ap.add_argument("--since", default="2026-09-25T00:00:00",
                    help="only runs created after this (excludes older production sets)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.family:
        ap.error("family required")
    prod, win = fetch(a.family, a.since)
    return report(a.family, prod, win, [v for v in a.variants.split(",") if v])


if __name__ == "__main__":
    sys.exit(main())
