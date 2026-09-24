#!/usr/bin/env python
"""Regenerate the numbers behind §7.4's round-5 disclosures (to-do 18h, 18i, 18j).

Two of the four sweeps were still searching when §7.4 was written, so its tables
must be recomputed on the final data rather than copied from a mid-sweep check.
This is the one command that does it.

  18i  WINNER SEPARATION.  The winner is the lowest `val_cvar_ce` among ELIGIBLE
       trials (§3.2.12), taken over the trials up to the K=15 stop.  A trial is
       TIED with it when |dCE| <= 2*sqrt(SE_w^2 + SE_t^2).  Reports the tied set
       and the architectures it spans -- the disclosure is two-sided: the TRIAL
       is an argmin over noise, the ARCHITECTURE is determined.
  18h  DEPTH COVERAGE.  Per sweep: how often each depth was sampled, and whether
       the depth-1 trials used sampler settings inside the range of that sweep's
       own eligible trials.  large_play is the case §7.4 discloses.
  18j  TRIAL COUNTS against the MR/PT baselines' 17-66 (§4.3.115).  The budget
       invariant is one-sided: the BNN must get no MORE search than the baselines.

The stop point is computed with check_sweep_convergence.frontier, so it cannot
drift from the tool that decides when a sweep has stopped.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/round5_disclosures.py
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/round5_disclosures.py --selftest
"""

import argparse
import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selection_gates as G                       # noqa: E402
from check_sweep_convergence import DEFAULT_PATIENCE, frontier   # noqa: E402

ENTITY, PROJECT = "champlin-university-of-arizona", "BNN-training"
SWEEPS = {"37wlya3i": "medium_play", "03iccldi": "medium_diverse",
          "obt9bwtz": "large_play", "3svmvmle": "large_diverse"}
ORDER = ("medium_play", "medium_diverse", "large_play", "large_diverse")
BASELINE_TRIALS = (17, 66)          # MR/PT completed sweeps, §4.3.115
SAMPLER_KEYS = ("sghmc_lr", "sghmc_lr_max", "mdecay")
D_IN = 37


def n_params(w, d):
    W = 2 ** w
    return W * (D_IN + 1) + (d - 1) * W * (W + 1) + (W + 1)


def fetch():
    import wandb
    api = wandb.Api(timeout=60)
    out = {}
    for sid, v in SWEEPS.items():
        sw = api.sweep(f"{ENTITY}/{PROJECT}/{sid}")
        rows = []
        for r in sw.runs:
            s, c = dict(r.summary), dict(r.config)
            J = s.get("val_cvar_ce_penalised")
            if J is None:
                continue
            rows.append({
                "t": str(r.created_at), "id": r.id, "J": J,
                "ce": s.get("val_cvar_ce"), "se": s.get("val_cvar_ce_se"),
                "w": c.get("width"), "d": c.get("depth"),
                "eligible": not G.gate_failures(s, gated=True),
                **{k: c.get(k) for k in SAMPLER_KEYS},
            })
        rows.sort(key=lambda x: x["t"])
        out[v] = {"rows": rows, "state": sw.state}
    return out


def truncate_at_stop(rows, patience=DEFAULT_PATIENCE):
    """(trials up to the stop, stop index or None).  Same rule as the tool."""
    trials = [(r["t"], r["J"], r["id"], r["eligible"]) for r in rows]
    _, _, _, trigger = frontier(trials, "minimize", patience, eligible_only=False)
    return (rows[:trigger] if trigger else rows), trigger


def tied_set(rows):
    """(winner, [tied trials]) on val_cvar_ce among eligible trials."""
    el = [r for r in rows if r["eligible"] and r["ce"] is not None
          and r["se"] is not None]
    if not el:
        return None, []
    el.sort(key=lambda r: r["ce"])
    w = el[0]
    tied = []
    for r in el[1:]:
        thr = 2.0 * math.sqrt(w["se"] ** 2 + r["se"] ** 2)
        if r["ce"] - w["ce"] <= thr:
            tied.append(dict(r, delta=r["ce"] - w["ce"], thr=thr))
    return w, tied


def runner_up_ratio(rows, winner):
    """delta / joint-2SE for the best eligible non-winner (separation margin)."""
    el = sorted((r for r in rows if r["eligible"] and r["ce"] is not None
                 and r["se"] is not None and r["id"] != winner["id"]),
                key=lambda r: r["ce"])
    if not el:
        return None, None
    r = el[0]
    return r, (r["ce"] - winner["ce"]) / (2.0 * math.sqrt(winner["se"] ** 2
                                                          + r["se"] ** 2))


def report(data):
    print("=" * 72)
    print("18i  WINNER SEPARATION  (winner = lowest val_cvar_ce among eligible,")
    print("     over trials up to the K=15 stop; tied if |dCE| <= joint 2*SE)")
    print("=" * 72)
    for v in ORDER:
        rows, trig = truncate_at_stop(data[v]["rows"])
        status = (f"STOPPED at trial {trig}" if trig
                  else f"LIVE ({len(data[v]['rows'])} trials) -- PROVISIONAL")
        w, tied = tied_set(rows)
        print(f"\n  {v}   [{status}]")
        if w is None:
            print("    no eligible trial")
            continue
        print(f"    winner  {w['id']}  w{w['w']}xd{w['d']}  "
              f"{n_params(w['w'], w['d']):,}p  "
              f"cvar_ce {w['ce']:.4f} +- {w['se']:.4f}")
        ru, ratio = runner_up_ratio(rows, w)
        if ru is not None:
            print(f"    runner-up {ru['id']} at {ru['ce']:.4f}: "
                  f"{ratio:.2f}x the joint 2*SE "
                  f"({'SEPARATED' if ratio > 1 else 'TIED'})")
        archs = collections.Counter((t["w"], t["d"]) for t in tied)
        archs[(w["w"], w["d"])] += 1
        print(f"    tied with the winner: {len(tied)}   "
              f"architectures in the tied set (incl. winner): "
              + ", ".join(f"w{a}xd{b} x{n}" for (a, b), n in sorted(archs.items())))

    print("\n" + "=" * 72)
    print("18h  DEPTH COVERAGE, and were the depth-1 trials inside the sweep's")
    print("     own ELIGIBLE sampler range?")
    print("=" * 72)
    d1_pool = collections.Counter()
    for v in ORDER:
        rows, _ = truncate_at_stop(data[v]["rows"])
        hist = collections.Counter(r["d"] for r in rows)
        el = [r for r in rows if r["eligible"]]
        rng = {k: (min(r[k] for r in el), max(r[k] for r in el))
               for k in SAMPLER_KEYS} if el else {}
        d1 = [r for r in rows if r["d"] == 1]
        d1_el = sum(r["eligible"] for r in d1)
        d1_pool[v] = (d1_el, len(d1))
        print(f"\n  {v}: {len(rows)} trials, depth histogram "
              f"{dict(sorted(hist.items()))}, depth-1 eligible {d1_el}/{len(d1)}")
        if rng:
            print("    eligible sampler range: " + ", ".join(
                f"{k} {lo:.3g}..{hi:.3g}" for k, (lo, hi) in rng.items()))
        for r in d1:
            if r["eligible"]:
                continue
            outside = [k for k in SAMPLER_KEYS
                       if rng and not (rng[k][0] <= r[k] <= rng[k][1])]
            print(f"    failed d1 trial {r['id']}: "
                  + ", ".join(f"{k} {r[k]:.3g}" for k in SAMPLER_KEYS)
                  + f"  -> OUTSIDE eligible range on {len(outside)}/3"
                  + (f" ({', '.join(outside)})" if outside else ""))
    for v in ORDER:
        others = [x for x in ORDER if x != v]
        e = sum(d1_pool[x][0] for x in others)
        n = sum(d1_pool[x][1] for x in others)
        if n:
            print(f"  depth-1 eligibility in the three sweeps OTHER than {v}: "
                  f"{e}/{n} = {100 * e / n:.0f}%")

    print("\n" + "=" * 72)
    print(f"18j  TRIAL COUNTS vs the MR/PT baselines' "
          f"{BASELINE_TRIALS[0]}-{BASELINE_TRIALS[1]}")
    print("=" * 72)
    for v in ORDER:
        rows, trig = truncate_at_stop(data[v]["rows"])
        n = len(rows)
        flag = ("  !! EXCEEDS the baseline maximum -- DISCLOSE (4.3.127 s4)"
                if n > BASELINE_TRIALS[1] else "")
        print(f"  {v:15s} {n:3d} trials "
              f"({'stopped' if trig else 'live'}){flag}")
    return 0


def selftest():
    # Synthetic sweep: winner 0.400+-0.005; 0.405+-0.005 is tied (0.005 <= 0.0141);
    # 0.430+-0.005 is separated; an INELIGIBLE 0.390 must be ignored entirely.
    mk = lambda i, ce, se, el, w=6, d=1: {"id": f"r{i}", "ce": ce, "se": se,
                                         "eligible": el, "w": w, "d": d}
    rows = [mk(1, 0.430, 0.005, True, 5), mk(2, 0.400, 0.005, True),
            mk(3, 0.405, 0.005, True), mk(4, 0.390, 0.005, False)]
    w, tied = tied_set(rows)
    assert w["id"] == "r2", w
    assert [t["id"] for t in tied] == ["r3"], tied
    ru, ratio = runner_up_ratio(rows, w)
    assert ru["id"] == "r3" and ratio < 1, (ru, ratio)
    # separated case: widen the gap
    rows2 = [mk(2, 0.400, 0.002, True), mk(3, 0.410, 0.002, True)]
    ru, ratio = runner_up_ratio(rows2, tied_set(rows2)[0])
    assert ratio > 1, ratio
    # stop rule: best at trial 1, 15 non-improving -> trigger at 16
    seq = [{"t": f"{i:03d}", "J": 0.5 if i == 1 else 0.6, "id": f"s{i}",
            "eligible": True} for i in range(1, 20)]
    kept, trig = truncate_at_stop(seq)
    assert trig == 16 and len(kept) == 16, (trig, len(kept))
    # parameter formula matches the handoff's quoted values
    assert n_params(4, 1) == 625 and n_params(6, 1) == 2497
    assert n_params(5, 2) == 2305 and n_params(7, 4) == 54529
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return report(fetch())


if __name__ == "__main__":
    sys.exit(main())
