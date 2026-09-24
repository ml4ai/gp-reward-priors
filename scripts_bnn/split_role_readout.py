#!/usr/bin/env python
"""What the MR/PT split-role change did, on the COMPLETED baseline sweeps (§7.4 D).

§4.3.107 (2026-09-15) changed MR/PT checkpoint selection: the TEST split now
picks the checkpoint and VAL scores it (`eval_loss_at_selected`).  Before, val
did both — it picked the checkpoint AND ranked the hyperparameters, so the sweep
objective was the minimum of val over epochs, an optimistically biased number.

§4.3.108 measured the effect mid-sweep, on a 10-trial subsample of 6 of the 8
sweeps.  This recomputes it on every finished trial of all 8 completed sweeps,
and answers the decision-relevant question directly: did the change alter
WHICH configuration won?

Per sweep:
  old objective   min over eval epochs of `eval_loss` (from run history)
  new objective   `eval_loss_at_selected` (what the sweep actually ranked on)
  optimism        new - old per run, >= 0 by construction; median reported
  pair flips      fraction of all trial pairs ordered differently by the two
  winner          argmin under each; if they differ, each one's rank under the
                  other objective

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/split_role_readout.py
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/split_role_readout.py --selftest
"""

import argparse
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ENTITY = "champlin-university-of-arizona"
SWEEPS = {
    "MR": {"ymckz130": "medium_play", "oxac6roc": "medium_diverse",
           "p8yawcs2": "large_play", "n1d9qry8": "large_diverse"},
    "PT": {"xg6zk118": "medium_play", "beyi619f": "medium_diverse",
           "jj1or8i4": "large_play", "mtyctxxe": "large_diverse"},
}
ORDER = ("medium_play", "medium_diverse", "large_play", "large_diverse")


def pair_flip_fraction(old, new):
    """Fraction of pairs (i<j) that the two objectives order differently.

    Exact ties under either objective are not counted as flips.
    """
    n = len(old)
    if n < 2:
        return float("nan"), 0
    flips = total = 0
    for i, j in itertools.combinations(range(n), 2):
        a, b = old[i] - old[j], new[i] - new[j]
        if a == 0 or b == 0:
            continue
        total += 1
        flips += (a > 0) != (b > 0)
    return (flips / total if total else float("nan")), total


def rank_of(values, idx):
    """1-based rank of values[idx] (1 = lowest)."""
    return int(np.sum(np.asarray(values) < values[idx])) + 1


def summarise(old, new, ids):
    old, new = np.asarray(old, float), np.asarray(new, float)
    w_old, w_new = int(np.argmin(old)), int(np.argmin(new))
    frac, npairs = pair_flip_fraction(old, new)
    return {
        "n": len(old), "old_best": float(old.min()), "new_best": float(new.min()),
        "optimism_median": float(np.median(new - old)),
        "optimism_max": float(np.max(new - old)),
        "flip_frac": frac, "n_pairs": npairs,
        "winner_old": ids[w_old], "winner_new": ids[w_new],
        "same_winner": w_old == w_new,
        "new_winner_rank_under_old": rank_of(old, w_new),
        "old_winner_rank_under_new": rank_of(new, w_old),
    }


def _one(run):
    new = run.summary.get("eval_loss_at_selected")
    if new is None:
        return None
    hist = [r["eval_loss"] for r in run.scan_history(keys=["eval_loss"],
                                                     page_size=10000)
            if r.get("eval_loss") is not None]
    if not hist:
        return None
    return run.id, float(min(hist)), float(new)


def fetch():
    import wandb
    api = wandb.Api(timeout=120)
    out = {}
    for fam, sweeps in SWEEPS.items():
        for sid, v in sweeps.items():
            runs = [r for r in api.sweep(f"{ENTITY}/{fam}-training/{sid}").runs
                    if r.state == "finished"]
            with ThreadPoolExecutor(max_workers=8) as ex:
                rows = [x for x in ex.map(_one, runs) if x is not None]
            out[(fam, v)] = rows
            print(f"[fetch] {fam} {v}: {len(rows)}/{len(runs)} runs with history",
                  file=sys.stderr)
    return out


def report(data):
    print("SPLIT-ROLE CHANGE on the completed MR/PT sweeps (4.3.107)\n")
    print("  old = min val over epochs (val picked the checkpoint AND ranked)")
    print("  new = val at the test-selected checkpoint (eval_loss_at_selected)\n")
    hdr = (f"  {'fam':3s} {'variant':15s} {'n':>3s} {'old best':>9s} "
           f"{'new best':>9s} {'opt med':>8s} {'opt max':>8s} "
           f"{'pairs flip':>11s}  winner")
    print(hdr)
    changed = 0
    for fam in ("MR", "PT"):
        for v in ORDER:
            rows = data[(fam, v)]
            ids = [r[0] for r in rows]
            s = summarise([r[1] for r in rows], [r[2] for r in rows], ids)
            changed += not s["same_winner"]
            win = ("SAME" if s["same_winner"] else
                   f"CHANGED (new winner was #{s['new_winner_rank_under_old']} "
                   f"under old; old winner is #{s['old_winner_rank_under_new']} "
                   f"under new)")
            print(f"  {fam:3s} {v:15s} {s['n']:3d} {s['old_best']:9.4f} "
                  f"{s['new_best']:9.4f} {s['optimism_median']:+8.4f} "
                  f"{s['optimism_max']:+8.4f} {100 * s['flip_frac']:9.1f}%  {win}")
    print(f"\n  winner changed in {changed} of 8 sweeps")
    return 0


def selftest():
    # identical objectives -> no flips, same winner
    s = summarise([0.3, 0.1, 0.2], [0.3, 0.1, 0.2], ["a", "b", "c"])
    assert s["flip_frac"] == 0 and s["same_winner"], s
    # full reversal -> every pair flips, winner changes
    s = summarise([0.1, 0.2, 0.3], [0.9, 0.8, 0.7], ["a", "b", "c"])
    assert s["flip_frac"] == 1.0 and not s["same_winner"], s
    assert s["winner_old"] == "a" and s["winner_new"] == "c", s
    assert s["new_winner_rank_under_old"] == 3, s
    assert s["old_winner_rank_under_new"] == 3, s
    # one swap among four -> 1 of 6 pairs
    f, n = pair_flip_fraction([1, 2, 3, 4], [2, 1, 3, 4])
    assert n == 6 and abs(f - 1 / 6) < 1e-12, (f, n)
    # ties are not flips and are excluded from the denominator
    f, n = pair_flip_fraction([1, 1, 2], [1, 2, 3])
    assert n == 2 and f == 0, (f, n)
    # optimism is new - old
    s = summarise([0.10, 0.20], [0.15, 0.20], ["a", "b"])
    assert abs(s["optimism_median"] - 0.025) < 1e-12, s
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
