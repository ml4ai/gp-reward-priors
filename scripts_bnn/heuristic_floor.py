#!/usr/bin/env python
"""heuristic_floor.py -- Chapter 3's trivial-heuristic floor for the reward models.

Chapter 3 (sec:c1-eval-baselines): "how well does a parameter-free, goal-agnostic
statistic of a segment predict the held-out human preference?  ... A learned
reward model that does not clear this line by a visible margin has not been
shown to have learned anything beyond gross motion.  Run it against every model
family on the identical split ... and report the margin."

THE STATISTIC: distance travelled = the PATH LENGTH of the ant's torso (x, y) over
the segment's valid steps, sum_t ||xy_{t+1} - xy_t||.  The segment that travels
further is predicted preferred.  No parameters.  This is the definition that
reproduces Chapter 4's quick pass on the pooled tie-free files (0.789, 0.882,
0.793, 0.880; `--pooled` re-prints it); net displacement does not.

THE COMPARISON, per (variant, seed): on that seed's BASE validation split, which
is identical at every sparsity level (validation never shrinks, §13.6),
    margin = model accuracy - floor accuracy,   both PER PAIR on the same pairs.
Reported per family, variant and level as the mean margin over seeds 1-10 with
the exact 10-seed bootstrap interval (eval_readout.exact_bootstrap).
DESCRIPTIVE: Chapter 3 registers no margin, so there is no verdict here.

Model accuracies come from what each training run logged:
  MR, PT  `eval_acc_reloaded`, the selected checkpoint on the validation split.
          It is an UNWEIGHTED mean over the unshuffled batches of 64 (e.g. 64 +
          13 pairs), not a per-pair accuracy.  The per-pair count is recovered
          exactly as the unique integers (a, b) with a/64 + b/r = 2 x logged;
          a run where that is not unique is reported, not guessed.
  BNN     `val_cvar_acc_c0` (posterior-mean reward) and `val_cvar_acc_c0p95`
          (the deployed CVaR at 0.95), already per pair.

    python scripts_bnn/heuristic_floor.py                # anchor + all sparsity levels
    python scripts_bnn/heuristic_floor.py --pooled       # reproduce the quick pass
    python scripts_bnn/heuristic_floor.py --selftest
"""

import argparse
import importlib.util
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import experiment_conditions as E                          # noqa: E402

SEEDS = list(E.EVAL_SEEDS)
BATCH = 64                          # MR/PT validation loader: batch_size 64, shuffle=False
LEVELS = ("anchor",) + E.SPARSITY
DATA = os.path.join(ROOT, "data", "antmaze")


def _readout():
    spec = importlib.util.spec_from_file_location(
        "eval_readout", os.path.join(os.path.dirname(ROOT), "results", "eval_readout.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------
def path_length(xy, mask):
    """xy [B, T, 2], mask [B, T] -> [B] torso path length over the valid steps."""
    step = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
    valid = (mask[:, 1:] > 0.5) & (mask[:, :-1] > 0.5)
    return (step * valid).sum(axis=1)


def floor_correct(path):
    """-> [B] per-pair correctness (1, 0, or 0.5 on an exact tie of the statistic)."""
    import h5py
    with h5py.File(path) as f:
        p1 = path_length(f["states"][:, :, :2], f["attn_mask"][:])
        p2 = path_length(f["states_2"][:, :, :2], f["attn_mask_2"][:])
        y = f["labels"][:]
    first = y[:, 0] > 0.5
    return np.where(p1 == p2, 0.5, (p1 > p2) == first).astype(float)


def val_file(variant, seed):
    env = E.env_of(variant)
    return os.path.join(DATA, env, "eval", f"seed_{seed}", f"{env}_pref_val_{seed}.hdf5")


# --------------------------------------------------------------------------
# Recovering a per-pair accuracy from a batch-averaged one
# --------------------------------------------------------------------------
def per_pair_from_batch_mean(logged, n, batch=BATCH, tol=1e-6):
    """Unique number of correct pairs consistent with an unweighted mean over the
    batches [0:64], [64:128], ... of an n-pair split.  -> (count, None) or
    (None, reason)."""
    sizes = [min(batch, n - i) for i in range(0, n, batch)]
    sols = []
    for counts in _grid(sizes):
        if abs(np.mean([c / s for c, s in zip(counts, sizes)]) - logged) < tol:
            sols.append(sum(counts))
    totals = sorted(set(sols))
    if len(totals) == 1:
        return totals[0], None
    return None, (f"{len(totals)} totals fit the logged {logged!r} over batches {sizes}"
                  if totals else f"no integer fit for {logged!r} over batches {sizes}")


def _grid(sizes):
    if not sizes:
        yield ()
        return
    for c in range(sizes[0] + 1):
        for rest in _grid(sizes[1:]):
            yield (c,) + rest


# --------------------------------------------------------------------------
# The models' accuracies
# --------------------------------------------------------------------------
_NAME = re.compile(r"antmaze_([a-z_]+?)_(bnn|mr|pt)(?:_(r\d+))?_eval_(\d+)$")


def model_accuracies(family, n_pairs, since):
    """{(level, variant, seed): {series: (correct, n) or reason}} from wandb."""
    import production_readout as PR
    prod, _ = PR.fetch(family, since)
    key = E.OUT_KEY[family]
    out, notes = {}, []
    for r in prod:
        if r["state"] != "finished":
            continue
        base = os.path.basename(os.path.normpath(str(r["config"].get(key) or "")))
        m = _NAME.fullmatch(base)
        if not m or m.group(2) != family:
            continue
        v, level, s = m.group(1), m.group(3) or "anchor", int(m.group(4))
        if s not in SEEDS or level not in LEVELS:
            continue
        k = (level, v, s)
        if k in out:
            notes.append(f"{family} {k}: two finished runs -- cell not read")
            out[k] = None
            continue
        n = n_pairs[(v, s)]
        sm = r["summary"]
        if family in ("mr", "pt"):
            c, why = per_pair_from_batch_mean(sm.get("eval_acc_reloaded"), n)
            out[k] = {family.upper(): (c, n) if c is not None else why}
        else:
            row = {}
            for series, kk in (("BNN mean", "val_cvar_acc_c0"), ("BNN CVaR", "val_cvar_acc_c0p95")):
                a = sm.get(kk)
                c = None if a is None else round(a * n)
                ok = a is not None and abs(c / n - a) < 1e-9
                row[series] = (c, n) if ok else f"{kk} = {a!r} is not a whole number of {n} pairs"
            out[k] = row
    return out, notes


# --------------------------------------------------------------------------
def cmd_pooled():
    for v in E.VARIANTS:
        env = E.env_of(v)
        c = floor_correct(os.path.join(DATA, env, f"{env}_pref_nt.hdf5"))
        print(f"  {v:15s} {c.size:4d} pairs  path-length accuracy {c.mean():.3f}")
    return 0


def cmd_floor(a, out=print):
    R = _readout()
    floor, n_pairs = {}, {}
    for v in E.VARIANTS:
        for s in SEEDS:
            c = floor_correct(val_file(v, s))
            floor[(v, s)], n_pairs[(v, s)] = c, c.size
    accs, notes = {}, []
    for fam in ("mr", "pt", "bnn"):
        acc, nt = model_accuracies(fam, n_pairs, a.since)
        notes += nt
        for k, row in acc.items():
            if row:
                accs.setdefault(k, {}).update(row)
    out("=" * 100)
    out("TRIVIAL-HEURISTIC FLOOR (Chapter 3): torso path length vs the human label, per pair,")
    out("on each seed's base validation split.  margin = model accuracy - floor accuracy.")
    out("Mean over seeds 1-10, exact 10-seed bootstrap 95% interval.  DESCRIPTIVE: no verdict.")
    out("=" * 100)
    for n in notes:
        out(f"  note: {n}")
    out("\nTHE FLOOR (identical at every level: validation never shrinks)")
    for v in E.VARIANTS:
        f = [floor[(v, s)].mean() for s in SEEDS]
        out(f"  {v:15s} {np.mean(f):.3f}  (seeds {min(f):.3f}-{max(f):.3f}; "
            f"{n_pairs[(v, 1)]} pairs per seed)")
    series_all = ("MR", "PT", "BNN mean", "BNN CVaR")
    for level in LEVELS:
        out(f"\n[{level}]")
        for v in E.VARIANTS:
            for ser in series_all:
                vals, why = [], None
                for s in SEEDS:
                    x = accs.get((level, v, s), {}).get(ser)
                    if x is None:
                        why = f"seed {s} missing"
                        break
                    if isinstance(x, str):
                        why = f"seed {s}: {x}"
                        break
                    vals.append(x[0] / x[1] - floor[(v, s)].mean())
                if why:
                    if not (level != "anchor" and ser.startswith("BNN") and "missing" in why):
                        out(f"  {v:15s} {ser:9s} not read -- {why}")
                    continue
                b = R.exact_bootstrap(vals)
                macc = np.mean([accs[(level, v, s)][ser][0] / accs[(level, v, s)][ser][1]
                                for s in SEEDS])
                out(f"  {v:15s} {ser:9s} accuracy {macc:.3f}   margin {b['mean']:+.3f}  "
                    f"[{b['lower']:+.3f}, {b['upper']:+.3f}]   above the floor on "
                    f"{sum(x > 0 for x in vals)}/10 seeds")
    out("\nBNN rows appear at a sparsity level once its ten models are finished.")
    return 0


# --------------------------------------------------------------------------
def selftest():
    # path length: a straight run, a stationary segment, masking
    xy = np.zeros((2, 4, 2))
    xy[0, :, 0] = [0, 1, 2, 3]
    m = np.ones((2, 4))
    assert np.allclose(path_length(xy, m), [3.0, 0.0])
    m2 = m.copy()
    m2[0, 3] = 0
    assert np.allclose(path_length(xy, m2), [2.0, 0.0])
    # back and forth counts as distance (the "runs in circles" property)
    xy2 = np.zeros((1, 3, 2))
    xy2[0, :, 0] = [0, 1, 0]
    assert np.allclose(path_length(xy2, np.ones((1, 3))), [2.0])
    # batch-mean recovery: the worked case, and against brute force
    assert per_pair_from_batch_mean((60 / 64 + 10 / 13) / 2, 77) == (70, None)
    rng = np.random.default_rng(0)
    for n in (54, 77, 107, 110, 55):
        sizes = [min(BATCH, n - i) for i in range(0, n, BATCH)]
        for _ in range(20):
            counts = [int(rng.integers(0, s + 1)) for s in sizes]
            logged = float(np.mean([c / s for c, s in zip(counts, sizes)]))
            got, why = per_pair_from_batch_mean(logged, n)
            assert got == sum(counts) or (got is None and "totals fit" in why), (n, counts, why)
    assert per_pair_from_batch_mean(0.123456, 77)[0] is None
    # names
    assert _NAME.fullmatch("antmaze_large_play_mr_r16_eval_3").groups() == ("large_play", "mr", "r16", "3")
    assert _NAME.fullmatch("antmaze_medium_diverse_bnn_eval_10").groups() == ("medium_diverse", "bnn", None, "10")
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--since", default="2026-09-24T19:00:00")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.pooled:
        return cmd_pooled()
    return cmd_floor(a)


if __name__ == "__main__":
    sys.exit(main())
