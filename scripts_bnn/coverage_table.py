#!/usr/bin/env python
"""Maze-cell coverage of the preference splits, per variant, seed and split.

Handoff to-do 15 (§4.3.110, §4.3.169).  §7.3 once disclosed that the evaluation
set "occupies 13 of 26 free cells on medium (50%) and 18 of 33 on large (55%)".
Both figures were wrong: the diagnostic that produced them shifted every point by
a whole grid cell, and read large against a hardcoded maze that is not the one
the runs use ((9,9)/33 free cells against the live (9,12)/46).  This recomputes
the coverage on the prior the runs actually build.

For each (variant, seed, split) it assigns every valid timestep of BOTH segments
of every preference pair to a free maze cell and counts the distinct cells.  No
chains and no GPU: it reads the split files and the maze layout only.

The layout comes from the live D4RL env named in each variant's production config
(`map_env_name`), exactly as training does.  Where D4RL is not installed the tool
falls back to the hardcoded map, which is identical to the live one for MEDIUM
and a DIFFERENT MAZE for LARGE.  Large results on the fallback are refused unless
--allow-fallback is given, and are then labelled.

Run from the repo root (gp_reward-priors):
    python scripts_bnn/coverage_table.py                       # all four, seeds 0-10
    python scripts_bnn/coverage_table.py --variants large_play,large_diverse
    python scripts_bnn/coverage_table.py --selftest
"""

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

VARIANTS = ("medium_play", "medium_diverse", "large_play", "large_diverse")
SPLITS = ("train", "val", "test")


def split_path(env, split, seed, data_root):
    return os.path.join(data_root, env, "eval", f"seed_{seed}",
                        f"{env}_pref_{split}_{seed}.hdf5")


def summarise(occ):
    """(min, median, max) of a list of occupied-cell counts."""
    a = np.asarray(occ)
    return int(a.min()), int(np.median(a)), int(a.max())


def run(variants, seeds, data_root, allow_fallback):
    _argv, sys.argv = sys.argv, [sys.argv[0]]       # the diagnostic parses nothing here
    import diagnose_sampling_tail as D
    sys.argv = _argv
    bad = 0
    for v in variants:
        cfg = yaml.safe_load(open(os.path.join(HERE, f"antmaze_{v}_bnn_antmaze_eval.yaml")))
        env = cfg["map_env_name"]
        with contextlib.redirect_stdout(io.StringIO()):
            _, fm, is_live = D._live_prior(cfg, quiet=True)
        n_cells = int(np.asarray(fm).sum())
        print(f"\n{v}  ({env}; layout {'LIVE' if is_live else 'HARDCODED FALLBACK'}, "
              f"{n_cells} free cells)")
        if not is_live and cfg.get("map_size") != "medium":
            msg = ("the hardcoded LARGE map is a different maze from the live one; "
                   "these numbers do not describe the runs")
            if not allow_fallback:
                print(f"  REFUSED: {msg}.  Run where D4RL is installed, "
                      f"or pass --allow-fallback to see them anyway.")
                bad += 1
                continue
            print(f"  !! {msg}")
        for split in SPLITS:
            occ, pairs, thin, outside = [], [], [], 0
            for s in seeds:
                ds = split_path(env, split, s, data_root)
                if not os.path.exists(ds):
                    print(f"  {split:5s} seed {s}: MISSING {ds}")
                    bad += 1
                    continue
                with contextlib.redirect_stdout(io.StringIO()):
                    r = D.split_cell_coverage(ds, cfg, label=split)
                occ.append(r["n_occupied"])
                sh = r["share"]
                thin.append(int(((sh > 0) & (sh < 0.25 / r["n_cells"])).sum()))
                outside += r["n_outside"]
                pairs.append(r["n_points"])
            if not occ:
                continue
            lo, med, hi = summarise(occ)
            print(f"  {split:5s} occupied of {n_cells}: min {lo} ({100 * lo / n_cells:.0f}%)  "
                  f"median {med} ({100 * med / n_cells:.0f}%)  max {hi} "
                  f"({100 * hi / n_cells:.0f}%)   thin cells, median {int(np.median(thin))}"
                  f"   points/seed {min(pairs)}-{max(pairs)}"
                  + (f"   !! {outside} point(s) outside any free cell" if outside else ""))
            print(f"        per seed {dict(zip(seeds, occ))}")
    print("\n'thin' = occupied cells holding under a quarter of the uniform share of points.")
    return 1 if bad else 0


def selftest():
    assert summarise([21, 22, 24, 22]) == (21, 22, 24)
    assert split_path("antmaze-large-play-v2", "val", 3, "data/antmaze").endswith(
        "antmaze-large-play-v2/eval/seed_3/antmaze-large-play-v2_pref_val_3.hdf5")
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import diagnose_sampling_tail as D
    sys.argv = _argv
    n_occ, share, n_out = D.cell_occupancy([0, 0, 2, 2, 2, 5, -1, 9], 6)
    assert n_occ == 3 and n_out == 2 and abs(share.sum() - 1) < 1e-12
    assert abs(share[2] - 0.5) < 1e-12 and share[1] == 0
    n_occ, share, n_out = D.cell_occupancy([], 4)
    assert n_occ == 0 and n_out == 0 and share.sum() == 0
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(11)))
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "antmaze"))
    ap.add_argument("--allow-fallback", action="store_true",
                    help="report LARGE on the hardcoded map (a different maze), labelled")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run([v for v in a.variants.split(",") if v],
               [int(s) for s in a.seeds.split(",") if s.strip()],
               a.data_root, a.allow_fallback)


if __name__ == "__main__":
    sys.exit(main())
