"""make_noise_sets.py — build nested label-noise copies of the preference splits.

For every antmaze variant and seed, take a split (<variant>_pref_<part>_<s>.hdf5)
and write copies in which a fraction of the preference labels is flipped (the
one-hot columns swapped).  Fractions are 0.1, 0.2, 0.3, 0.4, 0.45 and are NESTED:
one seed-matched permutation of the pairs is drawn per partition, and the first
round(p*N) indices are flipped relative to the ORIGINAL labels.  A higher level
flips a superset of a lower one, so no label is ever un-flipped.

The level goes after the seed with the decimal point removed (0.1 -> 01,
0.45 -> 045):
  data/antmaze/<variant>/eval/seed_<s>/noise/<variant>_pref_<part>_<s>_<pp>.hdf5

WHICH PARTITIONS (handoff §11, the label-noise experiment):
  train   every model in the experiment trains on these.
  test    the MR/PT checkpoint-selection split in the PRIMARY arm.
  val     NOT used.  The corrupted validation files on disk belong to an earlier
          design that §11 supersedes; validation is a measuring instrument and
          stays uncorrupted.  Never pass a noise/..._pref_val_... file to a run.

So the default here is `--parts test`: the one partition §11.5 says is missing.

THIS TOOL NEVER REWRITES A FILE.  The training files already on disk are the
ones every noise run reads, so each output is built in memory and then:
  absent                  -> written
  present and identical   -> left alone, reported "unchanged"
  present and DIFFERENT   -> left alone, reported, and the exit status is 1
The generator is deterministic in (seed, partition size), so re-running any
partition is a check rather than a regeneration.

Usage (seeds default to 1..10, the evaluation seeds; §11.1 has no seed-0 lineage):
    python scripts_bnn/make_noise_sets.py                    # test, seeds 1-10
    python scripts_bnn/make_noise_sets.py --parts train,test # also re-check train
    python scripts_bnn/make_noise_sets.py --dry-run
    python scripts_bnn/make_noise_sets.py 3 4 --parts test   # chosen seeds
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "antmaze")
PERCENTS = [0.1, 0.2, 0.3, 0.4, 0.45]  # ascending; nested flip prefixes
PARTS = ("train", "val", "test")


def pct_suffix(p):
    # 0.1 -> "01", 0.2 -> "02", 0.45 -> "045"
    return str(p).replace(".", "")


def flip_indices(n, seed, p):
    """The sorted indices flipped at level p: a prefix of ONE seeded permutation.

    Kept exactly as the original generator had it, because the training files
    on disk were made by it: default_rng(seed).permutation(n), first round(p*n).
    """
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[: int(round(p * n))])


def flipped_copy(src, flip_idx):
    """{key: array} for a noise file: the source with `labels` rows swapped."""
    out = {}
    for key in src.keys():
        data = src[key][:]
        if key == "labels":
            data = data.copy()
            data[flip_idx] = data[flip_idx][:, ::-1]  # swap one-hot columns
        out[key] = data
    return out


def same_as_file(arrays, path):
    """True if the HDF5 file at `path` holds exactly these datasets."""
    with h5py.File(path, "r") as f:
        if sorted(f.keys()) != sorted(arrays):
            return False
        return all(f[k].shape == a.shape and f[k].dtype == a.dtype
                   and np.array_equal(f[k][:], a) for k, a in arrays.items())


def write_file(arrays, path):
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as out:
        for key, data in arrays.items():
            out.create_dataset(key, data=data)
    os.replace(tmp, path)


def process_partition(src_path, out_dir, part, variant, seed, dry_run=False):
    """-> {"written": n, "unchanged": n, "DIFFERENT": [paths]}"""
    res = {"written": 0, "unchanged": 0, "DIFFERENT": []}
    with h5py.File(src_path, "r") as src:
        n = src["labels"].shape[0]
        for p in PERCENTS:
            dst = os.path.join(out_dir, f"{variant}_pref_{part}_{seed}_{pct_suffix(p)}.hdf5")
            arrays = flipped_copy(src, flip_indices(n, seed, p))
            if os.path.exists(dst):
                if same_as_file(arrays, dst):
                    res["unchanged"] += 1
                else:
                    res["DIFFERENT"].append(dst)
            else:
                if not dry_run:
                    write_file(arrays, dst)
                res["written"] += 1
    return res


def selftest():
    import tempfile
    assert pct_suffix(0.1) == "01" and pct_suffix(0.45) == "045"
    # nesting, sizes and determinism of the flip sets
    n, seed = 77, 3
    sets = [set(flip_indices(n, seed, p).tolist()) for p in PERCENTS]
    assert [len(s) for s in sets] == [int(round(p * n)) for p in PERCENTS] == [8, 15, 23, 31, 35]
    assert all(a < b for a, b in zip(sets, sets[1:])), "levels must nest strictly"
    assert set(flip_indices(n, seed, 0.2).tolist()) == sets[1]
    assert set(flip_indices(n, seed + 1, 0.2).tolist()) != sets[1]
    with tempfile.TemporaryDirectory() as d:
        src_path = os.path.join(d, "v_pref_test_3.hdf5")
        rng = np.random.default_rng(0)
        lab = np.zeros((n, 2)); lab[np.arange(n), rng.integers(0, 2, n)] = 1.0
        with h5py.File(src_path, "w") as f:
            f.create_dataset("labels", data=lab)
            f.create_dataset("states", data=rng.normal(size=(n, 4, 3)))
        out = os.path.join(d, "noise"); os.makedirs(out)
        r = process_partition(src_path, out, "test", "v", seed)
        assert r == {"written": 5, "unchanged": 0, "DIFFERENT": []}, r
        with h5py.File(os.path.join(out, "v_pref_test_3_02.hdf5"), "r") as f, \
                h5py.File(src_path, "r") as s:
            changed = np.flatnonzero((f["labels"][:] != s["labels"][:]).any(axis=1))
            assert set(changed.tolist()) == sets[1]
            assert np.array_equal(f["labels"][:][changed], s["labels"][:][changed][:, ::-1])
            assert np.array_equal(f["states"][:], s["states"][:])
        before = {p: os.path.getmtime(os.path.join(out, p)) for p in os.listdir(out)}
        r = process_partition(src_path, out, "test", "v", seed)        # a re-run is a check
        assert r == {"written": 0, "unchanged": 5, "DIFFERENT": []}, r
        assert before == {p: os.path.getmtime(os.path.join(out, p)) for p in os.listdir(out)}
        bad = os.path.join(out, "v_pref_test_3_03.hdf5")               # tamper with one
        with h5py.File(bad, "r+") as f:
            L = f["labels"][:]; L[0] = L[0][::-1]; f["labels"][...] = L
        r = process_partition(src_path, out, "test", "v", seed)
        assert r["DIFFERENT"] == [bad] and r["unchanged"] == 4 and r["written"] == 0, r
        with h5py.File(bad, "r") as f:                                  # and it was NOT rewritten
            assert np.array_equal(f["labels"][:], L)
        r = process_partition(src_path, os.path.join(d, "nowhere"), "test", "v", seed, dry_run=True)
        assert r["written"] == 5 and not os.path.exists(os.path.join(d, "nowhere"))
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seeds", nargs="*", type=int, help="default 1..10")
    ap.add_argument("--parts", default="test",
                    help="comma-separated subset of train,val,test (default: test)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    seeds = a.seeds or list(range(1, 11))
    parts = [p for p in a.parts.split(",") if p]
    for p in parts:
        if p not in PARTS:
            ap.error(f"unknown partition {p!r}; choose from {PARTS}")
    if "val" in parts:
        print("NOTE: corrupted VALIDATION files are not used by the experiment "
              "(handoff §11.4).  Checking them only.")
    bad = []
    variant_dirs = sorted(d for d in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(d))
    for vdir in variant_dirs:
        variant = os.path.basename(vdir)
        tot = {"written": 0, "unchanged": 0}
        for seed in seeds:
            seed_dir = os.path.join(vdir, "eval", f"seed_{seed}")
            out_dir = os.path.join(seed_dir, "noise")
            for part in parts:
                src_path = os.path.join(seed_dir, f"{variant}_pref_{part}_{seed}.hdf5")
                if not os.path.isfile(src_path):
                    print(f"[skip] missing {src_path}")
                    bad.append(src_path)
                    continue
                if not a.dry_run:
                    os.makedirs(out_dir, exist_ok=True)
                r = process_partition(src_path, out_dir, part, variant, seed, a.dry_run)
                tot["written"] += r["written"]
                tot["unchanged"] += r["unchanged"]
                bad += r["DIFFERENT"]
        print(f"[ok] {variant}: {','.join(parts)} x {len(seeds)} seeds x {len(PERCENTS)} levels -> "
              f"{'would write' if a.dry_run else 'written'} {tot['written']}, "
              f"unchanged {tot['unchanged']}")
    for b in bad:
        print(f"!! NOT as generated, left untouched: {b}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
