#!/usr/bin/env python
"""Verify the label-noise and data-reduction sets before any run reads them.

Handoff §11.6 asks for this check on the corrupted TEST files; it is applied here
to every set the two ablation experiments read, written independently of the
generators (it reads the files and re-derives what they should contain).

NOISE  (data/.../eval/seed_<s>/noise/<variant>_pref_<part>_<s>_<pp>.hdf5),
       parts train and test, levels 0.1 0.2 0.3 0.4 0.45:
  N1  same datasets, shapes and dtypes as the base split
  N2  everything except `labels` is identical to the base split
  N3  exactly round(p*N) label rows differ, and each is the one-hot swap
  N4  the levels NEST: every row flipped at p is flipped at every higher level
  N5  the flipped rows are the prefix of default_rng(seed).permutation(N)
  N6  base labels are one-hot (no ties), so "flip" is well defined

REDUCTION  (.../reduction/<variant>_pref_train_<s>_<n>.hdf5), sizes 16 32 64 128:
  R1  exactly n pairs, with the base split's datasets and dtypes
  R2  the pairs are rows of the base TRAINING split, in base order
  R3  they are the first n of default_rng(seed).permutation(N), so sets nest
  R4  validation and test are untouched by construction: no reduced val/test
      file exists

It also prints one DIGEST per experiment over every label array it read.  The
same command on the analysis machine and on the GPU box must print the same two
digests; that is how to tell the two copies of the data agree.

Run from the repo root (gp_reward-priors):
    python scripts_bnn/verify_experiment_sets.py
    python scripts_bnn/verify_experiment_sets.py --selftest
"""

import argparse
import glob
import hashlib
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCENTS = [0.1, 0.2, 0.3, 0.4, 0.45]
NOISE_PARTS = ("train", "test")
SIZES = [16, 32, 64, 128]


def pct_suffix(p):
    return str(p).replace(".", "")


def load(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][:] for k in f.keys()}


def flipped_rows(base_labels, labels):
    return np.flatnonzero((base_labels != labels).any(axis=1))


def check_noise_file(base, noisy, n, seed, p):
    """[problems] for one noise file against its base split."""
    probs = []
    if sorted(base) != sorted(noisy):
        return [f"datasets differ: {sorted(noisy)} vs base {sorted(base)}"]
    for k in base:
        if base[k].shape != noisy[k].shape or base[k].dtype != noisy[k].dtype:
            probs.append(f"N1 {k}: shape/dtype {noisy[k].shape}/{noisy[k].dtype} "
                         f"vs {base[k].shape}/{base[k].dtype}")
        elif k != "labels" and not np.array_equal(base[k], noisy[k]):
            probs.append(f"N2 {k} differs from the base split")
    if probs:
        return probs
    rows = flipped_rows(base["labels"], noisy["labels"])
    k_want = int(round(p * n))
    if rows.size != k_want:
        probs.append(f"N3 {rows.size} rows flipped, expected round({p}*{n}) = {k_want}")
    if not np.array_equal(noisy["labels"][rows], base["labels"][rows][:, ::-1]):
        probs.append("N3 a changed row is not the one-hot swap")
    want = np.sort(np.random.default_rng(seed).permutation(n)[:k_want])
    if not np.array_equal(rows, want):
        probs.append("N5 flipped rows are not the seeded permutation's prefix")
    return probs


def check_reduction_file(base, red, n_base, seed, size):
    probs = []
    if sorted(base) != sorted(red):
        return [f"datasets differ: {sorted(red)} vs base {sorted(base)}"]
    for k in base:
        if red[k].shape[0] != size:
            probs.append(f"R1 {k} has {red[k].shape[0]} rows, expected {size}")
        if red[k].dtype != base[k].dtype or red[k].shape[1:] != base[k].shape[1:]:
            probs.append(f"R1 {k} dtype/shape differs from the base split")
    if probs:
        return probs
    want = np.sort(np.random.default_rng(seed).permutation(n_base)[:size])
    for k in base:
        if not np.array_equal(red[k], base[k][want]):
            probs.append(f"R2/R3 {k}: not rows perm[:{size}] of the base split, in base order")
            break
    return probs


def one_hot(labels):
    return bool(labels.ndim == 2 and labels.shape[1] == 2
                and np.isin(labels, (0.0, 1.0)).all() and (labels.sum(axis=1) == 1).all())


def run(data_root, seeds):
    variants = sorted(os.path.basename(d) for d in glob.glob(os.path.join(data_root, "*"))
                      if os.path.isdir(d))
    bad = []
    dig = {"noise": hashlib.sha256(), "reduction": hashlib.sha256()}
    n_files = {"noise": 0, "reduction": 0}
    for v in variants:
        counts = {"train": set(), "test": set()}
        for s in seeds:
            sd = os.path.join(data_root, v, "eval", f"seed_{s}")
            for part in NOISE_PARTS:
                bp = os.path.join(sd, f"{v}_pref_{part}_{s}.hdf5")
                if not os.path.isfile(bp):
                    bad.append(f"missing base split {bp}")
                    continue
                base = load(bp)
                n = base["labels"].shape[0]
                counts[part].add(n)
                if not one_hot(base["labels"]):
                    bad.append(f"N6 {bp}: labels are not one-hot")
                prev = set()
                for p in PERCENTS:
                    fp = os.path.join(sd, "noise", f"{v}_pref_{part}_{s}_{pct_suffix(p)}.hdf5")
                    if not os.path.isfile(fp):
                        bad.append(f"missing {fp}")
                        continue
                    noisy = load(fp)
                    bad += [f"{fp}: {x}" for x in check_noise_file(base, noisy, n, s, p)]
                    rows = set(flipped_rows(base["labels"], noisy["labels"]).tolist()) \
                        if noisy["labels"].shape == base["labels"].shape else set()
                    if not prev <= rows:
                        bad.append(f"N4 {fp}: does not contain the level below")
                    prev = rows
                    dig["noise"].update(f"{v}|{s}|{part}|{p}|".encode())
                    dig["noise"].update(np.ascontiguousarray(noisy["labels"]).tobytes())
                    n_files["noise"] += 1
                if part == "train":
                    for size in SIZES:
                        fp = os.path.join(sd, "reduction", f"{v}_pref_train_{s}_{size}.hdf5")
                        if not os.path.isfile(fp):
                            bad.append(f"missing {fp}")
                            continue
                        red = load(fp)
                        bad += [f"{fp}: {x}" for x in check_reduction_file(base, red, n, s, size)]
                        dig["reduction"].update(f"{v}|{s}|{size}|".encode())
                        for k in sorted(red):
                            dig["reduction"].update(np.ascontiguousarray(red[k]).tobytes())
                        n_files["reduction"] += 1
            stray = [f for f in glob.glob(os.path.join(sd, "reduction", "*"))
                     if "_pref_train_" not in os.path.basename(f)]
            if stray:
                bad.append(f"R4 reduced non-training files exist: {stray[:2]}")
        flips = {part: sorted({int(round(p * n)) for n in counts[part]} for p in PERCENTS)
                 for part in NOISE_PARTS}
        print(f"  {v:28s} train pairs {sorted(counts['train'])}  test pairs {sorted(counts['test'])}")
        for part in NOISE_PARTS:
            ns = sorted(counts[part])
            if ns:
                print(f"      {part:5s} flips at p={PERCENTS}: "
                      f"{[int(round(p * ns[0])) for p in PERCENTS]}"
                      + (f" .. {[int(round(p * ns[-1])) for p in PERCENTS]}" if len(ns) > 1 else ""))
        full = sorted(counts["train"])
        if full:
            print(f"      reduction levels {SIZES} of a pool of {full[0]}"
                  + (f"-{full[-1]}" if len(full) > 1 else ""))
    print(f"\n  noise:     {n_files['noise']} files checked (train + test), "
          f"DIGEST {dig['noise'].hexdigest()[:16]}")
    print(f"  reduction: {n_files['reduction']} files checked, "
          f"DIGEST {dig['reduction'].hexdigest()[:16]}")
    for b in bad[:40]:
        print(f"  !! {b}")
    if len(bad) > 40:
        print(f"  !! ... and {len(bad) - 40} more")
    print("\nVERDICT: " + ("VERIFIED -- every set is what its generator specifies"
                           if not bad else f"NOT VERIFIED -- {len(bad)} problem(s)"))
    return 1 if bad else 0


def selftest():
    rng = np.random.default_rng(1)
    n, seed, p = 60, 4, 0.2
    lab = np.zeros((n, 2)); lab[np.arange(n), rng.integers(0, 2, n)] = 1.0
    base = {"labels": lab, "states": rng.normal(size=(n, 3))}
    assert one_hot(lab) and not one_hot(np.full((n, 2), 0.5))
    idx = np.sort(np.random.default_rng(seed).permutation(n)[: int(round(p * n))])
    good = {"labels": lab.copy(), "states": base["states"].copy()}
    good["labels"][idx] = good["labels"][idx][:, ::-1]
    assert check_noise_file(base, good, n, seed, p) == []
    few = {"labels": lab.copy(), "states": base["states"].copy()}       # one flip too few
    few["labels"][idx[1:]] = few["labels"][idx[1:]][:, ::-1]
    assert any(x.startswith("N3") for x in check_noise_file(base, few, n, seed, p))
    other = {"labels": lab.copy(), "states": base["states"].copy()}     # right count, wrong rows
    j = np.sort(np.random.default_rng(seed + 1).permutation(n)[: idx.size])
    other["labels"][j] = other["labels"][j][:, ::-1]
    assert any(x.startswith("N5") for x in check_noise_file(base, other, n, seed, p))
    moved = {"labels": good["labels"], "states": base["states"] + 1e-9}  # a state changed
    assert any(x.startswith("N2") for x in check_noise_file(base, moved, n, seed, p))
    want = np.sort(np.random.default_rng(seed).permutation(n)[:16])
    red = {k: v[want] for k, v in base.items()}
    assert check_reduction_file(base, red, n, seed, 16) == []
    assert check_reduction_file(base, {k: v[:16] for k, v in base.items()}, n, seed, 16)
    assert check_reduction_file(base, {k: v[want[:15]] for k, v in base.items()}, n, seed, 16)
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "antmaze"))
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(1, 11)))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run(a.data_root, [int(s) for s in a.seeds.split(",") if s.strip()])


if __name__ == "__main__":
    sys.exit(main())
