"""make_ties_sets.py -- the ties-kept arm's data (handoff §14.2 item 3, §14.20).

For each variant in TIES_VARIANTS and each seed 1-10, the dataset's TIED pairs (label
[0.5, 0.5]) are allocated to train / validation / test at 70 / 15 / 15 with a seeded
draw, and added to that seed's existing tie-free files.  The tie-free arm is the
anchor, unchanged, and the two arms share every non-tie pair.

    data/antmaze/<env>/eval/seed_<s>/ties/<env>_pref_train_<s>_ties.hdf5    base train + train ties
    data/antmaze/<env>/eval/seed_<s>/ties/<env>_pref_test_<s>_ties.hdf5     base test  + test ties
    data/antmaze/<env>/eval/seed_<s>/ties/<env>_pref_valties_<s>.hdf5       the validation ties ONLY

Validation is NOT altered for training: a ties-kept model is scored, like every
model here, on the base (tie-free) validation split, which is exactly the "accuracy
on the tie-free validation pairs" of §14.2.  The validation ties are kept apart for
the readout's "cross-entropy on the tied validation pairs".  The test split, which
picks the checkpoint, does carry its ties: only the data differs between the arms.

The draw: numpy default_rng(RNG_OFFSET + seed) permutes the tied rows (in file
order), then the same rounding as split_pref_nt_seeds.py.  Rows are appended after
the tie-free rows; training shuffles.  Deterministic: re-running writes identical
files, and --check verifies every file against its sources row by row.

    python scripts_bnn/make_ties_sets.py [--write] [--check]
"""
import argparse
import hashlib
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(HERE), "data", "antmaze")
TIES_VARIANTS = ("large_play", "medium_diverse")     # §14.2 item 3: 64% and 29% ties
SEEDS = tuple(range(1, 11))
RNG_OFFSET = 100_000                                 # keeps this draw apart from the split seeds
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15                    # as split_pref_nt_seeds.py


def env_of(v):
    return f"antmaze-{v.replace('_', '-')}-v2"


def is_tie(labels):
    return np.isclose(labels[:, 0], 0.5) & np.isclose(labels[:, 1], 0.5)


def allocate(n, seed):
    idx = np.random.default_rng(RNG_OFFSET + seed).permutation(n)
    n_train, n_val = int(round(TRAIN_FRAC * n)), int(round(VAL_FRAC * n))
    return {"train": idx[:n_train], "val": idx[n_train:n_train + n_val], "test": idx[n_train + n_val:]}


def paths(v, s, root=DATA_ROOT):
    env = env_of(v)
    sd = os.path.join(root, env, "eval", f"seed_{s}")
    td = os.path.join(sd, "ties")
    return {"base": {p: os.path.join(sd, f"{env}_pref_{p}_{s}.hdf5") for p in ("train", "val", "test")},
            "out": {"train": os.path.join(td, f"{env}_pref_train_{s}_ties.hdf5"),
                    "test": os.path.join(td, f"{env}_pref_test_{s}_ties.hdf5"),
                    "valties": os.path.join(td, f"{env}_pref_valties_{s}.hdf5")},
            "full": os.path.join(root, env, f"{env}_pref.hdf5")}


def build(v, s, root=DATA_ROOT):
    """{out name: {key: array}} for one (variant, seed)."""
    P = paths(v, s, root)
    with h5py.File(P["full"]) as f:
        full = {k: f[k][:] for k in f.keys()}
    ties = np.flatnonzero(is_tie(full["labels"]))
    alloc = allocate(len(ties), s)
    out = {}
    for part in ("train", "test"):
        with h5py.File(P["base"][part]) as b:
            base = {k: b[k][:] for k in b.keys()}
        assert set(base) == set(full), (v, s, part)
        out[part] = {k: np.concatenate([base[k], full[k][ties[alloc[part]]]]) for k in full}
    out["valties"] = {k: full[k][ties[alloc["val"]]] for k in full}
    return out, ties, alloc


def check(v, s, root=DATA_ROOT):
    """[problems] for the written files of one (variant, seed)."""
    P = paths(v, s, root)
    want, ties, alloc = build(v, s, root)
    probs = []
    for name, arrs in want.items():
        p = P["out"][name]
        if not os.path.isfile(p):
            probs.append(f"missing {p}")
            continue
        with h5py.File(p) as h:
            for k, a in arrs.items():
                if k not in h or h[k].dtype != a.dtype or not np.array_equal(h[k][:], a):
                    probs.append(f"{p}: key {k} differs")
    # the tie-free rows of each ties file are the base file's, and the ties partition
    with h5py.File(P["full"]) as f:
        L = f["labels"][:]
    got = np.concatenate([alloc[p] for p in ("train", "val", "test")])
    if sorted(got.tolist()) != list(range(len(ties))):
        probs.append("tied rows are not a partition")
    for part in ("train", "test"):
        with h5py.File(P["base"][part]) as b, h5py.File(P["out"][part]) as h:
            nb = b["labels"].shape[0]
            if not all(np.array_equal(b[k][:], h[k][:nb]) for k in b.keys()):
                probs.append(f"{part}: tie-free rows differ from the base file")
            if not is_tie(h["labels"][nb:]).all() or is_tie(h["labels"][:nb]).any():
                probs.append(f"{part}: rows misplaced (ties must follow the base rows)")
    with h5py.File(P["out"]["valties"]) as h:
        if not is_tie(h["labels"][:]).all():
            probs.append("valties holds a non-tie")
    return probs


def digest(p):
    """A content fingerprint (keys, dtypes, shapes, bytes), the same on any machine:
    HDF5 files embed creation times, so hashing the file bytes would not be."""
    h = hashlib.sha256()
    with h5py.File(p) as f:
        for k in sorted(f.keys()):
            a = f[k][:]
            h.update(f"{k}|{a.dtype}|{a.shape}|".encode())
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--data-root", default=DATA_ROOT)
    a = ap.parse_args()
    bad = 0
    for v in TIES_VARIANTS:
        for s in SEEDS:
            out, ties, alloc = build(v, s, a.data_root)
            P = paths(v, s, a.data_root)
            if a.write:
                os.makedirs(os.path.dirname(P["out"]["train"]), exist_ok=True)
                for name, arrs in out.items():
                    with h5py.File(P["out"][name], "w") as h:
                        for k, arr in arrs.items():
                            h.create_dataset(k, data=arr)
            line = (f"  {v:15s} seed {s:2d}: {len(ties)} ties -> train {len(alloc['train'])}, "
                    f"val {len(alloc['val'])}, test {len(alloc['test'])}; "
                    f"train {len(out['train']['labels'])} rows, test {len(out['test']['labels'])} rows")
            if a.check:
                probs = check(v, s, a.data_root)
                bad += bool(probs)
                line += "   CHECK " + ("OK  " + " ".join(digest(P["out"][n]) for n in ("train", "test", "valties"))
                                       if not probs else f"FAILED: {probs}")
            print(line)
    if a.check:
        print(f"\n  {'ALL OK' if not bad else f'{bad} (variant, seed) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
