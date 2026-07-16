"""make_reduction_sets.py — build nested data-reduction training sets.

For every antmaze variant and every seed (1..10), take that seed's training
partition (<variant>_pref_train_<s>.hdf5) and create reduced versions with
128, 64, 32, and 16 preference pairs.  The reductions are NESTED using the
matching seed: a single seeded permutation of the training pairs is drawn and
prefixes are taken, so the 16-set ⊂ 32-set ⊂ 64-set ⊂ 128-set ⊂ full train.

Outputs (same HDF5 layout as the source, subset of rows):
  data/antmaze/<variant>/eval/seed_<s>/reduction/<variant>_pref_train_<s>_<N>.hdf5
"""
import os
import glob
import h5py
import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "antmaze")
SEEDS = range(1, 11)
SIZES = [128, 64, 32, 16]  # descending; each nested within the previous


def write_partition(src, dst_path, indices):
    with h5py.File(dst_path, "w") as out:
        for key in src.keys():
            out.create_dataset(key, data=src[key][:][indices])


def main():
    variant_dirs = sorted(
        d for d in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(d)
    )
    for vdir in variant_dirs:
        variant = os.path.basename(vdir)
        for seed in SEEDS:
            seed_dir = os.path.join(vdir, "eval", f"seed_{seed}")
            src_path = os.path.join(seed_dir, f"{variant}_pref_train_{seed}.hdf5")
            if not os.path.isfile(src_path):
                print(f"[skip] no train set: {src_path}")
                continue

            out_dir = os.path.join(seed_dir, "reduction")
            os.makedirs(out_dir, exist_ok=True)

            with h5py.File(src_path, "r") as src:
                keys = list(src.keys())
                n = src[keys[0]].shape[0]
                assert n >= SIZES[0], f"{variant} seed {seed}: train N={n} < {SIZES[0]}"

                # One seeded permutation; nested prefixes give the nesting.
                perm = np.random.default_rng(seed).permutation(n)
                for size in SIZES:
                    idx = np.sort(perm[:size])
                    dst = os.path.join(
                        out_dir, f"{variant}_pref_train_{seed}_{size}.hdf5"
                    )
                    write_partition(src, dst, idx)
        print(f"[ok] {variant}: reductions {SIZES} x {len(list(SEEDS))} seeds")


if __name__ == "__main__":
    main()
