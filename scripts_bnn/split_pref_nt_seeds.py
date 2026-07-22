"""split_pref_nt_seeds.py — split each antmaze *_pref_nt.hdf5 preference dataset
into train/val/test partitions over random seeds (default 1..10; pass seed
numbers as command-line args to override, e.g. `split_pref_nt_seeds.py 0`).

For every variant under data/antmaze, the *_pref_nt.hdf5 file holds N preference
pairs along axis 0 across 9 datasets (states/states_2, actions/actions_2,
attn_mask/attn_mask_2, timesteps/timesteps_2, labels).  For each seed we shuffle
the pair indices and split 70/15/15 into train/val/test, writing each partition
back in the identical HDF5 layout (same keys/dtypes, subset of rows) so that
util.load_pref_data reads them unchanged.

Outputs:
  data/antmaze/<variant>/eval/seed_<s>/<variant>_pref_{train,val,test}_<s>.hdf5
"""
import os
import sys
import glob
import h5py
import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "antmaze")
# Seeds may be given on the command line (e.g. `split_pref_nt_seeds.py 0`);
# default is the original 1..10.
SEEDS = [int(a) for a in sys.argv[1:]] or list(range(1, 11))
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test gets the remainder


def split_indices(n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(TRAIN_FRAC * n))
    n_val = int(round(VAL_FRAC * n))
    n_test = n - n_train - n_val
    assert n_test > 0, f"non-positive test size for n={n}"
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


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
        src_path = os.path.join(vdir, f"{variant}_pref_nt.hdf5")
        if not os.path.isfile(src_path):
            print(f"[skip] no _pref_nt for {variant}")
            continue

        with h5py.File(src_path, "r") as src:
            keys = list(src.keys())
            n = src[keys[0]].shape[0]
            for k in keys:
                assert src[k].shape[0] == n, f"{variant}: key {k} has mismatched axis 0"

            for seed in SEEDS:
                splits = split_indices(n, seed)
                out_dir = os.path.join(vdir, "eval", f"seed_{seed}")
                os.makedirs(out_dir, exist_ok=True)
                for part, ind in splits.items():
                    dst = os.path.join(out_dir, f"{variant}_pref_{part}_{seed}.hdf5")
                    write_partition(src, dst, np.sort(ind))
            sizes = {p: len(v) for p, v in split_indices(n, 1).items()}
        print(f"[ok] {variant}: N={n} -> {sizes} x {len(list(SEEDS))} seeds")


if __name__ == "__main__":
    main()
