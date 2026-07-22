"""make_noise_sets.py — build nested label-noise training/validation sets.

For every antmaze variant and every seed (default 1..10; pass seed numbers as
command-line args to override, e.g. `make_noise_sets.py 0`), take that seed's training and
validation partitions and create noisy copies where a percentage of the
preference labels are flipped (one-hot columns swapped).  Flip percentages are
0.1, 0.2, 0.3, 0.4, 0.45 and are NESTED: a single seed-matched permutation of
the pairs is drawn per partition and the first round(p*N) indices are flipped
relative to the ORIGINAL labels.  Because higher-p sets flip a prefix superset
of the lower-p sets, each label is flipped at most once — later sets never undo
an earlier flip; the 0.2 set = the 0.1 flips plus an additional ~0.1 of new
flips, and so on.

File naming appends the percentage after the seed number with the decimal point
removed (0.1 -> 01, 0.45 -> 045):
  data/antmaze/<variant>/eval/seed_<s>/noise/<variant>_pref_train_<s>_<pp>.hdf5
  data/antmaze/<variant>/eval/seed_<s>/noise/<variant>_pref_val_<s>_<pp>.hdf5
"""
import os
import sys
import glob
import h5py
import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "antmaze")
# Seeds may be given on the command line; default is the original 1..10.
SEEDS = [int(a) for a in sys.argv[1:]] or list(range(1, 11))
PERCENTS = [0.1, 0.2, 0.3, 0.4, 0.45]  # ascending; nested flip prefixes


def pct_suffix(p):
    # 0.1 -> "01", 0.2 -> "02", 0.45 -> "045"
    return str(p).replace(".", "")


def write_flipped(src, dst_path, flip_idx):
    with h5py.File(dst_path, "w") as out:
        for key in src.keys():
            data = src[key][:]
            if key == "labels":
                data = data.copy()
                data[flip_idx] = data[flip_idx][:, ::-1]  # swap one-hot columns
            out.create_dataset(key, data=data)


def process_partition(src_path, out_dir, part, variant, seed):
    with h5py.File(src_path, "r") as src:
        n = src["labels"].shape[0]
        perm = np.random.default_rng(seed).permutation(n)
        for p in PERCENTS:
            k = int(round(p * n))
            flip_idx = np.sort(perm[:k])  # nested prefix -> no un-flipping
            dst = os.path.join(
                out_dir, f"{variant}_pref_{part}_{seed}_{pct_suffix(p)}.hdf5"
            )
            write_flipped(src, dst, flip_idx)


def main():
    variant_dirs = sorted(
        d for d in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(d)
    )
    for vdir in variant_dirs:
        variant = os.path.basename(vdir)
        for seed in SEEDS:
            seed_dir = os.path.join(vdir, "eval", f"seed_{seed}")
            out_dir = os.path.join(seed_dir, "noise")
            os.makedirs(out_dir, exist_ok=True)
            for part in ("train", "val"):
                src_path = os.path.join(
                    seed_dir, f"{variant}_pref_{part}_{seed}.hdf5"
                )
                if not os.path.isfile(src_path):
                    print(f"[skip] missing {src_path}")
                    continue
                process_partition(src_path, out_dir, part, variant, seed)
        print(f"[ok] {variant}: noise {PERCENTS} x train/val x {len(list(SEEDS))} seeds")


if __name__ == "__main__":
    main()
