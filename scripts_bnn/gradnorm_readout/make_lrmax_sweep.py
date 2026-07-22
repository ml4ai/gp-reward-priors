#!/usr/bin/env python
"""Generate a medium-play lr_max mini-sweep for the grad-clip readout.

Post-/T-fix, medium-play diverged during unclipped sampling under its OLD
hyperparameters (grad-norm max ~5e7).  This isolates ONE variable -- the
cyclical hot-phase peak lr_max -- to test whether a gentler lr_max stabilises it
(=> HP mismatch, the sweep will fix it) or whether it diverges even at low lr_max
(=> deeper kernel-conditioning issue, raise meas_jitter before sweeping).

Everything else is the post-fix regime: bt_pool=mean, clip_during_sampling=False,
resample_momentum=True, samples_per_cycle=1, chain_init_jitter=0.  Only
sghmc_lr_max varies.  lr_max must stay >= sghmc_lr (the cool-phase lr / cosine
floor), so the ladder is clamped above it.

Output: mp_lrmax<micro>_readout.yaml, one per rung (micro = round(lr_max*1e6)).
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# Geometric-ish ladder from the current lr_max (0.006402) down ~4x.
LR_MAX_LADDER = [0.006402047122411048, 0.0048, 0.0032, 0.0024, 0.0016]


def main():
    base = yaml.safe_load(
        open(os.path.join(HERE, "medium_play_readout.yaml"))
    )
    floor = float(base["sghmc_lr"])  # lr_max must exceed the cosine floor
    written = []
    for lr_max in LR_MAX_LADDER:
        if lr_max <= floor:
            print(f"skip lr_max={lr_max} (<= sghmc_lr floor {floor})")
            continue
        micro = int(round(lr_max * 1e6))
        cfg = dict(base)
        cfg["sghmc_lr_max"] = lr_max
        cfg["OUT_DIR"] = f"./exp/_gradnorm_readout/mp_lrmax{micro}"
        out = os.path.join(HERE, f"mp_lrmax{micro}_readout.yaml")
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        written.append((lr_max, os.path.relpath(out, REPO)))
    print(f"wrote {len(written)} configs (sghmc_lr floor = {floor}):")
    for lr_max, p in written:
        print(f"  lr_max={lr_max:.6f}  {p}")


if __name__ == "__main__":
    main()
