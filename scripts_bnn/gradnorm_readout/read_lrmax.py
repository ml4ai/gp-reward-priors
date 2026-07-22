#!/usr/bin/env python
"""Report the medium-play lr_max mini-sweep grad-norm stats, one row per rung.

Reads each mp_lrmax<micro>_readout.yaml (for its lr_max + OUT_DIR) and the
grad_norm_stats.pt the workers wrote, and prints the sampling-phase max / mean /
%>clip sorted by lr_max.  A rung "stabilises" when the max norm drops from ~1e7
to O(hundreds) and the mean to single digits -- that lr_max (and below) is the
safe region, confirming the divergence is HP-fixable.
"""
import glob
import os

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SEED = 1


def main():
    cfgs = sorted(glob.glob(os.path.join(HERE, "mp_lrmax*_readout.yaml")))
    if not cfgs:
        print("no mp_lrmax*_readout.yaml found -- run make_lrmax_sweep.py first")
        return
    rows = []
    for cfg_path in cfgs:
        cfg = yaml.safe_load(open(cfg_path))
        lr_max = float(cfg["sghmc_lr_max"])
        out = os.path.join(REPO, cfg["OUT_DIR"].lstrip("./") + f"_{SEED}", "sampling_f")
        files = sorted(glob.glob(os.path.join(out, "chain_*", "grad_norm_stats.pt")))
        if not files:
            rows.append((lr_max, None))
            continue
        cnt = nover = 0
        sm = mx = 0.0
        for f in files:
            d = torch.load(f, weights_only=False).get("sampling", {})
            cnt += d.get("count", 0)
            sm += d.get("sum", 0.0)
            nover += d.get("n_over_clip", 0)
            mx = max(mx, d.get("max", 0.0))
        rows.append((lr_max, (len(files), cnt, mx, sm / cnt if cnt else float("nan"),
                              100 * nover / cnt if cnt else float("nan"))))

    rows.sort(key=lambda r: r[0], reverse=True)
    print("medium-play lr_max mini-sweep (SAMPLING phase; bt_pool=mean, clip off)\n")
    print(f"{'lr_max':>10}{'chains':>8}{'steps':>9}{'max':>16}{'mean':>12}{'%>clip':>9}")
    print("-" * 64)
    for lr_max, s in rows:
        if s is None:
            print(f"{lr_max:>10.6f}   (no grad_norm_stats.pt yet)")
            continue
        nch, cnt, mx, mean, pct = s
        print(f"{lr_max:>10.6f}{nch:>8}{cnt:>9}{mx:>16.2f}{mean:>12.2f}{pct:>8.2f}%")
    print("\nStable rung: max ~O(1e2), mean single digits => HP-fixable (proceed to sweep).")
    print("Diverges even at the lowest lr_max => kernel conditioning; raise meas_jitter.")


if __name__ == "__main__":
    main()
