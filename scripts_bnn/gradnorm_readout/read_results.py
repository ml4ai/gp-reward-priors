#!/usr/bin/env python
"""Aggregate and print the per-chain grad-norm stats from a readout run.

Reads chain_<i>/grad_norm_stats.pt under each variant's readout OUT_DIR (which
gets `_<seed>` appended by the eval script) and prints, per variant and phase,
the pre-clip grad-norm max / mean / % of steps over the clip threshold.

Decision: sampling-phase "%>clip" ~ 0  -> clip inert during sampling, the CVaR
lower tail is clip-unbiased.  > 0  -> clip fires in sampling -> real tail bias,
proceed to the Bradley-Terry logit /T fix (Issue 3, Step 3).
"""
import glob
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TASKS = ["medium_play", "medium_diverse", "large_play", "large_diverse"]
SEED = 1               # readout configs all use seed 1
CLIP = 100.0           # the hard-coded clip threshold being instrumented


def main():
    print(f"clip threshold = {CLIP}\n")
    hdr = f"{'variant':16}{'phase':10}{'chains':>7}{'steps':>10}{'max':>12}{'mean':>10}{'%>clip':>9}"
    print(hdr)
    print("-" * len(hdr))
    for task in TASKS:
        out = os.path.join(REPO, "exp", "_gradnorm_readout",
                           f"{task}_{SEED}", "sampling_f")
        files = sorted(glob.glob(os.path.join(out, "chain_*", "grad_norm_stats.pt")))
        if not files:
            print(f"{task:16}(no grad_norm_stats.pt under {os.path.relpath(out, REPO)})")
            continue
        for phase in ("burnin", "sampling"):
            cnt = nover = 0
            sm = 0.0
            mx = 0.0
            for f in files:
                d = torch.load(f, weights_only=False).get(phase, {})
                cnt += d.get("count", 0)
                sm += d.get("sum", 0.0)
                nover += d.get("n_over_clip", 0)
                mx = max(mx, d.get("max", 0.0))
            if cnt == 0:
                continue
            print(f"{task:16}{phase:10}{len(files):>7}{cnt:>10}"
                  f"{mx:>12.2f}{sm / cnt:>10.2f}{100 * nover / cnt:>8.2f}%")
        print()
    print("Read the SAMPLING rows: %>clip ~ 0 => clip inert (CVaR unbiased); "
          ">0 => fires => do the BT-logit /T fix.")


if __name__ == "__main__":
    main()
