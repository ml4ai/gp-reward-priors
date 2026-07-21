#!/usr/bin/env python
"""Generate reduced-budget configs for the grad-clip instrumentation readout.

Copies each `antmaze_<variant>_bnn_antmaze_eval.yaml` and overrides ONLY the
sampling budget, so the readout uses the exact production schedule / prior /
data (seed 1) but runs a fraction of the compute.  We keep the full 5000-step
burn-in (so sampling-phase grad norms are representative) and just cut the
number of cycles — the pre-clip grad-norm firing RATE stabilises within a few
thousand sampling steps, which a dozen cycles supply.

Output: scripts_bnn/gradnorm_readout/<variant>_readout.yaml
The workers write chain_<i>/grad_norm_stats.pt regardless of the training
script; read_results.py aggregates them.  Nothing here changes sampler behaviour.
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# --- readout budget (edit these to trade coverage vs wall-clock) ---
NUM_SAMPLES = 12      # cycles collected after burn-in (sampling-phase length)
NUM_CHAINS = 2        # 2 gives a cross-chain view at minimal cost
N_DISCARDED = 0       # keep all cycles; sample quality is irrelevant here

# chains_per_gpu is paired with the launcher's GPU assignment: large variants
# get 2 GPUs (1 chain each), medium variants get 1 GPU (2 chains).
CHAINS_PER_GPU = {
    "medium_play": 2, "medium_diverse": 2,
    "large_play": 1, "large_diverse": 1,
}


def main():
    for task, cpg in CHAINS_PER_GPU.items():
        src = os.path.join(REPO, "scripts_bnn",
                           f"antmaze_{task}_bnn_antmaze_eval.yaml")
        cfg = yaml.safe_load(open(src))
        cfg["num_samples"] = NUM_SAMPLES
        cfg["num_chains"] = NUM_CHAINS
        cfg["n_discarded"] = N_DISCARDED
        cfg["chains_per_gpu"] = cpg
        cfg["warmup_log_every"] = 0            # quieter; burn-in length unchanged
        cfg["OUT_DIR"] = f"./exp/_gradnorm_readout/{task}"
        out = os.path.join(HERE, f"{task}_readout.yaml")
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"wrote {os.path.relpath(out, REPO)}  "
              f"(seed={cfg['seed']} chains={NUM_CHAINS} samples={NUM_SAMPLES} "
              f"chains_per_gpu={cpg})")


if __name__ == "__main__":
    main()
