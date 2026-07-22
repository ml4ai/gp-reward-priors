#!/usr/bin/env bash
# launch_hp_sweeps.sh — create and run the post-sampler-fix HP re-selection
# sweeps on the GPU box (Ubuntu, 6x RTX A6000, conda env `pt` activated).
#
# Usage (from anywhere; the script cd's to the repo root):
#   ./launch_hp_sweeps.sh phase1   # MR + PT + BNN warm-up tier  (12 sweeps)
#   ./launch_hp_sweeps.sh phase2   # BNN sampling tier            (4 sweeps)
#
# phase2 requires the FILL_ME placeholders in the
# scripts_bnn/sweep_antmaze_*_bnn_sampling_antmaze_eval.yaml templates to be
# replaced with the tier-1 winners first; the preflight refuses otherwise.
#
# Idempotent: sweep IDs are cached in exp/sweep_ids_<phase>.txt, so a re-run
# reuses the existing sweeps and only starts agents that are not already
# running (safe after a reboot or a killed agent).
#
# Exactly ONE wandb agent per sweep, by design:
#   * serial runs give the Bayes optimizer the full result history;
#   * the antmaze_eval scripts write checkpoints/OUT_DIR to deterministic
#     per-seed paths (..._eval_0), so two concurrent runs of the same sweep
#     would clobber each other's artifacts.  Do not add extra agents.
#
# Launch convention: each agent runs from its scripts_* directory (the sweep
# `program` is a bare filename).  The training scripts os.chdir("..") to the
# repo root at import, so config_path / data_root / measurement_dataset in the
# configs are repo-root-relative.  Launching agents from anywhere else breaks
# every relative path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PHASE="${1:-}"
if [[ "$PHASE" != "phase1" && "$PHASE" != "phase2" ]]; then
    echo "usage: $0 phase1|phase2" >&2
    exit 1
fi

# ---------------------------------------------------------------- sweep maps
# Entries are "key|sweep_yaml|gpu".  GPU packing (phase1): MR is a small MLP
# (all 4 agents share GPU 0), PT fits 2/GPU, BNN warm-up is a single-network
# phase (medium variants pair on GPU 3, large variants get their own GPU).
# phase2: each sampling run needs a full GPU (4 chains, chains_per_gpu=4);
# GPUs 4-5 stay free.
if [[ "$PHASE" == "phase1" ]]; then
    ENTRIES=(
        "mr_medium_play|scripts_mr/sweep_antmaze_medium_play_mr_antmaze_eval.yaml|0"
        "mr_medium_diverse|scripts_mr/sweep_antmaze_medium_diverse_mr_antmaze_eval.yaml|0"
        "mr_large_play|scripts_mr/sweep_antmaze_large_play_mr_antmaze_eval.yaml|0"
        "mr_large_diverse|scripts_mr/sweep_antmaze_large_diverse_mr_antmaze_eval.yaml|0"
        "pt_medium_play|scripts_pt/sweep_antmaze_medium_play_pt_antmaze_eval.yaml|1"
        "pt_medium_diverse|scripts_pt/sweep_antmaze_medium_diverse_pt_antmaze_eval.yaml|1"
        "pt_large_play|scripts_pt/sweep_antmaze_large_play_pt_antmaze_eval.yaml|2"
        "pt_large_diverse|scripts_pt/sweep_antmaze_large_diverse_pt_antmaze_eval.yaml|2"
        "bnn_warmup_medium_play|scripts_bnn/sweep_antmaze_medium_play_bnn_warmup_antmaze_eval.yaml|3"
        "bnn_warmup_medium_diverse|scripts_bnn/sweep_antmaze_medium_diverse_bnn_warmup_antmaze_eval.yaml|3"
        "bnn_warmup_large_play|scripts_bnn/sweep_antmaze_large_play_bnn_warmup_antmaze_eval.yaml|4"
        "bnn_warmup_large_diverse|scripts_bnn/sweep_antmaze_large_diverse_bnn_warmup_antmaze_eval.yaml|5"
    )
else
    ENTRIES=(
        "bnn_sampling_medium_play|scripts_bnn/sweep_antmaze_medium_play_bnn_sampling_antmaze_eval.yaml|0"
        "bnn_sampling_medium_diverse|scripts_bnn/sweep_antmaze_medium_diverse_bnn_sampling_antmaze_eval.yaml|1"
        "bnn_sampling_large_play|scripts_bnn/sweep_antmaze_large_play_bnn_sampling_antmaze_eval.yaml|2"
        "bnn_sampling_large_diverse|scripts_bnn/sweep_antmaze_large_diverse_bnn_sampling_antmaze_eval.yaml|3"
    )
fi

# ----------------------------------------------------------------- preflight
fail=0

command -v wandb >/dev/null 2>&1 || { echo "PREFLIGHT: wandb not on PATH (activate the pt env)" >&2; fail=1; }

if command -v nvidia-smi >/dev/null 2>&1; then
    ngpu=$(nvidia-smi --list-gpus | wc -l)
    if (( ngpu < 6 )); then
        echo "PREFLIGHT: only $ngpu GPUs visible (expected 6) — GPU map may not fit" >&2
        fail=1
    fi
else
    echo "PREFLIGHT: nvidia-smi not found" >&2
    fail=1
fi

python - <<'PYEOF' || fail=1
import importlib
for m in ("torch", "optbnn", "wandb", "pyrallis"):
    importlib.import_module(m)
PYEOF
(( fail )) && echo "PREFLIGHT: python import check failed (PYTHONPATH/env wrong?)" >&2

# seed-0 eval splits + tuning sets (data/ is gitignored: copied manually)
for variant in antmaze-medium-play-v2 antmaze-medium-diverse-v2 \
               antmaze-large-play-v2 antmaze-large-diverse-v2; do
    for split in train val test; do
        f="data/antmaze/${variant}/eval/seed_0/${variant}_pref_${split}_0.hdf5"
        [[ -f "$f" ]] || { echo "PREFLIGHT: missing $f" >&2; fail=1; }
    done
    t="data/antmaze/${variant}/${variant}_tuning_set.hdf5"
    [[ -f "$t" ]] || { echo "PREFLIGHT: missing $t (needed by the BNN prior)" >&2; fail=1; }
done

# sweep yamls exist; phase2 must have its tier-1 winners filled in
for entry in "${ENTRIES[@]}"; do
    yaml="${entry#*|}"; yaml="${yaml%%|*}"
    [[ -f "$yaml" ]] || { echo "PREFLIGHT: missing sweep yaml $yaml" >&2; fail=1; }
    if [[ "$PHASE" == "phase2" ]] && grep -q "FILL_ME" "$yaml" 2>/dev/null; then
        echo "PREFLIGHT: $yaml still contains FILL_ME — transcribe the tier-1 winners first" >&2
        fail=1
    fi
done

if (( fail )); then
    echo "Preflight failed; nothing launched." >&2
    exit 1
fi
echo "Preflight OK."

# ------------------------------------------------- sweep creation (idempotent)
mkdir -p exp/sweep_logs
IDS_FILE="exp/sweep_ids_${PHASE}.txt"
touch "$IDS_FILE"

get_or_create_sweep() {
    # echoes the sweep path (entity/project/id) for $1=key $2=yaml
    local key="$1" yaml="$2" cached out id
    cached=$(awk -v k="$key" '$1 == k {print $2}' "$IDS_FILE" | tail -1)
    if [[ -n "$cached" ]]; then
        echo "$cached"
        return 0
    fi
    if ! out=$(wandb sweep "$yaml" 2>&1); then
        printf '%s\n' "$out" >&2
        echo "ERROR: 'wandb sweep $yaml' failed" >&2
        return 1
    fi
    # wandb prints: "wandb: Run sweep agent with: wandb agent <entity>/<project>/<id>"
    id=$(printf '%s\n' "$out" | grep -oE 'wandb agent [^ ]+' | awk '{print $3}' | tail -1)
    if [[ -z "$id" ]]; then
        printf '%s\n' "$out" >&2
        echo "ERROR: could not parse sweep id for $yaml" >&2
        return 1
    fi
    echo "$key $id" >> "$IDS_FILE"
    echo "$id"
}

# --------------------------------------------------------------- agent launch
launched=0
for entry in "${ENTRIES[@]}"; do
    key="${entry%%|*}"
    rest="${entry#*|}"
    yaml="${rest%%|*}"
    gpu="${rest##*|}"
    workdir="$ROOT/$(dirname "$yaml")"   # agents run from their scripts_* dir

    sweep_id=$(get_or_create_sweep "$key" "$yaml")

    if pgrep -f "wandb agent.*${sweep_id}" >/dev/null 2>&1; then
        echo "[$key] agent for $sweep_id already running — skipping"
        continue
    fi

    log="$ROOT/exp/sweep_logs/agent_${key}.log"
    echo "[$key] GPU $gpu  wandb agent $sweep_id  (cwd: $workdir, log: $log)"
    (
        cd "$workdir"
        CUDA_VISIBLE_DEVICES="$gpu" nohup wandb agent "$sweep_id" >> "$log" 2>&1 &
    )
    launched=$((launched + 1))
done

echo
echo "$PHASE: $launched agent(s) launched (sweep IDs cached in $IDS_FILE)."
echo "Monitor with: tail -f exp/sweep_logs/agent_*.log"
