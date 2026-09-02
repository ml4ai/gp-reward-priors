#!/usr/bin/env bash
# launch_hp_sweeps.sh — create and run the HP selection sweeps on the GPU box
# (Ubuntu, 6x RTX A6000, conda env `pt` activated).
#
# Usage (from anywhere; the script cd's to the repo root):
#   ./launch_hp_sweeps.sh bnn        # round-3 BNN sweeps             (4 sweeps)
#   ./launch_hp_sweeps.sh baselines  # MR + PT stage 1                (8 sweeps)
#
# There is no combined mode: the two sets' GPU maps overlap (both use 0-2), so
# running them together would oversubscribe.  The baselines are complete in any
# case; that set exists to make them reproducible, not to be re-run.
#
# ROUND 2: the BNN's two-tier structure (warm-up tier -> sampling tier, launched
# here as phase1/phase2) is retired.  Architecture, prior strength and sampler
# schedule are now searched by ONE sweep per variant, exactly as the baselines
# are — see HANDOFF_HP_SELECTION.md sections 3 and 3.7.  The set names replace
# the old phase numbers because those numbers encoded the retired tiering.
#
# The BNN sweeps require their base config NOT to set `burn_in_lr`, so
# that burn-in inherits the swept `sghmc_lr`.  The preflight enforces this: a
# base config that sets it would silently reinstate the mismatch that ended
# round 1, and nothing downstream would show that it had.
#
# NOTE: the BNN base configs deliberately still carry a STATUS:
# SUPERSEDED-ROUND1 marker and round-1's values.  That is correct here — the
# sweep overrides all nine swept fields — and this launcher must NOT refuse on
# it, unlike train_rewards.sh, which trains from those values directly.
#
# Idempotent: sweep IDs are cached per set, so a re-run reuses the existing
# sweeps and only starts agents that are not already running (safe after a
# reboot or a killed agent).
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

# --- CPU thread caps (see HANDOFF_HP_SELECTION.md section 10.7) ---------------
# Must match train_rewards.sh.  Thread count changes floating-point reduction
# order, so the selection runs launched here and the seed 1-10 evaluation runs
# have to agree; a sweep run uncapped would not be comparable to the production
# runs that use its winner.  Capping is also simply faster (-27% measured).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

SET="${1:-}"
if [[ "$SET" != "bnn" && "$SET" != "baselines" ]]; then
    echo "usage: $0 bnn|baselines" >&2
    exit 1
fi

# ---------------------------------------------------------------- sweep maps
# Entries are "key|sweep_yaml|gpu".
#   bnn:       each trial is 4 chains with chains_per_gpu=4, i.e. one
#              full GPU per sweep -> GPUs 0-3, leaving 4-5 free.
#   baselines: MR is a small MLP (all 4 agents share GPU 0); PT fits 2/GPU.
#
# Sweep-id caches are per set.  `baselines` deliberately reads the historical
# exp/sweep_ids_phase1.txt: the MR/PT sweeps in it are complete round-1 sweeps
# that are unaffected by the round-2 redesign, and reusing their ids is what
# stops a re-run from creating duplicates.
#
# The BNN cache filename is versioned (now sweep_ids_bnn_round3.txt) ON PURPOSE.
# Bumped round2 -> round3 on 2026-09-01 with the round-3 redesign: the metric
# changed to val_cvar_ce, the search dropped 9 dimensions to 6, width capped at
# 9, and the per-trial budget moved to 32 chains x 120k steps.  Reusing the
# round-2 filename would have resumed the old sweeps with the old metric.
# The cache is keyed by entry name, so reusing a filename across a procedure
# change silently RESUMES the old sweeps -- which would carry the old selection
# metric and the old search ranges while appearing to start fresh.  Any change
# to the metric, the ranges or the per-trial budget must come with a new cache
# filename.
BNN_ENTRIES=(
    "bnn_medium_play|scripts_bnn/sweep_antmaze_medium_play_bnn_antmaze_eval.yaml|0"
    "bnn_medium_diverse|scripts_bnn/sweep_antmaze_medium_diverse_bnn_antmaze_eval.yaml|1"
    "bnn_large_play|scripts_bnn/sweep_antmaze_large_play_bnn_antmaze_eval.yaml|2"
    "bnn_large_diverse|scripts_bnn/sweep_antmaze_large_diverse_bnn_antmaze_eval.yaml|3"
)
BASELINE_ENTRIES=(
    "mr_medium_play|scripts_mr/sweep_antmaze_medium_play_mr_antmaze_eval.yaml|0"
    "mr_medium_diverse|scripts_mr/sweep_antmaze_medium_diverse_mr_antmaze_eval.yaml|0"
    "mr_large_play|scripts_mr/sweep_antmaze_large_play_mr_antmaze_eval.yaml|0"
    "mr_large_diverse|scripts_mr/sweep_antmaze_large_diverse_mr_antmaze_eval.yaml|0"
    "pt_medium_play|scripts_pt/sweep_antmaze_medium_play_pt_antmaze_eval.yaml|1"
    "pt_medium_diverse|scripts_pt/sweep_antmaze_medium_diverse_pt_antmaze_eval.yaml|1"
    "pt_large_play|scripts_pt/sweep_antmaze_large_play_pt_antmaze_eval.yaml|2"
    "pt_large_diverse|scripts_pt/sweep_antmaze_large_diverse_pt_antmaze_eval.yaml|2"
)

if [[ "$SET" == "bnn" ]]; then
    ENTRIES=("${BNN_ENTRIES[@]}")
    IDS_FILE="exp/sweep_ids_bnn_round3.txt"
else
    ENTRIES=("${BASELINE_ENTRIES[@]}")
    IDS_FILE="exp/sweep_ids_phase1.txt"
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

# sweep yamls exist and are launchable
for entry in "${ENTRIES[@]}"; do
    yaml="${entry#*|}"; yaml="${yaml%%|*}"
    if [[ ! -f "$yaml" ]]; then
        echo "PREFLIGHT: missing sweep yaml $yaml" >&2; fail=1; continue
    fi
    # Match FILL_ME only as a parameter VALUE — a yaml's header comments may
    # legitimately mention FILL_ME when describing this very guard.  The round-2
    # sweeps inherit nothing and so have none, but the check is generic.
    if grep -qE "value: *FILL_ME" "$yaml" 2>/dev/null; then
        echo "PREFLIGHT: $yaml still contains FILL_ME values" >&2
        fail=1
    fi
    # BNN sweeps: burn-in must inherit the swept `sghmc_lr`, so the base
    # config must not set `burn_in_lr`.  If it does, burn-in silently uses that
    # value, the sweep scores configurations it is not actually running, and
    # nothing downstream reveals it — this is the seam that ended round 1
    # (HANDOFF_HP_SELECTION.md section 3.7).  A null in the sweep cannot undo
    # it: wandb serialises null onto argv as "None" and pyrallis rejects that.
    if [[ "$SET" == "bnn" ]]; then
        base=$(awk '$1=="config_path:"{f=1;next} f&&$1=="value:"{v=$2; gsub(/"/,"",v); print v; exit}' "$yaml")
        if [[ -z "$base" ]]; then
            echo "PREFLIGHT: could not read config_path from $yaml" >&2; fail=1
        elif [[ ! -f "$base" ]]; then
            echo "PREFLIGHT: base config $base (from $yaml) not found" >&2; fail=1
        elif grep -qE '^[[:space:]]*burn_in_lr[[:space:]]*:' "$base"; then
            echo "PREFLIGHT: $base sets burn_in_lr — remove it." >&2
            echo "  The merged sweep needs burn-in to inherit the swept sghmc_lr;" >&2
            echo "  setting it here reinstates the round-1 mismatch invisibly." >&2
            fail=1
        fi
    fi
done

if (( fail )); then
    echo "Preflight failed; nothing launched." >&2
    exit 1
fi
echo "Preflight OK."

# ------------------------------------------------- sweep creation (idempotent)
mkdir -p exp/sweep_logs
touch "$IDS_FILE"   # set above, per sweep set

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
echo "$SET: $launched agent(s) launched (sweep IDs cached in $IDS_FILE)."
echo "Monitor with: tail -f exp/sweep_logs/agent_*.log"
