#!/usr/bin/env bash
# Phase-1 launcher: train the antmaze-eval reward models across seeds 1..10 and
# all four antmaze variants, GPU-optimally, for one method (bnn | mr | pt).
#
# Run from the gp_reward-priors submodule root.
#
# Each (variant, seed) is one training job.  The seed both selects the data split
# (data/antmaze/<variant>/eval/seed_<seed>/...) and fixes the deterministic
# output dir (exp/reward_learning/<variant>_<method>_eval_<seed>/), which the IQL
# eval stage (iql_eval.py, via reward_model_root) reads back by seed.
#
# Scheduling is slot-based: a "slot" is the set of GPU ids one concurrent job
# occupies.  Jobs are dispatched onto free slots; when a slot's job finishes the
# slot is reused, so #slots = max concurrency.
#
#   MR / PT  — one GPU per job, JOBS_PER_GPU jobs packed per GPU (small nets).
#              slots = (#GPUs * JOBS_PER_GPU), each slot a single GPU id.
#   BNN      — GPUS_PER_JOB GPUs per job (fSGHMC spreads its chains across them
#              via chains_per_gpu); slots = floor(#GPUs / GPUS_PER_JOB), each a
#              GPU group.  One job per group.
#
# Usage:
#   ./train_rewards.sh METHOD [GPU_LIST] [PACK]
#
#   METHOD    bnn | mr | pt                                     (required)
#   GPU_LIST  space-separated GPU ids (quote it).   Default: "0 1 2 3 4 5"
#   PACK      mr/pt: jobs per GPU.        Default: mr=3, pt=2
#             bnn:   GPUs per job.        Default: 1  (all 8 chains on one GPU)
#
# Env overrides:
#   VARIANTS  space-separated variant tokens.
#             Default: "medium_play medium_diverse large_play large_diverse"
#   SEEDS     space-separated seeds.      Default: "1 2 3 4 5 6 7 8 9 10"
#   NUM_CHAINS  bnn chains per run (must match the config). Default: 8
#
# Examples:
#   ./train_rewards.sh mr                       # MR, 6 GPUs, 3 jobs/GPU (18 concurrent)
#   ./train_rewards.sh pt "0 1 2 3 4 5" 2       # PT, 2 jobs/GPU (12 concurrent)
#   ./train_rewards.sh bnn                      # BNN, 1 GPU/job (6 concurrent)
#   ./train_rewards.sh bnn "0 1 2 3 4 5" 3      # BNN, 3 GPUs/job (2 concurrent, all 6 used)
#   SEEDS="1 2 3" ./train_rewards.sh mr         # only seeds 1-3
set -euo pipefail

# Needs bash >= 4.3 for `wait -n` and associative arrays (the job pool).
# macOS ships bash 3.2; the Linux run box is fine. Install a newer bash if needed.
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
  echo "ERROR: this launcher needs bash >= 4.3 (found $BASH_VERSION)." >&2
  exit 1
fi

# Uses whatever `python` is on PATH: activate your conda env before running.
# Override with e.g. PY=/path/to/python ./train_rewards.sh ...
PY="${PY:-python}"

# Submodule root (dir holding optbnn/). The training scripts import optbnn, but
# run from a working dir where their own `sys.path.insert(0, abspath(".."))` does
# not resolve to it — so put the root on PYTHONPATH explicitly. Uses the launcher's
# own location, independent of the caller's CWD.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Always run from the submodule root so the configs' relative paths resolve no
# matter where the launcher was invoked from: the --config_path yaml, and inside
# each config data_root ("data/antmaze") and measurement_dataset, are all relative
# to this directory.
cd "$ROOT"

if [[ ! -d scripts_bnn || ! -d data/antmaze ]]; then
  echo "ERROR: train_rewards.sh must live in the gp_reward-priors submodule root." >&2
  exit 1
fi

METHOD="${1:-}"
GPU_LIST="${2:-0 1 2 3 4 5}"
PACK_ARG="${3:-}"
VARIANTS="${VARIANTS:-medium_play medium_diverse large_play large_diverse}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
NUM_CHAINS="${NUM_CHAINS:-8}"

case "$METHOD" in
  bnn) SCRIPT="scripts_bnn/run_bnn_training_antmaze_eval.py"; CFG_DIR="scripts_bnn"; CFG_SUF="_bnn_antmaze_eval.yaml" ;;
  mr)  SCRIPT="scripts_mr/run_mr_training_antmaze_eval.py";   CFG_DIR="scripts_mr";  CFG_SUF="_mr_antmaze_eval.yaml"  ;;
  pt)  SCRIPT="scripts_pt/run_pt_training_antmaze_eval.py";   CFG_DIR="scripts_pt";  CFG_SUF="_pt_antmaze_eval.yaml"  ;;
  *)   echo "ERROR: METHOD must be bnn | mr | pt (got '${METHOD:-<empty>}')" >&2; exit 1 ;;
esac

read -ra GPUS <<< "$GPU_LIST"
NGPU=${#GPUS[@]}
LOGDIR="exp/train_logs/${METHOD}"; mkdir -p "$LOGDIR"

# --- build the slot array (each element = comma-separated GPU ids for one job) ---
SLOTS=()
EXTRA_ARGS=()
if [[ "$METHOD" == "bnn" ]]; then
  GPUS_PER_JOB="${PACK_ARG:-1}"
  (( GPUS_PER_JOB >= 1 )) || { echo "ERROR: PACK (GPUs/job) must be >= 1" >&2; exit 1; }
  (( NGPU >= GPUS_PER_JOB )) || { echo "ERROR: fewer GPUs ($NGPU) than GPUs/job ($GPUS_PER_JOB)" >&2; exit 1; }
  # chains pack onto the visible GPUs: chains_per_gpu = ceil(NUM_CHAINS / GPUS_PER_JOB)
  CPG=$(( (NUM_CHAINS + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))
  EXTRA_ARGS=(--chains_per_gpu "$CPG")
  for (( i=0; i + GPUS_PER_JOB <= NGPU; i += GPUS_PER_JOB )); do
    grp=$(IFS=,; echo "${GPUS[*]:i:GPUS_PER_JOB}")
    SLOTS+=("$grp")
  done
  echo "BNN: ${GPUS_PER_JOB} GPU(s)/job, chains_per_gpu=$CPG -> ${#SLOTS[@]} concurrent job(s)"
else
  JOBS_PER_GPU="${PACK_ARG:-$([[ "$METHOD" == mr ]] && echo 3 || echo 2)}"
  (( JOBS_PER_GPU >= 1 )) || { echo "ERROR: PACK (jobs/GPU) must be >= 1" >&2; exit 1; }
  for (( a=0; a<JOBS_PER_GPU; a++ )); do
    for g in "${GPUS[@]}"; do SLOTS+=("$g"); done
  done
  echo "$(echo "$METHOD" | tr '[:lower:]' '[:upper:]'): ${JOBS_PER_GPU} job(s)/GPU -> ${#SLOTS[@]} concurrent job(s)"
fi
NSLOTS=${#SLOTS[@]}

# --- enumerate jobs ---
JOBS=()   # each element: "variant seed"
for v in $VARIANTS; do
  cfg="${ROOT}/${CFG_DIR}/antmaze_${v}${CFG_SUF}"
  if [[ ! -f "$cfg" ]]; then echo "ERROR: config not found: $cfg" >&2; exit 1; fi
  for s in $SEEDS; do JOBS+=("$v $s"); done
done
echo "Root (cwd): $ROOT   data_root: ${ROOT}/data/antmaze"
echo "Method: $METHOD   variants: $VARIANTS   seeds: $SEEDS"
echo "Total jobs: ${#JOBS[@]}   slots: $NSLOTS   log dir: $LOGDIR"
echo

# --- dispatch: keep NSLOTS jobs in flight; reuse a slot when its job exits ---
declare -A SLOT_PID=()      # slot index -> pid of running job (unset if free)
declare -A PID_SLOT=()      # pid -> slot index

launch() {  # $1=slot_index  $2=variant  $3=seed
  local slot="$1" v="$2" s="$3"
  local gpus="${SLOTS[$slot]}"
  # Absolute path: pyrallis opens config_path relative to the process CWD, so an
  # absolute path makes config loading independent of where the run happens.
  local cfg="${ROOT}/${CFG_DIR}/antmaze_${v}${CFG_SUF}"
  local logf="$LOGDIR/${v}_seed${s}.log"
  echo "[gpu ${gpus}] ${METHOD} ${v} seed=${s}  -> ${logf}"
  # Absolute --data_root as well: the configs build train/val/test paths from it
  # (default "data/antmaze"), and pyrallis/h5py open those relative to the CWD.
  # Passing the absolute root makes the data paths resolve independent of CWD.
  CUDA_VISIBLE_DEVICES="$gpus" nohup "$PY" "$SCRIPT" \
      --config_path "$cfg" --data_root "${ROOT}/data/antmaze" \
      --seed "$s" "${EXTRA_ARGS[@]}" \
      > "$logf" 2>&1 &
  local pid=$!
  SLOT_PID[$slot]=$pid
  PID_SLOT[$pid]=$slot
}

free_slot() {  # echo index of a free slot, or nothing
  local i
  for (( i=0; i<NSLOTS; i++ )); do
    [[ -z "${SLOT_PID[$i]:-}" ]] && { echo "$i"; return; }
  done
}

FAILED=0
DONE=0
ji=0
NJOBS=${#JOBS[@]}
while (( ji < NJOBS )) || (( ${#PID_SLOT[@]} > 0 )); do
  # fill free slots
  while (( ji < NJOBS )); do
    slot=$(free_slot)
    [[ -z "$slot" ]] && break
    read -r v s <<< "${JOBS[$ji]}"
    launch "$slot" "$v" "$s"
    ji=$(( ji + 1 ))
  done
  # all slots busy (or no jobs left to fill) -> block until one finishes
  if (( ${#PID_SLOT[@]} > 0 )); then
    (( ji < NJOBS )) && echo "  ...${#PID_SLOT[@]}/${NSLOTS} slots busy, ${DONE} done, $(( NJOBS - ji )) queued; waiting for a job to finish"
    wait -n || FAILED=$(( FAILED + 1 ))
    for pid in "${!PID_SLOT[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        slot="${PID_SLOT[$pid]}"
        unset "SLOT_PID[$slot]"
        unset "PID_SLOT[$pid]"
        DONE=$(( DONE + 1 ))
        echo "  [${DONE}/${NJOBS} done] freed slot ${slot}  (running: ${#PID_SLOT[@]})"
      fi
    done
  fi
done

echo
if (( FAILED > 0 )); then
  echo "DONE with $FAILED failed job(s). Grep the logs:  grep -l Traceback $LOGDIR/*.log" >&2
  exit 1
fi
echo "All ${NJOBS} ${METHOD} training jobs completed. Logs: $LOGDIR/"
