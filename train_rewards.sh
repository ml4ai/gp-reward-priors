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
#   SEEDS     space-separated seeds.      Default: "0 1 2 3 4 5 6 7 8 9 10"
#             Seed 0 is the hyperparameter-selection lineage; 1-10 are the
#             evaluation lineages (see HANDOFF_HP_SELECTION.md).
#   NUM_CHAINS  bnn chains per run (must match the config). Default: 8
#   COND      experiment condition(s), space-separated.  Empty (the default) trains
#             the production (anchor) models, exactly as before.
#               r16 r32 r64 r128      Experiment 1, data sparsity
#               n01 n02 n03 n04 n045  Experiment 2, label noise, PRIMARY arm
#               n01o ... n045o        Experiment 2, SENSITIVITY arm (mr/pt, medium_play)
#             A condition changes only the split files a run reads and the
#             directory it writes, both defined in scripts_bnn/experiment_conditions.py
#             (handoff sections 11 and 13).  With COND set, SEEDS defaults to 1..10
#             (seed 0 is refused: nothing is re-selected under a condition), jobs
#             run condition by condition, and a preflight checks every split file,
#             every output directory and the BNN chain budget before anything starts.
#   ALLOW_EXISTING=1  with COND: allow writing into a non-empty model directory.
#   DRY_RUN=1   print every job's full command, in order, and launch nothing.
#
# Examples:
#   ./train_rewards.sh mr                       # MR, 6 GPUs, 3 jobs/GPU (18 concurrent)
#   ./train_rewards.sh pt "0 1 2 3 4 5" 2       # PT, 2 jobs/GPU (12 concurrent)
#   ./train_rewards.sh bnn                      # BNN, 1 GPU/job (6 concurrent)
#   ./train_rewards.sh bnn "0 1 2 3 4 5" 3      # BNN, 3 GPUs/job (2 concurrent, all 6 used)
#   SEEDS="1 2 3" ./train_rewards.sh mr         # only seeds 1-3
#   COND="r16 r32 r64 r128" NUM_CHAINS=128 ./train_rewards.sh bnn "2 3 4 5" 4
#                                               # sparsity BNN models, seeds 1-10
#   DRY_RUN=1 COND="r16" ./train_rewards.sh mr  # show the commands only
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

# --- CPU thread caps (see HANDOFF_HP_SELECTION.md section 10.7) ---------------
# torch defaults its intra-op pool to the core count (255 on leviathan), and a
# BNN run spawns one process per chain, so an 8-chain job alone asks for ~2000
# threads and ~317 of 255 cores.  Most of that is spin-wait, not work: measured
# A/B on one 8-chain medium_diverse job, 3h01m default vs 2h12m capped (-27%),
# with the concurrency penalty (~2.8x) on top of that.
#
# Set here rather than per-command so every run in a campaign shares them.
# Thread count changes floating-point reduction order, so selection and
# evaluation runs must agree: do NOT override for part of a campaign.
#
# 2, not 8, from round 3 onward -- pre-registered in section 7.3 on 2026-09-02,
# propagated here 2026-09-12.  Section 7.3 requires round 3's selection runs and
# the seeds 1-10 evaluation runs to share this setting, so this value must track
# launch_hp_sweeps.sh exactly.  The -27% A/B above was measured at 8 vs
# uncapped; 2 is a further 1.40x per GPU-hour over 8 under round-3 concurrency.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

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
COND="${COND:-}"     # experiment condition(s); empty = the production (anchor) models
if [[ -n "$COND" ]]; then
  SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"   # no seed-0 lineage under a condition
else
  SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10}"
fi
NUM_CHAINS="${NUM_CHAINS:-8}"
COND_PY="scripts_bnn/experiment_conditions.py"

case "$METHOD" in
  bnn) SCRIPT="scripts_bnn/run_bnn_training_antmaze_eval.py"; CFG_DIR="scripts_bnn"; CFG_SUF="_bnn_antmaze_eval.yaml" ;;
  mr)  SCRIPT="scripts_mr/run_mr_training_antmaze_eval.py";   CFG_DIR="scripts_mr";  CFG_SUF="_mr_antmaze_eval.yaml"  ;;
  pt)  SCRIPT="scripts_pt/run_pt_training_antmaze_eval.py";   CFG_DIR="scripts_pt";  CFG_SUF="_pt_antmaze_eval.yaml"  ;;
  nmr) SCRIPT="scripts_nmr/run_nmr_training_antmaze_eval.py"; CFG_DIR="scripts_nmr"; CFG_SUF="_nmr_antmaze_eval.yaml" ;;
  *)   echo "ERROR: METHOD must be bnn | mr | pt | nmr (got '${METHOD:-<empty>}')" >&2; exit 1 ;;
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

# Absolute measurement_dataset (the fSGHMC GP-prior "tuning set") for a config.
# BNN configs set it as a path relative to the submodule root; MR/PT configs have
# no such key, so this echoes nothing for them. Always exits 0 (see free_slot).
abs_meas() {  # $1 = config file
  local m
  m=$(sed -n 's/^[[:space:]]*measurement_dataset:[[:space:]]*//p' "$1" | awk '{print $1}' | head -1)
  [[ -z "$m" ]] && return 0
  [[ "$m" == /* ]] || m="${ROOT}/${m}"
  echo "$m"
  return 0
}

# --- enumerate jobs (and preflight the measurement set for BNN) ---
JOBS=()   # each element: "cond variant seed"; cond is "-" for the anchor models
MISSING_MEAS=()
SUPERSEDED=()
for v in $VARIANTS; do
  cfg="${ROOT}/${CFG_DIR}/antmaze_${v}${CFG_SUF}"
  if [[ ! -f "$cfg" ]]; then echo "ERROR: config not found: $cfg" >&2; exit 1; fi
  # A config still carrying selected values from a discarded selection round
  # must not silently train production models.  The marker is a comment, so it
  # disappears the moment the header is replaced with a real provenance block.
  grep -q 'SUPERSEDED-ROUND1' "$cfg" && SUPERSEDED+=("$v -> $cfg")
  # A base config whose swept fields are placeholders (NMR before its stage-1
  # winner, handoff 14.14) must never train production models.  No override.
  if grep -q 'UNSELECTED-BASE' "$cfg"; then
    echo "ERROR: $cfg is an UNSELECTED base config (placeholder hyperparameters)." >&2
    echo "Write the stage-1 winner with scripts_bnn/make_production_config.py first." >&2
    exit 1
  fi
  meas=$(abs_meas "$cfg")
  [[ -n "$meas" && ! -f "$meas" ]] && MISSING_MEAS+=("$v -> $meas")
  for s in $SEEDS; do JOBS+=("- $v $s"); done
done
if (( ${#SUPERSEDED[@]} > 0 )) && [[ "${ALLOW_SUPERSEDED:-0}" != "1" ]]; then
  echo "ERROR: config(s) marked SUPERSEDED-ROUND1 -- refusing to train:" >&2
  printf '  %s\n' "${SUPERSEDED[@]}" >&2
  echo "These still hold hyperparameters from a selection round that was" >&2
  echo "discarded (HANDOFF_HP_SELECTION.md section 3.7).  Transcribe the" >&2
  echo "round-2 merged-sweep winner and replace the header first." >&2
  echo "Override with ALLOW_SUPERSEDED=1 only if you know why you want this." >&2
  exit 1
fi
if (( ${#MISSING_MEAS[@]} > 0 )); then
  echo "ERROR: measurement (tuning) set file missing for:" >&2
  printf '  %s\n' "${MISSING_MEAS[@]}" >&2
  echo "These *_tuning_set.hdf5 files are separate from the eval/ splits and, like all" >&2
  echo "of data/, are gitignored -- copy them to the box (or set measurement_dataset)." >&2
  exit 1
fi
if [[ -n "$COND" ]]; then
  # Experiment conditions: the job list, its order and every check come from
  # experiment_conditions.py, so the launcher and the verifier cannot disagree.
  pf=("$COND_PY" preflight "$METHOD" --variants "$VARIANTS" --seeds "$SEEDS"
      --conds "$COND" --data-root "${ROOT}/data/antmaze"
      --models-root "${ROOT}/../exp/reward_learning" --num-chains "$NUM_CHAINS")
  [[ "${ALLOW_EXISTING:-0}" == "1" ]] && pf+=(--allow-existing)
  echo "Condition(s): $COND"
  if ! "$PY" "${pf[@]}"; then
    echo "ERROR: experiment preflight failed -- nothing was launched." >&2
    exit 1
  fi
  JOBS=()
  while read -r c v s; do
    [[ -n "$c" ]] && JOBS+=("$c $v $s")
  done < <("$PY" "$COND_PY" jobs "$METHOD" --variants "$VARIANTS" --seeds "$SEEDS" --conds "$COND")
  (( ${#JOBS[@]} > 0 )) || { echo "ERROR: no jobs for COND='$COND'" >&2; exit 1; }
fi
echo "Root (cwd): $ROOT   data_root: ${ROOT}/data/antmaze"
echo "Method: $METHOD   variants: $VARIANTS   seeds: $SEEDS"
echo "Total jobs: ${#JOBS[@]}   slots: $NSLOTS   log dir: $LOGDIR"
echo

# --- dispatch: keep NSLOTS jobs in flight; reuse a slot when its job exits ---
declare -A SLOT_PID=()      # slot index -> pid of running job (unset if free)
declare -A PID_SLOT=()      # pid -> slot index

# Fill CMD (the training command) and LOGF for one job.  $1=cond  $2=variant  $3=seed
CMD=(); LOGF=""
job_cmd() {
  local c="$1" v="$2" s="$3"
  # Absolute path: pyrallis opens config_path relative to the process CWD, so an
  # absolute path makes config loading independent of where the run happens.
  local cfg="${ROOT}/${CFG_DIR}/antmaze_${v}${CFG_SUF}"
  LOGF="$LOGDIR/${v}_seed${s}.log"
  # Absolute --data_root as well: the configs build train/val/test paths from it
  # (default "data/antmaze"), and pyrallis/h5py open those relative to the CWD.
  # Passing the absolute root makes the data paths resolve independent of CWD.
  # Same for the BNN measurement_dataset (empty flag for MR/PT, which lack it).
  local meas; meas=$(abs_meas "$cfg")
  local -a mflag=(); [[ -n "$meas" ]] && mflag=(--measurement_dataset "$meas")
  # An experiment condition adds the split overrides and the output directory,
  # last on the command line.  The anchor models (cond "-") add nothing.
  local -a cargs=()
  if [[ "$c" != "-" ]]; then
    mapfile -t cargs < <("$PY" "$COND_PY" args "$METHOD" "$v" "$s" "$c" --data-root "${ROOT}/data/antmaze")
    (( ${#cargs[@]} > 0 )) || { echo "ERROR: no arguments for condition $c ($v seed $s)" >&2; exit 1; }
    LOGF="$LOGDIR/${c}/${v}_seed${s}.log"
  fi
  CMD=("$PY" "$SCRIPT" --config_path "$cfg" --data_root "${ROOT}/data/antmaze" "${mflag[@]}"
       --seed "$s" "${EXTRA_ARGS[@]}" "${cargs[@]}")
}

launch() {  # $1=slot_index  $2=cond  $3=variant  $4=seed
  local slot="$1" c="$2" v="$3" s="$4"
  local gpus="${SLOTS[$slot]}"
  job_cmd "$c" "$v" "$s"
  local logf="$LOGF"
  mkdir -p "$(dirname "$logf")"
  echo "[gpu ${gpus}] ${METHOD} ${c/#-/anchor} ${v} seed=${s}  -> ${logf}"
  CUDA_VISIBLE_DEVICES="$gpus" nohup "${CMD[@]}" > "$logf" 2>&1 &
  local pid=$!
  SLOT_PID[$slot]=$pid
  PID_SLOT[$pid]=$slot
}

free_slot() {  # echo index of a free slot, or nothing
  local i
  for (( i=0; i<NSLOTS; i++ )); do
    [[ -z "${SLOT_PID[$i]:-}" ]] && { echo "$i"; return 0; }
  done
  # No free slot: still exit 0. Otherwise the C-style for loop's final failing
  # test would make this function return 1, and `slot=$(free_slot)` under
  # `set -e` would abort the whole launcher the moment all slots are busy.
  return 0
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY RUN -- nothing is launched.  ${#JOBS[@]} job(s), ${NSLOTS} at a time, in this order:"
  for j in "${JOBS[@]}"; do
    read -r c v s <<< "$j"
    job_cmd "$c" "$v" "$s"
    printf '  [%s %s seed %s] ' "${c/#-/anchor}" "$v" "$s"
    printf '%q ' "${CMD[@]}"
    printf '> %s\n' "$LOGF"
  done
  exit 0
fi

FAILED=0
DONE=0
ji=0
NJOBS=${#JOBS[@]}
while (( ji < NJOBS )) || (( ${#PID_SLOT[@]} > 0 )); do
  # fill free slots
  while (( ji < NJOBS )); do
    slot=$(free_slot)
    [[ -z "$slot" ]] && break
    read -r c v s <<< "${JOBS[$ji]}"
    launch "$slot" "$c" "$v" "$s"
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
