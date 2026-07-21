#!/usr/bin/env bash
# Grad-clip instrumentation readout (Issue 3, Step 1).
#
# Runs all four variants concurrently on disjoint GPUs with a reduced budget,
# wandb disabled, then prints the aggregated pre-clip grad-norm table.  The
# decisive number is the sampling-phase "%>clip": ~0 means the clip is inert
# during sampling (CVaR tail unbiased); >0 means it fires (proceed to BT /T).
#
# Run from anywhere; must be on the GPU box with the `irl` env active (or set
# PY=/path/to/irl/python).  6 A6000s assumed: large variants get 2 GPUs each
# (1 chain/GPU), medium variants 1 GPU each (2 chains/GPU).
set -u

cd "$(dirname "$0")/../.." || exit 1          # repo root
REPO="$(pwd)"
PY="${PY:-python}"                            # override with PY=... if needed
SCRIPT="scripts_bnn/run_bnn_training_antmaze_eval.py"
CFGDIR="scripts_bnn/gradnorm_readout"
LOGDIR="$CFGDIR/logs"
mkdir -p "$LOGDIR"

export WANDB_MODE=disabled
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

gpus_for() {   # GPU assignment: 4 variants over 6 GPUs (2 idle)
  case "$1" in
    large_play)     echo "0,1" ;;
    large_diverse)  echo "2,3" ;;
    medium_play)    echo "4"   ;;
    medium_diverse) echo "5"   ;;
  esac
}

pids=""
for task in large_play large_diverse medium_play medium_diverse; do
  gpus="$(gpus_for "$task")"
  echo "[launch] $task on GPU(s) $gpus  -> $LOGDIR/$task.log"
  CUDA_VISIBLE_DEVICES="$gpus" nohup "$PY" "$SCRIPT" \
      --config_path "$CFGDIR/${task}_readout.yaml" \
      > "$LOGDIR/$task.log" 2>&1 &
  pids="$pids $!"
done

echo "[launch] PIDs:$pids — waiting..."
wait $pids
echo "[launch] all runs finished; aggregating grad-norm stats"
"$PY" "$CFGDIR/read_results.py"
