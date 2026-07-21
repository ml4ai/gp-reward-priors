#!/usr/bin/env bash
# Grad-clip instrumentation readout (Issue 3, Step 1).
#
# Runs all four variants concurrently on disjoint GPUs with a reduced budget,
# wandb disabled, then prints the aggregated pre-clip grad-norm table.  The
# decisive number is the sampling-phase "%>clip": ~0 means the clip is inert
# during sampling (CVaR tail unbiased); >0 means it fires (proceed to BT /T).
#
# ACTIVATE the conda env that has torch + optbnn first (e.g. `conda activate pt`),
# then just run this — it uses `python` on PATH.  Override with PY=/path if needed.
# 6 A6000s assumed: large variants 2 GPUs each (1 chain/GPU), medium 1 GPU each.
set -u

cd "$(dirname "$0")/../.." || exit 1          # repo root
REPO="$(pwd)"
SCRIPT="$REPO/scripts_bnn/run_bnn_training_antmaze_eval.py"
CFGDIR="$REPO/scripts_bnn/gradnorm_readout"
LOGDIR="$CFGDIR/logs"
mkdir -p "$LOGDIR"

export WANDB_MODE=disabled
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# --- resolve interpreter: PY override, else python, else python3 --------------
if [ -n "${PY:-}" ]; then :; elif command -v python >/dev/null 2>&1; then PY=python;
elif command -v python3 >/dev/null 2>&1; then PY=python3; else
  echo "ERROR: no python on PATH. Activate the env with torch+optbnn, or set PY=." >&2; exit 1
fi

# --- preflight: interpreter must import torch + optbnn ------------------------
echo "[preflight] repo root : $REPO"
echo "[preflight] interpreter: $($PY -c 'import sys;print(sys.executable)' 2>/dev/null || echo "$PY (not runnable)")"
if ! "$PY" -c "import torch, optbnn" >/tmp/gradnorm_preflight.log 2>&1; then
  echo "ERROR: '$PY' cannot import torch + optbnn. This is why nothing ran." >&2
  echo "  Fix: activate the conda env that has them (e.g. 'conda activate pt')," >&2
  echo "       or run with PY=/path/to/that/env/bin/python" >&2
  echo "  ---- preflight error ----" >&2; sed 's/^/  /' /tmp/gradnorm_preflight.log >&2
  exit 1
fi
_ngpu="$("$PY" -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo '?')"
echo "[preflight] torch + optbnn OK; visible GPUs: $_ngpu (need 6 for the layout below)"

gpus_for() {   # GPU assignment: 4 variants over 6 GPUs (2 idle)
  case "$1" in
    large_play)     echo "0,1" ;;
    large_diverse)  echo "2,3" ;;
    medium_play)    echo "4"   ;;
    medium_diverse) echo "5"   ;;
  esac
}

# --- verify all configs are present (catches a missing git pull) -------------
tasks="large_play large_diverse medium_play medium_diverse"
for task in $tasks; do
  cfg="$CFGDIR/${task}_readout.yaml"
  if [ ! -f "$cfg" ]; then
    echo "ERROR: missing config $cfg — did you 'git pull' on this machine?" >&2; exit 1
  fi
done

pids=""
for task in $tasks; do
  gpus="$(gpus_for "$task")"
  cfg="$CFGDIR/${task}_readout.yaml"          # absolute path (CWD-independent)
  echo "[launch] $task on GPU(s) $gpus  -> $LOGDIR/$task.log"
  CUDA_VISIBLE_DEVICES="$gpus" nohup "$PY" "$SCRIPT" \
      --config_path "$cfg" \
      > "$LOGDIR/$task.log" 2>&1 &
  eval "pid_$task=$!"
  pids="$pids $!"
done

# --- grace check: catch processes that die on startup ------------------------
sleep 8
_dead=0
for task in $tasks; do
  eval "p=\$pid_$task"
  if ! kill -0 "$p" 2>/dev/null; then
    wait "$p" 2>/dev/null; rc=$?
    if [ "$rc" -ne 0 ]; then
      _dead=1
      echo "[error] $task (pid $p) exited early rc=$rc — last lines of its log:" >&2
      tail -n 15 "$LOGDIR/$task.log" | sed 's/^/    /' >&2
    fi
  fi
done
[ "$_dead" -eq 1 ] && echo "[error] at least one run died on startup; fix the above and re-run." >&2

echo "[launch] waiting for runs to finish (burn-in 5000 + 12 cycles)..."
wait $pids
echo "[launch] done; aggregating grad-norm stats"
"$PY" "$CFGDIR/read_results.py"
