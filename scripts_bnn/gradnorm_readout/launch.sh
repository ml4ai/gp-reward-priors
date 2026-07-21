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
#
# Paths mirror train_rewards.sh: absolute --config_path, --data_root, and
# --measurement_dataset, because pyrallis/h5py open them relative to the CWD.
# data/ is gitignored, so its eval splits + *_tuning_set.hdf5 must be copied to
# this box under $REPO/data/antmaze (this script preflights that they exist).
set -u

cd "$(dirname "$0")/../.." || exit 1          # repo root
REPO="$(pwd)"
SCRIPT="$REPO/scripts_bnn/run_bnn_training_antmaze_eval.py"
CFGDIR="$REPO/scripts_bnn/gradnorm_readout"
LOGDIR="$CFGDIR/logs"
DATA_ROOT="$REPO/data/antmaze"
mkdir -p "$LOGDIR"

export WANDB_MODE=disabled
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

cfg_field() { sed -n "s/^[[:space:]]*$2:[[:space:]]*//p" "$1" | awk '{print $1}' | head -1; }
abs_path()  { case "$1" in /*) echo "$1" ;; *) echo "$REPO/$1" ;; esac; }

# --- resolve interpreter: PY override, else python, else python3 --------------
if [ -n "${PY:-}" ]; then :; elif command -v python >/dev/null 2>&1; then PY=python;
elif command -v python3 >/dev/null 2>&1; then PY=python3; else
  echo "ERROR: no python on PATH. Activate the env with torch+optbnn, or set PY=." >&2; exit 1
fi

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

# --- preflight configs + DATA (eval split + measurement set) -----------------
tasks="large_play large_diverse medium_play medium_diverse"
_missing=0
for task in $tasks; do
  cfg="$CFGDIR/${task}_readout.yaml"
  if [ ! -f "$cfg" ]; then
    echo "ERROR: missing config $cfg — did you 'git pull' on this machine?" >&2; exit 1
  fi
  variant="$(cfg_field "$cfg" antmaze_variant)"
  seed="$(cfg_field "$cfg" seed)"
  meas="$(abs_path "$(cfg_field "$cfg" measurement_dataset)")"
  train_file="$DATA_ROOT/$variant/eval/seed_$seed/${variant}_pref_train_$seed.hdf5"
  for f in "$train_file" "$meas"; do
    if [ ! -f "$f" ]; then echo "  MISSING ($task): $f" >&2; _missing=1; fi
  done
done
if [ "$_missing" -eq 1 ]; then
  echo "ERROR: required data files are missing on this box.  data/ is gitignored," >&2
  echo "  so copy the eval/ splits and *_tuning_set.hdf5 into $DATA_ROOT." >&2
  exit 1
fi

pids=""
for task in $tasks; do
  gpus="$(gpus_for "$task")"
  cfg="$CFGDIR/${task}_readout.yaml"
  meas="$(abs_path "$(cfg_field "$cfg" measurement_dataset)")"
  echo "[launch] $task on GPU(s) $gpus  -> $LOGDIR/$task.log"
  CUDA_VISIBLE_DEVICES="$gpus" nohup "$PY" "$SCRIPT" \
      --config_path "$cfg" \
      --data_root "$DATA_ROOT" \
      --measurement_dataset "$meas" \
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
