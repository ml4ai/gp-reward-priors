#!/usr/bin/env bash
# Medium-play lr_max mini-sweep (post-/T-fix divergence check).
#
# Runs each mp_lrmax*_readout.yaml concurrently, one GPU per rung (2 chains/GPU),
# wandb disabled, then prints the sampling-phase grad-norm table by lr_max.
# Run make_lrmax_sweep.py first.  Activate the env with torch+optbnn (e.g.
# `conda activate pt`), then run; override with PY=/path if needed.
set -u

cd "$(dirname "$0")/../.." || exit 1
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

if [ -n "${PY:-}" ]; then :; elif command -v python >/dev/null 2>&1; then PY=python;
elif command -v python3 >/dev/null 2>&1; then PY=python3; else
  echo "ERROR: no python on PATH. Activate the env with torch+optbnn, or set PY=." >&2; exit 1
fi
if ! "$PY" -c "import torch, optbnn" >/tmp/gradnorm_preflight.log 2>&1; then
  echo "ERROR: '$PY' cannot import torch + optbnn (activate the env, or set PY=)." >&2
  sed 's/^/  /' /tmp/gradnorm_preflight.log >&2; exit 1
fi
_ngpu="$("$PY" -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo '?')"
echo "[preflight] repo=$REPO  interp=$($PY -c 'import sys;print(sys.executable)')  GPUs=$_ngpu"

configs="$(ls "$CFGDIR"/mp_lrmax*_readout.yaml 2>/dev/null)"
[ -z "$configs" ] && { echo "ERROR: no mp_lrmax*_readout.yaml; run make_lrmax_sweep.py first." >&2; exit 1; }

# All rungs are medium-play -> one data preflight (eval split + tuning set).
_variant="$(cfg_field "$(echo "$configs" | head -1)" antmaze_variant)"
_seed="$(cfg_field "$(echo "$configs" | head -1)" seed)"
_meas="$(abs_path "$(cfg_field "$(echo "$configs" | head -1)" measurement_dataset)")"
_train="$DATA_ROOT/$_variant/eval/seed_$_seed/${_variant}_pref_train_$_seed.hdf5"
for f in "$_train" "$_meas"; do
  [ -f "$f" ] || { echo "ERROR: missing data $f (data/ is gitignored; copy it to the box)." >&2; exit 1; }
done

gpu=0; pids=""
for cfg in $configs; do
  if [ "$gpu" -ge "${_ngpu:-6}" ]; then
    echo "ERROR: more rungs than GPUs ($_ngpu); trim LR_MAX_LADDER." >&2; exit 1
  fi
  meas="$(abs_path "$(cfg_field "$cfg" measurement_dataset)")"
  log="$LOGDIR/$(basename "$cfg" .yaml).log"
  echo "[launch] $(basename "$cfg") lr_max=$(cfg_field "$cfg" sghmc_lr_max) on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" "$SCRIPT" \
      --config_path "$cfg" --data_root "$DATA_ROOT" --measurement_dataset "$meas" \
      > "$log" 2>&1 &
  pids="$pids $!"
  gpu=$((gpu + 1))
done

echo "[launch] waiting..."
wait $pids
echo "[launch] done; aggregating"
"$PY" "$CFGDIR/read_lrmax.py"
