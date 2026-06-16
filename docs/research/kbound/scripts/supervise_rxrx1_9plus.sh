#!/bin/zsh
# Auto-restart supervisor for the RxRx1 K-Bound 9+ protocol (16 GB MPS survival).
# Generalizes supervise_rxrx1.sh to the LOCKED multi-model-seed 9+ grid. For EACH
# model seed it relaunches the runner with --resume whenever it dies (e.g. OOM-kill)
# BEFORE writing that seed's .done sentinel. Keys on .done (NOT exit code), so an
# incomplete-but-exit-0 pass is also retried. The runner skips completed cells and
# APPENDS to _partial.json (atomic flush, never truncated); run.log opened append (>>).
# SINGLE-INSTANCE LOCK: refuses to start if another 9+ supervisor is already alive.
set -u
PYBIN=/Users/pratik_n/.venv_wilds/bin/python
REPO=/Volumes/T9/uav/AutoML_Flagship_V8
RUNNER="$REPO/experiments/kbound/wilds/run_rxrx1_kbound.py"
DATA_ROOT=/Users/pratik_n/kbound_rxrx1_data
CKPT_ROOT=/Users/pratik_n/kbound_rxrx1_ckpt
RESULTS_ROOT="$REPO/experiments/kbound/results"
RUN_TAG=rxrx1_protocol_c_9plus
MODEL_SEEDS=(${=RXRX1_MODEL_SEEDS:-0 1 2})
MAX=200
SUPLOG="$RESULTS_ROOT/${RUN_TAG}_supervisor.log"
LOCK="$RESULTS_ROOT/${RUN_TAG}_supervisor.lock"
mkdir -p "$RESULTS_ROOT"
# --- single-instance guard ---
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[sup9] another supervisor PID $(cat "$LOCK") already alive; exiting $(date)" >> "$SUPLOG"
  exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
echo "[sup9] START $(date) pid=$$ host=$(hostname) seeds=${MODEL_SEEDS[*]}" >> "$SUPLOG"

for S in "${MODEL_SEEDS[@]}"; do
  RUN_NAME="${RUN_TAG}_modelseed${S}"
  OUT="$RESULTS_ROOT/$RUN_NAME"
  DONE="$OUT/.done"
  RUNLOG="$OUT/run.log"
  CKPT="$CKPT_ROOT/rxrx1_seed:${S}_epoch:best_model.pth"
  mkdir -p "$OUT"
  if [ ! -f "$CKPT" ]; then
    echo "[sup9] seed $S MISSING ckpt $CKPT; skipping" >> "$SUPLOG"; continue
  fi
  # kill any orphan runner for THIS run-name before (re)starting the single owner
  pkill -f "run_rxrx1_kbound.py .*--run-name $RUN_NAME" 2>/dev/null
  i=0
  while [ $i -lt $MAX ]; do
    i=$((i+1))
    if [ -f "$DONE" ]; then
      echo "[sup9] seed $S .done present before attempt $i; next seed $(date)" >> "$SUPLOG"; break
    fi
    echo "[sup9] seed $S attempt $i: launching runner $(date)" >> "$SUPLOG"
    caffeinate -is "$PYBIN" "$RUNNER" \
      --data-root "$DATA_ROOT" --ckpt "$CKPT" --split test \
      --seeds 0 1 2 3 4 5 6 7 8 9 \
      --compositions iid imbalanced single_class \
      --batch-regimes small tiny --aggressiveness mild aggressive \
      --n-eval 512 --n-batches 4 --episodic-steps 5 --episodic-batch 64 \
      --tau-star 0.52 --kappa 2.5 --sd-L 0.6 --delta 0.05 --device auto \
      --results-root "$RESULTS_ROOT" --run-name "$RUN_NAME" --resume >> "$RUNLOG" 2>&1
    code=$?
    echo "[sup9] seed $S attempt $i: runner exited code=$code $(date)" >> "$SUPLOG"
    if [ -f "$DONE" ]; then
      echo "[sup9] seed $S .done after attempt $i; next seed $(date)" >> "$SUPLOG"; break
    fi
    echo "[sup9] seed $S died before .done; resuming in 5s" >> "$SUPLOG"
    sleep 5
  done
done

echo "[sup9] ALL SEEDS PROCESSED $(date)" >> "$SUPLOG"
# overall completion marker iff every seed wrote .done
ALLDONE=1
for S in "${MODEL_SEEDS[@]}"; do
  [ -f "$RESULTS_ROOT/${RUN_TAG}_modelseed${S}/.done" ] || ALLDONE=0
done
if [ $ALLDONE -eq 1 ]; then
  echo "$(date) all model seeds: ${MODEL_SEEDS[*]}" > "$RESULTS_ROOT/${RUN_TAG}_ALL.done"
fi
echo "[sup9] EXIT $(date) all_done=$ALLDONE attempts_last=$i" >> "$SUPLOG"
