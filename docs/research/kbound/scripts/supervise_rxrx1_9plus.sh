#!/bin/zsh
# Auto-restart supervisor for the RxRx1 K-Bound 9+ protocol (16 GB MPS survival).
# Generalizes supervise_rxrx1.sh to the LOCKED multi-model-seed 9+ grid. For EACH
# model seed it relaunches the runner with --resume whenever it dies (e.g. OOM-kill)
# BEFORE writing that seed's hash-bound completion receipt. A receipt is accepted
# only when the runner verifies its result hash and complete ledger, so an
# incomplete-but-exit-0 pass is also retried. The runner skips completed cells and
# APPENDS to _partial.json (atomic flush, never truncated); run.log opened append (>>).
# SINGLE-INSTANCE LOCK: refuses to start if another 9+ supervisor is already alive.
# --- interpreter: $KBOUND_PYTHON, default python3 (was a hard-coded venv path).
# --- external (git-excluded) data volume: ONE documented variable, no default.
# --- defect D8: portable roots. No machine-local absolute paths in tracked code
# --- (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md). KB_REPO_ROOT is discovered
# --- from this script's own location; override with KBOUND_REPO_ROOT.
_kb_find_root() {
  d=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
  while [ "$d" != "/" ]; do
    [ -f "$d/pyproject.toml" ] && { printf '%s\n' "$d"; return 0; }
    d=$(dirname "$d")
  done
  echo "ERROR: repository root not found above $(dirname "${BASH_SOURCE[0]:-$0}")" >&2
  return 1
}
KB_REPO_ROOT="${KBOUND_REPO_ROOT:-$(_kb_find_root)}" || exit 1

: "${KBOUND_EXTERNAL_ROOT:?set KBOUND_EXTERNAL_ROOT to the volume holding the git-excluded datasets/checkpoints/caches (layout: docs/research/kbound/kbound_repro/paths.py, acquisition: DATA.md)}"
KB_EXTERNAL_ROOT="$KBOUND_EXTERNAL_ROOT"

KB_PYTHON="${KBOUND_PYTHON:-python3}"

set -u
PYBIN="$KB_PYTHON"
REPO="$KB_REPO_ROOT"
RUNNER="$REPO/experiments/kbound/wilds/run_rxrx1_kbound.py"
DATA_ROOT="$KB_EXTERNAL_ROOT/kbound_rxrx1_data"
CKPT_ROOT="$KB_EXTERNAL_ROOT/kbound_rxrx1_ckpt"
RESULTS_ROOT="$REPO/experiments/kbound/results"
RUN_TAG=rxrx1_protocol_c_9plus
MODEL_SEEDS=(${=RXRX1_MODEL_SEEDS:-0 1 2})
MAX=200
SUPLOG="$RESULTS_ROOT/${RUN_TAG}_supervisor.log"
LOCK="$RESULTS_ROOT/${RUN_TAG}_supervisor.lock"
ALLDONE_PATH="$RESULTS_ROOT/${RUN_TAG}_ALL.done"
mkdir -p "$RESULTS_ROOT"
rm -f "$ALLDONE_PATH"
# --- single-instance guard ---
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[sup9] another supervisor PID $(cat "$LOCK") already alive; exiting $(date)" >> "$SUPLOG"
  exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
echo "[sup9] START $(date) pid=$$ host=$(hostname) seeds=${MODEL_SEEDS[*]}" >> "$SUPLOG"

set_seed_context() {
  S="$1"
  RUN_NAME="${RUN_TAG}_modelseed${S}"
  OUT="$RESULTS_ROOT/$RUN_NAME"
  DONE="$OUT/.done"
  RUNLOG="$OUT/run.log"
  CKPT="$CKPT_ROOT/rxrx1_seed:${S}_epoch:best_model.pth"
  RUN_ARGS=(
    --data-root "$DATA_ROOT" --ckpt "$CKPT" --model-seed "$S" --split test
    --seeds 0 1 2 3 4 5 6 7 8 9
    --compositions iid imbalanced single_class
    --batch-regimes small tiny --aggressiveness mild aggressive
    --n-eval 512 --n-batches 4 --episodic-steps 5 --episodic-batch 64
    --tau-star 0.52 --kappa 2.5 --sd-L 0.6 --delta 0.05 --device auto
    --results-root "$RESULTS_ROOT" --run-name "$RUN_NAME"
  )
}

completion_receipt_valid() {
  [ -f "$DONE" ] || return 1
  "$PYBIN" "$RUNNER" --verify-completion "$DONE" "${RUN_ARGS[@]}" >> "$SUPLOG" 2>&1
}

for S in "${MODEL_SEEDS[@]}"; do
  set_seed_context "$S"
  mkdir -p "$OUT"
  if [ ! -f "$CKPT" ]; then
    echo "[sup9] seed $S MISSING ckpt $CKPT; skipping" >> "$SUPLOG"; continue
  fi
  discard_stale_receipt() {
    [ -f "$DONE" ] || return 0
    rm -f "$DONE"
    echo "[sup9] seed $S removed invalid/stale completion receipt $(date)" >> "$SUPLOG"
  }
  # kill any orphan runner for THIS run-name before (re)starting the single owner
  pkill -f "run_rxrx1_kbound.py .*--run-name $RUN_NAME" 2>/dev/null
  i=0
  while [ $i -lt $MAX ]; do
    i=$((i+1))
    if completion_receipt_valid; then
      echo "[sup9] seed $S valid completion receipt before attempt $i; next seed $(date)" >> "$SUPLOG"; break
    fi
    discard_stale_receipt
    echo "[sup9] seed $S attempt $i: launching runner $(date)" >> "$SUPLOG"
    caffeinate -is "$PYBIN" "$RUNNER" "${RUN_ARGS[@]}" --resume >> "$RUNLOG" 2>&1
    code=$?
    echo "[sup9] seed $S attempt $i: runner exited code=$code $(date)" >> "$SUPLOG"
    if completion_receipt_valid; then
      echo "[sup9] seed $S valid completion receipt after attempt $i; next seed $(date)" >> "$SUPLOG"; break
    fi
    discard_stale_receipt
    echo "[sup9] seed $S exited without a valid completion receipt; resuming in 5s" >> "$SUPLOG"
    sleep 5
  done
done

echo "[sup9] ALL SEEDS PROCESSED $(date)" >> "$SUPLOG"
# The overall marker is only a convenience signal for this supervisor invocation.
# Every per-seed receipt is revalidated first, and any stale overall marker was
# removed at startup.
ALLDONE=1
for S in "${MODEL_SEEDS[@]}"; do
  set_seed_context "$S"
  completion_receipt_valid || ALLDONE=0
done
if [ $ALLDONE -eq 1 ]; then
  echo "$(date) all model seeds validated: ${MODEL_SEEDS[*]}" > "$ALLDONE_PATH"
fi
echo "[sup9] EXIT $(date) all_done=$ALLDONE attempts_last=$i" >> "$SUPLOG"
