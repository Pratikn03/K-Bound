#!/bin/zsh
# Auto-restart supervisor for the RxRx1 K-Bound light sweep (16 GB MPS survival).
# Relaunches the runner with --resume whenever it dies (e.g. OOM-kill) BEFORE writing
# the .done sentinel. The runner skips completed cells and APPENDS; run.log is opened
# in append mode here (>>), so nothing is ever truncated across restarts.
# SINGLE-INSTANCE LOCK: refuses to start if another supervisor is already alive.
PYBIN=/Users/pratik_n/.venv_wilds/bin/python
SCRIPT="/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/wilds/run_rxrx1_kbound.py"
RUN_NAME=rxrx1_kbound_light_mps_internal
RESULTS_ROOT=/Users/pratik_n/kbound_rxrx1_results
OUT="$RESULTS_ROOT/$RUN_NAME"
DONE="$OUT/.done"
RUNLOG="$OUT/run.log"
SUPLOG="$OUT/supervisor.log"
LOCK="$OUT/supervisor.lock"
MAX=200
mkdir -p "$OUT"
# --- single-instance guard ---
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[supervisor] another supervisor PID $(cat "$LOCK") already alive; exiting $(date)" >> "$SUPLOG"
  exit 0
fi
# kill any orphan runners from a prior crash before we (re)start the single owner
pkill -f "run_rxrx1_kbound.py --resume --run-name $RUN_NAME" 2>/dev/null
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
echo "[supervisor] START $(date) pid=$$ host=$(hostname)" >> "$SUPLOG"
i=0
while [ $i -lt $MAX ]; do
  i=$((i+1))
  if [ -f "$DONE" ]; then echo "[supervisor] .done present before attempt $i; stop $(date)" >> "$SUPLOG"; break; fi
  echo "[supervisor] attempt $i: launching runner $(date)" >> "$SUPLOG"
  "$PYBIN" "$SCRIPT" --resume --run-name "$RUN_NAME" --results-root "$RESULTS_ROOT" >> "$RUNLOG" 2>&1
  code=$?
  echo "[supervisor] attempt $i: runner exited code=$code $(date)" >> "$SUPLOG"
  if [ -f "$DONE" ]; then echo "[supervisor] .done present after attempt $i; stop $(date)" >> "$SUPLOG"; break; fi
  echo "[supervisor] runner died before .done; resuming in 5s" >> "$SUPLOG"
  sleep 5
done
echo "[supervisor] EXIT $(date) attempts=$i" >> "$SUPLOG"
