#!/bin/bash
# supervise_oh.sh ROLE RUN_NAME [extra args...]
# Auto-restart wrapper for the Office-Home K-Bound sweep (MPS-survival harness):
# the runner resumes from _partial.json, so on any OOM/crash we just relaunch.
set -u
ROLE="$1"; RUN="$2"; shift 2
# Repo root from this script's own location (experiments/kbound/officehome/ ->
# three levels up) and interpreter from $KBOUND_PYTHON or PATH, so the harness
# is not tied to one machine.  Both were hard-coded until 2026-07-26
# (fix-queue item 30 / defect D8).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${KBOUND_PYTHON:-$(command -v python3)}"
OUT="$REPO/experiments/kbound/results/$RUN"
mkdir -p "$OUT"
DONE="$OUT/.${ROLE}.done"
# A marker from an earlier invocation must never survive into a new attempt.
rm -f "$DONE"
MAXTRIES=40
for i in $(seq 1 $MAXTRIES); do
  echo "[supervise] attempt $i/$MAXTRIES role=$ROLE run=$RUN $(date)"
  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is "$PY" \
    "$REPO/experiments/kbound/officehome/run_officehome_kbound.py" \
    --role "$ROLE" --run-name "$RUN" "$@"
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[supervise] role=$ROLE completed rc=0 on attempt $i"; touch "$DONE"; exit 0
  fi
  echo "[supervise] rc=$rc — relaunching in 8s (resume from _partial.json)"; sleep 8
done
echo "[supervise] gave up after $MAXTRIES attempts"; exit 1
