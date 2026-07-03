#!/usr/bin/env bash
# Protocol L full pipeline launcher (FMoW or PovertyMap).
#
#   bash docs/research/kbound/scripts/run_protocol_L.sh fmow
#   bash docs/research/kbound/scripts/run_protocol_L.sh poverty
#
# Phases: download (optional) -> dev GPU -> screen -> full val/test GPU -> analyze_F
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO"
PY="${KB_VENV:-$HOME/.venv_wilds}/bin/python"
DATASET="${1:?usage: run_protocol_L.sh fmow|poverty}"
DEVICE="${DEVICE:-mps}"
DOWNLOAD="${DOWNLOAD:-0}"

RUNNER="experiments/kbound/wilds/run_geoshift_kbound.py"
SCREEN="docs/research/kbound/scripts/screen_protocol_L.py"
ANALYZE="docs/research/kbound/scripts/analyze_F.py"

if [[ "$DOWNLOAD" == 1 ]]; then
  bash docs/research/kbound/scripts/download_wilds_fmow_poverty.sh
fi

echo "=== Phase 1: smoke (requires data present) ==="
PYTORCH_ENABLE_MPS_FALLBACK=1 "$PY" -u "$RUNNER" --dataset "$DATASET" --device "$DEVICE" --smoke

echo "=== Phase 2: dev screen on id_val (seeds 0-2) ==="
DEV_RUN="${DATASET}_protocol_L_dev"
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is "$PY" -u "$RUNNER" \
  --dataset "$DATASET" --device "$DEVICE" --split id_val \
  --run-name "$DEV_RUN" --seeds 0 1 2 \
  --max-groups 4 --n-eval 48 --max-train-batches 120

REC=$(ls -t experiments/kbound/results/"$DEV_RUN"/result_*.json | head -1)
if ! "$PY" "$SCREEN" --records "$REC" --candidate sar_online --dev-seeds 0 1 2; then
  echo "STOP: dev screen failed for $DATASET — do not run full GPU"
  exit 1
fi

echo "=== Phase 3: full val (source=id_val implicit in records; target=val) ==="
VAL_RUN="${DATASET}_protocol_L_val"
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is "$PY" -u "$RUNNER" \
  --dataset "$DATASET" --device "$DEVICE" --split val \
  --run-name "$VAL_RUN" --seeds 0 1 2 3 4 \
  --max-groups 6 --n-eval 64

echo "=== Phase 4: held-out test ==="
TEST_RUN="${DATASET}_protocol_L_test"
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is "$PY" -u "$RUNNER" \
  --dataset "$DATASET" --device "$DEVICE" --split test \
  --run-name "$TEST_RUN" --seeds 0 1 2 3 4 \
  --max-groups 6 --n-eval 64

TEST_REC=$(ls -t experiments/kbound/results/"$TEST_RUN"/result_*.json | head -1)
OUT="experiments/kbound/results/${DATASET}_protocol_L_v1"
mkdir -p "$OUT"
"$PY" "$ANALYZE" --records "$TEST_REC" --candidate sar_online \
  --estimator gbr --conformal global --dev-seeds 0 1 --test-seeds 2 3 4 \
  --output-dir "$OUT"

echo "Protocol L complete for $DATASET -> $OUT"
