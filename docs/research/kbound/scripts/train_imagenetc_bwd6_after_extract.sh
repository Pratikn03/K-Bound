#!/usr/bin/env bash
# Wait for requested ImageNet-C groups to be extracted, then run the six
# requested corruptions after the current v5 overnight queue stops using MPS.
set -uo pipefail

REPO="${REPO:-/Volumes/T9/uav/AutoML_Flagship_V8}"
ROOT="${ROOT:-$REPO/experiments/kbound/data/imagenet-c}"
OUT="${OUT:-$REPO/experiments/kbound/results/win_hunt_v5/imagenetc_bwd6_aggr}"
EXTRACT_LOG="${EXTRACT_LOG:-/Volumes/T9/uav/imagenetc_extract_requested_groups.log}"

cd "$REPO" || exit 1

echo "[$(date '+%F %T')] wait job started"
echo "root=$ROOT"
echo "out=$OUT"

while true; do
  if [ -f "$ROOT/.done_blur" ] && [ -f "$ROOT/.done_weather" ] && [ -f "$ROOT/.done_digital" ]; then
    echo "[$(date '+%F %T')] extraction markers present"
    break
  fi
  if grep -q '^!!' "$EXTRACT_LOG" 2>/dev/null; then
    echo "[$(date '+%F %T')] extraction log reports failure; not starting training"
    tail -40 "$EXTRACT_LOG" || true
    exit 2
  fi
  echo "[$(date '+%F %T')] waiting for .done_blur/.done_weather/.done_digital; free=$(df -g "$ROOT" | awk 'NR==2{print $4}')GB"
  sleep 300
done

while pgrep -f "bash docs/research/kbound/gapclose_wave5/run_v5_overnight.sh" >/dev/null; do
  echo "[$(date '+%F %T')] extraction done; waiting for current v5 queue to finish before using MPS"
  sleep 300
done

source ~/.venv_wilds/bin/activate
mkdir -p "$OUT"

echo "[$(date '+%F %T')] starting requested-six ImageNet-C run"
python docs/research/kbound/scripts/cifar_tent_mps_v2.py \
  --benchmarks imagenetc \
  --imagenetc-root "$ROOT" \
  --corruptions defocus_blur motion_blur snow fog brightness jpeg_compression \
  --severities 1 2 3 4 5 \
  --methods tent eata sar \
  --device mps \
  --seed 0 \
  --batch-regimes small \
  --aggressiveness aggressive \
  --adapt-lr 0.004 \
  --imagenetc-composition iid imbalanced single_class \
  --out-results "$OUT"

rc=$?
echo "[$(date '+%F %T')] requested-six ImageNet-C run finished rc=$rc"
exit "$rc"
