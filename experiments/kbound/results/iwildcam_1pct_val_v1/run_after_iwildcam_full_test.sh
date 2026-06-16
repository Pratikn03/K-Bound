#!/usr/bin/env bash
set -euo pipefail

cd /Volumes/T9/uav/AutoML_Flagship_V8

WAIT_PATTERN='run_iwildcam_kbound.py .*--run-name iwildcam_full_test'
echo "[iwildcam-1pct] queued at $(date)"
while pgrep -f "$WAIT_PATTERN" >/dev/null 2>&1; do
  echo "[iwildcam-1pct] waiting for active iwildcam_full_test at $(date)"
  sleep 60
done

echo "[iwildcam-1pct] starting at $(date)"
/Users/pratik_n/.venv_wilds/bin/python -u experiments/kbound/wilds/run_iwildcam_kbound.py \
  --device mps \
  --split val \
  --results-root experiments/kbound/results \
  --run-name iwildcam_1pct_val_v1 \
  --backbone resnet18 \
  --trainable head \
  --max-train-batches 56 \
  --train-bs 32 \
  --balanced-train \
  --max-locations 8 \
  --n-eval 48 \
  --n-batches 2 \
  --compositions iid imbalanced single_class \
  --batch-regimes tiny small \
  --aggressiveness mild aggressive \
  --candidates tent_online eata_online sar_online tent_episodic eata_episodic sar_episodic \
  --seeds 0 \
  --tau-star 0.52 \
  --eval-bs 48 \
  --episodic-batch 24 \
  --episodic-steps 3 \
  --retrain
echo "[iwildcam-1pct] finished at $(date)"
