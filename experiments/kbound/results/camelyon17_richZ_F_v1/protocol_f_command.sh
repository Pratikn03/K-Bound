#!/usr/bin/env bash
set -euo pipefail
cd /Volumes/T9/uav/AutoML_Flagship_V8
source /Users/pratik_n/.venv_wilds/bin/activate
export TMPDIR=/Volumes/T9/uav/tmp
export TORCH_HOME=/Volumes/T9/uav/torch_cache
mkdir -p "$TMPDIR" "$TORCH_HOME"
caffeinate -is python experiments/kbound/wilds/run_camelyon17_kbound.py \
  --data-root /Users/pratik_n/datasets/wilds \
  --f0-template 'experiments/kbound/results/camelyon17_fullscale_B_v1/f0_seed{seed}.pt' \
  --seeds 0 1 2 3 4 \
  --domains test val id_val \
  --compositions iid imbalanced single_class \
  --batch-regimes small \
  --aggressiveness mild aggressive \
  --n-eval 1024 \
  --n-batches 4 \
  --tau-star 0.52 \
  --kappa 2.5 \
  --sd-L 0.6 \
  --delta 0.05 \
  --device auto \
  --evidence-panel rich \
  --run-name camelyon17_richZ_F_v1
latest=$(ls -t experiments/kbound/results/camelyon17_richZ_F_v1/result_*.json | head -1)
python docs/research/kbound/scripts/analyze_F.py \
  --records "$latest" \
  --output-dir experiments/kbound/results/camelyon17_richZ_F_v1 \
  --estimator ppi_debias \
  --conformal mondrian \
  --dev-seeds 0 1 \
  --test-seeds 2 3 4
