#!/bin/bash
# ImageNet-C v5 — 1% subsample on MPS (GPU). Full 27-cell grid (3 noise x 3 sev x 3 comp x 3 methods),
# but ~50 images/cell instead of the full pool -> cells finish in seconds.
# This is a REDUCED-SCALE / directional run: it proves MPS works and gives a directional signal.
# It is NOT the frozen full-scale ImageNet-C number and must be reported as a 1% subsample if used.
set -u
cd /Volumes/T9/uav/AutoML_Flagship_V8 || exit 1
source ~/.venv_wilds/bin/activate
export TORCH_HOME=/Volumes/T9/uav/torch_cache

echo "== kill any lingering imagenet-c run =="
pkill -9 -f cifar_tent_mps_v2 2>/dev/null || true
sleep 2

OUT=experiments/kbound/results/win_hunt_v5/imagenetc_aggr_1pct
rm -rf "$OUT"

echo "== 1% MPS run: 3 noise x 3 sev x 3 comp x 3 methods, ~50 imgs/cell, caffeinated =="
caffeinate -is python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks imagenetc --methods tent eata sar --device mps --seed 0 --imagenetc-root "$HOME/imagenetc_local" --corruptions gaussian_noise shot_noise impulse_noise --arch resnet50 --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --imagenetc-composition iid imbalanced single_class --max-images 50 --out-results "$OUT"
