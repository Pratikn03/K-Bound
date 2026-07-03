#!/usr/bin/env bash
# Run ImageNet-C with all 4 candidates incl. SHOT, AFTER the GPU frees (one MPS job at a time).
set -u
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"
REPO=/Volumes/T9/uav/AutoML_Flagship_V8
PY="$HOME/.venv_wilds/bin/python"
OUT="$HOME/kbound_inr_results/imagenetc_noise_4methods"
QLOG="$OUT/queue.log"; mkdir -p "$OUT"
export MPLCONFIGDIR="$HOME/.cache/mpl" PYTHONUNBUFFERED=1 PYTHONPATH=.
cd "$REPO" || exit 1
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }
log "SHOT queue: waiting for any MPS job (panel / cifar_tent) to finish..."
while pgrep -f "run_imagenetr_kbound.py" >/dev/null 2>&1 || pgrep -f "cifar_tent_mps_v2.py" >/dev/null 2>&1; do sleep 120; done
log "GPU free -> ImageNet-C 4-method sweep (tent eata sar shot), ResNet-50"
caffeinate -is "$PY" docs/research/kbound/scripts/cifar_tent_mps_v2.py \
  --benchmarks imagenetc --imagenetc-root experiments/kbound/data/imagenet-c \
  --corruptions gaussian_noise shot_noise impulse_noise \
  --arch resnet50 --methods tent eata sar shot \
  --out-results "$OUT" --cooldown 3 --max-images 2000 >> "$QLOG" 2>&1
log "SHOT sweep exit=$? -> $OUT"
