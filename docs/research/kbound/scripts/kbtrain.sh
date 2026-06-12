#!/usr/bin/env bash
# Foolproof K-Bound training launcher. Run from ANY folder, e.g.:
#   bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/kbtrain.sh noise
#
# It always: cd's to the repo, activates ~/.venv_wilds (the one WITH torch+wilds),
# sets TMPDIR/TORCH_HOME to T9, wraps in caffeinate, and verifies the venv first.
set -uo pipefail
REPO=/Volumes/T9/uav/AutoML_Flagship_V8
VENV="$HOME/.venv_wilds"
IC=experiments/kbound/data/imagenet-c
IC_FAST="$HOME/kbound_inc"            # fast internal-SSD copy of ImageNet-C noise (prep_internal_noise.sh)
CAM_FAST="$HOME/kbound_cam/wilds"     # fast internal-SSD copy of Camelyon17 (prep_internal_camelyon.sh)
RES=experiments/kbound/results
S2=docs/research/kbound/scripts/cifar_tent_mps_v2.py
WILDS=docs/research/kbound/scripts/run_wilds_camelyon17.py

cd "$REPO" || { echo "ERROR: repo not found at $REPO"; exit 1; }
[ -d "$VENV" ] || { echo "ERROR: $VENV missing. Create it once with run_wilds.sh"; exit 1; }
source "$VENV/bin/activate"
export TMPDIR=/Volumes/T9/uav/tmp TORCH_HOME=/Volumes/T9/uav/torch_cache
mkdir -p "$TMPDIR" "$TORCH_HOME"
python -c "import torch, wilds" 2>/dev/null || { echo "ERROR: wrong venv (torch/wilds missing). Expected $VENV"; exit 1; }
echo ">> repo=$REPO  venv=$VENV  job=${1:-<none>}"

case "${1:-}" in
  noise)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_noise" ;;
  noise-full)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_noise_full" \
      --severities 1 2 3 4 5 --all-batch --max-images 2000 ;;
  noise-quick)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_noise" --quick ;;
  noise-smoke)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_smoke" \
      --quick --max-images 64 ;;
  camelyon)
    caffeinate -is python "$WILDS" --wilds-root experiments/kbound/data/wilds \
      --output-dir "$RES/wilds" --seeds 0 1 2 3 --epochs 4 --steps 10 --lr 1e-3 --retrain ;;
  camelyon-smoke)
    caffeinate -is python "$WILDS" --wilds-root experiments/kbound/data/wilds \
      --output-dir "$RES/wilds_smoke" --seeds 0 1 2 3 --epochs 1 --steps 5 --lr 1e-3 --frac 0.001 --retrain ;;
  camelyon-fast)
    [ -d "$CAM_FAST/camelyon17_v1.0/patches" ] || { echo "ERROR: internal copy not found at $CAM_FAST -- run prep_internal_camelyon.sh first."; exit 1; }
    caffeinate -is python "$WILDS" --wilds-root "$CAM_FAST" \
      --output-dir "$RES/wilds" --seeds 0 1 2 3 --epochs 4 --steps 10 --lr 1e-3 --retrain ;;
  camelyon-fast-1pct)
    [ -d "$CAM_FAST/camelyon17_v1.0/patches" ] || { echo "ERROR: internal copy not found at $CAM_FAST -- run prep_internal_camelyon.sh first."; exit 1; }
    caffeinate -is python "$WILDS" --wilds-root "$CAM_FAST" \
      --output-dir "$RES/wilds_1pct" --seeds 0 1 2 3 --epochs 1 --steps 5 --lr 1e-3 --frac 0.01 --retrain ;;
  noiseblur)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_noiseblur" ;;
  vit)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch vit_b16 --methods tent eata sar --out-results "$RES/imagenetc_noise_vit" ;;
  noise-fast)
    [ -d "$IC_FAST/gaussian_noise/1" ] || { echo "ERROR: internal copy not found at $IC_FAST -- run prep_internal_noise.sh first."; exit 1; }
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC_FAST" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_noise" --cooldown 3 ;;
  noise-fast-1pct)
    # FULL grid (3 corruptions x 3 sev x 3 methods, all 108 conditions) at ~1% images/cell.
    [ -d "$IC_FAST/gaussian_noise/1" ] || { echo "ERROR: internal copy not found at $IC_FAST -- run prep_internal_noise.sh first."; exit 1; }
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC_FAST" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_1pct" \
      --max-images 50 ;;
  noise-fast-01pct)
    # 0.1% SMOKE: full 3x3 grid, ~8 imgs/cell -> proves the pipeline end-to-end in ~1-2 min.
    # Writes to a SEPARATE dir so it NEVER contaminates the full run's resume checkpoint.
    [ -d "$IC_FAST/gaussian_noise/1" ] || { echo "ERROR: internal copy not found at $IC_FAST -- run prep_internal_noise.sh first."; exit 1; }
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC_FAST" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_smoke01" \
      --max-images 8 ;;
  vit-fast)
    [ -d "$IC_FAST/gaussian_noise/1" ] || { echo "ERROR: internal copy not found at $IC_FAST -- run prep_internal_noise.sh first."; exit 1; }
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC_FAST" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch vit_b16 --methods tent eata sar --out-results "$RES/imagenetc_noise_vit" ;;
  cifar101)
    # CIFAR-10.1 NATURAL distribution shift (reuses experiments/kbound/cifar/resnet18_cifar.pt; auto-downloads ~30MB .npy)
    caffeinate -is python "$S2" --benchmarks cifar101 \
      --data-root experiments/kbound/cifar --methods tent eata sar \
      --out-results "$RES/cifar101" ;;
  cifar101-quick)
    caffeinate -is python "$S2" --benchmarks cifar101 \
      --data-root experiments/kbound/cifar --methods tent eata sar \
      --out-results "$RES/cifar101_quick" --quick ;;
  *)
    echo "usage: bash kbtrain.sh [noise|noise-fast|noise-fast-1pct|noise-fast-01pct|noise-full|noise-quick|noise-smoke|camelyon|camelyon-fast|camelyon-fast-1pct|camelyon-smoke|cifar101|cifar101-quick|noiseblur|vit|vit-fast]"; exit 1 ;;
esac
