#!/usr/bin/env bash
# Phase-A sequential, memory-safe training queue (Mac/MPS). ONE MPS job at a time.
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

set -u
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"
REPO="$KB_REPO_ROOT"
PY="$HOME/.venv_wilds/bin/python"
S2="docs/research/kbound/scripts/cifar_tent_mps_v2.py"
IC="experiments/kbound/data/imagenet-c"
RES="experiments/kbound/results"
QLOG="$REPO/$RES/_train_queue.log"
export MPLCONFIGDIR="$KB_EXTERNAL_ROOT/tmp"/mpl PYTHONUNBUFFERED=1 PYTHONPATH=.
cd "$REPO" || exit 1
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }
wait_for(){ while pgrep -f "$1" >/dev/null 2>&1; do sleep 60; done; }

log "QUEUE START. Waiting for the ImageNet-R light run (one MPS job at a time)..."
wait_for "run_imagenetr_kbound.py.*light_mps_internal"
log "ImageNet-R done/absent -> starting Phase A."

log "JOB1: ViT-B/16 on ImageNet-C noise -> imagenetc_noise_vit"
caffeinate -is "$PY" "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
  --corruptions gaussian_noise shot_noise impulse_noise \
  --arch vit_b16 --methods tent eata sar \
  --out-results "$RES/imagenetc_noise_vit" --cooldown 3 --max-images 2000 >> "$QLOG" 2>&1
log "JOB1 exit=$?"

if [ ! -d "$IC/glass_blur" ]; then
  log "JOB2a: extracting blur.tar (6.6G, one-time)"
  ( cd "$IC" && caffeinate -is tar xf blur.tar ) >> "$QLOG" 2>&1; log "extract exit=$?"
fi
log "JOB2: ImageNet-C blur (ResNet-50) -> imagenetc_blur"
caffeinate -is "$PY" "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
  --corruptions defocus_blur glass_blur motion_blur zoom_blur \
  --arch resnet50 --methods tent eata sar \
  --out-results "$RES/imagenetc_blur" --cooldown 3 --max-images 2000 >> "$QLOG" 2>&1
log "JOB2 exit=$?"
log "QUEUE DONE -> imagenetc_noise_vit, imagenetc_blur (+ ImageNet-R light)."
