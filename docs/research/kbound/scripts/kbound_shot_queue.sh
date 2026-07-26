#!/usr/bin/env bash
# Run ImageNet-C with all 4 candidates incl. SHOT, AFTER the GPU frees (one MPS job at a time).
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

set -u
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"
REPO="$KB_REPO_ROOT"
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
