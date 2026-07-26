#!/usr/bin/env bash
# Protocol E — official gentle SAR on ImageNet-C (3 seeds, same 36-cell grid as sarfix).
# Registered: research_lock/OFFICIAL_SAR_SCHEDULE_PROTOCOL_E_v1.yaml
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
REPO="$KB_REPO_ROOT"
PY="$REPO/.venv/bin/python"
SCRIPT="$REPO/docs/research/kbound/scripts/cifar_tent_mps_v2.py"
IC="$REPO/experiments/kbound/data/imagenet-c"
BASE="$REPO/experiments/kbound/results/imagenetc_official_sar_E_v1"
cd "$REPO" || exit 1
mkdir -p "$BASE"
export TMPDIR="$KB_EXTERNAL_ROOT/tmp" TORCH_HOME="$KB_EXTERNAL_ROOT/torch_cache"
mkdir -p "$TMPDIR" "$TORCH_HOME"

echo "[$(date '+%F %T')] PROTOCOL E QUEUE START" >> "$BASE/launch.log"
for S in 0 1 2; do
  OUT="$BASE/seed$S"
  mkdir -p "$OUT"
  if [[ -f "$OUT/decisive_tta_results.json" ]] && [[ -f "$OUT/result_manifest.json" ]]; then
    echo "[$(date '+%F %T')] SEED $S SKIP (already complete)" >> "$BASE/launch.log"
    continue
  fi
  echo "[$(date '+%F %T')] SEED $S START -> $OUT" >> "$BASE/launch.log"
  caffeinate -is "$PY" "$SCRIPT" \
    --benchmarks imagenetc \
    --imagenetc-root "$IC" \
    --corruptions gaussian_noise shot_noise impulse_noise \
    --arch resnet50 \
    --methods tent eata sar \
    --sar-lr 2.5e-4 \
    --sar-freeze-layer4 \
    --seed "$S" \
    --cooldown 3 \
    --out-results "$OUT" \
    --out-figs "$OUT/figs" \
    >> "$OUT/seed${S}.log" 2>&1
  RC=$?
  echo "[$(date '+%F %T')] SEED $S DONE rc=$RC" >> "$BASE/launch.log"
done
echo "[$(date '+%F %T')] PROTOCOL E QUEUE COMPLETE" >> "$BASE/launch.log"
