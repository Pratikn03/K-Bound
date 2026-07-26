#!/usr/bin/env bash
# Item 12: multi-seed ImageNet-C, seeds 1-4 (seed 0 = locked win_hunt_v5/imagenetc_aggr run).
# Protocol replicated EXACTLY from the authoritative seed-0 manifest
# (experiments/kbound/results/win_hunt_v5/imagenetc_aggr/result_manifest.json, git 87bf90a):
#   mechanism-faithful SAR (shared lr; no --sar-lr, no --sar-freeze-layer4)
#   3 noise corruptions x severities {1,3,5} x {iid,imbalanced,single_class} = 27 cells/method
#   batch regime small, aggressiveness aggressive, adapt_lr 0.004, ResNet-50, mps.
# Only the seed varies. Env: conda 'aetta' python (local .venv torch is broken on py3.14).
# --- interpreter: $KBOUND_PYTHON, default python3 (was a hard-coded venv path).
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

KB_PYTHON="${KBOUND_PYTHON:-python3}"

set -u
REPO="$KB_REPO_ROOT"
PY=""$KB_PYTHON""
SCRIPT="$REPO/docs/research/kbound/scripts/cifar_tent_mps_v2.py"
IC=""$KB_EXTERNAL_ROOT/imagenetc_local""   # same root the authoritative seed-0 run used (all 3 noise corr, sev 1-5, 23GB)
BASE="$REPO/experiments/kbound/results/win_hunt_v5_imagenetc_ms"
export TORCH_HOME="$KB_EXTERNAL_ROOT/torch_cache" TMPDIR="$KB_EXTERNAL_ROOT/tmp"
mkdir -p "$BASE" "$TMPDIR"
cd "$REPO" || exit 1

echo "[$(date '+%F %T')] ITEM12 QUEUE START (seeds 1-4)" >> "$BASE/launch.log"
for S in 1 2 3 4; do
  OUT="$BASE/seed$S"
  mkdir -p "$OUT"
  if [[ -f "$OUT/decisive_tta_results.json" ]]; then
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
    --device mps \
    --seed "$S" \
    --severities 1 3 5 --max-images 4000 \
    --imagenetc-composition iid imbalanced single_class \
    --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 \
    --out-results "$OUT" \
    >> "$OUT/seed${S}.log" 2>&1
  echo "[$(date '+%F %T')] SEED $S DONE rc=$?" >> "$BASE/launch.log"
done
echo "[$(date '+%F %T')] ITEM12 QUEUE COMPLETE" >> "$BASE/launch.log"
# After completion: per-seed paired bootstrap + aggregate (see CAMERA_READY_RUNBOOK.md Item 12);
# criterion: beats-both per seed (both paired CIs exclude 0, Holm), FA_u <= alpha every seed.
