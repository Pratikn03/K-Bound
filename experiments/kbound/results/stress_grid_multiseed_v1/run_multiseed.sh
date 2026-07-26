#!/usr/bin/env bash
# PROTOCOL A multi-seed stress grid launcher (seeds 0..4, sequential = one MPS job at a time).
# Confirmatory 432-condition grid reproduced byte-identically via: --benchmarks cifar10c --quick
# --methods tent eata sar. Grid constants in cifar_tent_mps_v2.py are UNCHANGED.
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
REPO="$KB_REPO_ROOT"
PY="$REPO/.venv/bin/python"
SCRIPT="$REPO/docs/research/kbound/scripts/cifar_tent_mps_v2.py"
BASE="$REPO/experiments/kbound/results/stress_grid_multiseed_v1"
DATA="$REPO/experiments/kbound/cifar"
cd "$REPO" || exit 1
echo "[$(date '+%F %T')] MULTISEED QUEUE START (sequential, one MPS job at a time)" >> "$BASE/launch.log"
for S in 0 1 2 3 4; do
  OUT="$BASE/seed$S"
  echo "[$(date '+%F %T')] SEED $S START -> $OUT" >> "$BASE/launch.log"
  "$PY" "$SCRIPT" --benchmarks cifar10c --quick --methods tent eata sar \
      --seed "$S" --data-root "$DATA" \
      --out-results "$OUT" --out-figs "$OUT" >> "$OUT/seed$S.log" 2>&1
  echo "[$(date '+%F %T')] SEED $S DONE rc=$? " >> "$BASE/launch.log"
done
echo "[$(date '+%F %T')] MULTISEED QUEUE COMPLETE" >> "$BASE/launch.log"
