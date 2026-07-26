#!/usr/bin/env bash
# CIFAR-10.1 v6 multi-seed quick grid (seeds 0..4, sequential MPS).
# Same cell grid as cifar101_quick: small/tiny x iid/imbalanced/single_class x mild/aggressive x 2 repeats.
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
BASE="$REPO/experiments/kbound/results/cifar101_multiseed_v1"
DATA="$REPO/experiments/kbound/cifar"
cd "$REPO" || exit 1
mkdir -p "$BASE"
echo "[$(date '+%F %T')] CIFAR-10.1 MULTISEED QUEUE START" >> "$BASE/launch.log"
for S in 0 1 2 3 4; do
  OUT="$BASE/seed$S"
  mkdir -p "$OUT"
  if [[ -f "$OUT/result_manifest.json" ]] && [[ -f "$OUT/decisive_tta_results.json" ]]; then
    echo "[$(date '+%F %T')] SEED $S SKIP (already complete)" >> "$BASE/launch.log"
    continue
  fi
  echo "[$(date '+%F %T')] SEED $S START -> $OUT" >> "$BASE/launch.log"
  caffeinate -is "$PY" "$SCRIPT" --benchmarks cifar101 --quick --methods tent eata sar \
      --seed "$S" --data-root "$DATA" \
      --out-results "$OUT" --out-figs "$OUT/figs" >> "$OUT/seed$S.log" 2>&1
  RC=$?
  echo "[$(date '+%F %T')] SEED $S DONE rc=$RC" >> "$BASE/launch.log"
done
echo "[$(date '+%F %T')] CIFAR-10.1 MULTISEED QUEUE COMPLETE" >> "$BASE/launch.log"
