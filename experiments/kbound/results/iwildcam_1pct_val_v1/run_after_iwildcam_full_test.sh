#!/usr/bin/env bash
# --- interpreter: $KBOUND_PYTHON, default python3 (was a hard-coded venv path).
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

KB_PYTHON="${KBOUND_PYTHON:-python3}"

set -euo pipefail

cd "$KB_REPO_ROOT"

WAIT_PATTERN='run_iwildcam_kbound.py .*--run-name iwildcam_full_test'
echo "[iwildcam-1pct] queued at $(date)"
while pgrep -f "$WAIT_PATTERN" >/dev/null 2>&1; do
  echo "[iwildcam-1pct] waiting for active iwildcam_full_test at $(date)"
  sleep 60
done

echo "[iwildcam-1pct] starting at $(date)"
"$KB_PYTHON" -u experiments/kbound/wilds/run_iwildcam_kbound.py \
  --device mps \
  --split val \
  --results-root experiments/kbound/results \
  --run-name iwildcam_1pct_val_v1 \
  --backbone resnet18 \
  --trainable head \
  --max-train-batches 56 \
  --train-bs 32 \
  --balanced-train \
  --max-locations 8 \
  --n-eval 48 \
  --n-batches 2 \
  --compositions iid imbalanced single_class \
  --batch-regimes tiny small \
  --aggressiveness mild aggressive \
  --candidates tent_online eata_online sar_online tent_episodic eata_episodic sar_episodic \
  --seeds 0 \
  --tau-star 0.52 \
  --eval-bs 48 \
  --episodic-batch 24 \
  --episodic-steps 3 \
  --retrain
echo "[iwildcam-1pct] finished at $(date)"
