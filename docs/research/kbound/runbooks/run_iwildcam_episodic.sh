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

set -u; R="$KB_REPO_ROOT"
CKPT="$R/experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt"
OUT="$R/experiments/kbound/results/multiseed/iwildcam"; PY="$KB_PYTHON"
LOG="$OUT/iwildcam_episodic.log"; say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "iWildCam tent_episodic 5-seed (real ResNet-50 ERM ckpt) — matches paper's Protocol-H candidate"
cd "$R/experiments/kbound/wilds"
caffeinate -is "$PY" run_iwildcam_aetta.py --ckpt "$CKPT" --backbone resnet50 --split test \
  --seeds 0 1 2 3 4 --max-locations 4 --candidates tent_episodic --device auto \
  --results-root "$OUT" --run-name iwildcam_episodic_v1 >> "$LOG" 2>&1
say "run rc=$?"
"$KB_PYTHON" "$R/docs/research/kbound/scripts/extract_multiseed_natural.py" --track iwildcam \
  --result "$OUT/iwildcam_episodic_v1/**/result_*.json" --candidates tent_episodic \
  --out-dir "$OUT/extracted_episodic" >> "$LOG" 2>&1
say "extract rc=$? — DONE"
