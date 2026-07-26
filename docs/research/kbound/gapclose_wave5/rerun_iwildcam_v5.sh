#!/bin/bash
# iWildCam v5 aggressive rerun with the REAL ResNet-50 ERM f0 (id_val 0.72 / test 0.69).
# Single python invocation — no line-continuation to break on paste.
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

set -e
cd "$KB_REPO_ROOT"
source "${KBOUND_VENV:-$HOME/.venv_wilds}/bin/activate"   # override with KBOUND_VENV

CKPT=experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt
echo "f0 ckpt: $CKPT ($([ -f "$CKPT" ] && echo present || echo MISSING))"

# clear any invalid stub run so the rerun starts clean
rm -rf experiments/kbound/results/win_hunt_v5_iwildcam

python experiments/kbound/wilds/run_iwildcam_kbound.py --data-root experiments/kbound/data/wilds --ckpt "$CKPT" --backbone resnet50 --split test --seeds 0 1 --run-name win_hunt_v5_iwildcam --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --candidates tent_online eata_online sar_online
