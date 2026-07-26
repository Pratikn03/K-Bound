#!/usr/bin/env bash
# quick_check.sh -- 0.1% SMOKE of both pending datasets, end-to-end, in a few minutes.
# Confirms the pipelines (incl. progress.log + checkpoint) are healthy BEFORE the full run.
# Writes ONLY to *_smoke* dirs -> never touches real results or their resume checkpoints.
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

set -uo pipefail
K="$KB_REPO_ROOT"/docs/research/kbound/scripts/kbtrain.sh
echo "########## QUICK 0.1% SMOKE 1/2 : ImageNet-C noise (full 3x3 grid, ~8 imgs/cell) ##########"
if bash "$K" noise-fast-01pct; then echo "  -> ImageNet-C pipeline: OK"; else echo "  -> ImageNet-C pipeline: FAILED"; exit 1; fi
echo
echo "########## QUICK 0.1% SMOKE 2/2 : CIFAR-10.1 natural shift (quick subset) ##########"
if bash "$K" cifar101-quick; then echo "  -> CIFAR-10.1 pipeline: OK"; else echo "  -> CIFAR-10.1 pipeline: FAILED"; exit 1; fi
echo
echo "=================================================================="
echo " BOTH PIPELINES HEALTHY (numbers are meaningless at 0.1%)."
echo " Now launch the real runs:"
echo "   bash $KB_REPO_ROOT/docs/research/kbound/scripts/full_run.sh"
echo "=================================================================="
