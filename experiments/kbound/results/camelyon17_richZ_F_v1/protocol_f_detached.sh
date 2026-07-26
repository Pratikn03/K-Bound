#!/usr/bin/env bash
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

set -euo pipefail
cd "$KB_REPO_ROOT"
export TMPDIR="$KB_EXTERNAL_ROOT/tmp"
export TORCH_HOME="$KB_EXTERNAL_ROOT/torch_cache"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$TORCH_HOME"
"$KB_PYTHON" -u experiments/kbound/wilds/run_camelyon17_kbound.py \
  --data-root "$KB_EXTERNAL_ROOT/datasets/wilds" \
  --f0-template 'experiments/kbound/results/camelyon17_fullscale_B_v1/f0_seed{seed}.pt' \
  --seeds 0 1 2 3 4 \
  --domains test val id_val \
  --compositions iid imbalanced single_class \
  --batch-regimes small \
  --aggressiveness mild aggressive \
  --n-eval 1024 \
  --n-batches 4 \
  --tau-star 0.52 \
  --kappa 2.5 \
  --sd-L 0.6 \
  --delta 0.05 \
  --device auto \
  --evidence-panel rich \
  --run-name camelyon17_richZ_F_v1
latest=$(ls -t experiments/kbound/results/camelyon17_richZ_F_v1/result_*.json | head -1)
"$KB_PYTHON" -u docs/research/kbound/scripts/analyze_F.py \
  --records "$latest" \
  --output-dir experiments/kbound/results/camelyon17_richZ_F_v1 \
  --estimator ppi_debias \
  --conformal mondrian \
  --dev-seeds 0 1 \
  --test-seeds 2 3 4
