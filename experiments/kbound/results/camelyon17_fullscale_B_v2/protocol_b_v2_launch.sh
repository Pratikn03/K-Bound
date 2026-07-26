#!/usr/bin/env bash
# Protocol B v2 — integrity fix: full composition grid via run_camelyon17_kbound.py
# (v1 used run_wilds_camelyon17.py and only wrote per-seed aggregates).
#
# Requires ~/.venv_wilds (torch + wilds) — same env as Protocol F GPU runs.
#
# From uav/:
#   bash <repo>/experiments/kbound/results/camelyon17_fullscale_B_v2/protocol_b_v2_launch.sh
# From the repository root:
#   bash experiments/kbound/results/camelyon17_fullscale_B_v2/protocol_b_v2_launch.sh
#
# EXPECTED OUTCOME: scientific negative (bias-limited sparse Z) even at n=1024.
# --- external (git-excluded) data volume: ONE documented variable, no default.
: "${KBOUND_EXTERNAL_ROOT:?set KBOUND_EXTERNAL_ROOT to the volume holding the git-excluded datasets/checkpoints/caches (layout: docs/research/kbound/kbound_repro/paths.py, acquisition: DATA.md)}"
KB_EXTERNAL_ROOT="$KBOUND_EXTERNAL_ROOT"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$ROOT"

# GPU WILDS env (matches camelyon17_richZ_F_v1/protocol_f_command.sh)
VENV_WILDS="${VENV_WILDS:-$HOME/.venv_wilds}"
if [[ -x "$VENV_WILDS/bin/python" ]]; then
  PYTHON="$VENV_WILDS/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]] && "$ROOT/.venv/bin/python" -c "import wilds" 2>/dev/null; then
  PYTHON="$ROOT/.venv/bin/python"
else
  echo "ERROR: need a venv with torch+wilds."
  echo "  Expected: $VENV_WILDS"
  echo "  Protocol F used: source $VENV_WILDS/bin/activate"
  echo "  Or install: pip install wilds  (in your GPU venv)"
  exit 1
fi

export TMPDIR="${TMPDIR:-"$KB_EXTERNAL_ROOT/tmp"}"
export TORCH_HOME="${TORCH_HOME:-"$KB_EXTERNAL_ROOT/torch_cache"}"
mkdir -p "$TMPDIR" "$TORCH_HOME"

DATA_ROOT="${WILDS_DATA_ROOT:-"$KB_EXTERNAL_ROOT/datasets/wilds"}"
F0_TEMPLATE="experiments/kbound/results/camelyon17_fullscale_B_v1/f0_seed{seed}.pt"
RUN_NAME="camelyon17_fullscale_B_v2"
OUT="$ROOT/experiments/kbound/results/$RUN_NAME"
mkdir -p "$OUT"

echo "ROOT=$ROOT"
echo "PYTHON=$PYTHON"
echo "OUT=$OUT"
echo "DATA_ROOT=$DATA_ROOT"
echo "TMPDIR=$TMPDIR"

# caffeinate keeps Mac awake during long MPS run (optional; drop if unavailable)
RUN=(caffeinate -is "$PYTHON" -u experiments/kbound/wilds/run_camelyon17_kbound.py)
if ! command -v caffeinate >/dev/null 2>&1; then
  RUN=("$PYTHON" -u experiments/kbound/wilds/run_camelyon17_kbound.py)
fi

"${RUN[@]}" \
  --data-root "$DATA_ROOT" \
  --f0-template "$F0_TEMPLATE" \
  --seeds 0 1 2 3 4 \
  --domains test val id_val \
  --compositions iid imbalanced single_class \
  --batch-regimes small \
  --aggressiveness mild aggressive \
  --n-eval 1024 --n-batches 4 \
  --tau-star 0.52 --kappa 2.5 --delta 0.05 --sd-L 0.6 \
  --evidence-panel base \
  --device auto \
  --run-name "$RUN_NAME" \
  2>&1 | tee "$OUT/run_B_v2.log"

echo ""
echo "Done. Analyze with:"
echo "  cd $ROOT"
echo "  $PYTHON docs/research/kbound/theory_v2/realdata/eps_recal/eps_recal_camelyon.py \\"
echo "    --records experiments/kbound/results/$RUN_NAME/result_*.json --label sparse_B_v2"
