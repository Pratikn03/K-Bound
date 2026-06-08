#!/usr/bin/env bash
# run_wilds.sh — Environment bootstrap + WILDS Camelyon17 KGA experiment.
#
# Mirrors run_decisive_tta.sh's env bootstrap (uv venv, python 3.12, torch 2.5.1).
# Run from the repo root:
#   bash docs/research/kbound/scripts/run_wilds.sh
#
# After the first run the .venv_wilds venv is reused; skip --epochs to use
# cached f0 checkpoints.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

# ── 1. Bootstrap venv ────────────────────────────────────────────────────────
# NOTE: venv lives in $HOME (APFS). Creating it on the exFAT T9 drive fails
# (ensurepip cannot create the dir layout). Reused across runs.
VENV="$HOME/.venv_wilds"
if [ ! -d "$VENV" ]; then
    echo "==> Creating $VENV with uv (python 3.12)..."
    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    uv venv "$VENV" --python 3.12
fi

source "$VENV/bin/activate"

# ── 2. Install deps ──────────────────────────────────────────────────────────
echo "==> Installing dependencies..."
# uv-created venvs ship no `pip`; use `uv pip` into the active venv.
PIP="uv pip"; command -v uv >/dev/null 2>&1 || PIP="python -m pip"

# torch 2.5.1 — default PyPI wheels include MPS on Apple silicon (do NOT pin the cpu index)
if python3 -c "import torch" 2>/dev/null; then
    echo "    torch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
else
    $PIP install -q torch==2.5.1 torchvision==0.20.1
    echo "    torch installed"
fi

$PIP install -q wilds numpy scipy scikit-learn pandas matplotlib

# ── 3. Smoke test (no GPU needed) ────────────────────────────────────────────
echo "==> Smoke-testing analysis core (no torch forward pass)..."
python3 docs/research/kbound/scripts/run_wilds_camelyon17.py --smoke-test
echo "    Smoke test OK"

# ── 4. Run experiment ────────────────────────────────────────────────────────
WILDS_ROOT="${WILDS_ROOT:-$REPO_ROOT/experiments/kbound/data/wilds}"
OUTPUT_DIR="experiments/kbound/results/wilds"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "==> Running WILDS Camelyon17 KGA (5 seeds)..."
echo "    WILDS data root: $WILDS_ROOT"
echo "    Output dir:      $OUTPUT_DIR"
echo "    (data already present locally; WILDS will not re-download)"
echo ""

python3 docs/research/kbound/scripts/run_wilds_camelyon17.py \
    --wilds-root   "$WILDS_ROOT"     \
    --output-dir   "$OUTPUT_DIR"     \
    --seeds        0 1 2 3 4         \
    --epochs       5                 \
    --steps        10                \
    --lr           1e-3

echo ""
echo "==> Done. Results in $OUTPUT_DIR/wilds_camelyon17_kga.json"
