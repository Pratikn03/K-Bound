#!/usr/bin/env bash
# NATURAL_WIN_PROTOCOL_v1 launcher (pre-registered: research_lock/NATURAL_WIN_PROTOCOL_v1.yaml)
# Run on the Mac (MPS). Wave-5 instruments are already wired into the runners:
#   - panel_capture.py  -> c_ij / n_D serialized per condition (tau' gate input)
#   - kga/evidence_v2.py -> Z_ev2 features (Camelyon rich mode)
#   - per_condition_serialize.py -> pass-through of the new fields
# Analysis is the pre-committed gapclose_wave5/natural_win_analysis.py.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export PYTHONWARNINGS=ignore
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/experiments/kbound/wilds${PYTHONPATH:+:$PYTHONPATH}"

# ---- interpreters (house convention, matches run_remaining_gpu_experiments.sh)
PY="${PY:-$ROOT/.venv/bin/python}"          # repo venv: numpy/torch/sklearn
[ -x "$PY" ] || PY="$(command -v python3)"  # fallback: system python3
CPY="$PY"                                   # Camelyon needs torch + wilds
if [ -x "$HOME/.venv_wilds/bin/python" ]; then CPY="$HOME/.venv_wilds/bin/python"; fi
echo "PY=$PY"
echo "CPY=$CPY (Camelyon arm)"
"$PY" -c "import numpy, sklearn" || { echo "ERROR: $PY lacks numpy/sklearn. Set PY=/path/to/python"; exit 3; }
"$CPY" -c "import wilds" 2>/dev/null || echo "WARN: $CPY has no 'wilds' — Camelyon arm will fail. pip install wilds, or set up ~/.venv_wilds"

echo "== [0/4] wiring self-check (no GPU work)"
"$PY" - <<'EOF'
import sys, numpy as np
sys.path.insert(0, "experiments/kbound/wilds"); sys.path.insert(0, ".")
import panel_capture as pc
from kga.evidence_v2 import extract_all
f = pc.panel_fields(np.random.default_rng(0).integers(0, 5, size=(4, 64)))
assert len(f["c_ij"]) == 4 and isinstance(f["n_D"], int)
assert len(pc.ev2_vector(np.random.default_rng(1).normal(size=(32, 8)))) == 5
sys.path.insert(0, "docs/research/kbound/gapclose_wave5")
import tau_selfnorm, radius_v2, natural_win_analysis  # noqa
print("wiring OK: panel_capture + evidence_v2 + tau_selfnorm + radius_v2 + analysis")
EOF

RESULTS=experiments/kbound/results
CAM_RUN=natural_win_v1_camelyon
INR_RUN=natural_win_v1_imagenetr

# ---- data roots (house convention, matches run_remaining_gpu_experiments.sh;
#      the runners' built-in defaults are stale)
WILDS_DATA_ROOT="${WILDS_DATA_ROOT:-$HOME/datasets/wilds}"
IMAGENETR_DIR="${IMAGENETR_DIR:-$ROOT/experiments/kbound/data/imagenet-r}"
echo "WILDS_DATA_ROOT=$WILDS_DATA_ROOT"
echo "IMAGENETR_DIR=$IMAGENETR_DIR"
if [ ! -d "$WILDS_DATA_ROOT/camelyon17_v1.0" ]; then
  echo "ERROR: camelyon17_v1.0 not found under $WILDS_DATA_ROOT"
  echo "  Either point at your existing copy:   WILDS_DATA_ROOT=/path/to/wilds bash scripts/run_natural_win_v1.sh"
  echo "  Or download it (~10 GB) with:"
  echo "    $CPY -c \"from wilds import get_dataset; get_dataset(dataset='camelyon17', download=True, root_dir='$WILDS_DATA_ROOT')\""
  echo "  Skipping the Camelyon arm is possible with: SKIP_CAM=1 bash scripts/run_natural_win_v1.sh"
  [ "${SKIP_CAM:-0}" = "1" ] || exit 3
fi
[ -d "$IMAGENETR_DIR" ] || { echo "ERROR: ImageNet-R data not found at $IMAGENETR_DIR"; exit 3; }

if [ "${SKIP_CAM:-0}" != "1" ]; then
  echo "== [1/4] PRIMARY ARM: Camelyon17, seeds 0-3, rich evidence"
  echo "   (protocol amended pre-unblinding: f0_seed4.pt never existed; see YAML)"
  "$CPY" experiments/kbound/wilds/run_camelyon17_kbound.py \
    --data-root "$WILDS_DATA_ROOT" \
    --seeds 0 1 2 3 \
    --evidence-panel rich \
    --device mps \
    --run-name "$CAM_RUN" \
    --serialize-per-condition
else
  echo "== [1/4] SKIPPED (SKIP_CAM=1)"
fi

echo "== [2/4] SECONDARY ARM: ImageNet-R diverse 10-backbone panel, seeds 0-2"
"$PY" experiments/kbound/wilds/run_imagenetr_kbound.py \
  --imagenetr-dir "$IMAGENETR_DIR" \
  --panel diverse_backbones \
  --seeds 0 1 2 \
  --device mps \
  --run-name "$INR_RUN" \
  --serialize-per-condition

echo "== [3/4] locate per_condition outputs"
CAM_DIR=$(dirname "$(ls -t "$RESULTS/$CAM_RUN"/per_condition_camelyon17_*_seed0.json 2>/dev/null | head -1 || true)")
INR_DIR=$(dirname "$(ls -t "$RESULTS/$INR_RUN"/per_condition_imagenet-r_*_seed0.json 2>/dev/null | head -1 || true)")
echo "camelyon: ${CAM_DIR:-<missing>}"; echo "imagenetr: ${INR_DIR:-<missing>}"

echo "== [4/4] pre-committed analysis (held-out scoring, ONCE)"
if [ -d "$CAM_DIR" ]; then
  "$PY" docs/research/kbound/gapclose_wave5/natural_win_analysis.py \
    --run-dir "$CAM_DIR" --dataset camelyon17
fi
if [ -d "$INR_DIR" ]; then
  "$PY" docs/research/kbound/gapclose_wave5/natural_win_analysis.py \
    --run-dir "$INR_DIR" --dataset imagenet-r --panel
fi

echo
echo "Done. Verdict JSONs: NATURAL_WIN_v1_*.json inside the run dirs."
echo "Per protocol: WIN / NO-HARM / FAIL are all final — no re-tuning."
