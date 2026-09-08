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

# Explicit absolute runtime/data paths; CPY may select a separate WILDS runtime.
# With SKIP_CAM=1, neither CPY nor WILDS_DATA_ROOT is needed or inspected.
PY="${PY:?Set PY to the selected Python executable}"
[[ "$PY" == /* && -f "$PY" && -x "$PY" ]] || { echo "Invalid PY: $PY" >&2; exit 3; }
IMAGENETR_DIR="${IMAGENETR_DIR:?Set IMAGENETR_DIR to existing ImageNet-R data}"
[[ "$IMAGENETR_DIR" == /* && -d "$IMAGENETR_DIR" ]] || { echo "Invalid IMAGENETR_DIR: $IMAGENETR_DIR" >&2; exit 3; }
if [ "${SKIP_CAM:-0}" != "1" ]; then
  CPY="${CPY:-$PY}"
  [[ "$CPY" == /* && -f "$CPY" && -x "$CPY" ]] || { echo "Invalid CPY: $CPY" >&2; exit 3; }
  WILDS_DATA_ROOT="${WILDS_DATA_ROOT:?Set WILDS_DATA_ROOT to existing WILDS data}"
  [[ "$WILDS_DATA_ROOT" == /* && -d "$WILDS_DATA_ROOT/camelyon17_v1.0" ]] || { echo "Missing camelyon17_v1.0 under WILDS_DATA_ROOT: $WILDS_DATA_ROOT" >&2; exit 3; }
fi
echo "PY=$PY"
"$PY" -c "import numpy, sklearn" || { echo "ERROR: $PY lacks numpy/sklearn. Set PY=/path/to/python"; exit 3; }
if [ "${SKIP_CAM:-0}" != "1" ]; then
  echo "CPY=$CPY (Camelyon arm)"
  "$CPY" -c "import wilds" 2>/dev/null || { echo "ERROR: CPY lacks wilds; select a prepared runtime" >&2; exit 3; }
fi

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

echo "WILDS_DATA_ROOT=${WILDS_DATA_ROOT:-<not requested>}"
echo "IMAGENETR_DIR=$IMAGENETR_DIR"

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
require_analysis_dir() {
  local selected="$1" directory
  if [[ -z "$selected" || ! -f "$selected" ]]; then
    echo "ERROR: expected per_condition output is missing; refusing analysis" >&2
    return 3
  fi
  directory="$(cd "$(dirname "$selected")" && pwd -P)" || return 3
  if [[ "$directory" == "$(cd "$ROOT" && pwd -P)" ]]; then
    echo "ERROR: analysis output resolves to the repository root" >&2
    return 3
  fi
  printf '%s\n' "$directory"
}
CAM_DIR=""
if [ "${SKIP_CAM:-0}" != "1" ]; then
  CAM_DIR=$(require_analysis_dir "$(ls -t "$RESULTS/$CAM_RUN"/per_condition_camelyon17_*_seed0.json 2>/dev/null | head -1 || true)")
fi
INR_DIR=$(require_analysis_dir "$(ls -t "$RESULTS/$INR_RUN"/per_condition_imagenet-r_*_seed0.json 2>/dev/null | head -1 || true)")
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
