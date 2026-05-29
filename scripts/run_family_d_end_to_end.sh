#!/usr/bin/env bash
# End-to-end Family-D pipeline (v3 frozen + v4 exploratory).
#
# Stages:
#   0. Verify cached features + fusion CSV exist (rebuild if requested with --rebuild-features).
#   1. v3 execute → archive → inference → primary decision.
#   2. v4 execute → inference → exploratory decision.
#
# Usage:
#   bash scripts/run_family_d_end_to_end.sh            # both v3 and v4 with cached features
#   bash scripts/run_family_d_end_to_end.sh --rebuild-features
#   bash scripts/run_family_d_end_to_end.sh --v3-only
#   bash scripts/run_family_d_end_to_end.sh --v4-only

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=src
PY=.venv/bin/python

REBUILD_FEATURES=0
RUN_V3=1
RUN_V4=1
for arg in "$@"; do
  case "$arg" in
    --rebuild-features) REBUILD_FEATURES=1 ;;
    --v3-only)          RUN_V4=0 ;;
    --v4-only)          RUN_V3=0 ;;
    *) echo "unknown arg: $arg" && exit 2 ;;
  esac
done

echo "==================================================="
echo " Family-D pipeline: v3=${RUN_V3} v4=${RUN_V4} rebuild=${REBUILD_FEATURES}"
echo "==================================================="

# Stage 0 — features + fusion CSV
if [[ "$REBUILD_FEATURES" -eq 1 ]]; then
  echo "[stage 0] extracting features (MPS, ~minutes)..."
  PYTHONPATH=$PYTHONPATH $PY src/scripts/family_d_v2_extract_features.py
  echo "[stage 0] building fusion CSV..."
  PYTHONPATH=$PYTHONPATH $PY src/scripts/family_d_v2_build_fusion_csv.py
else
  echo "[stage 0] checking cached features..."
  ls experiments/phase2/family_d/features/*.npz 2>/dev/null \
    | grep -v "/\._" | wc -l | xargs -I {} echo "  found {} category NPZs"
  test -f experiments/fusion/eyecandies_inputs.csv \
    || { echo "  fusion CSV missing -> rerun with --rebuild-features"; exit 1; }
  echo "  fusion CSV ok"
fi

# Stage 1 — v3 (frozen)
if [[ "$RUN_V3" -eq 1 ]]; then
  echo "[stage 1] v3 execution (frozen contract, 30 seeds, D-EYE-1+D-EYE-2+D-EYE-3)..."
  PYTHONPATH=$PYTHONPATH $PY src/scripts/family_d_v2_execute.py \
    --endpoints D-EYE-1,D-EYE-2,D-EYE-3 --seeds 30 --seed-start 42
  echo "[stage 1] v3 inference (one-time test-label read, Holm K=2)..."
  PYTHONPATH=$PYTHONPATH $PY src/scripts/family_d_v2_inference.py
  echo "[stage 1] v3 family decision:"
  cat experiments/phase2/family_d/family_d_v2_family_decision.txt
fi

# Stage 2 — v4 (exploratory)
if [[ "$RUN_V4" -eq 1 ]]; then
  echo "[stage 2] v4 execution (exploratory contract, 60 seeds, soft corruption)..."
  PYTHONPATH=$PYTHONPATH $PY src/scripts/family_d_v4_execute.py \
    --endpoints D-EYE-1v4,D-EYE-2v4,D-EYE-3v4 --seeds 60 --seed-start 42
  echo "[stage 2] v4 inference (AUC + Brier, Holm K=2)..."
  PYTHONPATH=$PYTHONPATH $PY src/scripts/family_d_v4_inference.py
  echo "[stage 2] v4 family decision:"
  cat experiments/phase2/family_d/family_d_v4_family_decision.txt
fi

echo
echo "==================================================="
echo " Pipeline complete. Decisions:"
echo "==================================================="
[[ -f experiments/phase2/family_d/family_d_v2_family_decision.txt ]] && \
  echo " v3 (primary held-out): $(cat experiments/phase2/family_d/family_d_v2_family_decision.txt)"
[[ -f experiments/phase2/family_d/family_d_v4_family_decision.txt ]] && \
  echo " v4 (exploratory):      $(cat experiments/phase2/family_d/family_d_v4_family_decision.txt)"
echo
echo "Manuscript-relevant docs:"
echo "  docs/research/phase2/FAMILY_D_V3_FINAL_DECISION_AUDITED.md  (primary, frozen)"
echo "  docs/research/phase2/FAMILY_D_V4_EXPLORATORY_DECISION.md    (v4, exploratory)"
