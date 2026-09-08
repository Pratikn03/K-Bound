#!/usr/bin/env bash
# =============================================================================
# run_remaining_gpu_experiments.sh
#
# One command to close the two remaining NEEDS-GPU K-Bound gaps on an Apple-silicon
# Mac (MPS) -- or CUDA / CPU by overriding $DEVICE:
#
#   GAP B4  ImageNet-R multi-seed (>=3 seeds), 48-condition diverse-backbone grid,
#           OOM-hardened (lazy per-backbone load, --resume, small frozen-eval batch),
#           per-condition arrays serialized in the stress_grid_multiseed schema.
#
#   GAP B1  Camelyon17 SAR completion. The candidate set already includes
#           {tent,eata,sar} x {online,episodic}; this finishes the B_v2 sweep and
#           serializes per-condition arrays (incl. the SAR columns).
#
# After each run it executes the torch-free multi-seed paired-CI analysis
# (experiments/kbound/wilds/multiseed_paired_ci.py) over the freshly written
# per-condition files, producing MULTISEED_ANALYSIS_RESULTS.json.
#
# NOTHING here overwrites a committed/canonical result: ImageNet-R writes to
# imagenetr_protocol_d_multiseed_v1/, Camelyon17 resumes into the existing
# camelyon17_fullscale_B_v2/ checkpoint (a non-canonical, in-progress run dir).
#
# USAGE (from repo root or anywhere):
#   bash scripts/run_remaining_gpu_experiments.sh                 # both, MPS, defaults
#   ONLY=imagenetr bash scripts/run_remaining_gpu_experiments.sh  # just ImageNet-R
#   ONLY=camelyon  bash scripts/run_remaining_gpu_experiments.sh  # just Camelyon17
#   DEVICE=cuda    bash scripts/run_remaining_gpu_experiments.sh  # CUDA box
#   SEEDS="0 1 2 3 4" bash scripts/run_remaining_gpu_experiments.sh
#
# REQUIRED CONFIG: absolute existing runtime and requested-arm data paths.
# PY must be set. CPY optionally selects a separate prepared WILDS runtime;
# absent CPY uses the explicitly selected PY, with no other runtime discovery.
# ENV OVERRIDES (scientific defaults retained):
#   DEVICE=auto                 auto|mps|cuda|cpu  (auto -> MPS on Apple silicon)
#   SEEDS="0 1 2"               >=3 recommended for the multi-seed CIs
#   PY=<python executable>     required python with torch
#   CPY=<python executable>    optional python with torch + wilds
#   IMAGENETR_DIR=<data root>   required for ImageNet-R
#   WILDS_DATA_ROOT=<data root> required for Camelyon17
#   INR_BATCH=24                ImageNet-R frozen-eval batch (lower if MPS OOMs)
#   INR_NEVAL=500               ImageNet-R class-balanced eval pool size
#   NBOOT=10000                 paired-bootstrap resamples for the CI analysis
# =============================================================================
set -euo pipefail

# ---- locate repo root (this script lives in <repo>/scripts/) ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-0 1 2}"
ONLY="${ONLY:-both}"
case "$ONLY" in
  imagenetr|inr|b4|B4|camelyon|cam|b1|B1|both) ;;
  *) echo "ERROR: unknown ONLY mode: $ONLY" >&2; exit 2 ;;
esac
NBOOT="${NBOOT:-10000}"
INR_BATCH="${INR_BATCH:-24}"
INR_NEVAL="${INR_NEVAL:-500}"

# pick a python with torch
PY="${PY:?Set PY to the selected Python executable}"
if [[ "$PY" != /* || ! -f "$PY" || ! -x "$PY" ]]; then
  echo "ERROR: python not found/executable at PY=$PY"
  echo "       set PY=/path/to/python-with-torch (and wilds for Camelyon17)"
  exit 1
fi

# Validate every requested arm before any Python invocation or training.
case "$ONLY" in
  imagenetr|inr|b4|B4|both)
    : "${IMAGENETR_DIR:?Set IMAGENETR_DIR to existing ImageNet-R data}"
    [[ "$IMAGENETR_DIR" == /* && -d "$IMAGENETR_DIR" ]] || { echo "Invalid IMAGENETR_DIR: $IMAGENETR_DIR" >&2; exit 2; } ;;
esac
case "$ONLY" in
  camelyon|cam|b1|B1|both)
    CPY="${CPY:-$PY}"
    [[ "$CPY" == /* && -f "$CPY" && -x "$CPY" ]] || { echo "Invalid CPY: $CPY" >&2; exit 2; }
    : "${WILDS_DATA_ROOT:?Set WILDS_DATA_ROOT to existing WILDS data}"
    [[ "$WILDS_DATA_ROOT" == /* && -d "$WILDS_DATA_ROOT/camelyon17_v1.0" ]] || { echo "Missing camelyon17_v1.0 under WILDS_DATA_ROOT: $WILDS_DATA_ROOT" >&2; exit 2; } ;;
esac

# keep all heavy torch caches / tmp off the slow data drive if the user has set them
export TMPDIR="${TMPDIR:-$ROOT/tmp}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.torch_cache}"
mkdir -p "$TMPDIR" "$TORCH_HOME"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/experiments/kbound/wilds${PYTHONPATH:+:$PYTHONPATH}"

# caffeinate keeps the Mac awake during long MPS runs (optional)
CAF=()
if command -v caffeinate >/dev/null 2>&1; then CAF=(caffeinate -is); fi

echo "ROOT=$ROOT"
echo "PY=$PY"
echo "DEVICE=$DEVICE  SEEDS=[$SEEDS]  NBOOT=$NBOOT  ONLY=$ONLY"
"$PY" -c "import torch;print('torch',torch.__version__,'mps',torch.backends.mps.is_available(),'cuda',torch.cuda.is_available())"
case "$ONLY" in
  camelyon|cam|b1|B1|both)
    if ! "$CPY" -c "import wilds" 2>/dev/null; then
      echo "[B1] ERROR: $CPY has no 'wilds'. Select a prepared CPY (or PY) runtime."
      echo "[B1] Cannot complete the requested Camelyon17 stage."
      exit 1
    fi ;;
esac
echo

# =============================================================================
# GAP B4 -- ImageNet-R multi-seed diverse-backbone panel
# =============================================================================
run_imagenetr () {
  local RUN_NAME="imagenetr_protocol_d_multiseed_v1"
  local OUT="$ROOT/experiments/kbound/results/$RUN_NAME"
  mkdir -p "$OUT"
  echo "============================================================"
  echo "[B4] ImageNet-R multi-seed -> $OUT"
  echo "     imagenetr_dir=$IMAGENETR_DIR  seeds=[$SEEDS]  frozen_eval_batch=$INR_BATCH"
  echo "============================================================"
  "${CAF[@]}" "$PY" -u experiments/kbound/wilds/run_imagenetr_kbound.py \
    --panel diverse_backbones \
    --imagenetr-dir "$IMAGENETR_DIR" \
    --seeds $SEEDS \
    --compositions iid imbalanced single_class \
    --batch-regimes small tiny \
    --aggressiveness mild aggressive \
    --n-eval "$INR_NEVAL" --n-batches 4 \
    --frozen-eval-batch "$INR_BATCH" \
    --tau-star 0.52 --kappa 2.5 --sd-L 0.6 --delta 0.05 \
    --device "$DEVICE" \
    --resume \
    --serialize-per-condition \
    --run-name "$RUN_NAME" \
    2>&1 | tee "$OUT/run_multiseed.log"

  echo "[B4] multi-seed paired-CI analysis (diverse-backbone panel) ..."
  # each frozen backbone is its own 'method' column in the per-condition files
  local METHODS
  METHODS="$("$PY" - "$OUT" <<'PYEOF'
import sys, os, glob, re
out=sys.argv[1]
m=set()
for p in glob.glob(os.path.join(out,"per_condition_imagenet-r_*_seed*.json")):
    b=os.path.basename(p)
    mm=re.match(r"per_condition_imagenet-r_(.+)_seed\d+\.json$", b)
    if mm: m.add(mm.group(1))
print(" ".join(sorted(m)))
PYEOF
)"
  if [[ -n "$METHODS" ]]; then
    "$PY" -u experiments/kbound/wilds/multiseed_paired_ci.py \
      --run-dir "$OUT" --dataset imagenet-r \
      --methods $METHODS --seeds $SEEDS --nboot "$NBOOT" \
      2>&1 | tee -a "$OUT/run_multiseed.log"
  else
    echo "[B4] WARN: no per-condition files found to analyze (did the run write any cells?)"
  fi
  echo "[B4] DONE. Result + per-condition files + MULTISEED_ANALYSIS_RESULTS.json under: $OUT"
}

# =============================================================================
# GAP B1 -- Camelyon17 SAR completion (candidate set already includes SAR)
# =============================================================================
run_camelyon () {
  local RUN_NAME="camelyon17_fullscale_B_v2"
  local OUT="$ROOT/experiments/kbound/results/$RUN_NAME"
  local F0_TEMPLATE="experiments/kbound/results/camelyon17_fullscale_B_v1/f0_seed{seed}.pt"
  mkdir -p "$OUT"
  # CPY was explicitly resolved and checked before either requested arm.
  echo "============================================================"
  echo "[B1] Camelyon17 SAR completion (resume) -> $OUT"
  echo "     data_root=$WILDS_DATA_ROOT  seeds=[$SEEDS]  candidates include SAR"
  echo "============================================================"
  "${CAF[@]}" "$CPY" -u experiments/kbound/wilds/run_camelyon17_kbound.py \
    --data-root "$WILDS_DATA_ROOT" \
    --f0-template "$F0_TEMPLATE" \
    --seeds $SEEDS \
    --domains test val id_val \
    --compositions iid imbalanced single_class \
    --batch-regimes small \
    --aggressiveness mild aggressive \
    --n-eval 1024 --n-batches 4 \
    --tau-star 0.52 --kappa 2.5 --delta 0.05 --sd-L 0.6 \
    --evidence-panel base \
    --device "$DEVICE" \
    --resume \
    --serialize-per-condition \
    --run-name "$RUN_NAME" \
    2>&1 | tee -a "$OUT/run_B_v2.log"

  echo "[B1] multi-seed paired-CI analysis (tent/eata/sar) ..."
  "$CPY" -u experiments/kbound/wilds/multiseed_paired_ci.py \
    --run-dir "$OUT" --dataset camelyon17 \
    --methods tent eata sar --seeds $SEEDS --nboot "$NBOOT" \
    2>&1 | tee -a "$OUT/run_B_v2.log"
  echo "[B1] DONE. Result + per-condition files (incl. SAR) + MULTISEED_ANALYSIS_RESULTS.json under: $OUT"
}

case "$ONLY" in
  imagenetr|inr|b4|B4) run_imagenetr ;;
  camelyon|cam|b1|B1)  run_camelyon ;;
  both)                run_imagenetr; run_camelyon ;;
esac

echo
echo "ALL REQUESTED GPU EXPERIMENTS COMPLETE."
echo "Verify success with the schema check in docs/research/kbound/RUN_ON_MAC.md."
