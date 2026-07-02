#!/bin/bash
# 1% AETTA-detector PREVIEW: source(id_val)+target(val), small grid, online candidates.
set -e
cd /Volumes/T9/uav/AutoML_Flagship_V8
PY=~/.venv_wilds/bin/python
CK=experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt
run () {
  PYTORCH_ENABLE_MPS_FALLBACK=1 $PY -u experiments/kbound/wilds/run_iwildcam_aetta.py \
    --ckpt "$CK" --backbone resnet50 --device mps \
    --max-locations 4 --compositions iid single_class --batch-regimes tiny \
    --aggressiveness mild aggressive --candidates tent_online eata_online sar_online \
    --n-eval 48 --n-batches 2 --seeds 0 --mc-M 8 --mc-p 0.4 \
    --split "$1" --run-name "$2"
}
echo "[aetta-preview] start $(date +%H:%M:%S)"
run id_val iwildcam_aetta_prev_idval
run val    iwildcam_aetta_prev_val
SRC=$(ls experiments/kbound/results/iwildcam_aetta_prev_idval/result_*.json | head -1)
TGT=$(ls experiments/kbound/results/iwildcam_aetta_prev_val/result_*.json | head -1)
$PY experiments/kbound/wilds/analyze_iwildcam_detector.py --source "$SRC" --target "$TGT" \
    --out experiments/kbound/results/iwildcam_aetta_prev_val/VERDICT_detector_preview.json
echo "AETTA_PREVIEW_DONE rc=$? $(date +%H:%M:%S)"
