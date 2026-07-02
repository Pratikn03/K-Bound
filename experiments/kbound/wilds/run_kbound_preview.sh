#!/bin/bash
# REAL iWildCam K-Bound PREVIEW (small grid) on the trained ResNet-50 f0.
# Runs SOURCE (id_val) + TARGET (val), then the source-calibrated analyzer.
# MPS; do NOT run while f0 training holds the GPU.
set -e
cd /Volumes/T9/uav/AutoML_Flagship_V8
PY=~/.venv_wilds/bin/python
CK=${CK:-experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt}
CANDS="tent_online eata_online sar_online tent_episodic eata_episodic sar_episodic"
run () {
  PYTORCH_ENABLE_MPS_FALLBACK=1 $PY -u experiments/kbound/wilds/run_iwildcam_kbound.py \
    --device mps --ckpt "$CK" --backbone resnet50 \
    --max-locations 3 --compositions iid single_class --batch-regimes tiny \
    --aggressiveness mild aggressive --candidates $CANDS \
    --n-eval 48 --n-batches 2 --seeds 0 --tau-star 0.52 \
    --eval-bs 48 --episodic-batch 24 --episodic-steps 3 \
    --split "$1" --run-name "$2"
}
echo "[preview] CK=$CK"
run id_val iwildcam_preview_idval
run val   iwildcam_preview_val
SRC=$(ls experiments/kbound/results/iwildcam_preview_idval/result_*.json | head -1)
TGT=$(ls experiments/kbound/results/iwildcam_preview_val/result_*.json | head -1)
$PY experiments/kbound/wilds/analyze_iwildcam_kbound.py --source "$SRC" --target "$TGT" \
    --out experiments/kbound/results/iwildcam_preview_val/VERDICT_preview.json
echo "PREVIEW_DONE rc=$?"
