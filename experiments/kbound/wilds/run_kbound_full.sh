#!/bin/bash
# REAL iWildCam K-Bound FULL evaluation on the trained ResNet-50 f0.
# 6 candidates {tent,eata,sar}x{online,episodic}; comps iid/imbalanced/single_class;
# regimes tiny/small; aggr mild/aggressive; >=2 seeds. SOURCE=id_val, TARGET=val(+test).
# Source-calibrated analyzer. MPS; runs in background via caffeinate.
# Order: source -> val target -> VAL VERDICT -> test target -> TEST VERDICT
# (so the primary val verdict is produced before the optional test target).
set -e
cd /Volumes/T9/uav/AutoML_Flagship_V8
PY=~/.venv_wilds/bin/python
CK=${CK:-experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt}
MAXLOC=${MAXLOC:-6}
SEEDS=${SEEDS:-0 1}
CANDS="tent_online eata_online sar_online tent_episodic eata_episodic sar_episodic"
run () {
  PYTORCH_ENABLE_MPS_FALLBACK=1 $PY -u experiments/kbound/wilds/run_iwildcam_kbound.py \
    --device mps --ckpt "$CK" --backbone resnet50 \
    --max-locations $MAXLOC --compositions iid imbalanced single_class \
    --batch-regimes tiny small --aggressiveness mild aggressive --candidates $CANDS \
    --n-eval 48 --n-batches 2 --seeds $SEEDS --tau-star 0.52 \
    --eval-bs 48 --episodic-batch 24 --episodic-steps 3 \
    --split "$1" --run-name "$2"
}
echo "[full] CK=$CK MAXLOC=$MAXLOC SEEDS=$SEEDS start=$(date +%H:%M:%S)"
run id_val iwildcam_full_idval
run val    iwildcam_full_val
SRC=$(ls experiments/kbound/results/iwildcam_full_idval/result_*.json | head -1)
TGV=$(ls experiments/kbound/results/iwildcam_full_val/result_*.json | head -1)
$PY experiments/kbound/wilds/analyze_iwildcam_kbound.py --source "$SRC" --target "$TGV" \
    --out experiments/kbound/results/iwildcam_full_val/VERDICT_val.json
echo "VAL_VERDICT_DONE $(date +%H:%M:%S)"
run test   iwildcam_full_test
TGT=$(ls experiments/kbound/results/iwildcam_full_test/result_*.json | head -1)
$PY experiments/kbound/wilds/analyze_iwildcam_kbound.py --source "$SRC" --target "$TGT" \
    --out experiments/kbound/results/iwildcam_full_test/VERDICT_test.json
echo "FULL_DONE rc=$? $(date +%H:%M:%S)"
