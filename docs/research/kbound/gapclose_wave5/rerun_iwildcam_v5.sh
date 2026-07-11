#!/bin/bash
# iWildCam v5 aggressive rerun with the REAL ResNet-50 ERM f0 (id_val 0.72 / test 0.69).
# Single python invocation — no line-continuation to break on paste.
set -e
cd /Volumes/T9/uav/AutoML_Flagship_V8
source ~/.venv_wilds/bin/activate

CKPT=experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt
echo "f0 ckpt: $CKPT ($([ -f "$CKPT" ] && echo present || echo MISSING))"

# clear any invalid stub run so the rerun starts clean
rm -rf experiments/kbound/results/win_hunt_v5_iwildcam

python experiments/kbound/wilds/run_iwildcam_kbound.py --data-root experiments/kbound/data/wilds --ckpt "$CKPT" --backbone resnet50 --split test --seeds 0 1 --run-name win_hunt_v5_iwildcam --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --candidates tent_online eata_online sar_online
