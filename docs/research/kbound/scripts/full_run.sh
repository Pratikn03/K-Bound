#!/usr/bin/env bash
# full_run.sh  --  the REAL runs, back-to-back, one GPU job at a time. Set-and-forget.
# Runs ImageNet-C noise (full, fast internal copy, ResNet-50), then Camelyon17 4x4.
#
#   bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/full_run.sh
#
# BEFORE running: plug in + lid open  (or: sudo pmset -c disablesleep 1).
# Run quick_check.sh first to confirm both pipelines are healthy.
set -uo pipefail
K=/Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/kbtrain.sh
R=/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results
echo "start: $(date)"

echo "##########################################################"
echo "# FULL RUN 1/2 : ImageNet-C noise  (ResNet-50, full grid, internal SSD)"
echo "##########################################################"
bash "$K" noise-fast
echo "  -> ImageNet-C done: $(ls -1 "$R/imagenetc_noise/" 2>/dev/null | tr '\n' ' ')"

echo
echo "##########################################################"
echo "# FULL RUN 2/2 : Camelyon17  (4 seeds x 4 epochs, --retrain, internal SSD)"
echo "##########################################################"
bash "$K" camelyon-fast
echo "  -> Camelyon done: $(ls -1 "$R/wilds/" 2>/dev/null | tr '\n' ' ')"

echo
echo "=========================================================="
echo " ALL DONE @ $(date)"
echo "   ImageNet-C : $R/imagenetc_noise/decisive_tta_results.json"
echo "   Camelyon   : $R/wilds/wilds_camelyon17_kga.json"
echo " Paste both result blocks for the paper. (Optional next: kbtrain.sh vit-fast)"
echo "=========================================================="
