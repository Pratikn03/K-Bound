#!/usr/bin/env bash
# quick_check.sh -- 0.1% SMOKE of both pending datasets, end-to-end, in a few minutes.
# Confirms the pipelines (incl. progress.log + checkpoint) are healthy BEFORE the full run.
# Writes ONLY to *_smoke* dirs -> never touches real results or their resume checkpoints.
set -uo pipefail
K=/Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/kbtrain.sh
echo "########## QUICK 0.1% SMOKE 1/2 : ImageNet-C noise (full 3x3 grid, ~8 imgs/cell) ##########"
if bash "$K" noise-fast-01pct; then echo "  -> ImageNet-C pipeline: OK"; else echo "  -> ImageNet-C pipeline: FAILED"; exit 1; fi
echo
echo "########## QUICK 0.1% SMOKE 2/2 : CIFAR-10.1 natural shift (quick subset) ##########"
if bash "$K" cifar101-quick; then echo "  -> CIFAR-10.1 pipeline: OK"; else echo "  -> CIFAR-10.1 pipeline: FAILED"; exit 1; fi
echo
echo "=================================================================="
echo " BOTH PIPELINES HEALTHY (numbers are meaningless at 0.1%)."
echo " Now launch the real runs:"
echo "   bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/full_run.sh"
echo "=================================================================="
