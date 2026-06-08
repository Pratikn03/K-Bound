#!/usr/bin/env bash
# quick_check.sh  --  ~1% SMOKE of both remaining runs, end-to-end, in a few minutes.
# Confirms the pipelines are healthy BEFORE the multi-hour full run.
# Safe: writes only to *_smoke result dirs; does NOT touch real results.
#
#   bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/quick_check.sh
#
set -uo pipefail
K=/Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/kbtrain.sh

echo "##########################################################"
echo "# QUICK CHECK 1/2 : ImageNet-C noise  (FULL grid 3x3x3, ~1% = 50 imgs/cell, internal SSD)"
echo "##########################################################"
if bash "$K" noise-fast-1pct; then echo "  -> ImageNet-C pipeline: OK"; else echo "  -> ImageNet-C pipeline: FAILED"; exit 1; fi

echo
echo "##########################################################"
echo "# QUICK CHECK 2/2 : Camelyon17  (4 seeds x 1 epoch x 1% data, internal SSD)"
echo "##########################################################"
if bash "$K" camelyon-fast-1pct; then echo "  -> Camelyon pipeline: OK"; else echo "  -> Camelyon pipeline: FAILED"; exit 1; fi

echo
echo "=========================================================="
echo " BOTH PIPELINES HEALTHY (numbers are meaningless at this scale)."
echo " Now launch the real runs:"
echo "   bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/full_run.sh"
echo "=========================================================="
