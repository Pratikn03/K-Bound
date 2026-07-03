#!/usr/bin/env bash
# full_run.sh -- the REAL runs on the full datasets, one GPU job at a time.
#   ImageNet-C noise (full, ResNet-50, cooldown 3s, checkpoint/resume) then CIFAR-10.1 (full).
# RESUMABLE: if the Mac sleeps/overheats/shuts down, just RE-RUN this -- finished cells skip.
set -uo pipefail
K=/Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/kbtrain.sh
R=/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results
echo "start: $(date)"
echo "WATCH progress live in another terminal:"
echo "   tail -f $R/imagenetc_noise/progress.log"
echo
echo "########## FULL 1/2 : ImageNet-C noise (full grid, internal SSD, resumable) ##########"
bash "$K" noise-fast
echo "  -> ImageNet-C: $(ls -1 "$R/imagenetc_noise/" 2>/dev/null | tr '\n' ' ')"
echo
echo "########## FULL 2/2 : CIFAR-10.1 natural shift (full) ##########"
bash "$K" cifar101
echo "  -> CIFAR-10.1: $(ls -1 "$R/cifar101/" 2>/dev/null | tr '\n' ' ')"
echo
echo "=================================================================="
echo " ALL DONE @ $(date)"
echo "   ImageNet-C : $R/imagenetc_noise/decisive_tta_results.json"
echo "   CIFAR-10.1 : $R/cifar101/decisive_tta_results.json"
echo " Paste both === blocks (and the progress.log tail) for the paper."
echo "=================================================================="
