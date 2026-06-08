#!/usr/bin/env bash
# download_imagenetc_safe.sh — SPACE-SAFE ImageNet-C downloader (macOS).
#
# Downloads each corruption tar, extracts it, then DELETES the tar BEFORE the
# next one. Peak disk use stays ~40 GB instead of the ~140 GB the all-at-once
# approach needed (that is what filled your drive last time).
#
# Source: Zenodo record 2235448 (Hendrycks & Dietterich, ICLR 2019).
#
# Usage:
#   bash download_imagenetc_safe.sh                 # all 5 groups (~62 GB final)
#   CORRUPTIONS="noise digital" bash download_imagenetc_safe.sh   # just a subset
#   DATA_ROOT=/some/other/path bash download_imagenetc_safe.sh
#
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/data/imagenet-c}"
# Which groups to fetch (space separated). Default = all. Override to save space.
CORRUPTIONS="${CORRUPTIONS:-noise blur weather digital extra}"
BASE="https://zenodo.org/records/2235448/files"
MIN_FREE_GB="${MIN_FREE_GB:-45}"   # refuse to start a tar if free space below this

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT" || { echo "cannot cd $DATA_ROOT"; exit 1; }

free_gb () { df -g "$DATA_ROOT" | awk 'NR==2{print $4}'; }

echo "Target : $DATA_ROOT"
echo "Groups : $CORRUPTIONS"
echo "Free   : $(free_gb) GB"
echo "------------------------------------------------------------"

for c in $CORRUPTIONS; do
  if [ -f ".done_$c" ]; then echo "[$c] already extracted, skipping"; continue; fi

  f=$(free_gb)
  if [ "$f" -lt "$MIN_FREE_GB" ]; then
    echo "!! Only ${f} GB free (need >= ${MIN_FREE_GB}). Free space or trim CORRUPTIONS, then re-run."
    exit 1
  fi

  echo "==> [$c] downloading ${c}.tar  (resumable)"
  curl -L -C - -o "${c}.tar" "${BASE}/${c}.tar?download=1" \
    || { echo "!! download failed for $c — re-run to resume"; exit 1; }

  echo "==> [$c] extracting ..."
  if tar xf "${c}.tar" -C "$DATA_ROOT"; then
    rm -f "${c}.tar"          # delete tar immediately to reclaim space
    touch ".done_$c"
    echo "==> [$c] DONE.  Free now: $(free_gb) GB"
  else
    echo "!! extraction failed for $c (probably out of space). Tar kept: ${c}.tar"
    exit 1
  fi
  echo
done

echo "============================================================"
echo "All requested groups complete."
echo "Layout: $DATA_ROOT/<corruption>/<severity 1-5>/<class>/*.JPEG"
echo "Run:    python docs/research/kbound/scripts/cifar_tent_mps_v2.py \\"
echo "          --benchmarks imagenetc --imagenetc-root $DATA_ROOT --arch resnet50 --quick"
