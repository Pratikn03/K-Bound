#!/usr/bin/env bash
# Resume/download and extract the ImageNet-C groups needed for:
# defocus_blur, motion_blur, snow, fog, brightness, jpeg_compression.
#
# Official ImageNet-C is packaged by group:
#   blur.tar    -> defocus_blur, glass_blur, motion_blur, zoom_blur
#   weather.tar -> snow, frost, fog, brightness
#   digital.tar -> contrast, elastic_transform, pixelate, jpeg_compression
#
# The script deletes a group tar only after successful extraction.
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/data/imagenet-c}"
IMAGENETC_GROUPS="${IMAGENETC_GROUPS:-blur weather digital}"
BASE="${BASE:-https://zenodo.org/records/2235448/files}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT" || { echo "cannot cd $DATA_ROOT"; exit 1; }

free_gb() { df -g "$DATA_ROOT" | awk 'NR==2{print $4}'; }
tar_valid() { [ -f "$1" ] && tar tf "$1" >/dev/null 2>&1; }

echo "Target : $DATA_ROOT"
echo "Groups : $IMAGENETC_GROUPS"
echo "Free   : $(free_gb) GB"
echo "------------------------------------------------------------"

for group in $IMAGENETC_GROUPS; do
  if [ -f ".done_${group}" ]; then
    echo "[$group] already extracted, skipping"
    continue
  fi

  f=$(free_gb)
  if [ "$f" -lt "$MIN_FREE_GB" ]; then
    echo "!! Only ${f} GB free; need >= ${MIN_FREE_GB} GB before ${group}."
    exit 1
  fi

  if tar_valid "${group}.tar"; then
    echo "==> [$group] using existing valid ${group}.tar"
  else
    echo "==> [$group] downloading ${group}.tar (resumable)"
    curl -fL -C - -o "${group}.tar" "${BASE}/${group}.tar?download=1" || {
      echo "!! download failed for $group; re-run to resume"
      exit 1
    }
    if ! tar_valid "${group}.tar"; then
      echo "!! ${group}.tar is not a valid tar after download"
      exit 1
    fi
  fi

  echo "==> [$group] extracting ..."
  if tar xf "${group}.tar" -C "$DATA_ROOT"; then
    rm -f "${group}.tar" "._${group}.tar"
    touch ".done_${group}"
    echo "==> [$group] DONE. Free now: $(free_gb) GB"
  else
    echo "!! extraction failed for $group; tar kept at $DATA_ROOT/${group}.tar"
    exit 1
  fi
  echo
done

echo "============================================================"
echo "Requested ImageNet-C groups complete."
for c in defocus_blur motion_blur snow fog brightness jpeg_compression; do
  if [ -d "$DATA_ROOT/$c" ]; then
    echo "OK $c"
  else
    echo "MISSING $c"
  fi
done
