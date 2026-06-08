#!/usr/bin/env bash
# prep_internal_noise.sh
# Copy the ImageNet-C *noise* data the standard grid needs (gaussian/shot/impulse,
# severities 1/3/5) from the slow exFAT T9 drive to the fast internal SSD (~/kbound_inc),
# so the TTA run loads ~10x faster. Resumable (rsync): if interrupted, just re-run.
set -uo pipefail

SRC=/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/data/imagenet-c
DST="$HOME/kbound_inc"

echo "============================================================"
echo " Copying ImageNet-C noise (sev 1/3/5) -> internal SSD"
echo "   from: $SRC"
echo "   to:   $DST"
echo " Free internal BEFORE: $(df -h / | awk 'NR==2{print $4}')"
echo "============================================================"
mkdir -p "$DST"

for c in gaussian_noise shot_noise impulse_noise; do
  for s in 1 3 5; do
    if [ -d "$SRC/$c/$s" ]; then
      echo "==> $c/$s ..."
      mkdir -p "$DST/$c"
      rsync -a --exclude '._*' "$SRC/$c/$s" "$DST/$c/"
    else
      echo "!! source missing: $c/$s (skipping)"
    fi
  done
done

echo "============================================================"
echo " Free internal AFTER: $(df -h / | awk 'NR==2{print $4}')"
echo " Verify (each severity should show ~1000 class dirs):"
ok=1
for c in gaussian_noise shot_noise impulse_noise; do
  for s in 1 3 5; do
    n=$(ls -1 "$DST/$c/$s" 2>/dev/null | grep -v '^\._' | wc -l | tr -d ' ')
    printf "   %-16s sev%s : %s class dirs\n" "$c" "$s" "$n"
    [ "${n:-0}" -lt 900 ] && ok=0
  done
done
echo "============================================================"
if [ "$ok" = 1 ]; then
  echo " READY. Start training with:"
  echo "   bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/kbtrain.sh noise-fast"
else
  echo " WARNING: some severities look incomplete (<900 class dirs)."
  echo " Re-run this script (rsync resumes) before training."
fi
