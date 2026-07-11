#!/bin/bash
# ImageNet-C v5 — run from INTERNAL SSD. The external exFAT drive was ~65 min/cell (~6 days);
# internal NVMe should be a few min/cell. Self-contained + idempotent: safe to re-run.
#   - kills any lingering run
#   - clears the OLD internal copy and OLD output so nothing merges
#   - copies data T9 -> internal (~/imagenetc_local) with ditto (fast for many small files)
#   - auto scope: FULL 15 if >=60 GB free on /, else the 3-noise protocol-E set
#   - launches under caffeinate so the machine/drive never sleep-stalls again
set -u
SRC=/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/data/imagenet-c
DST=$HOME/imagenetc_local
REPO=/Volumes/T9/uav/AutoML_Flagship_V8
OUT=$REPO/experiments/kbound/results/win_hunt_v5/imagenetc_aggr

cd "$REPO" || exit 1
source ~/.venv_wilds/bin/activate
export TORCH_HOME=/Volumes/T9/uav/torch_cache   # reuse the already-cached resnet50 weights

echo "== kill any lingering imagenet-c run =="
pkill -9 -f cifar_tent_mps_v2 2>/dev/null || true
sleep 2

echo "== clear OLD internal copy + OLD output (fresh start) =="
rm -rf "$DST" "$OUT"
mkdir -p "$DST"

FREE_GB=$(df -g / | awk 'NR==2{print $4}')
echo "== internal free space: ${FREE_GB} GB =="

if [ "$FREE_GB" -ge 60 ]; then
  echo "== scope: FULL 15 corruptions =="
  CORRS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
  for d in gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur contrast elastic_transform; do
    echo "  copying $d ..."; ditto "$SRC/$d" "$DST/$d"
  done
  echo "  copying weather.tar + digital.tar ..."; cp "$SRC/weather.tar" "$SRC/digital.tar" "$DST/"
else
  echo "== scope: 3 noise corruptions (internal < 60 GB; matches protocol E) =="
  CORRS="gaussian_noise shot_noise impulse_noise"
  for d in gaussian_noise shot_noise impulse_noise; do
    echo "  copying $d ..."; ditto "$SRC/$d" "$DST/$d"
  done
fi

echo "== internal copy contents: =="; ls -1 "$DST"
echo "== launching run from internal disk (caffeinated) =="
caffeinate -is python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks imagenetc --methods tent eata sar --device mps --seed 0 --imagenetc-root "$DST" --corruptions $CORRS --arch resnet50 --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --imagenetc-composition iid imbalanced single_class --out-results "$OUT"
