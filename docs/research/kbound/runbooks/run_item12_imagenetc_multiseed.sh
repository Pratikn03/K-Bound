#!/usr/bin/env bash
# Item 12: multi-seed ImageNet-C (seeds 1-4; seed 0 is the locked 2026-07-09 full-scale run).
# Protocol identical to the authoritative 27-cell configuration: ResNet-50, 3 noise corruptions,
# severities {1,3,5}, compositions {iid, imbalanced, single_class}, Tent/EATA/SAR
# (mechanism-faithful SAR, shared lr). Only the seed varies.
#
# GPU (or Apple mps) + ImageNet-C noise subset required
# (scripts/download_imagenetc_safe.sh / extract_imagenetc_requested_groups.sh).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
K="$REPO/docs/research/kbound"
IC="${IC:-$REPO/experiments/kbound/data/imagenet-c}"
PY="${PY:-python3}"
DEVICE="${DEVICE:-cuda}"   # or mps

for S in 1 2 3 4; do
  OUTDIR="$REPO/experiments/kbound/results/imagenetc_official_sar_E_v1_s$S"
  if [ -f "$OUTDIR/DONE" ]; then echo "[item12] seed $S already done"; continue; fi
  mkdir -p "$OUTDIR"
  "$PY" "$K/scripts/cifar_tent_mps_v2.py" \
    --benchmarks imagenetc --imagenetc-root "$IC" --arch resnet50 \
    --methods tent eata sar --device "$DEVICE" --seed "$S" \
    --severities 1 3 5 \
    --imagenetc-composition iid imbalanced single_class \
    --out-results "$OUTDIR" \
    2>&1 | tee -a "$OUTDIR/progress.log"
  touch "$OUTDIR/DONE"
done

# Per-seed paired bootstrap (same 2-comparison family + Holm as the seed-0 lock), then aggregate.
for S in 1 2 3 4; do
  "$PY" "$K/scripts/percondition_bootstrap.py" \
    --results "$REPO/experiments/kbound/results/imagenetc_official_sar_E_v1_s$S" \
    --candidate sar --alpha 0.10 || echo "[item12] bootstrap for seed $S needs the per-seed JSON path — check scorer output layout"
done
"$PY" "$K/scripts/multiseed_aggregate.py" \
  --pattern "$REPO/experiments/kbound/results/imagenetc_official_sar_E_v1_s*" \
  --out "$REPO/experiments/kbound/results/imagenetc_multiseed_v1.json" || \
  echo "[item12] aggregate: wire the per-seed schema into multiseed_aggregate.py if flags differ"

echo "[item12] Criterion: beats-both per seed (both paired CIs exclude 0, Holm), FA_u<=alpha every seed."
echo "[item12] On success, paper tier: ImageNet-C SAR -> locked (5 seeds); drop the one-seed caveats."
