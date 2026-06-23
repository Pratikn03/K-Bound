#!/usr/bin/env bash
# Protocol E (FULL ImageNet-C, tar-streaming) - official gentle SAR, 3 seeds.
# Registered: research_lock/imagenetc_protocol_E_v1.yaml
# Full 19 corruptions x severities {4,5} x compositions {iid,imbalanced,single_class},
# served by TAR-STREAMING (no extraction). DO NOT run while ELARA-Opt Stage 2 is on MPS.
set -u
REPO=/Volumes/T9/uav/AutoML_Flagship_V8
PY="$HOME/.venv_wilds/bin/python"
SCRIPT="$REPO/docs/research/kbound/scripts/cifar_tent_mps_v2.py"
IC="$REPO/experiments/kbound/data/imagenet-c"
BASE="$REPO/experiments/kbound/results/imagenetc_protocol_E_full_v1"
CORR="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression speckle_noise gaussian_blur spatter saturate"
cd "$REPO" || exit 1
mkdir -p "$BASE"
export TMPDIR=/Volumes/T9/uav/tmp TORCH_HOME=/Volumes/T9/uav/torch_cache
mkdir -p "$TMPDIR" "$TORCH_HOME"

# --- QUEUE GUARD: refuse to start while ELARA-Opt Stage 2 (or any MPS job) is running ---
if pgrep -f "elara_stage2_run.py" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] ELARA-Opt Stage 2 still running on MPS -> NOT starting (queue). Re-run when it finishes." | tee -a "$BASE/launch.log"
  exit 3
fi

# --- SINGLE-INSTANCE LOCK: refuse to start if another Protocol-E launcher is alive ---
# Prevents a second `bash run_protocol_E_full.sh` from spawning a parallel worker on the
# same seed dir (the collision that happened before). Stale locks (dead pid) are ignored.
LOCK="$BASE/.run.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date '+%F %T')] REFUSING: another Protocol-E launcher is active (pid $(cat "$LOCK")). Only one run at a time." | tee -a "$BASE/launch.log"
  exit 4
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
echo "[$(date '+%F %T')] lock acquired (pid $$)" >> "$BASE/launch.log"

echo "[$(date '+%F %T')] PROTOCOL E (full) QUEUE START" >> "$BASE/launch.log"
for S in 0 1 2; do
  OUT="$BASE/seed$S"
  mkdir -p "$OUT"
  if [[ -f "$OUT/result_manifest.json" ]] && [[ -f "$OUT/decisive_tta_results.json" ]]; then
    echo "[$(date '+%F %T')] SEED $S SKIP (already complete)" >> "$BASE/launch.log"; continue
  fi
  echo "[$(date '+%F %T')] SEED $S START -> $OUT" >> "$BASE/launch.log"
  caffeinate -is "$PY" "$SCRIPT" \
    --benchmarks imagenetc \
    --imagenetc-root "$IC" \
    --corruptions $CORR \
    --severities 4 5 \
    --imagenetc-composition iid imbalanced single_class \
    --arch resnet50 \
    --methods tent eata sar \
    --sar-lr 2.5e-4 --sar-freeze-layer4 \
    --seed "$S" \
    --cooldown 3 \
    --out-results "$OUT" \
    --out-figs "$OUT/figs" \
    >> "$OUT/seed${S}.log" 2>&1
  echo "[$(date '+%F %T')] SEED $S DONE rc=$?" >> "$BASE/launch.log"
done
echo "[$(date '+%F %T')] PROTOCOL E (full) QUEUE COMPLETE" >> "$BASE/launch.log"
