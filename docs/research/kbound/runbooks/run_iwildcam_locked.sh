#!/usr/bin/env bash
# iWildCam 5-seed with the REAL locked ERM ResNet-50 checkpoint (from T9, copied internal).
# Fixes the chain's junk finder run (throwaway ResNet-18, 0.6% acc). Waits for the item-11
# AETTA TTA to free the GPU, then runs run_iwildcam_aetta.py (--ckpt required, backbone rn50).
set -u
R="$HOME/Documents/AutoML_Flagship_V8"
CKPT="$R/experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt"
OUT="$R/experiments/kbound/results/multiseed/iwildcam"
LOG="$OUT/iwildcam_locked.log"; PY=/opt/anaconda3/envs/aetta/bin/python
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "WAIT: item-11 AETTA TTA to free the GPU"
while pgrep -f "eval_results_0_dist1|tta_item11|run_item11_finish|main.py.*cifar10outdist" >/dev/null 2>&1; do sleep 180; done
say "GPU idle — iWildCam locked 5-seed (real ResNet-50 ERM ckpt)"
[ -f "$CKPT" ] || { say "FATAL: ckpt missing $CKPT"; exit 2; }
caffeinate -is "$PY" "$R/experiments/kbound/wilds/run_iwildcam_aetta.py" \
  --ckpt "$CKPT" --backbone resnet50 --split test --seeds 0 1 2 3 4 \
  --max-locations 4 --device auto \
  --results-root "$OUT" --run-name iwildcam_locked_v1 >> "$LOG" 2>&1
say "iwildcam run rc=$?"
/opt/anaconda3/bin/python3 "$R/docs/research/kbound/scripts/extract_multiseed_natural.py" --track iwildcam \
  --result "$OUT/iwildcam_locked_v1/**/result_*.json" --candidates tent_online \
  --out-dir "$OUT/extracted_locked" >> "$LOG" 2>&1
say "iwildcam extract rc=$? — DONE (real-model 5-seed)"
