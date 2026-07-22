#!/usr/bin/env bash
set -u; R="$HOME/Documents/AutoML_Flagship_V8"
CKPT="$R/experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt"
OUT="$R/experiments/kbound/results/multiseed/iwildcam"; PY=/opt/anaconda3/envs/aetta/bin/python
LOG="$OUT/iwildcam_episodic.log"; say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "iWildCam tent_episodic 5-seed (real ResNet-50 ERM ckpt) — matches paper's Protocol-H candidate"
cd "$R/experiments/kbound/wilds"
caffeinate -is "$PY" run_iwildcam_aetta.py --ckpt "$CKPT" --backbone resnet50 --split test \
  --seeds 0 1 2 3 4 --max-locations 4 --candidates tent_episodic --device auto \
  --results-root "$OUT" --run-name iwildcam_episodic_v1 >> "$LOG" 2>&1
say "run rc=$?"
/opt/anaconda3/bin/python3 "$R/docs/research/kbound/scripts/extract_multiseed_natural.py" --track iwildcam \
  --result "$OUT/iwildcam_episodic_v1/**/result_*.json" --candidates tent_episodic \
  --out-dir "$OUT/extracted_episodic" >> "$LOG" 2>&1
say "extract rc=$? — DONE"
