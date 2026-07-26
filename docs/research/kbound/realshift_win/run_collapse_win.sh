#!/usr/bin/env bash
# ============================================================================
# GPU run: chase a CI-robust PURE-label-free beats-both on a real shift (iWildCam).
#
# Strategy (from docs/research/kbound/realshift_win/PROTOCOL_realshift_win.md):
#   - generate a TWO-SIDED panel (benign iid/mild windows where TTA helps +
#     trap single_class/aggressive windows where it collapses) so the deployed
#     adapter is genuinely mixed (not net-harmful) -> neither trivial policy is oracle
#   - n >= 240 conditions (2 seeds x 10 locs x 3 comps x 2 regimes x 2 aggr = 240)
#   - good ResNet-50 ERM f0 (NOT the head-only chance model)
#   - detector = collapse-entropy (computed post-hoc from logged preds; shown to
#     transfer source->OOD at ~0.64 where calibrated detectors hit ~0.43)
#   - dev-lock on id_val (source), score val (OOD) ONCE through the locked verifier
#
# Reuses your existing experiments/kbound/wilds/run_iwildcam_kbound.py unchanged.
# Set DEVICE=cuda on your GPU box (use mps on a Mac). Resumable via _partial.json.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../../../.."                      # -> repo root
PY="${PY:-$HOME/.venv_wilds/bin/python}"
DEVICE="${DEVICE:-cuda}"
F0="experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt"
RES="experiments/kbound/results"

[ -f "$F0" ] || { echo "missing f0 checkpoint: $F0"; exit 1; }

COMMON="--backbone resnet50 --ckpt $F0 --seeds 0 1 --max-locations 10 \
  --compositions iid imbalanced single_class --batch-regimes tiny small \
  --aggressiveness mild aggressive --candidates tent_online eata_online sar_online \
  --n-eval 128 --n-batches 4 --device $DEVICE"

echo '=== [1/3] SOURCE panel (id_val = calibration/dev-lock) ==='
$PY experiments/kbound/wilds/run_iwildcam_kbound.py $COMMON --split id_val \
  --run-name collapse_win_source --out "$RES/collapse_win_source.json"

echo '=== [2/3] TARGET panel (val = held-out OOD, scored once) ==='
$PY experiments/kbound/wilds/run_iwildcam_kbound.py $COMMON --split val \
  --run-name collapse_win_target --out "$RES/collapse_win_target.json"

echo '=== [3/3] VERDICT (locked verifier; collapse-entropy detector) ==='
$PY docs/research/kbound/realshift_win/analyze_collapse_win.py \
  --source "$RES/collapse_win_source.json" \
  --target "$RES/collapse_win_target.json"

echo
echo "Done. A '*** CI-ROBUST BEATS-BOTH ***' line is the win; anything else is an honest null."
echo "If the TARGET panel is not two-sided (harm fraction <0.25 or >0.60), adjust the composition"
echo "mix (more/less single_class+aggressive) and rerun BOTH panels — never tune on the target."
