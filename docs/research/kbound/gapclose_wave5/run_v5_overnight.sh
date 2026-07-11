#!/bin/bash
# WIN_HUNT_v5 FULL queue: ALL nine datasets, natural-first, sequential.
# Failures logged and skipped; completed CIFAR seeds auto-skipped; never fatal.
cd /Volumes/T9/uav/AutoML_Flagship_V8 || exit 1
source ~/.venv_wilds/bin/activate
export TMPDIR=/Volumes/T9/uav/tmp
export TORCH_HOME=/Volumes/T9/uav/torch_cache
mkdir -p "$TMPDIR" "$TORCH_HOME"
WILDS_ROOT=/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/data/wilds
PACS_ROOT=/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/domainbed
OFFICEHOME_ROOT=/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/data/office_home
OFFICEHOME_SPLITS=/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/win_hunt_v5/officehome_splits.json
RXRX1_CKPT=/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/rxrx1_internal_backup/rxrx1_seed:0_epoch:best_model.pth
IMAGENETC_ROOT=/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/data/imagenet-c
step () { echo ""; echo "==================== [$(date '+%H:%M:%S')] $1 ===================="; }

step "1/9 CAMELYON17 (natural)"
python experiments/kbound/wilds/run_camelyon17_kbound.py --data-root "$WILDS_ROOT" --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --online-only || echo "[FAIL] camelyon17"

step "2/9 IMAGENET-R (natural)"
python experiments/kbound/wilds/run_imagenetr_kbound.py --panel shared_tta --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --online-only || echo "[FAIL] imagenet-r"

step "3/9 OFFICE-HOME (natural, 3 roles)"
if [ -d "$OFFICEHOME_ROOT/Real_World" ]; then
  python experiments/kbound/officehome/run_officehome_kbound.py --data-root "$OFFICEHOME_ROOT" --splits "$OFFICEHOME_SPLITS" --fresh --role source --batch-regimes small --adapt-lr 0.004 --candidates tent_online_aggressive eata_online_aggressive sar_online_aggressive || echo "[FAIL] officehome source"
  python experiments/kbound/officehome/run_officehome_kbound.py --data-root "$OFFICEHOME_ROOT" --splits "$OFFICEHOME_SPLITS" --role target_val --batch-regimes small --adapt-lr 0.004 --candidates tent_online_aggressive eata_online_aggressive sar_online_aggressive || echo "[FAIL] officehome target_val"
  python experiments/kbound/officehome/run_officehome_kbound.py --data-root "$OFFICEHOME_ROOT" --splits "$OFFICEHOME_SPLITS" --role target_test --batch-regimes small --adapt-lr 0.004 --candidates tent_online_aggressive eata_online_aggressive sar_online_aggressive || echo "[FAIL] officehome target_test"
else
  echo "[NOT_RUN] officehome: raw data not found at $OFFICEHOME_ROOT (materialize with experiments/kbound/officehome/materialize_officehome.py)"
fi

step "4/9 IWILDCAM (natural, existing f0 checkpoint, no retraining)"
# Real ERM ResNet-50 f0 (id_val acc 0.72 / test 0.69) — the protocol_H run was built on this.
IWC_CKPT=experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt
echo "iwildcam f0 ckpt: ${IWC_CKPT} ($([ -f "$IWC_CKPT" ] && echo present || echo MISSING))"
python experiments/kbound/wilds/run_iwildcam_kbound.py --data-root "$WILDS_ROOT" --ckpt "$IWC_CKPT" --backbone resnet50 --split test --seeds 0 1 --run-name win_hunt_v5_iwildcam --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --candidates tent_online eata_online sar_online || \
  echo "[NOT_RUN] iwildcam: ckpt or data issue at $WILDS_ROOT"

step "5/9 RXRX1 (natural, two data-root attempts)"
python experiments/kbound/wilds/run_rxrx1_kbound.py --data-root "$WILDS_ROOT" --ckpt "$RXRX1_CKPT" --results-root experiments/kbound/results/win_hunt_v5 --run-name rxrx1_aggr --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --online-only || \
  echo "[NOT_RUN] rxrx1: data not found or incomplete at $WILDS_ROOT"

step "6/9 PACS 4 splits (natural/domain-gen)"
python docs/research/kbound/scripts/pacs_vlcs_runner.py --dataset PACS --root "$PACS_ROOT" --batch-regimes tiny --aggressiveness aggressive --adapt-lr 0.004 --out experiments/kbound/results/win_hunt_v5/pacs_aggr/pacs_result.json || echo "[FAIL] pacs"

step "7/9 CIFAR-10.1 seeds 0-4 (natural shift)"
for S in 0 1 2 3 4; do
  if [ -f "experiments/kbound/results/win_hunt_v5/cifar101_aggr/seed$S/result_manifest.json" ]; then echo "[SKIP] cifar101 seed $S complete"; continue; fi
  python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks cifar101 --data-root experiments/kbound/cifar --methods tent eata sar --device mps --seed "$S" --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --out-results experiments/kbound/results/win_hunt_v5/cifar101_aggr/seed$S || echo "[FAIL] cifar101 seed $S"
done

step "8/9 CIFAR-10-C seeds 0-4 (skips completed)"
for S in 0 1 2 3 4; do
  if [ -f "experiments/kbound/results/win_hunt_v5/cifar10c_aggr/seed$S/result_manifest.json" ]; then echo "[SKIP] cifar10c seed $S complete"; continue; fi
  python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks cifar10c --data-root experiments/kbound/cifar --methods tent eata sar --device mps --seed "$S" --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --out-results experiments/kbound/results/win_hunt_v5/cifar10c_aggr/seed$S || echo "[FAIL] cifar10c seed $S"
done

step "9/9 IMAGENET-C"
python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks imagenetc --imagenetc-root "$IMAGENETC_ROOT" --methods tent eata sar --device mps --seed 0 --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --imagenetc-composition iid imbalanced single_class --out-results experiments/kbound/results/win_hunt_v5/imagenetc_aggr || echo "[FAIL] imagenetc"

step "QUEUE COMPLETE — summary of problems (if any):"
grep -E "\[(FAIL|NOT_RUN)\]" /Volumes/T9/uav/v5_overnight.log 2>/dev/null || echo "none"
