#!/usr/bin/env bash
# Gated, resumable closure for the remaining K-Bound empirical runs.
# Usage:
#   bash docs/research/kbound/runbooks/finish_empirical_training.sh preflight
#   bash docs/research/kbound/runbooks/finish_empirical_training.sh smoke
#   caffeinate -is bash docs/research/kbound/runbooks/finish_empirical_training.sh run
#   bash docs/research/kbound/runbooks/finish_empirical_training.sh finalize
# --- interpreter: $KBOUND_PYTHON, default python3 (was a hard-coded venv path).
KB_PYTHON="${KBOUND_PYTHON:-python3}"

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
K="$ROOT/docs/research/kbound"
RES="$ROOT/experiments/kbound/results"
AETTA="$ROOT/AETTA"
PY="${PY:-"$KB_PYTHON"}"
CONDA="${CONDA:-/opt/anaconda3/bin/conda}"
MODE="${1:-preflight}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/kbound-matplotlib}"
mkdir -p "$MPLCONFIGDIR"
INR_DIR="${KBOUND_IMAGENETR_ROOT:-${INR_FAST:-$ROOT/experiments/kbound/data/imagenet-r}}"
CLASS_INDEX="${IMAGENET_CLASS_INDEX:-$ROOT/experiments/kbound/data/imagenet_class_index.json}"
if [[ ! -f "$CLASS_INDEX" && -f "$ROOT/external/poem/cotta/imagenet/robustbench/data/imagenet_class_to_id_map.json" ]]; then
  CLASS_INDEX="$ROOT/external/poem/cotta/imagenet/robustbench/data/imagenet_class_to_id_map.json"
fi
PACS_ROOT="${PACS_ROOT:-$ROOT/experiments/kbound/domainbed}"
INR3="$RES/imagenetr_protocol_d_seed3_v1"
INRALL="$RES/imagenetr_protocol_d_multiseed_v1"
PACS0="$RES/win_hunt_v5/pacs_aggr/pacs_result.json"
PACSALL="$RES/pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json"
METHODS=(convnext_base convnext_tiny efficientnet_b0 efficientnet_b3 resnet101 resnet152 resnext101_32x8d swin_b swin_t vit_b_16)

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
need_file() { [[ -e "$1" ]] || { say "MISSING: $1"; exit 2; }; }

preflight() {
  need_file "$AETTA/log/cifar10/Src/tgt_test/reproduce_src_0/cp/cp_last.pth.tar"
  need_file "$PACS_ROOT/PACS"
  need_file "$INR_DIR"
  need_file "$CLASS_INDEX"
  "$CONDA" run -n aetta python -c 'import torch, torchvision, sklearn; print(torch.__version__, torchvision.__version__, sklearn.__version__)'
  cd "$ROOT"
  pytest -q AETTA/tests/test_dnn_state_initialization.py
  "$PY" "$ROOT/experiments/kbound/wilds/run_imagenetr_kbound.py" --panel diverse_backbones \
    --imagenetr-dir "$INR_DIR" --class-index "$CLASS_INDEX" --seeds 3 --compositions iid imbalanced single_class \
    --batch-regimes small tiny --aggressiveness mild aggressive --n-eval 500 --n-batches 4 \
    --frozen-eval-batch 32 --device mps --run-name imagenetr_protocol_d_seed3_v1 --dry-run
  say "PREFLIGHT PASS"
}

smoke() {
  preflight
  say "AETTA Src/GDE queue smoke (128 samples, isolated seed 99)"
  cd "$AETTA"
  "$CONDA" run -n aetta python main.py --gpu_idx 0 --dataset cifar10outdist --method Src \
    --tgt gaussian_noise-5 --model resnet18 --epoch 0 \
    --load_checkpoint_path log/cifar10/Src/tgt_test/reproduce_src_0/cp/cp_last.pth.tar \
    --seed 99 --nsample 128 --update_every_x 64 --log_prefix kbound_queue_smoke \
    --acc_est_method gde softmax_score
  say "PACS CPU smoke"
  cd "$ROOT"
  "$CONDA" run -n aetta python "$K/scripts/pacs_vlcs_runner.py" --dataset PACS \
    --root "$PACS_ROOT" --device cpu --methods tent --seed 99 --smoke --out /tmp/kbound_pacs_smoke.json
  need_file /tmp/kbound_pacs_smoke.json
  say "SMOKE PASS"
}

run_long() {
  preflight
  say "AETTA TENT then Src; max two workers on the single MPS device"
  cd "$AETTA"
  AETTA_MAX_JOBS="${AETTA_MAX_JOBS:-2}" "$CONDA" run -n aetta bash tta_item11_TENT.sh
  AETTA_MAX_JOBS="${AETTA_MAX_JOBS:-2}" "$CONDA" run -n aetta bash tta_item11_Src.sh
  if rg -l 'Traceback \(most recent call last\)' raw_logs/cifar10outdist_eval_results_0_job*.txt >/dev/null; then
    say "AETTA failed: traceback found in raw logs"; exit 3
  fi

  run_remaining_seeds
  say "LONG RUNS COMPLETE"
}

run_remaining_seeds() {
  preflight
  say "PACS missing seeds 1 and 2 (locked seed-0 operating point)"
  cd "$ROOT"
  for seed in 1 2; do
    if [[ -s "$RES/pacs_seed${seed}.json" ]]; then
      "$CONDA" run -n aetta python "$K/scripts/validate_closure_seed.py" pacs \
        --file "$RES/pacs_seed${seed}.json" --seed "$seed"
      say "PACS seed $seed is complete and valid; skip"
      continue
    fi
    "$CONDA" run -n aetta python "$K/scripts/pacs_vlcs_runner.py" --dataset PACS \
      --root "$PACS_ROOT" --device mps --methods tent eata sar --seed "$seed" \
      --adapt-lr 0.004 --batch-regimes tiny --aggressiveness aggressive \
      --out "$RES/pacs_seed${seed}.json"
    "$CONDA" run -n aetta python "$K/scripts/validate_closure_seed.py" pacs \
      --file "$RES/pacs_seed${seed}.json" --seed "$seed"
  done
  "$CONDA" run -n aetta python "$K/scripts/aggregate_pacs_multiseed.py" \
    --seed0 "$PACS0" --seed1 "$RES/pacs_seed1.json" --seed2 "$RES/pacs_seed2.json" \
    --out "$PACSALL"

  say "ImageNet-R missing seed 3 (resumable)"
  "$PY" "$ROOT/experiments/kbound/wilds/run_imagenetr_kbound.py" --panel diverse_backbones \
    --imagenetr-dir "$INR_DIR" --class-index "$CLASS_INDEX" --seeds 3 --compositions iid imbalanced single_class \
    --batch-regimes small tiny --aggressiveness mild aggressive --n-eval 500 --n-batches 4 \
    --frozen-eval-batch 32 --tau-star 0.52 --kappa 2.5 --sd-L 0.6 --delta 0.05 \
    --device mps --results-root "$RES" --run-name imagenetr_protocol_d_seed3_v1
  "$PY" "$K/scripts/validate_closure_seed.py" imagenetr --run-dir "$INR3" --seed 3
  for method in "${METHODS[@]}"; do
    need_file "$INR3/per_condition_imagenet-r_${method}_seed3.json"
    cp "$INR3/per_condition_imagenet-r_${method}_seed3.json" "$INRALL/"
  done
  "$PY" "$ROOT/experiments/kbound/wilds/multiseed_paired_ci.py" --run-dir "$INRALL" \
    --dataset imagenet-r --methods "${METHODS[@]}" --seeds 0 1 2 3 \
    --out "$INRALL/MULTISEED_ANALYSIS_RESULTS.json"
  say "PACS 3-seed and ImageNet-R 4-seed closure complete"
}

finalize() {
  need_file "$RES/pacs_seed1.json"
  need_file "$RES/pacs_seed2.json"
  need_file "$PACSALL"
  need_file "$INRALL/per_condition_imagenet-r_resnet101_seed3.json"
  "$PY" "$K/scripts/validate_closure_seed.py" imagenetr --run-dir "$INRALL" --seed 3
  cd "$ROOT"
  "$PY" "$K/scripts/empirical_closure.py"
  python3 "$K/formal/formal_audit.py" --strict-core
  cd "$K"
  latexmk -pdf -interaction=nonstopmode -halt-on-error kbound_short.tex
  mkdir -p /tmp/kbound_final_pdf
  pdftoppm -png -r 110 kbound_short.pdf /tmp/kbound_final_pdf/page >/dev/null 2>&1
  [[ "$(find /tmp/kbound_final_pdf -name 'page-*.png' | wc -l | tr -d ' ')" -eq 23 ]]
  say "FINALIZE PASS: audit and 23-page PDF rebuilt"
}

case "$MODE" in
  preflight) preflight ;;
  smoke) smoke ;;
  remaining-seeds) run_remaining_seeds ;;
  close-seeds) run_remaining_seeds; finalize ;;
  run) run_long ;;
  finalize) finalize ;;
  all) smoke; run_long; finalize ;;
  *) echo "usage: $0 {preflight|smoke|remaining-seeds|close-seeds|run|finalize|all}"; exit 2 ;;
esac
