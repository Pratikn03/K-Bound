#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
PY="${KBOUND_SAR_PYTHON:-/opt/anaconda3/envs/ag311/bin/python}"
DEVICE="${KB_DEVICE:-mps}"
DATA="${KBOUND_CIFAR_ROOT:-$ROOT/experiments/kbound/cifar}"
OUT="${KBOUND_SAR_OUTPUT:-$ROOT/experiments/kbound/results/cifar10c_sar_rebuild_v2}"
RUNNER="$ROOT/docs/research/kbound/scripts/cifar_tent_mps_v2.py"
VALIDATOR="$ROOT/docs/research/kbound/scripts/validate_cifar10c_sar_rebuild.py"
MODE="${1:-preflight}"

check_hash() {
  local expected="$1" path="$2"
  local observed
  observed="$(shasum -a 256 "$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || {
    echo "HASH MISMATCH: $path" >&2
    exit 2
  }
}

preflight() {
  [[ -x "$PY" ]] || { echo "Missing runtime: $PY" >&2; exit 2; }
  [[ -d "$DATA/CIFAR-10-C" ]] || { echo "Missing CIFAR-10-C under $DATA" >&2; exit 2; }
  check_hash f1687904d36114340ae7da055197f6bd44c08e2f617d17703a52824765e62dbc "$RUNNER"
  check_hash 43333456a795bbe679966c14812f9964d8b3bf060d30ca2b3d5051cb8c9d7491 "$DATA/resnet18_cifar.pt"
  check_hash e6d972b1238665d8ef54aae5affe8e292dda1eb88a6840bf0f5988cdb649da7b "$DATA/CIFAR-10-C/labels.npy"
  "$PY" -c 'import platform, torch, torchvision, numpy, sklearn; print({"machine": platform.machine(), "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": numpy.__version__, "sklearn": sklearn.__version__, "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available()})'
  if [[ "$DEVICE" == "mps" ]]; then
    "$PY" -c 'import torch; assert torch.backends.mps.is_available(), "MPS unavailable in selected runtime"'
  fi
  echo "PREFLIGHT PASS: device=$DEVICE output=$OUT"
}

run_seed() {
  local seed="$1" target="$OUT/seed$seed"
  mkdir -p "$target"
  if [[ -f "$target/per_condition_cifar10c_sar_seed${seed}.json" ]]; then
    echo "Seed $seed output exists; validating before skip"
    "$PY" "$VALIDATOR" --root "$ROOT" --results "$OUT" --allow-partial
    return
  fi
  echo "START seed=$seed target=$target"
  "$PY" "$RUNNER" --benchmarks cifar10c --quick --methods sar \
    --device "$DEVICE" --seed "$seed" --max-cells 0 \
    --data-root "$DATA" --cifar10-ckpt "$DATA/resnet18_cifar.pt" \
    --out-results "$target" --out-figs "$target" 2>&1 | tee "$target/seed${seed}.log"
  "$PY" "$VALIDATOR" --root "$ROOT" --results "$OUT" --allow-partial
}

case "$MODE" in
  preflight)
    preflight ;;
  smoke)
    preflight
    target="$OUT/smoke_seed0"
    mkdir -p "$target"
    # --max-cells 4 -> 8 conditions (4 cells x 2 repeats): the leave-one-out benefit
    # model (decide_kga) needs a non-trivial training fold; max-cells 1 (2 conditions)
    # collapses the fold and raises sklearn "weights sum to zero". This only sizes the
    # smoke pipeline check; the full 'run' path (--max-cells 0) is unchanged.
    "$PY" "$RUNNER" --benchmarks cifar10c --quick --methods sar \
      --device "$DEVICE" --seed 0 --max-cells 4 \
      --data-root "$DATA" --cifar10-ckpt "$DATA/resnet18_cifar.pt" \
      --out-results "$target" --out-figs "$target" ;;
  run)
    preflight
    for seed in ${KBOUND_SAR_SEEDS:-0 1 2 3 4}; do run_seed "$seed"; done ;;
  finalize)
    "$PY" "$VALIDATOR" --root "$ROOT" --results "$OUT"
    "$PY" "$ROOT/docs/research/kbound/scripts/percondition_bootstrap.py" \
      --root "$OUT" --pattern 'per_condition_cifar10c_sar_seed*.json' ;;
  *)
    echo "Usage: $0 {preflight|smoke|run|finalize}" >&2
    exit 2 ;;
esac
