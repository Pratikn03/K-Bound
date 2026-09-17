#!/usr/bin/env bash
# One-command launcher for the locked RxRx1 9+ evidence protocol.
#
# Default protocol:
#   5 base-model seeds x 10 condition seeds x
#   3 compositions x 2 batch regimes x 2 aggressiveness settings x 6 adapters.
#
# Run after other GPU/MPS work is finished:
#   bash docs/research/kbound/scripts/kbtrain.sh rxrx1-9plus
#
# Safe preview:
#   bash docs/research/kbound/scripts/run_rxrx1_9plus.sh --dry-run
set -euo pipefail

usage() {
  cat <<'EOF'
usage: run_rxrx1_9plus.sh [--dry-run] [--allow-concurrent]

Environment overrides:
  RXRX1_MODEL_SEEDS       default: "0 1 2 3 4"
  RXRX1_CONDITION_SEEDS   default: "0 1 2 3 4 5 6 7 8 9"
  RXRX1_N_EVAL            default: 512
  RXRX1_N_BATCHES         default: 4
  RXRX1_DATA_ROOT         default: $HOME/kbound_rxrx1_data
  RXRX1_CKPT_ROOT         default: $HOME/kbound_rxrx1_ckpt
  RXRX1_RESULTS_ROOT      default: <repo>/experiments/kbound/results
  RXRX1_RUN_TAG           default: rxrx1_protocol_c_9plus

Expected checkpoints:
  $RXRX1_CKPT_ROOT/rxrx1_seed:<MODEL_SEED>_epoch:best_model.pth
EOF
}

DRY_RUN=0
ALLOW_CONCURRENT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --allow-concurrent)
      ALLOW_CONCURRENT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RUNNER="$REPO/experiments/kbound/wilds/run_rxrx1_kbound.py"
VENV="${RXRX1_VENV:-$HOME/.venv_wilds}"
DATA_ROOT="${RXRX1_DATA_ROOT:-$HOME/kbound_rxrx1_data}"
CKPT_ROOT="${RXRX1_CKPT_ROOT:-$HOME/kbound_rxrx1_ckpt}"
RESULTS_ROOT="${RXRX1_RESULTS_ROOT:-$REPO/experiments/kbound/results}"
RUN_TAG="${RXRX1_RUN_TAG:-rxrx1_protocol_c_9plus}"
SPLIT="${RXRX1_SPLIT:-test}"
N_EVAL="${RXRX1_N_EVAL:-512}"
N_BATCHES="${RXRX1_N_BATCHES:-4}"
EPISODIC_STEPS="${RXRX1_EPISODIC_STEPS:-5}"
EPISODIC_BATCH="${RXRX1_EPISODIC_BATCH:-64}"
TAU_STAR="${RXRX1_TAU_STAR:-0.52}"
KAPPA="${RXRX1_KAPPA:-2.5}"
SD_L="${RXRX1_SD_L:-0.6}"
DELTA="${RXRX1_DELTA:-0.05}"
DEVICE="${RXRX1_DEVICE:-auto}"

MODEL_SEEDS_STR="${RXRX1_MODEL_SEEDS:-0 1 2 3 4}"
CONDITION_SEEDS_STR="${RXRX1_CONDITION_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
COMPOSITIONS_STR="${RXRX1_COMPOSITIONS:-iid imbalanced single_class}"
BATCH_REGIMES_STR="${RXRX1_BATCH_REGIMES:-small tiny}"
AGGRESSIVENESS_STR="${RXRX1_AGGRESSIVENESS:-mild aggressive}"

MODEL_SEEDS_STR="${MODEL_SEEDS_STR//,/ }"
CONDITION_SEEDS_STR="${CONDITION_SEEDS_STR//,/ }"
COMPOSITIONS_STR="${COMPOSITIONS_STR//,/ }"
BATCH_REGIMES_STR="${BATCH_REGIMES_STR//,/ }"
AGGRESSIVENESS_STR="${AGGRESSIVENESS_STR//,/ }"

read -r -a MODEL_SEEDS <<< "$MODEL_SEEDS_STR"
read -r -a CONDITION_SEEDS <<< "$CONDITION_SEEDS_STR"
read -r -a COMPOSITIONS <<< "$COMPOSITIONS_STR"
read -r -a BATCH_REGIMES <<< "$BATCH_REGIMES_STR"
read -r -a AGGRESSIVENESS <<< "$AGGRESSIVENESS_STR"

quote_cmd() {
  local item
  item="$1"
  [[ "$item" == "$REPO"* ]] && item="<repo>${item#"$REPO"}"
  [[ "$item" == "$HOME"* ]] && item='${HOME}'"${item#"$HOME"}"
  printf '%q' "$item"
  shift || true
  for item in "$@"; do
    [[ "$item" == "$REPO"* ]] && item="<repo>${item#"$REPO"}"
    [[ "$item" == "$HOME"* ]] && item='${HOME}'"${item#"$HOME"}"
    printf ' %q' "$item"
  done
  printf '\n'
}

condition_count=$((${#MODEL_SEEDS[@]} * ${#CONDITION_SEEDS[@]} * ${#COMPOSITIONS[@]} * ${#BATCH_REGIMES[@]} * ${#AGGRESSIVENESS[@]}))
record_count=$((condition_count * 6))

echo "RxRx1 Protocol-C / 9+ launcher"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  repo             = <repo>"
  echo "  runner           = <repo>/experiments/kbound/wilds/run_rxrx1_kbound.py"
else
  echo "  repo             = $REPO"
  echo "  runner           = $RUNNER"
fi
echo "  model seeds      = ${MODEL_SEEDS[*]}"
echo "  condition seeds  = ${CONDITION_SEEDS[*]}"
echo "  n_eval/n_batches = $N_EVAL / $N_BATCHES"
echo "  grid             = ${#COMPOSITIONS[@]} compositions x ${#BATCH_REGIMES[@]} batch regimes x ${#AGGRESSIVENESS[@]} aggressiveness"
echo "  planned records  = $record_count adapter records over $condition_count conditions"
echo "  results root     = $RESULTS_ROOT"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: no training or evaluation will start."
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  [[ -f "$RUNNER" ]] || { echo "ERROR: runner missing: $RUNNER" >&2; exit 2; }
  [[ -d "$VENV" ]] || { echo "ERROR: venv missing: $VENV" >&2; exit 2; }
  [[ -d "$DATA_ROOT/rxrx1_v1.0" ]] || {
    echo "ERROR: RxRx1 data not found at $DATA_ROOT/rxrx1_v1.0" >&2
    exit 2
  }
  if [[ "$ALLOW_CONCURRENT" -eq 0 ]] && pgrep -f "run_wilds_camelyon17.py|run_rxrx1_kbound.py" >/dev/null 2>&1; then
    echo "ERROR: another K-Bound GPU/MPS job is running. Re-run after it finishes, or pass --allow-concurrent." >&2
    exit 3
  fi
fi

missing=()
for model_seed in "${MODEL_SEEDS[@]}"; do
  ckpt="$CKPT_ROOT/rxrx1_seed:${model_seed}_epoch:best_model.pth"
  if [[ ! -f "$ckpt" ]]; then
    missing+=("$ckpt")
  fi
done
if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "Missing RxRx1 base-model checkpoint(s):"
  printf '  %s\n' "${missing[@]}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "Train or place those checkpoints first, then re-run this launcher." >&2
    exit 2
  fi
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  cd "$REPO"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  python -c "import torch, torchvision, wilds" >/dev/null
  mkdir -p "$RESULTS_ROOT"
fi

for model_seed in "${MODEL_SEEDS[@]}"; do
  ckpt="$CKPT_ROOT/rxrx1_seed:${model_seed}_epoch:best_model.pth"
  run_name="${RUN_TAG}_modelseed${model_seed}"
  cmd=(
    python "$RUNNER"
    --data-root "$DATA_ROOT"
    --ckpt "$ckpt"
    --split "$SPLIT"
    --seeds "${CONDITION_SEEDS[@]}"
    --compositions "${COMPOSITIONS[@]}"
    --batch-regimes "${BATCH_REGIMES[@]}"
    --aggressiveness "${AGGRESSIVENESS[@]}"
    --n-eval "$N_EVAL"
    --n-batches "$N_BATCHES"
    --episodic-steps "$EPISODIC_STEPS"
    --episodic-batch "$EPISODIC_BATCH"
    --tau-star "$TAU_STAR"
    --kappa "$KAPPA"
    --sd-L "$SD_L"
    --delta "$DELTA"
    --device "$DEVICE"
    --results-root "$RESULTS_ROOT"
    --run-name "$run_name"
    --resume
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    quote_cmd "${cmd[@]}"
    continue
  fi

  log_path="$RESULTS_ROOT/${run_name}.log"
  echo "Starting model seed $model_seed -> $run_name"
  echo "  log: $log_path"
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -is "${cmd[@]}" 2>&1 | tee -a "$log_path"
  else
    "${cmd[@]}" 2>&1 | tee -a "$log_path"
  fi
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "RxRx1 Protocol-C launcher complete. Results are under $RESULTS_ROOT/${RUN_TAG}_modelseed*/"
fi
