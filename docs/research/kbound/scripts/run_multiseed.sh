#!/usr/bin/env bash
# Run development-only natural-shift diagnostics at seeds 0-4, then extract
# per-condition files + multi-seed diagnostic aggregates. This entry point never
# opens a target-test split; held-out scoring requires a separately audited,
# externally receipted one-shot workflow that is not automated here.
#
#   bash docs/research/kbound/scripts/run_multiseed.sh camelyon
#   bash docs/research/kbound/scripts/run_multiseed.sh officehome     # lightest (mps-ok)
#   bash docs/research/kbound/scripts/run_multiseed.sh pacs
#   bash docs/research/kbound/scripts/run_multiseed.sh iwildcam
#   bash docs/research/kbound/scripts/run_multiseed.sh rxrx1          # heaviest (~46GB data)
#   bash docs/research/kbound/scripts/run_multiseed.sh source-checkpoints
#   bash docs/research/kbound/scripts/run_multiseed.sh extract-only   # CPU: extract+forest from existing results
#   bash docs/research/kbound/scripts/run_multiseed.sh all
#
# Pipeline per WILDS/Office-Home track:
#   1) GPU runner -> result_*.json
#   2) extract_multiseed_natural.py -> per_condition_*_seed*.json + multiseed_*.json
#   3) make_multiseed_natural_forest.py -> fig + LaTeX table (after extract-only / all)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
K="$REPO/docs/research/kbound"
PY="${PY:-python3}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DEVICE="${KBOUND_DEVICE:-mps}"
DRY_RUN="${KBOUND_DRY_RUN:-0}"
ALLOW_MISSING_SOURCES="${KBOUND_ALLOW_MISSING_SOURCES:-0}"
OUT="${OUT:-$REPO/experiments/kbound/results/multiseed}"
WILDS_ROOT="${WILDS_ROOT:-$HOME/datasets/wilds}"
PACS_ROOT="${PACS_ROOT:-$HOME/kbound_pacs}"      # parent dir containing PACS/
OFFICEHOME_ROOT="${OFFICEHOME_ROOT:-$HOME/kbound_officehome}"
OFFICEHOME_SPLITS="${OFFICEHOME_SPLITS:-$OUT/officehome/officehome_splits.json}"
mkdir -p "$OUT"
which="${1:-camelyon}"
CURRENT_AGGREGATES=()

dry_run_inventory() {
  echo "K-Bound multi-seed DRY RUN"
  echo "  track=$which"
  echo "  model_seeds=$SEEDS"
  echo "  device=$DEVICE"
  echo "  output=$OUT"
  echo "  WILDS_ROOT=$WILDS_ROOT"
  echo "  OFFICEHOME_ROOT=$OFFICEHOME_ROOT"
  echo "  PACS_ROOT=$PACS_ROOT"
  case "$which" in
    officehome)
      echo "  plan: one ResNet50 source training per model seed"
      echo "  plan: checkpoint hash audit, then target_val only with fixed stream seed 0"
      ;;
    iwildcam)
      echo "  plan: one ResNet50 ERM source training per model seed"
      echo "  plan: checkpoint hash audit, then val only with fixed stream seed 0"
      ;;
    source-checkpoints)
      echo "  plan: train Office-Home and iWildCam source models for every model seed"
      echo "  plan: stop after checkpoint hash audits; do not run KGA evaluation"
      ;;
    pacs)
      echo "  plan: one replayable Tent/EATA/SAR run per model seed; validate every per-cell file"
      ;;
    camelyon)
      echo "  plan: RETIRED — no target data will be opened"
      ;;
    rxrx1)
      echo "  plan: RETIRED — no target data will be opened"
      ;;
    all)
      echo "  plan: development-safe Office-Home, PACS, and iWildCam only"
      echo "  plan: Camelyon17 and RxRx1 held-out paths remain disabled"
      ;;
    extract-only)
      echo "  plan: CPU-only extraction from existing artifacts"
      ;;
    *) echo "unknown track: $which" >&2; return 2 ;;
  esac
  echo "DRY RUN ONLY: no model, dataset, or result was modified."
}

if [[ "$DRY_RUN" == "1" ]]; then
  dry_run_inventory
  exit $?
fi

checkpoint_log_matches() {
  local metadata="$1"
  local checkpoint="$2"
  local hash_field="$3"
  [[ -f "$metadata" && -f "$checkpoint" ]] || return 1
  "$PY" - "$metadata" "$checkpoint" "$hash_field" <<'PY'
import hashlib
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        document = json.load(handle)
except (OSError, ValueError):
    raise SystemExit(1)
if document.get("execution_complete") is not True:
    raise SystemExit(1)
digest = hashlib.sha256()
try:
    with open(sys.argv[2], "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
except OSError:
    raise SystemExit(1)
expected = document.get(sys.argv[3])
raise SystemExit(0 if isinstance(expected, str) and expected == digest.hexdigest() else 1)
PY
}

extract_track() {
  # Usage: extract_track <track> <model|stream> <result_glob> [<result_glob> ...]
  # Optional env EXTRACT_CANDIDATES="cand1 cand2" overrides locked adapters.
  local track="$1"; shift
  local seed_kind="$1"; shift
  local extr="$OUT/$track/extracted"
  mkdir -p "$extr"
  local globs=("$@") matched=()
  local g match
  for g in "${globs[@]}"; do
    while IFS= read -r match; do
      [[ -n "$match" ]] && matched+=("$match")
    done < <(compgen -G "$g" || true)
  done
  if [ ${#matched[@]} -eq 0 ]; then
    if [[ "$ALLOW_MISSING_SOURCES" == "1" ]]; then
      echo "!! optional track $track has no matching sources: ${globs[*]}"
      return 0
    fi
    echo "FATAL: requested track $track has no matching sources: ${globs[*]}" >&2
    return 66
  fi
  echo "== extract+aggregate $track (${#matched[@]} result files) =="
  if [ -n "${EXTRACT_CANDIDATES:-}" ]; then
    # shellcheck disable=SC2086
    "$PY" "$K/scripts/extract_multiseed_natural.py" \
        --track "$track" --result "${matched[@]}" --out-dir "$extr" \
        --seed-kind "$seed_kind" --expected-seeds $SEEDS \
        --candidates $EXTRACT_CANDIDATES
  else
    "$PY" "$K/scripts/extract_multiseed_natural.py" \
        --track "$track" --result "${matched[@]}" --out-dir "$extr" \
        --seed-kind "$seed_kind" --expected-seeds $SEEDS
  fi
  local manifest="$extr/extract_manifest_${track}.json"
  local manifest_output
  if ! manifest_output="$("$PY" - "$manifest" "$extr" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
directory = pathlib.Path(sys.argv[2]).resolve()
with manifest_path.open(encoding="utf-8") as handle:
    manifest = json.load(handle)
for name in manifest.get("aggregates", []):
    path = (directory / name).resolve()
    if path.parent != directory or not path.is_file():
        raise SystemExit(f"invalid current aggregate path: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if (manifest.get("aggregate_sha256") or {}).get(name) != digest:
        raise SystemExit(f"stale current aggregate hash: {path}")
    print(path)
PY
  )"; then
    echo "FATAL: current aggregate manifest validation failed: $manifest" >&2
    return 65
  fi
  while IFS= read -r aggregate; do
    [[ -n "$aggregate" ]] && CURRENT_AGGREGATES+=("$aggregate")
  done <<< "$manifest_output"
}

build_forest() {
  echo "== lineage-verified development diagnostic forest + LaTeX table =="
  if [ ${#CURRENT_AGGREGATES[@]} -eq 0 ]; then
    echo "FATAL: this invocation produced no explicit aggregates; stale output globs are forbidden" >&2
    return 66
  fi
  "$PY" "$K/scripts/make_multiseed_natural_forest.py" \
      --scope development-diagnostic \
      --agg "${CURRENT_AGGREGATES[@]}" \
      --out-fig "$K/figures/fig_natural_forest_multiseed.png" \
      --out-tex "$K/figures/tab_multiseed_natural.tex" \
      --out-json "$K/figures/multiseed_natural_forest_payload.json"
}

run_camelyon() {
  echo "RETIRED: this legacy Camelyon path opens held-out data without the v3 external-audit gate." >&2
  echo "No target data were opened and no aggregate was emitted." >&2
  return 64
}

train_officehome_checkpoints() {
  local ckpt_dir="$OUT/officehome/checkpoints"
  local s
  mkdir -p "$ckpt_dir"
  echo "== Office-Home independent source-model seeds: $SEEDS =="
  for s in $SEEDS; do
    if ! checkpoint_log_matches \
       "$ckpt_dir/f0_meta_seed${s}.json" \
       "$ckpt_dir/f0_resnet50_rw_seed${s}.pt" checkpoint_sha256; then
      "$PY" "$REPO/experiments/kbound/officehome/train_f0_officehome.py" \
        --data-root "$OFFICEHOME_ROOT" --splits "$OFFICEHOME_SPLITS" \
        --out "$ckpt_dir" --seed "$s" --device "$DEVICE" --workers 0
    fi
  done
  # shellcheck disable=SC2086
  "$PY" "$K/scripts/audit_independent_checkpoints.py" \
    --checkpoint-template "$ckpt_dir/f0_resnet50_rw_seed{seed}.pt" --seeds $SEEDS \
    --out "$OUT/officehome/CHECKPOINT_AUDIT.json"
}

run_officehome() { # independent source checkpoints; fixed stream seed isolates model variation
  local ckpt_dir="$OUT/officehome/checkpoints"
  local s
  train_officehome_checkpoints
  for s in $SEEDS; do
    "$PY" "$REPO/experiments/kbound/officehome/run_officehome_kbound.py" \
      --role target_val --model-seed "$s" --seeds 0 --device "$DEVICE" \
      --ckpt "$ckpt_dir/f0_resnet50_rw_seed${s}.pt" \
      --candidates sar_online_aggressive \
      --run-name "officehome_modelseed${s}_target_val" \
      --results-root "$OUT/officehome"
  done
  # LOO is a development-only diagnostic. Target-test labels are deliberately
  # excluded; a separate locked validation->test scorer is required for test.
  extract_track officehome model \
      "$OUT/officehome/officehome_modelseed*_target_val/result_target_val_*.json"
}

run_pacs() {       # PREREQ: $PACS_ROOT/PACS/ images (DomainBed layout); trains ERM in-run
  echo "== PACS tent/eata/sar replayable closure panel (seeds $SEEDS) =="
  mkdir -p "$OUT/pacs"
  for s in $SEEDS; do
    "$PY" "$K/scripts/pacs_vlcs_runner.py" --dataset PACS --root "$PACS_ROOT" \
        --methods tent eata sar --seed "$s" --adapt-lr 0.004 \
        --batch-regimes tiny --aggressiveness aggressive \
        --out "$OUT/pacs/pacs_seed${s}.json"
    "$PY" "$K/scripts/validate_closure_seed.py" pacs \
        --file "$OUT/pacs/pacs_seed${s}.json" --seed "$s"
  done
  echo ">> PACS per-seed summaries and hash-locked v2 decision records in $OUT/pacs"
}

train_iwildcam_checkpoints() {
  local ckpt_dir="$OUT/iwildcam/checkpoints"
  local s
  mkdir -p "$ckpt_dir"
  echo "== iWildCam independent source-model seeds: $SEEDS =="
  for s in $SEEDS; do
    if ! checkpoint_log_matches \
       "$ckpt_dir/trainlog_seed${s}.json" \
       "$ckpt_dir/f0_resnet50_erm_seed${s}.pt" best_checkpoint_sha256; then
      "$PY" "$REPO/experiments/kbound/wilds/train_iwildcam_f0.py" \
        --data-root "$WILDS_ROOT" --out-dir "$ckpt_dir" --seed "$s" \
        --device "$DEVICE" --workers 0
    fi
  done
  # shellcheck disable=SC2086
  "$PY" "$K/scripts/audit_independent_checkpoints.py" \
    --checkpoint-template "$ckpt_dir/f0_resnet50_erm_seed{seed}.pt" --seeds $SEEDS \
    --out "$OUT/iwildcam/CHECKPOINT_AUDIT.json"
}

run_iwildcam() {   # train/recover five recipe-identical source checkpoints before scoring
  local ckpt_dir="$OUT/iwildcam/checkpoints"
  local s
  train_iwildcam_checkpoints
  for s in $SEEDS; do
    "$PY" "$REPO/experiments/kbound/wilds/run_iwildcam_kbound.py" \
      --split val --train-seed "$s" --seeds 0 --device "$DEVICE" \
      --ckpt "$ckpt_dir/f0_resnet50_erm_seed${s}.pt" --backbone resnet50 \
      --trainable full --candidates tent_episodic \
      --run-name "iwildcam_modelseed${s}_val" \
      --results-root "$OUT/iwildcam"
  done
  extract_track iwildcam model \
      "$OUT/iwildcam/iwildcam_modelseed*_val/result_*.json"
}

run_source_checkpoints() {
  train_officehome_checkpoints
  train_iwildcam_checkpoints
  echo "== independent source checkpoint training and hash audits complete =="
}

run_rxrx1() {
  echo "RETIRED: this command would open the RxRx1 target-test split without a pre-opening lock." >&2
  echo "No target data were opened and no aggregate was emitted." >&2
  return 64
}

extract_only() {
  # CPU path: only one explicit development partition from the current
  # independent-checkpoint runs. Historical target-test files are not eligible.
  extract_track officehome model \
      "$OUT/officehome/officehome_modelseed*_target_val/result_target_val_*.json"
  extract_track iwildcam model \
      "$OUT/iwildcam/iwildcam_modelseed*_val/result_*.json"
  echo "!! RxRx1 archives are target-test/stream-seed only; no LOO aggregate emitted"
  build_forest
}

case "$which" in
  camelyon)     run_camelyon ;;
  officehome)   run_officehome; build_forest ;;
  pacs)         run_pacs ;;
  iwildcam)     run_iwildcam; build_forest ;;
  source-checkpoints) run_source_checkpoints ;;
  rxrx1)        run_rxrx1 ;;
  extract-only) extract_only ;;
  all)          run_officehome; run_pacs; run_iwildcam; build_forest ;;
  *) echo "usage: $0 {camelyon|officehome|pacs|iwildcam|source-checkpoints|rxrx1|extract-only|all}"; exit 2 ;;
esac
echo "done -> $OUT"
