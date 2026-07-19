#!/usr/bin/env bash
# Re-run natural-shift TTA protocols at seeds 0-4, then extract Camelyon-style
# per_condition files + multi-seed aggregates (forest/table ready).
#
#   bash docs/research/kbound/scripts/run_multiseed.sh camelyon
#   bash docs/research/kbound/scripts/run_multiseed.sh officehome     # lightest (mps-ok)
#   bash docs/research/kbound/scripts/run_multiseed.sh pacs
#   bash docs/research/kbound/scripts/run_multiseed.sh iwildcam
#   bash docs/research/kbound/scripts/run_multiseed.sh rxrx1          # heaviest (~46GB data)
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
OUT="${OUT:-$REPO/experiments/kbound/results/multiseed}"
WILDS_ROOT="${WILDS_ROOT:-$HOME/datasets/wilds}"
PACS_ROOT="${PACS_ROOT:-$HOME/kbound_pacs}"      # parent dir containing PACS/
mkdir -p "$OUT"
which="${1:-camelyon}"

extract_track() {
  # Usage: extract_track <track> <result_glob> [<result_glob> ...]
  # Optional env EXTRACT_CANDIDATES="cand1 cand2" overrides locked adapters.
  local track="$1"; shift
  local extr="$OUT/$track/extracted"
  mkdir -p "$extr"
  local globs=("$@") matched=()
  local g
  for g in "${globs[@]}"; do
    # shellcheck disable=SC2086
    if compgen -G "$g" > /dev/null; then
      matched+=($g)
    fi
  done
  if [ ${#matched[@]} -eq 0 ]; then
    echo "!! skip extract $track — no files match: ${globs[*]}"
    return 0
  fi
  echo "== extract+aggregate $track (${#matched[@]} result files) =="
  if [ -n "${EXTRACT_CANDIDATES:-}" ]; then
    # shellcheck disable=SC2086
    "$PY" "$K/scripts/extract_multiseed_natural.py" \
        --track "$track" --result "${matched[@]}" --out-dir "$extr" \
        --candidates $EXTRACT_CANDIDATES
  else
    "$PY" "$K/scripts/extract_multiseed_natural.py" \
        --track "$track" --result "${matched[@]}" --out-dir "$extr"
  fi
}

build_forest() {
  echo "== multi-seed forest + LaTeX table =="
  "$PY" "$K/scripts/make_multiseed_natural_forest.py" \
      --agg \
        "$OUT/multiseed_camelyon17_*.json" \
        "$OUT/*/extracted/multiseed_*.json" \
        "$OUT/camelyon/multiseed_*.json" \
      --out-fig "$K/figures/fig_natural_forest_multiseed.png" \
      --out-tex "$K/figures/tab_multiseed_natural.tex" \
      --out-json "$K/figures/multiseed_natural_forest_payload.json" || true
}

run_camelyon() {   # PREREQ: WILDS Camelyon17 under $WILDS_ROOT (fully wired; emits per_condition files)
  echo "== Camelyon17 (seeds $SEEDS) =="
  "$PY" "$K/scripts/run_wilds_camelyon17.py" --wilds-root "$WILDS_ROOT" --seeds $SEEDS \
      --output-dir "$OUT/camelyon" --evidence-panel rich
  for c in tent eata sar; do
    "$PY" "$K/scripts/multiseed_natural.py" --dataset camelyon17 --candidate "$c" \
        --dir "$OUT/camelyon" --out "$OUT/multiseed_camelyon17_${c}.json" || true
  done
}

run_officehome() { # PREREQ: ~/kbound_officehome data + results/officehome_f0/f0_resnet50_rw_seed0.pt
  echo "== Office-Home target_test (seeds $SEEDS) -- mps-ok, lightest =="
  "$PY" "$REPO/experiments/kbound/officehome/run_officehome_kbound.py" \
      --role target_test --seeds $SEEDS --device mps --results-root "$OUT/officehome"
  extract_track officehome "$OUT/officehome/**/result_*.json"
}

run_pacs() {       # PREREQ: $PACS_ROOT/PACS/ images (DomainBed layout); trains ERM in-run
  echo "== PACS tent/eata/sar (seeds $SEEDS) =="
  mkdir -p "$OUT/pacs"
  for s in $SEEDS; do
    "$PY" "$K/scripts/pacs_vlcs_runner.py" --dataset PACS --root "$PACS_ROOT" \
        --methods tent eata sar --seed "$s" --out "$OUT/pacs/pacs_seed${s}.json"
  done
  echo ">> PACS per-seed results in $OUT/pacs (DomainBed schema; not yet in extract_multiseed_natural)"
}

run_iwildcam() {   # PREREQ: iWildCam WILDS data under experiments/kbound/data/wilds (~12GB) + F0
  echo "== iWildCam test (seeds $SEEDS) =="
  "$PY" "$REPO/experiments/kbound/wilds/run_iwildcam_kbound.py" \
      --split test --seeds $SEEDS --device auto --results-root "$OUT/iwildcam"
  extract_track iwildcam "$OUT/iwildcam/**/result_*.json"
}

run_rxrx1() {      # PREREQ: ~/kbound_rxrx1_data (~46GB) + ~/kbound_rxrx1_ckpt/rxrx1_seed:0_...pth
  echo "== RxRx1 test (seeds $SEEDS) -- heaviest =="
  "$PY" "$REPO/experiments/kbound/wilds/run_rxrx1_kbound.py" \
      --split test --seeds $SEEDS --device auto --results-root "$OUT/rxrx1"
  extract_track rxrx1 "$OUT/rxrx1/**/result_*.json"
}

extract_only() {
  # CPU path: reuse existing monolithic results (prefer fresh multiseed tree, else archive).
  # One coherent source per track — do not merge id_val+test or modelseed folders.
  if compgen -G "$OUT/officehome/**/result_*.json" > /dev/null; then
    extract_track officehome "$OUT/officehome/**/result_*.json"
  else
    extract_track officehome \
        "$REPO/experiments/kbound/results/officehome_protocol_m_repl_targettest/result_*.json"
  fi
  if compgen -G "$OUT/iwildcam/**/result_*.json" > /dev/null; then
    extract_track iwildcam "$OUT/iwildcam/**/result_*.json"
  else
    extract_track iwildcam \
        "$REPO/experiments/kbound/results/iwildcam_full_idval/result_*.json"
  fi
  if compgen -G "$OUT/rxrx1/**/result_*.json" > /dev/null; then
    extract_track rxrx1 "$OUT/rxrx1/**/result_*.json"
  else
    extract_track rxrx1 \
        "$REPO/experiments/kbound/results/win_hunt_v5/rxrx1_aggr/result_*.json"
  fi
  # Refresh Camelyon aggregates if per_condition files already exist somewhere.
  if compgen -G "$REPO/experiments/kbound/results/**/per_condition_camelyon17_*_seed*.json" > /dev/null; then
    for c in tent eata sar; do
      "$PY" "$K/scripts/multiseed_natural.py" --dataset camelyon17 --candidate "$c" \
          --dir "$REPO/experiments/kbound/results" --out "$OUT/multiseed_camelyon17_${c}.json" || true
    done
  fi
  build_forest
}

case "$which" in
  camelyon)     run_camelyon; build_forest ;;
  officehome)   run_officehome; build_forest ;;
  pacs)         run_pacs ;;
  iwildcam)     run_iwildcam; build_forest ;;
  rxrx1)        run_rxrx1; build_forest ;;
  extract-only) extract_only ;;
  all)          run_camelyon; run_officehome; run_pacs; run_iwildcam; run_rxrx1; build_forest ;;
  *) echo "usage: $0 {camelyon|officehome|pacs|iwildcam|rxrx1|extract-only|all}"; exit 2 ;;
esac
echo "done -> $OUT"
