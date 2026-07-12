#!/usr/bin/env bash
# Re-run the natural-shift TTA protocols at seeds 0-4 to GENERATE the per-seed logs that are missing
# on disk (iWildCam / Office-Home / RxRx1 / PACS), plus the fully-wired Camelyon17. Needs a GPU
# (or Apple mps) and each track's dataset + source (F0) checkpoint -- see PREREQS per function.
#
#   bash docs/research/kbound/scripts/run_multiseed.sh camelyon
#   bash docs/research/kbound/scripts/run_multiseed.sh officehome     # lightest (mps-ok)
#   bash docs/research/kbound/scripts/run_multiseed.sh pacs
#   bash docs/research/kbound/scripts/run_multiseed.sh iwildcam
#   bash docs/research/kbound/scripts/run_multiseed.sh rxrx1          # heaviest (~46GB data)
#   bash docs/research/kbound/scripts/run_multiseed.sh all
#
# NOTE ON AGGREGATION: the WILDS/OfficeHome runners write the split-conformal *result* schema
# (records + routing_b_multicandidate.regret_vs_oracle + false_adapt_rate), NOT the
# per_condition_<ds>_<cand>_seed*.json that multiseed_natural.py reads. So this script runs the seeds
# and STOPS; send me one per-seed JSON and I wire the exact extractor (regret_kga from routing_b,
# adapt/freeze from the records' oracle) + aggregate all 5 + fold the row into the paper.
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
  echo ">> Office-Home per-seed results in $OUT/officehome ; send me one to aggregate + fold in."
}

run_pacs() {       # PREREQ: $PACS_ROOT/PACS/ images (DomainBed layout); trains ERM in-run
  echo "== PACS tent/eata/sar (seeds $SEEDS) =="
  mkdir -p "$OUT/pacs"
  for s in $SEEDS; do
    "$PY" "$K/scripts/pacs_vlcs_runner.py" --dataset PACS --root "$PACS_ROOT" \
        --methods tent eata sar --seed "$s" --out "$OUT/pacs/pacs_seed${s}.json"
  done
  echo ">> PACS per-seed results in $OUT/pacs ; send me one to aggregate + fold in."
}

run_iwildcam() {   # PREREQ: iWildCam WILDS data under experiments/kbound/data/wilds (~12GB) + F0
  echo "== iWildCam test (seeds $SEEDS) =="
  "$PY" "$REPO/experiments/kbound/wilds/run_iwildcam_kbound.py" \
      --split test --seeds $SEEDS --device auto --results-root "$OUT/iwildcam"
  echo ">> iWildCam per-seed results in $OUT/iwildcam ; send me one to aggregate + fold in."
}

run_rxrx1() {      # PREREQ: ~/kbound_rxrx1_data (~46GB) + ~/kbound_rxrx1_ckpt/rxrx1_seed:0_...pth
  echo "== RxRx1 test (seeds $SEEDS) -- heaviest =="
  "$PY" "$REPO/experiments/kbound/wilds/run_rxrx1_kbound.py" \
      --split test --seeds $SEEDS --device auto --results-root "$OUT/rxrx1"
  echo ">> RxRx1 per-seed results in $OUT/rxrx1 ; send me one to aggregate + fold in."
}

case "$which" in
  camelyon)   run_camelyon ;;
  officehome) run_officehome ;;
  pacs)       run_pacs ;;
  iwildcam)   run_iwildcam ;;
  rxrx1)      run_rxrx1 ;;
  all)        run_camelyon; run_officehome; run_pacs; run_iwildcam; run_rxrx1 ;;
  *) echo "usage: $0 {camelyon|officehome|pacs|iwildcam|rxrx1|all}"; exit 2 ;;
esac
echo "done -> $OUT"
