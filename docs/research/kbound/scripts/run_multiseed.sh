#!/usr/bin/env bash
# Re-run the natural-shift protocols at seeds 0-4 and aggregate into a multi-seed no-harm table.
# Needs a GPU + the WILDS/DomainBed data. Camelyon17 is fully wired (its runner takes --seeds);
# the other tracks are templated -- confirm the runner path/flags before enabling them.
#
#   bash docs/research/kbound/scripts/run_multiseed.sh camelyon
#   bash docs/research/kbound/scripts/run_multiseed.sh all
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"      # repo root
K="$REPO/docs/research/kbound"
WILDS_ROOT="${WILDS_ROOT:-$HOME/datasets/wilds}"
OUT="${OUT:-$REPO/experiments/kbound/results/multiseed}"
PY="${PY:-python3}"
SEEDS="0 1 2 3 4"
mkdir -p "$OUT"
which="${1:-camelyon}"

run_camelyon() {
  echo "== Camelyon17 (seeds $SEEDS) =="
  "$PY" "$K/scripts/run_wilds_camelyon17.py" \
      --wilds-root "$WILDS_ROOT" --seeds $SEEDS \
      --output-dir "$OUT/camelyon" --evidence-panel rich
  # adjust the glob to match the runner's per-seed output filenames:
  "$PY" "$K/scripts/multiseed_aggregate.py" --track Camelyon17 \
      --glob "$OUT/camelyon/**/*seed*.json" --out "$OUT/multiseed_Camelyon17.json"
}

# --- templates: uncomment + fix the runner path/flags for your setup -------------------
run_iwildcam() {
  echo "== iWildCam (seeds $SEEDS) =="
  for s in $SEEDS; do
    : # "$PY" "$K/scripts/run_wilds_iwildcam.py" --seed "$s" --wilds-root "$WILDS_ROOT" \
      #     --candidate tent_episodic --output-dir "$OUT/iwildcam/seed$s"
  done
  "$PY" "$K/scripts/multiseed_aggregate.py" --track iWildCam \
      --glob "$OUT/iwildcam/**/*seed*.json" --out "$OUT/multiseed_iWildCam.json"
}
run_officehome() {
  echo "== Office-Home (seeds $SEEDS) =="
  for s in $SEEDS; do
    : # "$PY" "$REPO/experiments/kbound/officehome/run_officehome_kbound.py" --role target_test \
      #     --seeds "$s" --candidates sar_online_aggressive --device mps
  done
  "$PY" "$K/scripts/multiseed_aggregate.py" --track OfficeHome \
      --glob "$OUT/officehome/**/*seed*.json" --out "$OUT/multiseed_OfficeHome.json"
}
run_rxrx1() {
  echo "== RxRx1 (seeds $SEEDS) =="
  for s in $SEEDS; do
    : # bash "$K/scripts/run_rxrx1_9plus.sh" "$s"   # confirm this script accepts a seed arg
  done
  "$PY" "$K/scripts/multiseed_aggregate.py" --track RxRx1 \
      --glob "$OUT/rxrx1/**/*seed*.json" --out "$OUT/multiseed_RxRx1.json"
}
# ---------------------------------------------------------------------------------------

case "$which" in
  camelyon)   run_camelyon ;;
  iwildcam)   run_iwildcam ;;
  officehome) run_officehome ;;
  rxrx1)      run_rxrx1 ;;
  all)        run_camelyon; run_iwildcam; run_officehome; run_rxrx1 ;;
  *) echo "usage: $0 {camelyon|iwildcam|officehome|rxrx1|all}"; exit 2 ;;
esac
echo "done -> $OUT/multiseed_*.json ; send those to fold multi-seed rows into the paper."
