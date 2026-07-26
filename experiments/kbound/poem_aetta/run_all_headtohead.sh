#!/usr/bin/env bash
# run_all_headtohead.sh - one command to run the pre-registered mixed head-to-head
# (KGA vs POEM vs AETTA + trivials + oracle) on the cached CIFAR-10-C stress-grid
# records, for the MIXED-PRIMARY (TENT) and SECONDARY (TENT+EATA pooled) sets.
#
# Pre-registration: docs/research/kbound/MIXED_BENCHMARK_PROTOCOL.md
# How to read the result: docs/research/kbound/RUN_ON_MAC_POEM_AETTA.md (WIN/TIE/LOSE
# are all valid pre-committed outcomes; the run decides, not us).
#
# This is the CACHED arm: pure numpy/scipy (KGA decisions taken from the cached
# kga_decision field). It needs NO torch and runs in seconds on the Mac CPU. The
# OPTIONAL fresh/official-repo arm (per-sample POEM, dropout-AETTA) is described in the
# RUN_ON_MAC doc and is NOT run here.
#
# Overrides via env: REPO, RECORDS_DIR, OUT_DIR, SEEDS, PY.
# --- defect D8: portable roots. No machine-local absolute paths in tracked code
# --- (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md). KB_REPO_ROOT is discovered
# --- from this script's own location; override with KBOUND_REPO_ROOT.
_kb_find_root() {
  d=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
  while [ "$d" != "/" ]; do
    [ -f "$d/pyproject.toml" ] && { printf '%s\n' "$d"; return 0; }
    d=$(dirname "$d")
  done
  echo "ERROR: repository root not found above $(dirname "${BASH_SOURCE[0]:-$0}")" >&2
  return 1
}
KB_REPO_ROOT="${KBOUND_REPO_ROOT:-$(_kb_find_root)}" || exit 1

set -euo pipefail

REPO="${REPO:-$KB_REPO_ROOT}"
RECORDS_DIR="${RECORDS_DIR:-$REPO/experiments/kbound/results/stress_grid_multiseed_v1}"
OUT_DIR="${OUT_DIR:-$REPO/experiments/kbound/results/mixed_headtohead_v1}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PY="${PY:-${REPO}/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi
DATASET="cifar10c"

cd "$REPO"
export PYTHONPATH="$PWD:$PWD/src:$PWD/experiments/kbound/wilds:$PWD/experiments/kbound/poem_aetta"
H="experiments/kbound/poem_aetta/run_mixed_headtohead.py"

echo "############################################################################"
echo "# MIXED HEAD-TO-HEAD (cached arm). Pre-registered protocol; run decides W/T/L."
echo "#   records: $RECORDS_DIR"
echo "#   out    : $OUT_DIR"
echo "#   seeds  : $SEEDS"
echo "############################################################################"

# --- 0. CPU/synthetic apparatus verification (proves the machinery, decides nothing) ---
echo; echo "### [0/3] synthetic apparatus verification (torch-free) ###"
"$PY" experiments/kbound/poem_aetta/verify_headtohead.py

# --- 1. MIXED-PRIMARY = TENT (the headline comparison) ---
echo; echo "### [1/3] MIXED-PRIMARY = TENT (headline) ###"
"$PY" "$H" \
  --records-dir "$RECORDS_DIR" --dataset "$DATASET" --adapter tent \
  --seeds $SEEDS --out-dir "$OUT_DIR" --set-name cifar10c_tent_primary

# --- 2. SECONDARY = TENT + EATA pooled (composition stress) ---
echo; echo "### [2/3] SECONDARY = TENT+EATA pooled ###"
"$PY" "$H" \
  --records-dir "$RECORDS_DIR" --dataset "$DATASET" --pool-adapters tent eata \
  --seeds $SEEDS --out-dir "$OUT_DIR" --set-name cifar10c_tent_eata_pooled

# --- 3. SECONDARY = EATA alone (lower harmful base rate) ---
echo; echo "### [3/3] SECONDARY = EATA alone ###"
"$PY" "$H" \
  --records-dir "$RECORDS_DIR" --dataset "$DATASET" --adapter eata \
  --seeds $SEEDS --out-dir "$OUT_DIR" --set-name cifar10c_eata_secondary

echo
echo "############################################################################"
echo "# DONE. Results + verdict in: $OUT_DIR"
echo "#   HEADTOHEAD_RESULTS_cifar10c_tent_primary.json   <- PRIMARY headline verdict"
echo "#   HEADTOHEAD_RESULTS_cifar10c_tent_eata_pooled.json"
echo "#   HEADTOHEAD_RESULTS_cifar10c_eata_secondary.json"
echo "# Read RUN_ON_MAC_POEM_AETTA.md sec 'how to read it' before interpreting."
echo "############################################################################"
