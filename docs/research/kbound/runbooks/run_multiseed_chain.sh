#!/usr/bin/env bash
# Chained multi-seed queue: waits for item-11 GPU phases to finish, then runs the
# remaining natural-shift multi-seed tracks sequentially (one GPU job at a time):
#   1) Office-Home target_test (seeds 0-4)  [lightest]
#   2) iWildCam test (seeds 0-4)            [12GB data via T9 symlink]
#   3) RxRx1 test (seeds 0-4)               [heaviest; data read from T9]
#   PACS: SKIPPED (no PACS data on internal or T9; arm later via HF download).
# Monitor: tail -f experiments/kbound/results/multiseed/chain.log
# --- interpreter: $KBOUND_PYTHON, default python3 (was a hard-coded venv path).
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

KB_PYTHON="${KBOUND_PYTHON:-python3}"

set -u
R="$KB_REPO_ROOT"
K="$R/docs/research/kbound"
T9="$KB_REPO_ROOT/experiments/kbound"
OUT="$R/experiments/kbound/results/multiseed"
PYBIN=""$KB_PYTHON""
LOG="$OUT/chain.log"
mkdir -p "$OUT"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "CHAIN START — waiting for item-11 GPU phases (source training + AETTA TTA runs)"
while pgrep -f "method Src|tta_item11" >/dev/null 2>&1; do sleep 300; done
say "GPU idle — beginning multi-seed chain"

say "PREP: checkpoints + data symlinks from T9"
mkdir -p "$R/experiments/kbound/results/officehome_f0" "$R/experiments/kbound/results/iwildcam_f0_erm"
[ -f "$R/experiments/kbound/results/officehome_f0/f0_resnet50_rw_seed0.pt" ] || \
  cp "$T9/results/officehome_f0/f0_resnet50_rw_seed0.pt" "$R/experiments/kbound/results/officehome_f0/" 2>>"$LOG"
cp -n "$T9/results/iwildcam_f0_erm/"f0_resnet50_erm_seed0*.pt "$R/experiments/kbound/results/iwildcam_f0_erm/" 2>>"$LOG"
mkdir -p "$R/experiments/kbound/data"
[ -e "$R/experiments/kbound/data/wilds" ] || ln -s "$T9/data/wilds" "$R/experiments/kbound/data/wilds"
say "PREP done"

say "TRACK 1/3: Office-Home (seeds 0-4)"
( cd "$R" && caffeinate -is env PY="$PYBIN" bash "$K/scripts/run_multiseed.sh" officehome ) >> "$LOG" 2>&1
say "Office-Home rc=$?"

say "TRACK 2/3: iWildCam (seeds 0-4)"
( cd "$R" && caffeinate -is env PY="$PYBIN" bash "$K/scripts/run_multiseed.sh" iwildcam ) >> "$LOG" 2>&1
say "iWildCam rc=$?"

say "TRACK 3/3: RxRx1 (seeds 0-4; data from T9)"
( cd "$R" && caffeinate -is "$PYBIN" "$R/experiments/kbound/wilds/run_rxrx1_kbound.py" \
    --split test --seeds 0 1 2 3 4 --device auto \
    --data-root "$T9/data/wilds" --results-root "$OUT/rxrx1" ) >> "$LOG" 2>&1
say "RxRx1 rc=$?"

say "PACS: SKIPPED — no PACS data found (~/kbound_pacs/PACS absent; not on T9). Arm via scripts/export_pacs_hf.py when desired."
say "CHAIN COMPLETE — per-seed results under $OUT/{officehome,iwildcam,rxrx1}; next: schema extractor + aggregation, then paper fold-in"
