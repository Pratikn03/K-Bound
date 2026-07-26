#!/usr/bin/env bash
# Locked OfficeHome Protocol M replication.
#
# This does NOT search the held-out test.  It reruns the already selected
# candidate/config on fresh GPU record artifacts:
#   candidate = sar_online_aggressive
#   estimator = gbr
#   conformal = global
#   seeds     = 2 3 4
#
# Outputs:
#   experiments/kbound/results/officehome_protocol_m_repl_targetval/
#   experiments/kbound/results/officehome_protocol_m_repl_targettest/
#   experiments/kbound/results/officehome_protocol_m_repl_holdout/
# --- external (git-excluded) data volume: ONE documented variable, no default.
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

: "${KBOUND_EXTERNAL_ROOT:?set KBOUND_EXTERNAL_ROOT to the volume holding the git-excluded datasets/checkpoints/caches (layout: docs/research/kbound/kbound_repro/paths.py, acquisition: DATA.md)}"
KB_EXTERNAL_ROOT="$KBOUND_EXTERNAL_ROOT"

set -euo pipefail

REPO="$KB_REPO_ROOT"
PY="${KB_VENV:-$HOME/.venv_wilds}/bin/python"
RUNNER="$REPO/experiments/kbound/officehome/run_officehome_kbound.py"
SCORER="$REPO/docs/research/kbound/scripts/score_kbound_holdout.py"
RES="$REPO/experiments/kbound/results"
LOG="$RES/officehome_protocol_m_repl.log"

cd "$REPO"
mkdir -p "$RES"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
export TMPDIR="$KB_EXTERNAL_ROOT/tmp"
export TORCH_HOME="$KB_EXTERNAL_ROOT/torch_cache"
mkdir -p "$TMPDIR" "$TORCH_HOME"

echo "=== OfficeHome Protocol M replication start $(date) ===" | tee "$LOG"
echo "candidate=sar_online_aggressive seeds=2 3 4 n_eval=320 n_batches=2" | tee -a "$LOG"

run_role() {
  local role="$1"
  local run_name="$2"
  echo "=== GPU role=$role run=$run_name $(date) ===" | tee -a "$LOG"
  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is "$PY" -u "$RUNNER" \
    --role "$role" \
    --run-name "$run_name" \
    --seeds 2 3 4 \
    --compositions iid imbalanced single_class \
    --batch-regimes tiny small \
    --candidates sar_online_aggressive \
    --n-eval 320 \
    --n-batches 2 \
    --device mps 2>&1 | tee -a "$LOG"
}

run_role target_val officehome_protocol_m_repl_targetval
run_role target_test officehome_protocol_m_repl_targettest

VAL_REC="$(ls -t "$RES"/officehome_protocol_m_repl_targetval/result_target_val_*.json | head -1)"
TEST_REC="$(ls -t "$RES"/officehome_protocol_m_repl_targettest/result_target_test_*.json | head -1)"

echo "=== CPU holdout score $(date) ===" | tee -a "$LOG"
"$PY" "$SCORER" \
  --cal-records "$VAL_REC" \
  --test-records "$TEST_REC" \
  --candidate sar_online_aggressive \
  --estimator gbr \
  --conformal global \
  --output-dir "$RES/officehome_protocol_m_repl_holdout" 2>&1 | tee -a "$LOG"

echo "=== OfficeHome Protocol M replication done $(date) ===" | tee -a "$LOG"
