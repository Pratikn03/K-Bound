#!/usr/bin/env bash
# Fail-closed entry point for the prospective K-Bound closure program.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

PROTOCOL="${KBOUND_CLOSURE_PROTOCOL:-$ROOT/research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml}"
LOCK="${KBOUND_CLOSURE_LOCK:-$ROOT/experiments/kbound/results/prospective_closure_v1/protocol_lock.json}"
MODE="${1:-preflight}"

select_python() {
  if [[ -n "${KBOUND_PYTHON:-}" ]]; then
    printf '%s\n' "$KBOUND_PYTHON"
  elif [[ -x "$HOME/.venv_wilds/bin/python" ]]; then
    printf '%s\n' "$HOME/.venv_wilds/bin/python"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$ROOT/.venv/bin/python"
  else
    command -v python3
  fi
}

PY="$(select_python)"
export PYTHONPATH="$ROOT:$ROOT/docs/research/kbound/kbound_pkg:$ROOT/docs/research/kbound/edge/src${PYTHONPATH:+:$PYTHONPATH}"

require_python_312() {
  "$PY" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"K-Bound closure requires Python 3.12; selected {sys.executable} reports "
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
print(f"Python gate: PASS ({sys.executable}, {sys.version.split()[0]})")
PY
}

preflight() {
  require_python_312
  "$PY" docs/research/kbound/scripts/validate_closure_protocol.py --protocol "$PROTOCOL"
  "$PY" - "$ROOT" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
data_root = Path(os.environ["KBOUND_DATA_ROOT"])  # must be set; no machine-local default
output_root = root / "experiments/kbound/results/prospective_closure_v1"
canonical = root / "experiments/kbound/results/reconciled_panels_v1"
print(f"Repository: {root}")
print(f"Dataset root: {data_root} ({'present' if data_root.is_dir() else 'missing'})")
print(f"Prospective output root: {output_root}")
if output_root.resolve() == canonical.resolve():
    raise SystemExit("prospective output must not overwrite the canonical release")
print("Output isolation gate: PASS")
PY
}

smoke() {
  preflight
  "$PY" -m pytest -q \
    tests/test_kga_experiment_contract.py \
    tests/test_kga_canonical_rule.py \
    tests/test_manuscript_claim_consistency.py
}

sealed_stage() {
  local stage="$1"
  require_python_312
  if [[ "${KBOUND_EXECUTE:-0}" == "1" ]]; then
    "$PY" docs/research/kbound/scripts/run_closure_stage.py "$stage" \
      --protocol "$PROTOCOL" --lock "$LOCK" --execute
  else
    "$PY" docs/research/kbound/scripts/run_closure_stage.py "$stage" \
      --protocol "$PROTOCOL" --lock "$LOCK"
  fi
}

release_gate() {
  require_python_312
  "$PY" docs/research/kbound/scripts/validate_closure_protocol.py \
    --protocol "$PROTOCOL" --require-sealed
  bash docs/research/kbound/runbooks/release_candidate.sh all
}

case "$MODE" in
  preflight) preflight ;;
  smoke) smoke ;;
  train) sealed_stage train ;;
  evaluate) sealed_stage evaluate ;;
  release) release_gate ;;
  *)
    echo "usage: $0 {preflight|smoke|train|evaluate|release}" >&2
    exit 2
    ;;
esac
