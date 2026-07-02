#!/usr/bin/env bash
# Production + monorepo verification (works from any cwd).
# Usage:
#   bash /Volumes/T9/uav/AutoML_Flagship_V8/scripts/production_health.sh
#   bash scripts/production_health.sh   # if already in repo root
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "ERROR: no .venv/bin/python and no python3 on PATH" >&2
    echo "Create venv: cd $ROOT && python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt" >&2
    exit 1
  fi
fi

export PYTHONPATH="${ROOT}:${ROOT}/src:${ROOT}/docs/research/kbound/kbound_pkg:${ROOT}/docs/research/kbound/edge/src"

echo "=== Repo: $ROOT ==="
echo "=== Python: $("$PY" --version 2>&1) ==="
echo ""

echo "=== 1/4 Gate P production audit ==="
"$PY" src/scripts/audit_gate_p_production.py
echo ""

echo "=== 2/4 KGA API + model governance tests ==="
"$PY" -m pytest tests/test_kga_api_routes.py tests/test_model_governance.py -q --tb=no
echo ""

echo "=== 3/4 Load baseline (P15) ==="
"$PY" deploy/loadtest/run_baseline.py
echo ""

echo "=== 4/4 Monorepo health ==="
bash "$ROOT/scripts/monorepo_health.sh"
echo ""
echo "=== production_health: ALL PASS ==="
