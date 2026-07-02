#!/usr/bin/env bash
# One-command monorepo health gate (fast, no GPU).
# Usage:
#   bash scripts/monorepo_health.sh          # quick (~2 min)
#   bash scripts/monorepo_health.sh --full   # + reproduce_submission (~70s more)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

export PYTHONPATH="${ROOT}:${ROOT}/src:${ROOT}/docs/research/kbound/kbound_pkg:${ROOT}/docs/research/kbound/edge/src"

echo "=== Monorepo health (quick) ==="
echo "PYTHONPATH ok; running kga + kbound research tests..."

"$PY" -m pytest \
  tests/test_kga_package.py \
  tests/test_certificate_drift_guard.py \
  tests/test_smoke_trichotomy.py \
  docs/research/kbound/kbound_pkg/tests \
  docs/research/kbound/tests \
  docs/research/kbound/edge/tests \
  -q --tb=no

echo "=== Canonical script wrappers ==="
"$PY" -c "
from pathlib import Path
from src.scripts.kbound._canonical import canonical_script
for name in ('cifar_tent_mps_v2.py', 'cifar_tent_online.py'):
    p = canonical_script(name)
    assert p.is_file(), p
    print('OK', name)
"

echo "=== Hermetic smoke ==="
PYTHON="$PY" bash scripts/smoke_kbound.sh

if [[ "${1:-}" == "--full" ]]; then
  echo "=== Full submission repro ==="
  bash docs/research/kbound/scripts/reproduce_submission.sh
fi

echo "=== Gate P production audit ==="
"$PY" src/scripts/audit_gate_p_production.py | tail -3

echo "=== Monorepo health: PASS ==="
