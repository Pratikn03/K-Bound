#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PY="${KBOUND_PYTHON:-$HOME/.venv_wilds/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: research Python is not executable: $PY" >&2
  echo "Set KBOUND_PYTHON to a Python with torch, torchvision, and scikit-learn." >&2
  exit 2
fi

exec "$PY" "$ROOT/experiments/kbound/training/run_multiseed.py" "$@"
