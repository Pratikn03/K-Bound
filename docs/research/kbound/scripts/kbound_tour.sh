#!/usr/bin/env bash
# K-Bound complete picture tour (macOS-friendly — no bare `python` required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${ROOT}"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="${HOME}/.venv_wilds/bin/python"
fi
if [[ ! -x "${PY}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    echo "ERROR: need .venv/bin/python, ~/.venv_wilds/bin/python, or python3 on PATH" >&2
    exit 1
  fi
fi

exec "${PY}" "${ROOT}/docs/research/kbound/scripts/kbound_tour.py" "$@"
