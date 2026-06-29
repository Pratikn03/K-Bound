#!/usr/bin/env bash
# Full theory audit wrapper (macOS-friendly).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="${HOME}/.venv_wilds/bin/python"
[[ -x "${PY}" ]] || PY=python3
exec "${PY}" "${ROOT}/docs/research/kbound/scripts/theory_audit_full.py" "$@"
