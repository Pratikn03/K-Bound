#!/usr/bin/env bash
# Wrapper: physical capture → edge/scripts/01_capture_real_session.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash -c 'cd "'"${ROOT}"'/docs/research/kbound/edge/scripts" && "'"${ROOT}"'/.venv/bin/python" 01_capture_real_session.py "$@"' _ "$@"
