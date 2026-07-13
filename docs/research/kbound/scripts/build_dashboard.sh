#!/usr/bin/env bash
# Build TypeScript dashboard + refresh artifact snapshot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
DASH="$ROOT/docs/research/kbound/dashboard"
PY="${PYTHON:-python3}"

cd "$DASH"
if [[ ! -d node_modules ]]; then
  npm install --no-audit --no-fund
fi
npm run build

cd "$ROOT"
$PY docs/research/kbound/scripts/build_dashboard_snapshot.py
echo "[build_dashboard] OK -> dashboard/js/*.js + dashboard/data/snapshot.json"
