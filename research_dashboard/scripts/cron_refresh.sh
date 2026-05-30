#!/usr/bin/env bash
# Example cron entry (every 15 minutes):
# */15 * * * * /path/to/AutoML_Flagship_V8/research_dashboard/scripts/cron_refresh.sh >> /tmp/elara_dashboard.log 2>&1

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${ROOT}/research_dashboard/build/elara_research_snapshot"

if [[ ! -x "${BIN}" ]]; then
  "${ROOT}/research_dashboard/build.sh"
fi

exec "${BIN}" --repo-root "${ROOT}"
