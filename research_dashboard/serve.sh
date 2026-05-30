#!/usr/bin/env bash
# Serve repo root so the dashboard can fetch audit JSON for gate drill-down.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8765}"
echo "Repo root: ${ROOT}"
echo "Dashboard: http://localhost:${PORT}/research_dashboard/web/?live=1"
echo "Press Ctrl+C to stop."
cd "${ROOT}"
exec python3 -m http.server "${PORT}"
