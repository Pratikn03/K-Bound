#!/usr/bin/env bash
# Rebuild the maintained K-Bound release from committed compact artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
KBOUND="${ROOT}/docs/research/kbound"
PY="${PYTHON:-python3}"

cd "${ROOT}"

echo "==> Python integrity suite"
"${PY}" -m pytest -q

echo "==> Canonical paper macros"
"${PY}" "${KBOUND}/scripts/make_tables.py"

echo "==> Deterministic dashboard snapshot"
"${PY}" "${KBOUND}/scripts/build_dashboard_snapshot.py"

echo "==> TypeScript dashboard"
cd "${KBOUND}/dashboard"
npm ci
npm run build

echo "==> Authoritative short paper"
cd "${KBOUND}"
latexmk -pdf -interaction=nonstopmode -halt-on-error kbound_short.tex
cp kbound_short.pdf kbound_short_final_draft.pdf

if grep -Eq 'undefined references|Citation .* undefined|Reference .* undefined|There were undefined' kbound_short.log; then
  echo "Unresolved LaTeX reference or citation" >&2
  exit 1
fi

if [[ "${RUN_LEAN:-0}" == "1" ]]; then
  echo "==> Lean strict-core verification"
  cd "${KBOUND}/formal"
  bash build.sh
else
  echo "==> Lean build skipped; run with RUN_LEAN=1 for kernel verification"
fi

echo "==> Release reproduction complete"
echo "PDF: ${KBOUND}/kbound_short_final_draft.pdf"
echo "Dashboard: ${KBOUND}/kbound_dashboard.html"
echo "Physical-study evidence remains pending until the separate publication gate passes."
