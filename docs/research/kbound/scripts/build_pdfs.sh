#!/usr/bin/env bash
# Build the authoritative K-Bound short paper.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export COPYFILE_DISABLE=1

echo "==> Generating table numbers from canonical manifest"
if [[ -f scripts/make_tables.py ]]; then
  python3 scripts/make_tables.py
fi

echo "==> Building kbound_short.pdf"
latexmk -pdf -interaction=nonstopmode -halt-on-error kbound_short.tex
cp kbound_short.pdf kbound_short_final_draft.pdf

echo "==> Done:"
ls -lh kbound_short_final_draft.pdf
