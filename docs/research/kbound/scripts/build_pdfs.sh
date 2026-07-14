#!/usr/bin/env bash
# Build kbound_short.pdf and kbound.pdf (IEEE + long paper).
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

echo "==> Building kbound.pdf"
latexmk -pdf -interaction=nonstopmode -halt-on-error kbound.tex

echo "==> Done:"
ls -lh kbound_short.pdf kbound.pdf
