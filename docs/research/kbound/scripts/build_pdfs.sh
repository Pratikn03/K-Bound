#!/usr/bin/env bash
# Build K-Bound short paper PDF + Word (and optionally the long PDF).
#
# Usage:
#   bash docs/research/kbound/scripts/build_pdfs.sh           # short PDF + DOCX
#   BUILD_LONG=1 bash docs/research/kbound/scripts/build_pdfs.sh  # also kbound.pdf
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export COPYFILE_DISABLE=1
PY="${PYTHON:-python3}"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: missing required tool '$1'" >&2
    exit 1
  }
}

need latexmk
need pandoc

echo "==> Generating table numbers from canonical manifest (if available)"
if [[ -f scripts/make_tables.py ]]; then
  "$PY" scripts/make_tables.py
fi

echo "==> Building kbound_short.pdf"
latexmk -pdf -interaction=nonstopmode -halt-on-error kbound_short.tex
cp -f kbound_short.pdf kbound_short_final_draft.pdf

echo "==> Building kbound_short.docx (pandoc from LaTeX)"
# Pandoc may warn on complex math (\rm, align*); DOCX is still produced.
PANDOC_LOG="$(mktemp -t kbound-pandoc.XXXXXX)"
trap 'rm -f "$PANDOC_LOG"' EXIT
pandoc kbound_short.tex \
  -o kbound_short.docx \
  --from latex \
  --resource-path=.:figures:paper:paper/generated:paper/sections \
  --standalone \
  2>"$PANDOC_LOG"
grep -v '^$' "$PANDOC_LOG" >&2 || true
cp -f kbound_short.docx kbound_short_final_draft.docx

if [[ "${BUILD_LONG:-0}" == "1" ]]; then
  echo "==> Building kbound.pdf (long paper)"
  latexmk -pdf -interaction=nonstopmode -halt-on-error kbound.tex
fi

echo "==> Done:"
ls -lh kbound_short.pdf kbound_short_final_draft.pdf \
  kbound_short.docx kbound_short_final_draft.docx
if [[ "${BUILD_LONG:-0}" == "1" ]]; then
  ls -lh kbound.pdf
fi

echo ""
echo "PDF:  $ROOT/kbound_short.pdf"
echo "Word: $ROOT/kbound_short.docx"
