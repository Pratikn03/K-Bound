#!/usr/bin/env bash
# Rebuild ELARA-U paper assets and compile the ELARA-U paper PDF.
# CWD-safe: always resolves project paths from this script's location.

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT="$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"

cd "$ROOT"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> Project root: $ROOT"
echo "==> Python: $(which python)"
echo

FIG_DIR="$ROOT/docs/research/figures"
BUILD_DIR="$ROOT/.tex_build_universal"
OUT_PDF_DIR="$ROOT/output/pdf"

mkdir -p "$FIG_DIR" "$BUILD_DIR" "$OUT_PDF_DIR"

echo "==> Regenerate ELARA-U tables"
PYTHONPATH=src python src/scripts/elara_u/emit_tables.py

echo "==> Regenerate ELARA-U figures"
PYTHONPATH=src python src/scripts/elara_u/emit_figures.py

echo
echo "==> Compile ELARA-U Paper"
cd "$ROOT/docs/research"
latexmk -pdf -g -interaction=nonstopmode -outdir="$BUILD_DIR" ELARA_U_PAPER_v0.tex >/dev/null

cp "$BUILD_DIR/ELARA_U_PAPER_v0.pdf" "$OUT_PDF_DIR/ELARA_U_PAPER_v0.pdf"
echo "==> Wrote $OUT_PDF_DIR/ELARA_U_PAPER_v0.pdf"

ELARA_PAGES=$(python -c "from pypdf import PdfReader; print(len(PdfReader('$OUT_PDF_DIR/ELARA_U_PAPER_v0.pdf').pages))" 2>/dev/null || echo "?")
echo "==> Paper Pages: $ELARA_PAGES"

echo "==> Compile ELARA-U Thesis Chapter"
latexmk -pdf -g -interaction=nonstopmode -outdir="$BUILD_DIR" ELARA_U_THESIS_CHAPTER.tex >/dev/null

cp "$BUILD_DIR/ELARA_U_THESIS_CHAPTER.pdf" "$OUT_PDF_DIR/ELARA_U_THESIS_CHAPTER.pdf"
echo "==> Wrote $OUT_PDF_DIR/ELARA_U_THESIS_CHAPTER.pdf"

THESIS_PAGES=$(python -c "from pypdf import PdfReader; print(len(PdfReader('$OUT_PDF_DIR/ELARA_U_THESIS_CHAPTER.pdf').pages))" 2>/dev/null || echo "?")
echo "==> Thesis Pages: $THESIS_PAGES"

