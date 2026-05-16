#!/usr/bin/env bash
# Rebuild paper assets and compile the paper and thesis PDFs.
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

LA_RESULTS="$ROOT/experiments/fusion/craf_real_results.json"
LA_META="$ROOT/experiments/fusion/real_domain_fusion_metadata.json"
MVTEC_RESULTS="$ROOT/experiments/fusion/mvtec3d_results.json"
MVTEC_META="$ROOT/experiments/fusion/mvtec3d_fusion_metadata.json"
HEALTHCARE_GAP_REPORT="$ROOT/experiments/fusion/healthcare_gap4_deployment_audit_validation.json"
FIG_DIR="$ROOT/docs/research/figures"
TBL_DIR="$ROOT/docs/research/tables"
BUILD_DIR="$ROOT/.tex_build"
THESIS_BUILD_DIR="$ROOT/.tex_build_thesis"
OUT_PDF_DIR="$ROOT/output/pdf"

mkdir -p "$FIG_DIR" "$TBL_DIR" "$BUILD_DIR" "$THESIS_BUILD_DIR" "$OUT_PDF_DIR"

if [ -f "$LA_RESULTS" ]; then
  echo "==> Regenerate LA-benchmark assets (elara_*)"
  PYTHONPATH=src python src/scripts/generate_craf_paper_assets.py \
    --input "$LA_RESULTS" \
    --metadata "$LA_META" \
    --figures-dir "$FIG_DIR" \
    --tables-dir "$TBL_DIR"
else
  echo "==> Skipping LA assets: $LA_RESULTS not found"
fi

if [ -f "$MVTEC_RESULTS" ]; then
  echo "==> Regenerate MVTec3D assets (mvtec3d_*)"
  PYTHONPATH=src python src/scripts/emit_mvtec3d_assets.py \
    --input "$MVTEC_RESULTS" \
    --metadata "$MVTEC_META" \
    --figures-dir "$FIG_DIR" \
    --tables-dir "$TBL_DIR"
else
  echo "==> Skipping MVTec3D assets: $MVTEC_RESULTS not found"
fi

if [ -f "$HEALTHCARE_GAP_REPORT" ]; then
  echo "==> Regenerate healthcare gap-closure assets"
  PYTHONPATH=src python src/scripts/generate_healthcare_gap_assets.py \
    --report "$HEALTHCARE_GAP_REPORT" \
    --figures-dir "$FIG_DIR" \
    --tables-dir "$TBL_DIR"
else
  echo "==> Skipping healthcare gap assets: $HEALTHCARE_GAP_REPORT not found"
fi

echo
echo "==> Compile PDF (from docs/research/ so \\graphicspath resolves)"
cd "$ROOT/docs/research"
latexmk -pdf -g -interaction=nonstopmode -outdir="$BUILD_DIR" PAPER_DRAFT_v1.tex >/dev/null

cp "$BUILD_DIR/PAPER_DRAFT_v1.pdf" "$OUT_PDF_DIR/PAPER_DRAFT_v1.pdf"
echo "==> Wrote $OUT_PDF_DIR/PAPER_DRAFT_v1.pdf"

PAGES=$(python -c "from pypdf import PdfReader; print(len(PdfReader('$OUT_PDF_DIR/PAPER_DRAFT_v1.pdf').pages))" 2>/dev/null || echo "?")
echo "==> Pages: $PAGES"

echo
echo "==> Compile thesis chapter"
latexmk -pdf -g -interaction=nonstopmode -outdir="$THESIS_BUILD_DIR" THESIS_CHAPTER_v1.tex >/dev/null

cp "$THESIS_BUILD_DIR/THESIS_CHAPTER_v1.pdf" "$OUT_PDF_DIR/THESIS_CHAPTER_v1.pdf"
echo "==> Wrote $OUT_PDF_DIR/THESIS_CHAPTER_v1.pdf"

THESIS_PAGES=$(python -c "from pypdf import PdfReader; print(len(PdfReader('$OUT_PDF_DIR/THESIS_CHAPTER_v1.pdf').pages))" 2>/dev/null || echo "?")
echo "==> Pages: $THESIS_PAGES"
