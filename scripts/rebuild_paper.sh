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
MVTEC_PATCHCORE_RESULTS="$ROOT/experiments/fusion/mvtec3d_patchcore_results.json"
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

if [ -f "$MVTEC_PATCHCORE_RESULTS" ]; then
  echo "==> Regenerate MVTec3D PatchCore assets (mvtec3d_patchcore_*)"
  PYTHONPATH=src python src/scripts/emit_mvtec3d_assets.py \
    --input "$MVTEC_PATCHCORE_RESULTS" \
    --figures-dir "$FIG_DIR" \
    --tables-dir "$TBL_DIR" \
    --asset-prefix "mvtec3d_patchcore_"
else
  echo "==> Skipping MVTec3D PatchCore assets: $MVTEC_PATCHCORE_RESULTS not found"
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

echo "==> Regenerate public MVTec protocol comparison"
PYTHONPATH=src python src/scripts/emit_milestone1_comparison.py \
  --repo-root "$ROOT" \
  --output "$TBL_DIR/mvtec3d_milestone1_comparison.tex"

echo "==> Regenerate MVTec 3D-AD SOTA demarcation table + figure"
PYTHONPATH=src python src/scripts/emit_mvtec3d_sota_demarcation.py \
  --repo-root "$ROOT" \
  --table-output "$TBL_DIR/mvtec3d_sota_demarcation.tex" \
  --figure-output "$FIG_DIR/mvtec3d_sota_demarcation.png"

echo "==> Regenerate per-category breakdown tables"
PYTHONPATH=src python src/scripts/emit_per_category_breakdown.py \
  --repo-root "$ROOT" \
  --tables-dir "$TBL_DIR"

echo "==> Regenerate cross-benchmark milestone-2 master (DeLong + Holm)"
PYTHONPATH=src python src/scripts/emit_milestone2_cross_benchmark.py \
  --repo-root "$ROOT" \
  --output "$TBL_DIR/milestone2_cross_benchmark.tex"

echo "==> Regenerate Phase-D theorem-stack validation assets"
PYTHONPATH=src python src/scripts/emit_k_of_d_corruption_table.py \
  --output "$TBL_DIR/elara_k_domain_corruption_results.tex" \
  --figure-output "$FIG_DIR/elara_k_domain_corruption.png"
PYTHONPATH=src python src/scripts/validate_category_mixture_t2.py \
  --output experiments/fusion/category_mixture_t2_validation.json
PYTHONPATH=src python src/scripts/emit_category_mixture_t2_table.py \
  --output "$TBL_DIR/category_mixture_t2.tex"
PYTHONPATH=src python src/scripts/audit_switching_certificate_t5.py \
  --repo-root "$ROOT" \
  --output experiments/fusion/switching_certificate_t5_audit.json
PYTHONPATH=src python src/scripts/emit_switching_certificate_t5_table.py \
  --output "$TBL_DIR/switching_certificate_t5.tex"
PYTHONPATH=src python src/scripts/emit_risk_dominance_t4_table.py \
  --output "$TBL_DIR/risk_dominance_t4_prevalence.tex"
PYTHONPATH=src python src/scripts/emit_ks_power_t6_table.py \
  --output "$TBL_DIR/ks_power_t6.tex"
if [[ ! -f experiments/fusion/meta_router_pac_audit.json ]]; then
  PYTHONPATH=src python src/scripts/audit_meta_router_pac.py \
    --output experiments/fusion/meta_router_pac_audit.json
fi
PYTHONPATH=src python src/scripts/emit_meta_router_pac_t7_table.py \
  --output "$TBL_DIR/meta_router_pac_t7.tex"
PYTHONPATH=src python src/scripts/audit_gate_decision_rule_e2e.py \
  --repo-root "$ROOT" \
  --output experiments/fusion/gate_decision_rule_e2e_audit.json
PYTHONPATH=src python src/scripts/emit_gate_decision_rule_table.py \
  --output "$TBL_DIR/gate_decision_rule_e2e.tex"
PYTHONPATH=src python src/scripts/emit_theory_experiment_mapping.py \
  --repo-root "$ROOT" \
  --output "$TBL_DIR/elara_theory_experiment_mapping.tex"
PYTHONPATH=src python src/scripts/validate_theorem_stack.py \
  --repo-root "$ROOT" \
  --output experiments/fusion/theorem_stack_validation_report.json
PYTHONPATH=src python src/scripts/emit_fourth_benchmark_scaffold.py \
  --output "$TBL_DIR/fourth_benchmark_scaffold.tex"

echo "==> Phase 1.1: strip degenerate canonical PR-AUC / ECE / Brier from promoted tables and figures"
PYTHONPATH=src python src/scripts/phase1_1_canonical_cleanup.py \
  --repo-root "$ROOT"

echo "==> Phase 1.1: rebuild metrics manifest + validator macros"
PYTHONPATH=src python src/scripts/build_metrics_manifest.py \
  --repo-root "$ROOT" \
  --output "$ROOT/docs/research/metrics_manifest.json" \
  --macros-output "$ROOT/docs/research/generated/elara_verified_metrics_macros.tex"

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
