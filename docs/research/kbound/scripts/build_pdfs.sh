#!/usr/bin/env bash
# Validate sealed evidence, regenerate presentation assets, and build the maintained compact PDF.
# Optionally build the synchronized maintained long companion from kbound_tmlr.tex.
#
# Usage:
#   bash docs/research/kbound/scripts/build_pdfs.sh
#   BUILD_LONG_TMLR=1 bash docs/research/kbound/scripts/build_pdfs.sh
#   BUILD_DOCX=1 bash docs/research/kbound/scripts/build_pdfs.sh
#
# BUILD_HISTORICAL_TMLR remains a backward-compatible alias for BUILD_LONG_TMLR.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"
cd "$ROOT"
export COPYFILE_DISABLE=1

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
else
  PY="python3"
fi

BUILD_LONG_TMLR="${BUILD_LONG_TMLR:-${BUILD_HISTORICAL_TMLR:-0}}"
case "$BUILD_LONG_TMLR" in
  0|1) ;;
  *)
    echo "ERROR: BUILD_LONG_TMLR must be 0 or 1" >&2
    exit 1
    ;;
esac

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: missing required tool '$1'" >&2
    exit 1
  }
}

build_pdf() {
  local driver="$1"
  local log="$2"
  local jobname="${3:-}"
  echo "==> Building ${jobname:-${driver%.tex}}.pdf"
  local command=(latexmk -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error)
  if [[ -n "$jobname" ]]; then
    command+=("-jobname=$jobname")
  fi
  command+=("$driver")
  if ! "${command[@]}" >"$log" 2>&1; then
    tail -80 "$log" >&2
    return 1
  fi
}

need latexmk

if [[ ! -f kbound_short_original_build.log && -f kbound_tmlr.log ]]; then
  cp -f kbound_tmlr.log kbound_short_original_build.log
fi

echo "==> Validating frozen release authorities"
# A manuscript build is a presentation operation.  It must not rewrite sealed
# scientific evidence with a new timestamp, Git head, or local package version.
# Release-data regeneration and resealing are separate, explicit runbook steps.
(cd "$REPO" && "$PY" "$ROOT/scripts/validate_canonical_release_data.py")
(cd "$REPO" && PYTHONPATH="$REPO/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" src/scripts/validate_manuscript_claims.py)

echo "==> Regenerating canonical numbers and figures"
"$PY" scripts/make_tables.py
"$PY" scripts/plot_canonical_decision_frontier.py
"$PY" scripts/plot_conceptual_regime_geometry.py

build_pdf kbound_submission.tex kbound_submission_build_driver.log kbound_short_final_draft
cp -f kbound_short_final_draft.log kbound_short_final_build.log
if [[ "$BUILD_LONG_TMLR" == "1" ]]; then
  echo "==> Building synchronized maintained long TMLR companion"
  build_pdf kbound_tmlr.tex kbound_tmlr_build.log
fi

if [[ "${BUILD_DIAGNOSTIC_IEEE:-0}" == "1" ]]; then
  build_pdf kbound_short.tex kbound_full_ieee_diagnostic_build.log
  cp -f kbound_short.pdf kbound_full_ieee_diagnostic.pdf
fi

# The maintained outputs are written in place. Historical compatibility PDFs
# are deliberately not refreshed: they are not release deliverables.
chmod 0644 kbound_short_final_draft.pdf
if [[ "$BUILD_LONG_TMLR" == "1" ]]; then
  chmod 0644 kbound_tmlr.pdf
fi

if [[ "${BUILD_DOCX:-0}" == "1" ]]; then
  echo "==> Building styled compact DOCX from the synchronized source"
  "$PY" scripts/build_docx.py --output kbound_short_final_draft.docx
fi

echo "==> Maintained compact outputs"
ls -lh \
  kbound_short_final_draft.pdf
if [[ "$BUILD_LONG_TMLR" == "1" ]]; then
  echo "==> Maintained synchronized long output"
  ls -lh kbound_tmlr.pdf
fi
