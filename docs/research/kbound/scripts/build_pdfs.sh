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

publish_derived() {
  # Copy only a newly built local artifact.  Rename a fresh sibling over the
  # destination so an old cloud placeholder is never opened or hydrated.
  local source="$1"
  local name="$2"
  local staged
  case "$name" in
    kbound_short_final_draft.pdf|kbound_short_final_draft.log|\
    kbound_submission_build_driver.log|kbound_short_final_build.log|\
    kbound_tmlr.pdf|kbound_tmlr.log|kbound_tmlr_build.log|\
    kbound_short.pdf|kbound_short.log|kbound_full_ieee_diagnostic.pdf|\
    kbound_full_ieee_diagnostic_build.log) ;;
    *) echo "ERROR: refusing unexpected derived-output name: $name" >&2; return 1 ;;
  esac
  if [[ "$source" != "$BUILD_TMP_DIR/"* || ! -s "$source" || -L "$source" || -d "$ROOT/$name" ]]; then
    echo "ERROR: cannot publish local derived artifact: $source -> $name" >&2
    return 1
  fi
  staged="$(mktemp "$ROOT/.${name}.publish.XXXXXX")"
  cp "$source" "$staged"
  chmod 0644 "$staged"
  mv -f "$staged" "$ROOT/$name"
}

build_pdf() {
  local driver="$1"
  local log="$2"
  local jobname="${3:-}"
  local stem="${jobname:-${driver%.tex}}"
  local driver_log="$BUILD_TMP_DIR/$log"
  local suffix
  echo "==> Building $stem.pdf"
  # On its first pass TeX/Hyperref may otherwise find a cwd sidecar when the
  # corresponding output-directory file does not exist yet.  These are new,
  # empty build intermediates, never copies of the previous build's evidence.
  for suffix in aux out toc lof lot loa lol bbl; do
    : >"$BUILD_TMP_DIR/$stem.$suffix"
  done
  local command=(latexmk -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error
    "-outdir=$BUILD_TMP_DIR" "-auxdir=$BUILD_TMP_DIR")
  if [[ -n "$jobname" ]]; then
    command+=("-jobname=$jobname")
  fi
  command+=("$driver")
  if ! "${command[@]}" >"$driver_log" 2>&1; then
    echo "ERROR: build failed; local diagnostics retained in $BUILD_TMP_DIR" >&2
    tail -80 "$driver_log" >&2
    return 1
  fi
  if [[ ! -s "$BUILD_TMP_DIR/$stem.pdf" || ! -s "$BUILD_TMP_DIR/$stem.log" ]]; then
    echo "ERROR: successful compiler did not produce a PDF and TeX log in $BUILD_TMP_DIR" >&2
    return 1
  fi
  if ! pdfinfo "$BUILD_TMP_DIR/$stem.pdf" >"$BUILD_TMP_DIR/$stem.pdfinfo" 2>&1; then
    echo "ERROR: built PDF failed validation; local diagnostics retained in $BUILD_TMP_DIR" >&2
    tail -80 "$BUILD_TMP_DIR/$stem.pdfinfo" >&2
    return 1
  fi
  publish_derived "$BUILD_TMP_DIR/$stem.pdf" "$stem.pdf"
  publish_derived "$BUILD_TMP_DIR/$stem.log" "$stem.log"
  publish_derived "$driver_log" "$log"
}

need latexmk
need pdfinfo

echo "==> Regenerating receipt-bound So2Sat manuscript numbers"
"$PY" scripts/build_so2sat_numbers.py

echo "==> Validating frozen release authorities"
# A manuscript build is a presentation operation.  It must not rewrite sealed
# scientific evidence with a new timestamp, Git head, or local package version.
# Release-data regeneration and resealing are separate, explicit runbook steps.
(cd "$REPO" && "$PY" "$ROOT/scripts/validate_canonical_release_data.py")
"$PY" scripts/build_current_policy_interval_diagnostics.py --check
(cd "$REPO" && PYTHONPATH="$REPO/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" src/scripts/validate_manuscript_claims.py)

echo "==> Regenerating canonical numbers and figures"
"$PY" scripts/make_tables.py
"$PY" scripts/plot_canonical_decision_frontier.py
"$PY" scripts/plot_conceptual_regime_geometry.py
"$PY" scripts/make_submission_figures.py --frontier-only
"$PY" scripts/plot_kga_interval_rule.py

# Keep the paper working directory for relative TeX inputs, but never place
# latexmk's intermediate files or redirected logs beside cloud-backed sources.
# The fresh directory is retained for diagnosis on success and failure alike.
BUILD_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kbound-pdf-build.XXXXXX")"
BUILD_TMP_DIR="$(cd "$BUILD_TMP_DIR" && pwd -P)"
echo "==> Local LaTeX intermediates and diagnostics: $BUILD_TMP_DIR"

build_pdf kbound_submission.tex kbound_submission_build_driver.log kbound_short_final_draft
publish_derived "$BUILD_TMP_DIR/kbound_short_final_draft.log" kbound_short_final_build.log
if [[ "$BUILD_LONG_TMLR" == "1" ]]; then
  echo "==> Building synchronized maintained long TMLR companion"
  build_pdf kbound_tmlr.tex kbound_tmlr_build.log
fi

if [[ "${BUILD_DIAGNOSTIC_IEEE:-0}" == "1" ]]; then
  build_pdf kbound_short.tex kbound_full_ieee_diagnostic_build.log
  publish_derived "$BUILD_TMP_DIR/kbound_short.pdf" kbound_full_ieee_diagnostic.pdf
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

# Dashboard generation precedes compilation in the release runbook. Refresh
# only the actual PDF page count and scoped theorem strip after the build;
# retain every empirical/edge field and the canonical evidence generation date.
echo "==> Refreshing dashboard presentation metadata from the built compact PDF"
"$PY" scripts/build_dashboard_snapshot.py --metadata-only

echo "==> Maintained compact outputs"
ls -lh \
  kbound_short_final_draft.pdf
if [[ "$BUILD_LONG_TMLR" == "1" ]]; then
  echo "==> Maintained synchronized long output"
  ls -lh kbound_tmlr.pdf
fi
