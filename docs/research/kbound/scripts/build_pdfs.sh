#!/usr/bin/env bash
# Validate sealed evidence, regenerate presentation assets, and build the maintained compact PDF.
# Optionally build the synchronized maintained long companion from kbound_tmlr.tex.
#
# Usage:
#   bash docs/research/kbound/scripts/build_pdfs.sh
#   BUILD_LONG_TMLR=1 bash docs/research/kbound/scripts/build_pdfs.sh
#   BUILD_SHORT_MAIN=1 bash docs/research/kbound/scripts/build_pdfs.sh
#   BUILD_SHORT_SUPPLEMENT=1 bash docs/research/kbound/scripts/build_pdfs.sh
#   BUILD_FULL_REPORT=1 bash docs/research/kbound/scripts/build_pdfs.sh
#   BUILD_DOCX=1 bash docs/research/kbound/scripts/build_pdfs.sh
#
# BUILD_HISTORICAL_TMLR remains a backward-compatible alias for BUILD_LONG_TMLR.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"
cd "$ROOT"
export COPYFILE_DISABLE=1
# Freeze TeX's clock for byte-reproducible release PDFs.  The epoch is the
# start of the final release day in the repository's declared local timezone
# (2026-09-04 00:00:00 America/Chicago).  Individual drivers also suppress the
# random trailer identifier and engine-path metadata.
export SOURCE_DATE_EPOCH=1788498000
export FORCE_SOURCE_DATE=1
export TZ=UTC

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
else
  PY="python3"
fi

BUILD_LONG_TMLR="${BUILD_LONG_TMLR:-${BUILD_HISTORICAL_TMLR:-0}}"
BUILD_SHORT_MAIN="${BUILD_SHORT_MAIN-0}"
BUILD_SHORT_SUPPLEMENT="${BUILD_SHORT_SUPPLEMENT-0}"
BUILD_FULL_REPORT="${BUILD_FULL_REPORT-0}"
KBOUND_AUTHORIZE_PROTECTED_SO2SAT="${KBOUND_AUTHORIZE_PROTECTED_SO2SAT:-0}"
if [[ -n "${SOURCE_SNAPSHOT_COMMIT:-}" ]]; then
  KBOUND_SOURCE_SNAPSHOT_COMMIT="$SOURCE_SNAPSHOT_COMMIT"
elif [[ -e "$REPO/.git" ]]; then
  KBOUND_SOURCE_SNAPSHOT_COMMIT="$(git -C "$REPO" rev-parse --short=12 HEAD)"
else
  KBOUND_SOURCE_SNAPSHOT_COMMIT="000000000000"
  echo "WARNING: no Git metadata; using an unsealed nonrelease source identity" >&2
fi
if [[ ! "$KBOUND_SOURCE_SNAPSHOT_COMMIT" =~ ^[0-9a-f]{12}$ ]]; then
  echo "ERROR: SOURCE_SNAPSHOT_COMMIT must be exactly 12 lowercase hex characters" >&2
  exit 1
fi

validate_binary_build_flag() {
  local name="$1"
  local value="$2"
  case "$value" in
    0|1) ;;
    *)
      echo "ERROR: $name must be 0 or 1" >&2
      exit 1
      ;;
  esac
}

# Validate every build selector before resolving release tools so malformed
# opt-ins fail without consulting the local LaTeX installation.
validate_binary_build_flag BUILD_LONG_TMLR "$BUILD_LONG_TMLR"
validate_binary_build_flag BUILD_SHORT_MAIN "$BUILD_SHORT_MAIN"
validate_binary_build_flag BUILD_SHORT_SUPPLEMENT "$BUILD_SHORT_SUPPLEMENT"
validate_binary_build_flag BUILD_FULL_REPORT "$BUILD_FULL_REPORT"
validate_binary_build_flag KBOUND_AUTHORIZE_PROTECTED_SO2SAT "$KBOUND_AUTHORIZE_PROTECTED_SO2SAT"

resolve_release_tool() {
  local logical_name="$1"
  local command_name="$2"
  local override_name="KBOUND_TOOL_${logical_name}"
  local override="${!override_name:-}"
  local candidate
  if [[ -n "$override" ]]; then
    case "$override" in
      /*) ;;
      *) echo "ERROR: $override_name must name an absolute non-symlink executable" >&2; return 1 ;;
    esac
    if [[ ! -f "$override" || ! -x "$override" || -L "$override" ]]; then
      echo "ERROR: $override_name must name an absolute non-symlink executable" >&2
      return 1
    fi
    printf '%s\n' "$override"
    return
  fi
  candidate="$(command -v "$command_name" 2>/dev/null || true)"
  if [[ -z "$candidate" || ! -f "$candidate" || ! -x "$candidate" ]]; then
    echo "ERROR: missing required tool '$command_name'" >&2
    return 1
  fi
  printf '%s\n' "$candidate"
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
    kbound_tmlr.pdf|kbound_tmlr.log|kbound_tmlr_build.log) ;;
    kbound_short_main.pdf|kbound_short_main.log|kbound_short_main_build.log|\
    kbound_short_supplement.pdf|kbound_short_supplement.log|kbound_short_supplement_build.log|\
    kbound_full_report.pdf|kbound_full_report.log|kbound_full_report_build.log) ;;
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
  local command=("$PERL" "$LATEXMK" -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error
    "-pdflatex=$PDFLATEX_COMMAND"
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
  if ! "$PDFINFO" "$BUILD_TMP_DIR/$stem.pdf" >"$BUILD_TMP_DIR/$stem.pdfinfo" 2>&1; then
    echo "ERROR: built PDF failed validation; local diagnostics retained in $BUILD_TMP_DIR" >&2
    tail -80 "$BUILD_TMP_DIR/$stem.pdfinfo" >&2
    return 1
  fi
  publish_derived "$BUILD_TMP_DIR/$stem.pdf" "$stem.pdf"
  publish_derived "$BUILD_TMP_DIR/$stem.log" "$stem.log"
  publish_derived "$driver_log" "$log"
}

LATEXMK="$(resolve_release_tool LATEXMK latexmk)"
PERL="$(resolve_release_tool PERL perl)"
PDFLATEX="$(resolve_release_tool PDFLATEX pdflatex)"
PDFINFO="$(resolve_release_tool PDFINFO pdfinfo)"
# The verified realpath behind the conventional `pdflatex` symlink is the
# `pdftex` executable. Bind the LaTeX format explicitly so resolving the
# symlink cannot silently switch the engine to plain TeX via argv[0].
printf -v PDFLATEX_COMMAND '%q -fmt=pdflatex %%O %%S' "$PDFLATEX"

echo "==> Validating frozen release authorities"
# A manuscript build is a presentation operation.  It must not rewrite sealed
# scientific evidence with a new timestamp, Git head, or local package version.
# Release-data regeneration and resealing are separate, explicit runbook steps.
if [[ "$KBOUND_AUTHORIZE_PROTECTED_SO2SAT" == "1" ]]; then
  echo "==> Explicit authorization: refreshing and validating protected So2Sat development authority"
  "$PY" scripts/build_so2sat_numbers.py
  (cd "$REPO" && "$PY" "$ROOT/scripts/validate_canonical_release_data.py")
else
  echo "==> Protected So2Sat development authority remains outside the public build closure"
fi
"$PY" scripts/build_current_policy_interval_diagnostics.py --check
# The claim validator expands every active TeX include.  Generate the shared
# release identity first so a clean checkout does not depend on an older local
# build product being present.
"$PY" scripts/generate_release_identity.py \
  --source-snapshot-commit "$KBOUND_SOURCE_SNAPSHOT_COMMIT"
if [[ "$KBOUND_AUTHORIZE_PROTECTED_SO2SAT" == "1" ]]; then
  (cd "$REPO" && PYTHONPATH="$REPO/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" src/scripts/validate_manuscript_claims.py --authorize-protected-so2sat)
else
  (cd "$REPO" && PYTHONPATH="$REPO/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" src/scripts/validate_manuscript_claims.py)
fi

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
if [[ "$BUILD_SHORT_MAIN" == "1" ]]; then
  echo "==> Building standalone short main paper"
  build_pdf kbound_short_main.tex kbound_short_main_build.log
fi
if [[ "$BUILD_SHORT_SUPPLEMENT" == "1" ]]; then
  echo "==> Building standalone short supplement"
  build_pdf kbound_short_supplement.tex kbound_short_supplement_build.log
fi
if [[ "$BUILD_FULL_REPORT" == "1" ]]; then
  echo "==> Building full technical report"
  build_pdf kbound_full_report.tex kbound_full_report_build.log
fi

# The maintained outputs are written in place. Historical compatibility PDFs
# are deliberately not refreshed: they are not release deliverables.
chmod 0644 kbound_short_final_draft.pdf
if [[ "$BUILD_LONG_TMLR" == "1" ]]; then
  chmod 0644 kbound_tmlr.pdf
fi
if [[ "$BUILD_SHORT_MAIN" == "1" ]]; then
  chmod 0644 kbound_short_main.pdf
fi
if [[ "$BUILD_SHORT_SUPPLEMENT" == "1" ]]; then
  chmod 0644 kbound_short_supplement.pdf
fi
if [[ "$BUILD_FULL_REPORT" == "1" ]]; then
  chmod 0644 kbound_full_report.pdf
fi

# Publish a single unambiguous release set only when every maintained driver
# was rebuilt in this invocation. Partial developer builds must never combine
# fresh and older PDFs under the current release names.
if [[ "$BUILD_LONG_TMLR" == "1" && "$BUILD_SHORT_MAIN" == "1" && \
      "$BUILD_SHORT_SUPPLEMENT" == "1" && "$BUILD_FULL_REPORT" == "1" ]]; then
  echo "==> Publishing one stable current PDF set"
  "$PY" scripts/publish_current_pdfs.py \
    --paper-dir "$ROOT" \
    --release-dir "$ROOT/release/current" \
    --output-dir "$REPO/output/pdf"
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
if [[ "$BUILD_SHORT_MAIN" == "1" ]]; then
  ls -lh kbound_short_main.pdf
fi
if [[ "$BUILD_SHORT_SUPPLEMENT" == "1" ]]; then
  ls -lh kbound_short_supplement.pdf
fi
if [[ "$BUILD_FULL_REPORT" == "1" ]]; then
  ls -lh kbound_full_report.pdf
fi
