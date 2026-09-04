#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"
ORIGINAL_DIR="$SCRIPT_DIR/originals"
TEMPLATE="$SCRIPT_DIR/watermark_wrapper.tex"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kbound-superseded.XXXXXX")"

cleanup() {
  rm -rf -- "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p -- "$ORIGINAL_DIR"

preserve_original() {
  local source_pdf="$1"
  local historical_name="$2"
  local preserved_pdf="$ORIGINAL_DIR/$historical_name"

  if [[ ! -f "$source_pdf" ]]; then
    printf 'Missing historical source: %s\n' "$source_pdf" >&2
    return 1
  fi

  if [[ -e "$preserved_pdf" ]]; then
    if ! cmp -s -- "$source_pdf" "$preserved_pdf"; then
      printf 'Refusing to overwrite non-identical preserved original: %s\n' "$preserved_pdf" >&2
      return 1
    fi
  else
    cp -p -- "$source_pdf" "$preserved_pdf"
  fi
}

make_warning_copy() {
  local historical_name="$1"
  local stem="${historical_name%.pdf}"
  local source_pdf="$ORIGINAL_DIR/$historical_name"
  local warning_pdf="$SCRIPT_DIR/${stem}_SUPERSEDED.pdf"
  local build_dir="$TMP_DIR/$stem"

  mkdir -p -- "$build_dir"
  ln -s -- "$source_pdf" "$build_dir/source.pdf"
  (
    cd -- "$build_dir"
    pdflatex \
      -halt-on-error \
      -interaction=batchmode \
      -file-line-error \
      -jobname="${stem}_SUPERSEDED" \
      "\\def\\SourcePdf{source.pdf}\\input{$TEMPLATE}" >/dev/null
  )
  cp -p -- "$build_dir/${stem}_SUPERSEDED.pdf" "$warning_pdf"
}

preserve_original \
  "$REPO_ROOT/docs/research/kbound/archive/stale_publication_builds_2026-09-02/tree/docs/research/kbound/kbound_short.pdf" \
  "kbound_short.pdf"
preserve_original \
  "$REPO_ROOT/docs/research/kbound/archive/stale_publication_builds_2026-09-02/tree/docs/research/kbound/kbound.pdf" \
  "kbound.pdf"
preserve_original \
  "$REPO_ROOT/archive/legacy_elara/audits/integrity_2026-06-20/backup_tex_20260620_212018/kbound/kbound_submission.pdf" \
  "kbound_submission.pdf"

make_warning_copy "kbound_short.pdf"
make_warning_copy "kbound.pdf"
make_warning_copy "kbound_submission.pdf"

printf 'Built superseded warning copies in %s\n' "$SCRIPT_DIR"
