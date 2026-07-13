#!/usr/bin/env bash
# Build the K-Bound Lean formalization from the pinned Lake manifest.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

clean_appledouble() {
  # macOS AppleDouble files can break Lean package loading and should never be
  # committed. Normal builds skip `.lake`; set CLEAN_LAKE_APPLEDOUBLE=1 for a
  # deep dependency-cache cleanup.
  if [ "${CLEAN_LAKE_APPLEDOUBLE:-0}" = "1" ]; then
    if find . -name '._*' -print -quit 2>/dev/null | grep -q .; then
      echo "Removing macOS ._ junk files under formal/ including .lake ..."
      find . -name '._*' -exec rm -f {} +
    fi
    return
  fi

  local found=0
  for d in . KBound KBound/Probability; do
    if [ -d "$d" ] && find "$d" -maxdepth 1 -name '._*' -print -quit 2>/dev/null | grep -q .; then
      found=1
    fi
  done
  if [ "$found" = "1" ]; then
    echo "Removing macOS ._ junk files under formal source files ..."
    find . -maxdepth 1 -name '._*' -exec rm -f {} +
    find KBound -maxdepth 1 -name '._*' -exec rm -f {} +
    find KBound/Probability -maxdepth 1 -name '._*' -exec rm -f {} +
  fi
}

clean_appledouble

if [ "${RUN_LAKE_UPDATE:-0}" = "1" ]; then
  echo "Updating the Lake manifest (cache fetch failures fall back to source builds) ..."
  lake update || true
  clean_appledouble
else
  echo "Skipping lake update; using pinned lake-manifest.json. Set RUN_LAKE_UPDATE=1 to refresh."
fi

echo "Building KBound (first run compiles Mathlib from source; expect 15–40 min) ..."
lake build KBound
clean_appledouble

echo "Auditing mechanized theorem coverage and documented probability-layer limits ..."
python3 formal_audit.py

echo ""
echo "OK: all currently mechanized K-Bound theorems typechecked."
echo "NOTE: this is a strict-core audit, not a full foundational Mathlib probability development."
echo "See KBound/TheoremMap.lean for paper-label → Lean-name mapping."
