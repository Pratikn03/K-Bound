#!/usr/bin/env bash
# Build K-Bound Lean formalization on macOS + external T9 drive.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# macOS AppleDouble files on exFAT/APFS external disks can break Lean package
# loading and should never be committed.
if find . -name '._*' -print -quit 2>/dev/null | grep -q .; then
  echo "Removing macOS ._ junk files under formal/ ..."
  find . -name '._*' -delete
fi

echo "Updating Lake manifest (cache fetch may fail on T9 — that is OK) ..."
lake update || true

echo "Building KBound (first run compiles Mathlib from source; expect 15–40 min) ..."
lake build KBound

echo "Auditing mechanized theorem coverage and documented probability-layer limits ..."
python3 formal_audit.py

echo ""
echo "OK: all currently mechanized K-Bound theorems typechecked."
echo "NOTE: this is a strict-core audit, not a full foundational Mathlib probability development."
echo "See KBound/TheoremMap.lean for paper-label → Lean-name mapping."
