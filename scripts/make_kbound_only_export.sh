#!/usr/bin/env bash
# Clean standalone KBOUND-ONLY repo from tracked files (no src/uais, no ELARA-U,
# no untracked build junk, no raw data). Non-destructive: full repo untouched.
set -euo pipefail
SRC="/Volumes/T9/uav/AutoML_Flagship_V8"; DST="${1:-/Volumes/T9/uav/kbound_only}"
rm -rf "$DST"; mkdir -p "$DST" "$DST/tests"; cd "$SRC"
git archive HEAD kga src/scripts/kbound docs/research/kbound research_lock experiments/kbound scripts \
  | ( cd "$DST" && tar -x )
for t in test_kga_package test_certificate_drift_guard test_smoke_trichotomy test_data_manifest; do
  cp "tests/$t.py" "$DST/tests/" 2>/dev/null || true; done
for f in requirements.txt requirements.lock.txt pyproject.toml README.md DATA.md LICENSE CITATION.cff .gitignore; do
  cp "$f" "$DST/" 2>/dev/null || true; done
echo "exported -> $DST"
