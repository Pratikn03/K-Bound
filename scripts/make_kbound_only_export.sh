#!/usr/bin/env bash
# Clean standalone KBOUND-ONLY repo from tracked files (no src/uais, no ELARA-U,
# no untracked build junk, no raw data). Non-destructive: full repo untouched.
# --- external (git-excluded) data volume: ONE documented variable, no default.
# --- defect D8: portable roots. No machine-local absolute paths in tracked code
# --- (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md). KB_REPO_ROOT is discovered
# --- from this script's own location; override with KBOUND_REPO_ROOT.
_kb_find_root() {
  d=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
  while [ "$d" != "/" ]; do
    [ -f "$d/pyproject.toml" ] && { printf '%s\n' "$d"; return 0; }
    d=$(dirname "$d")
  done
  echo "ERROR: repository root not found above $(dirname "${BASH_SOURCE[0]:-$0}")" >&2
  return 1
}
KB_REPO_ROOT="${KBOUND_REPO_ROOT:-$(_kb_find_root)}" || exit 1

: "${KBOUND_EXTERNAL_ROOT:?set KBOUND_EXTERNAL_ROOT to the volume holding the git-excluded datasets/checkpoints/caches (layout: docs/research/kbound/kbound_repro/paths.py, acquisition: DATA.md)}"
KB_EXTERNAL_ROOT="$KBOUND_EXTERNAL_ROOT"

set -euo pipefail
SRC="$KB_REPO_ROOT"; DST="${1:-$KB_EXTERNAL_ROOT/kbound_only}"
rm -rf "$DST"; mkdir -p "$DST" "$DST/tests"; cd "$SRC"
git archive HEAD kga src/scripts/kbound docs/research/kbound research_lock experiments/kbound scripts \
  | ( cd "$DST" && tar -x )
for t in test_kga_package test_certificate_drift_guard test_smoke_trichotomy test_data_manifest; do
  cp "tests/$t.py" "$DST/tests/" 2>/dev/null || true; done
for f in requirements.txt requirements.lock.txt pyproject.toml README.md DATA.md LICENSE CITATION.cff .gitignore; do
  cp "$f" "$DST/" 2>/dev/null || true; done
echo "exported -> $DST"
