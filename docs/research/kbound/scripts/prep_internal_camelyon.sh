#!/usr/bin/env bash
# prep_internal_camelyon.sh
# Copy Camelyon17 (patches + metadata + RELEASE marker) from the slow exFAT T9 drive to the
# fast internal SSD (~/kbound_cam) so the 4x4 training loads ~10x faster. Resumable (rsync).
# Does NOT copy archive.tar.gz (9 GB, not needed for training). ~12-15 GB of small PNGs.
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

set -uo pipefail
SRC="$KB_REPO_ROOT"/experiments/kbound/data/wilds/camelyon17_v1.0
DST="$HOME/kbound_cam/wilds/camelyon17_v1.0"

echo "============================================================"
echo " Copying Camelyon17 patches -> internal SSD"
echo "   from: $SRC"
echo "   to:   $DST"
echo " Free internal BEFORE: $(df -h / | awk 'NR==2{print $4}')"
echo " (~12-15 GB of small PNGs off exFAT -- expect 30-90 min; rsync resumes if interrupted)"
echo "============================================================"
mkdir -p "$DST"

echo "==> patches/ ..."
rsync -a --exclude '._*' "$SRC/patches" "$DST/"
echo "==> metadata.csv ..."
rsync -a "$SRC/metadata.csv" "$DST/" 2>/dev/null || echo "  WARNING: metadata.csv not copied"
if [ -f "$SRC/RELEASE_v1.0.txt" ]; then cp "$SRC/RELEASE_v1.0.txt" "$DST/"; else echo "Camelyon17 v1.0 (local copy)" > "$DST/RELEASE_v1.0.txt"; fi

echo "============================================================"
echo " Free internal AFTER: $(df -h / | awk 'NR==2{print $4}')"
echo " patch node-dirs on internal: $(ls -1 "$DST/patches" 2>/dev/null | grep -v '^\._' | wc -l | tr -d ' ')"
echo " metadata.csv: $([ -f "$DST/metadata.csv" ] && echo present || echo MISSING)"
echo " RELEASE marker: $([ -f "$DST/RELEASE_v1.0.txt" ] && echo present || echo MISSING)"
echo "============================================================"
echo " READY. Train (quick check, then full) with:"
echo "   bash $KB_REPO_ROOT/docs/research/kbound/scripts/kbtrain.sh camelyon-fast-1pct"
echo "   bash $KB_REPO_ROOT/docs/research/kbound/scripts/kbtrain.sh camelyon-fast"
