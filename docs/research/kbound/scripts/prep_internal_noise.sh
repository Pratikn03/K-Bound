#!/usr/bin/env bash
# prep_internal_noise.sh
# Copy the ImageNet-C *noise* data the standard grid needs (gaussian/shot/impulse,
# severities 1/3/5) from the slow exFAT T9 drive to the fast internal SSD (~/kbound_inc),
# so the TTA run loads ~10x faster. Resumable (rsync): if interrupted, just re-run.
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

SRC="$KB_REPO_ROOT"/experiments/kbound/data/imagenet-c
DST="$HOME/kbound_inc"

echo "============================================================"
echo " Copying ImageNet-C noise (sev 1/3/5) -> internal SSD"
echo "   from: $SRC"
echo "   to:   $DST"
echo " Free internal BEFORE: $(df -h / | awk 'NR==2{print $4}')"
echo "============================================================"
mkdir -p "$DST"

for c in gaussian_noise shot_noise impulse_noise; do
  for s in 1 3 5; do
    if [ -d "$SRC/$c/$s" ]; then
      echo "==> $c/$s ..."
      mkdir -p "$DST/$c"
      rsync -a --exclude '._*' "$SRC/$c/$s" "$DST/$c/"
    else
      echo "!! source missing: $c/$s (skipping)"
    fi
  done
done

echo "============================================================"
echo " Free internal AFTER: $(df -h / | awk 'NR==2{print $4}')"
echo " Verify (each severity should show ~1000 class dirs):"
ok=1
for c in gaussian_noise shot_noise impulse_noise; do
  for s in 1 3 5; do
    n=$(ls -1 "$DST/$c/$s" 2>/dev/null | grep -v '^\._' | wc -l | tr -d ' ')
    printf "   %-16s sev%s : %s class dirs\n" "$c" "$s" "$n"
    [ "${n:-0}" -lt 900 ] && ok=0
  done
done
echo "============================================================"
if [ "$ok" = 1 ]; then
  echo " READY. Start training with:"
  echo "   bash $KB_REPO_ROOT/docs/research/kbound/scripts/kbtrain.sh noise-fast"
else
  echo " WARNING: some severities look incomplete (<900 class dirs)."
  echo " Re-run this script (rsync resumes) before training."
fi
