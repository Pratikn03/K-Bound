#!/usr/bin/env bash
# verify_imagenetc_tars.sh - wait for the ImageNet-C tar download to finish, then
# md5-verify every tar vs Zenodo 2235448, structure-check, and clean exFAT ._ junk.
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
D="$KB_REPO_ROOT"/experiments/kbound/data/imagenet-c
REF="$D/_zenodo_md5sums.txt"
OUT="$D/_VERIFICATION.txt"
: > "$OUT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }

log "waiting for extra.tar to finish + be a valid tar ..."
for i in $(seq 1 160); do
  if [ -f "$D/extra.tar" ] && tar tf "$D/extra.tar" >/dev/null 2>&1; then break; fi
  sleep 30
done

log "=== AppleDouble (._*) + .DS_Store cleanup ==="
n=$(find "$D" -name '._*' 2>/dev/null | wc -l | tr -d ' ')
find "$D" -name '._*' -delete 2>/dev/null; find "$D" -name '.DS_Store' -delete 2>/dev/null
log "removed $n AppleDouble files"

log "=== md5 verify vs Zenodo (blur/weather/digital/extra) ==="
cd "$D" || exit 1
allok=1
for c in blur weather digital extra; do
  want=$(awk -v k="${c}.tar" '$2==k{print $1}' "$REF")
  got=$(md5 -q "${c}.tar" 2>/dev/null)
  if [ "$want" = "$got" ] && [ -n "$got" ]; then log "OK   ${c}.tar  md5=$got"; else log "FAIL ${c}.tar  want=$want got=$got"; allok=0; fi
done

log "=== tar structure sanity (first member of each) ==="
for c in blur weather digital extra; do
  first=$(tar tf "${c}.tar" 2>/dev/null | head -1)
  log "${c}.tar first-member: $first"
done

log "=== noise extracted dirs (severities 4,5) ==="
for c in gaussian_noise shot_noise impulse_noise; do
  for s in 4 5; do
    k=$(ls "$D/$c/$s" 2>/dev/null | grep -v '^\._' | wc -l | tr -d ' ')
    log "$c/$s class-dirs=$k"
  done
done

log "ALL_MD5_OK=$allok"
log "VERIFICATION_DONE"
