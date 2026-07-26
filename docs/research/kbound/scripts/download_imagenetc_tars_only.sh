#!/usr/bin/env bash
# download_imagenetc_tars_only.sh — TAR-STREAMING ImageNet-C fetch (NO extraction).
#
# Keeps the .tar archives on disk for DIRECT tarfile streaming, deliberately
# avoiding extraction: T9 is exFAT with a 128 KB allocation block, and the ~4.75M
# tiny ImageNet-C JPEGs suffer ~10-15x cluster slack when extracted (measured:
# one noise corruption = 62 GB extracted). Streaming from the tar = ~62 GB total.
#
# Source: Zenodo record 2235448 (Hendrycks & Dietterich, ICLR 2019). Resumable.
#
# Usage:
#   bash download_imagenetc_tars_only.sh                  # weather digital extra (~36 GB)
#   TARS="noise" bash download_imagenetc_tars_only.sh     # also (re)fetch another group
# NOTE: do NOT name the var GROUPS — that is a reserved bash array (gid list).
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
DATA_ROOT="${DATA_ROOT:-$KB_REPO_ROOT/experiments/kbound/data/imagenet-c}"
TARS="${TARS:-weather digital extra}"
BASE="https://zenodo.org/records/2235448/files"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
LOG="${LOG:-$DATA_ROOT/_tarstream_download.log}"
mkdir -p "$DATA_ROOT"; cd "$DATA_ROOT" || { echo "cannot cd $DATA_ROOT"; exit 1; }
free_gb(){ df -g "$DATA_ROOT" | awk 'NR==2{print $4}'; }
echo "[$(date '+%F %T')] START tars=[$TARS] free=$(free_gb)GB -> $DATA_ROOT" | tee -a "$LOG"
for c in $TARS; do
  if [ -f "${c}.tar" ] && tar tf "${c}.tar" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] [$c] present + valid tar, skip" | tee -a "$LOG"; continue
  fi
  f=$(free_gb)
  if [ "$f" -lt "$MIN_FREE_GB" ]; then
    echo "[$(date '+%F %T')] ABORT: only ${f}GB free (< ${MIN_FREE_GB}GB)" | tee -a "$LOG"; exit 1
  fi
  echo "[$(date '+%F %T')] [$c] downloading ${c}.tar (resumable curl -C -) ..." | tee -a "$LOG"
  if curl -fL -C - -o "${c}.tar" "${BASE}/${c}.tar?download=1" >>"$LOG" 2>&1; then
    sz=$(stat -f%z "${c}.tar" 2>/dev/null)
    md5v=$(md5 -q "${c}.tar" 2>/dev/null)
    echo "[$(date '+%F %T')] [$c] DONE bytes=$sz md5=$md5v free=$(free_gb)GB" | tee -a "$LOG"
  else
    echo "[$(date '+%F %T')] [$c] FAILED (re-run to resume)" | tee -a "$LOG"; exit 1
  fi
done
echo "[$(date '+%F %T')] ALL DONE free=$(free_gb)GB" | tee -a "$LOG"
