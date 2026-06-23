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
set -uo pipefail
DATA_ROOT="${DATA_ROOT:-/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/data/imagenet-c}"
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
