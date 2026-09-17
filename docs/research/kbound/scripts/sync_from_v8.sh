#!/usr/bin/env bash
set -e

SRC="/Users/pratik_n/Documents/AutoML_Flagship_V8"
DST="/Users/pratik_n/Documents/AutoML_Flagship_V8 2"

echo "=========================================================="
echo "SAFE CONSOLIDATION: Syncing missing files from V8 -> V8 2"
echo "=========================================================="

if [ ! -d "$SRC" ]; then
    echo "ERROR: Source directory $SRC not found."
    exit 1
fi

# 1. Copy the new preflight runbook
if [ -f "$SRC/docs/research/kbound/runbooks/run_task3_natural.sh" ]; then
    mkdir -p "$DST/docs/research/kbound/runbooks"
    cp -v "$SRC/docs/research/kbound/runbooks/run_task3_natural.sh" "$DST/docs/research/kbound/runbooks/"
    chmod +x "$DST/docs/research/kbound/runbooks/run_task3_natural.sh"
    echo "[OK] Synced run_task3_natural.sh"
fi

# 2. Copy official baseline repos in external/ if present in source and missing in dest
mkdir -p "$DST/external"
for pkg in poem_official aetta_official ttaline_official; do
    if [ -d "$SRC/external/$pkg" ] && [ ! -d "$DST/external/$pkg" ]; then
        echo "Copying external/$pkg..."
        cp -a "$SRC/external/$pkg" "$DST/external/"
        echo "[OK] Synced external/$pkg"
    fi
done

# 3. Sync any newly generated experiments results without overwriting existing ones
if [ -d "$SRC/experiments/kbound/results" ]; then
    echo "Syncing new experiment results (update-only, no overwrite of newer files)..."
    rsync -av --update --ignore-existing \
        --exclude="*.cache" --exclude="*.tmp" \
        "$SRC/experiments/kbound/results/" "$DST/experiments/kbound/results/"
    echo "[OK] Synced experiment results."
fi

echo "=========================================================="
echo "Sync completed successfully into: $DST"
echo "=========================================================="
