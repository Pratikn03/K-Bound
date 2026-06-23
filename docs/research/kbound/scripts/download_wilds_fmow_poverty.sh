#!/usr/bin/env bash
# Download WILDS FMoW + PovertyMap to T9 for Protocol L (third natural-shift attempt).
#
#   bash docs/research/kbound/scripts/download_wilds_fmow_poverty.sh
#
# Needs ~70 GB free (FMoW ~54 GB compressed + Poverty ~13 GB). Run AFTER freeing T9 space.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/experiments/kbound/data/wilds}"
VENV="${KB_VENV:-$HOME/.venv_wilds}"
PY="${PYTHON:-$VENV/bin/python}"

mkdir -p "$DATA_ROOT"
echo "=============================================================="
echo " WILDS FMoW + PovertyMap downloader"
echo "   data root : $DATA_ROOT"
df -h "$DATA_ROOT" | tail -1
echo "=============================================================="

if [[ ! -x "$PY" ]]; then
  echo "ERROR: need $VENV with wilds installed."
  exit 1
fi

download_one() {
  local name="$1"
  echo ""
  echo "==> Downloading WILDS ${name} (this may take hours) ..."
  "$PY" - "$name" "$DATA_ROOT" <<'PY'
import sys
from wilds import get_dataset
name, root = sys.argv[1], sys.argv[2]
get_dataset(dataset=name, download=True, root_dir=root)
print(f"    {name} ready under {root}")
PY
}

if [[ "${SKIP_FMOW:-0}" != 1 ]]; then
  download_one fmow
fi
if [[ "${SKIP_POVERTY:-0}" != 1 ]]; then
  download_one poverty
fi

echo ""
echo "Done. Verify with:"
echo "  $PY experiments/kbound/wilds/run_geoshift_kbound.py --dataset fmow --dry-run"
echo "  $PY experiments/kbound/wilds/run_geoshift_kbound.py --dataset poverty --dry-run"
