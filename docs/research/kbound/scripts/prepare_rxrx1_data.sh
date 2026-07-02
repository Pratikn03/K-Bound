#!/usr/bin/env bash
# Prepare WILDS RxRx1 for the full 9-dataset panel.
# Target: ~/kbound_rxrx1_data/rxrx1_v1.0  (~33 GB download)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="python3"
DEST="${RXRX1_DATA_ROOT:-$HOME/kbound_rxrx1_data}"
DATA="$DEST/rxrx1_v1.0"

if [ -d "$DATA" ]; then
  echo "OK: RxRx1 already at $DATA"
  exit 0
fi

echo "RxRx1 not found at $DATA"
echo ""
echo "Option A — WILDS official download (recommended):"
echo "  mkdir -p $DEST && cd $DEST"
echo "  $PY -m pip install wilds"
echo "  $PY - <<'PY'"
echo "from wilds import get_dataset"
echo "get_dataset(dataset='rxrx1', download=True, root_dir='$DEST')"
echo "PY"
echo ""
echo "Option B — if you already have WILDS data elsewhere:"
echo "  export RXRX1_DATA_ROOT=/path/to/parent"
echo "  ln -s /path/to/rxrx1_v1.0 $DATA"
echo ""
echo "Then verify:"
echo "  ls $DATA/images | head"
echo "  bash $HERE/kbtrain.sh smoke-all-v2   # should not SKIP RxRx1"

exit 1
