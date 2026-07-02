#!/usr/bin/env bash
# Full 5-seed showcase over all 9 datasets (many hours). Requires RxRx1 data.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"

SEEDS="${KB_SEEDS:-0 1 2 3 4}"
DEVICE="${KB_DEVICE:-mps}"
IC_IMG="${KB_IC_MAXIMG:-2000}"

bash "$HERE/prepare_rxrx1_data.sh" || {
  echo "WARN: RxRx1 missing — full panel will skip step 5"
}

echo ">> full panel  seeds=[$SEEDS]  device=$DEVICE  IC_MAXIMG=$IC_IMG"
cd "$ROOT"
KB_SEEDS="$SEEDS" KB_DEVICE="$DEVICE" KB_IC_MAXIMG="$IC_IMG" \
  caffeinate -is bash "$HERE/run_final_showcase.sh" --device "$DEVICE" --seeds "$SEEDS"
