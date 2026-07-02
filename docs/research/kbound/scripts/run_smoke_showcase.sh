#!/usr/bin/env bash
# Multiseed smoke: theory + routing + ~1% GPU panel + analysis report.
# Does NOT overwrite experiments/kbound/results/ locked headlines.
#
#   bash docs/research/kbound/scripts/run_smoke_showcase.sh
#   KB_SMOKE_SEEDS="0 1 2" bash .../run_smoke_showcase.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="python3"

DEVICE="${KB_DEVICE:-mps}"
SEEDS="${KB_SMOKE_SEEDS:-0 1}"

echo ">> smoke-showcase  device=$DEVICE  KB_SMOKE_SEEDS=[$SEEDS]"
KB_SMOKE_SEEDS="$SEEDS" KB_DEVICE="$DEVICE" bash "$HERE/kbtrain.sh" smoke-all-v2

SMOKE_ROOT="$(ls -td "$ROOT"/experiments/kbound/results/smoke_ms_* 2>/dev/null | head -1)"
if [ -z "${SMOKE_ROOT:-}" ]; then
  echo "ERROR: no smoke_ms_* output dir"; exit 1
fi
echo ">> smoke root: $SMOKE_ROOT"

# Mini locked analysis if both seeds produced per_condition files
NA_SEEDS=0
for s in $SEEDS; do
  [ -f "$SMOKE_ROOT/stress_grid_multiseed_v1/seed${s}/per_condition_cifar10c_tent_seed${s}.json" ] && NA_SEEDS=$((NA_SEEDS+1))
done
if [ "$NA_SEEDS" -ge 2 ]; then
  echo ">> mini LOCKED_ANALYSIS on smoke CIFAR ($NA_SEEDS seeds)"
  KBOUND_STRESS_GRID_ROOT="$SMOKE_ROOT/stress_grid_multiseed_v1" \
  KBOUND_STRESS_SEEDS="$SEEDS" \
    "$PY" "$SMOKE_ROOT/stress_grid_multiseed_v1/_locked_analysis_script.py" \
    && echo ">> smoke locked analysis OK" || echo "WARN: smoke locked analysis failed (need all 3 adapters x seeds)"
fi

"$PY" "$HERE/smoke_pipeline_report.py" --smoke-root "$SMOKE_ROOT" --seeds-expected "$(echo $SEEDS | wc -w | tr -d ' ')"
RC=$?
echo ">> report exit=$RC (0=all 9 datasets in manifest)"
exit "$RC"
