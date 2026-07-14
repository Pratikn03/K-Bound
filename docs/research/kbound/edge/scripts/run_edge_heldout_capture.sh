#!/usr/bin/env bash
# Phase 2b: after source gate passes — calibration (S03-S06), seal, held-out (S07-S10), publish.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="python3"
CFG="$HERE/../configs/edge_real_phone_v1.yaml"
CAMERA="${EDGE_CAMERA:-1}"
PHONE="${EDGE_PHONE_ID:-phone_a}"

say(){ printf "\n== %s ==\n" "$*"; }

for sess in S03 S04 S05 S06; do
  say "Calibration capture $sess"
  $PY "$HERE/01_capture_real_session.py" --config "$CFG" --session "$sess" \
    --phone-id "$PHONE" --camera "$CAMERA"
done

say "Seal calibration (before any held-out recording)"
$PY "$HERE/02_validate_real_dataset.py" --config "$CFG" \
  --through calibration_conformal --seal-through calibration_conformal --strict

for sess in S07 S08 S09 S10; do
  say "Held-out / replication capture $sess"
  SESSION_PHONE="$PHONE"
  if [[ "$sess" == "S09" || "$sess" == "S10" ]]; then
    SESSION_PHONE="phone_b"
  fi
  $PY "$HERE/01_capture_real_session.py" --config "$CFG" --session "$sess" \
    --phone-id "$SESSION_PHONE" --camera "$CAMERA"
done

say "Full publication pipeline (anti-leakage audit + TeX export)"
cd "$ROOT"
bash docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh

say "DONE — check experiments/kbound/results/edge_real_phone_v1/camera_tables_values.tex"
