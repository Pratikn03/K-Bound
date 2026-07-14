#!/usr/bin/env bash
# Phase 2 gate: record S01+S02 only, train f0, verify S02 balanced-acc >= 0.80.
# Stop here if the gate fails — do not record held-out sessions yet.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="python3"
CFG="$HERE/../configs/edge_real_phone_v1.yaml"
CAMERA="${EDGE_CAMERA:-1}"
PHONE="${EDGE_PHONE_ID:-phone_a}"

say(){ printf "\n== %s ==\n" "$*"; }

say "S01 source_train capture (120 clips)"
$PY "$HERE/01_capture_real_session.py" --config "$CFG" --session S01 \
  --phone-id "$PHONE" --camera "$CAMERA"

say "S02 source_val capture (40 clips) — this is the 0.80 gate"
$PY "$HERE/01_capture_real_session.py" --config "$CFG" --session S02 \
  --phone-id "$PHONE" --camera "$CAMERA"

say "Validate dev splits through S02"
$PY "$HERE/02_validate_real_dataset.py" --config "$CFG" \
  --through source_val --strict

say "Train source model f0 (no bypass)"
$PY "$HERE/03_train_source_model.py" --config "$CFG"

METRICS="$ROOT/experiments/kbound/results/edge_real_phone_v1/source_val_metrics.json"
if [ ! -f "$METRICS" ]; then
  # fallback: scan for latest metrics export
  METRICS="$(find "$ROOT/experiments/kbound/results/edge_real_phone_v1" -name '*val*metrics*.json' 2>/dev/null | head -1 || true)"
fi

say "Source gate check"
$PY - <<PY
import json, sys, os
paths = [
    "$METRICS",
    os.path.join("$ROOT", "experiments/kbound/results/edge_real_phone_v1/heldout_metrics.json"),
]
for p in paths:
    if not p or not os.path.isfile(p):
        continue
    d = json.load(open(p))
    bal = d.get("balanced_acc") or d.get("val_balanced_acc") or (d.get("metrics") or {}).get("balanced_acc")
    mf1 = d.get("macro_f1") or (d.get("metrics") or {}).get("macro_f1")
    if bal is not None:
        print(f"  {os.path.basename(p)}: balanced_acc={bal:.4f} macro_f1={mf1}")
        if float(bal) >= 0.80:
            print("PASS: source gate >= 0.80 — proceed to S03-S10")
            sys.exit(0)
        print("FAIL: balanced_acc < 0.80 — improve capture and re-shoot S01/S02")
        sys.exit(1)
print("WARN: could not find val metrics JSON; inspect training log manually")
sys.exit(2)
PY
