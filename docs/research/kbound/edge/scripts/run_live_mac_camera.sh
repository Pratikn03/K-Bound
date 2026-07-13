#!/usr/bin/env bash
# Launch K-Bound Edge live shadow demo with the best live camera feed.
# iPhone Continuity Camera is usually index 1; MacBook built-in is index 0.
# Override: CAMERA_INDEX=1 bash .../run_live_mac_camera.sh
# Grant camera access: System Settings → Privacy & Security → Camera → Terminal/Cursor.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
SCRIPT="docs/research/kbound/edge/scripts/07_shadow_live.py"
CFG="docs/research/kbound/edge/configs/edge_real_phone_v1.yaml"
SHADOW="docs/research/kbound/edge/configs/edge_shadow_v1.yaml"

pick_camera() {
  "$PY" - <<'PY'
import sys
sys.path.insert(0, "docs/research/kbound/edge/src")
from kbound_edge.capture import list_camera_probe
rows = list_camera_probe(1)
live = [(i, v) for i, v in rows if v is not None]
if not live:
    print("ERROR: no readable camera (index 0-1)", file=sys.stderr)
    sys.exit(1)
for idx, var in rows:
    if var is None:
        print(f"  index {idx}: unreadable", file=sys.stderr)
    else:
        tag = " (frozen/static)" if var < 0.5 else " (live — use this)"
        print(f"  index {idx}: motion={var:.3f}{tag}", file=sys.stderr)
best = max(live, key=lambda x: x[1])[0]
print(best)
PY
}

if [ "${CAMERA_INDEX:-auto}" = "auto" ]; then
  echo "Probing cameras (motion test; higher = live feed)..." >&2
  CAMERA="$(pick_camera)"
  echo "Selected camera index ${CAMERA}" >&2
else
  CAMERA="${CAMERA_INDEX}"
  echo "Using camera index ${CAMERA} (manual override)" >&2
fi

echo "Starting live shadow window (press 'q' in the OpenCV window to quit)..."
echo "Uses --demo calibrator until real S03–S06 calibration exists." >&2
exec "$PY" "$SCRIPT" \
  --config "$CFG" \
  --shadow-config "$SHADOW" \
  --camera "$CAMERA" \
  --view window \
  --demo \
  "$@"
