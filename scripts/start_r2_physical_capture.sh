#!/usr/bin/env bash
# Start physical camera R2 — interactive kickoff (from repo root).
#
#   bash scripts/start_r2_physical_capture.sh
#   bash scripts/start_r2_physical_capture.sh pilot
#   bash scripts/start_r2_physical_capture.sh preflight
#   bash scripts/start_r2_physical_capture.sh session S01
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
EDGE="${ROOT}/docs/research/kbound/edge"
SCRIPTS="${EDGE}/scripts"
CFG="${EDGE}/configs/edge_real_phone_v1.yaml"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "$PY" ]] || die "No .venv — run: python3 -m venv .venv && .venv/bin/pip install torch torchvision opencv-python scikit-learn joblib pyyaml"

cmd="${1:-menu}"

preflight() {
  "$PY" "${SCRIPTS}/preflight_r2.py" "${SKIP_CAMERA_PROBE:+--skip-camera}"
}

prepare_protocol() {
  echo "=== Preparing protocol lock + checklists ==="
  "$PY" "${SCRIPTS}/00_prepare_real_protocol.py" --config "$CFG"
}

pick_camera() {
  CAMERA_INDEX="${CAMERA_INDEX:-auto}"
  if [[ "$CAMERA_INDEX" == "auto" ]]; then
    bash "${SCRIPTS}/run_live_mac_camera.sh" --help >/dev/null 2>&1 || true
    CAMERA_INDEX="$("$PY" - <<'PY'
import sys
sys.path.insert(0, "docs/research/kbound/edge/src")
from kbound_edge.capture import list_camera_probe
rows = list_camera_probe(4)
live = [(i, v) for i, v in rows if v is not None and v > 0.5]
print(max(live, key=lambda x: x[1])[0] if live else 0)
PY
)"
  fi
  echo "$CAMERA_INDEX"
}

run_pilot() {
  local cam
  cam="$(pick_camera)"
  echo "=== Pilot capture (4 clips, camera index ${cam}) ==="
  echo "Grant camera access if prompted: System Settings → Privacy → Camera → Terminal"
  "$PY" "${SCRIPTS}/01_capture_real_session.py" \
    --config "$CFG" --pilot --phone-id phone_a --camera "$cam" --max-items 4
}

run_session() {
  local sess="${1:?session id e.g. S01}"
  local phone="${2:-}"
  if [[ -z "$phone" ]]; then
  case "$sess" in
    S09|S10) phone="phone_b" ;;
    *) phone="phone_a" ;;
  esac
  fi
  local cam
  cam="$(pick_camera)"
  echo "=== Capture ${sess} (phone ${phone}, camera ${cam}) ==="
  "$PY" "${SCRIPTS}/01_capture_real_session.py" \
    --config "$CFG" --session "$sess" --phone-id "$phone" --camera "$cam"
}

run_shadow() {
  echo "=== Live shadow dashboard (press q to quit) ==="
  bash "${SCRIPTS}/run_live_mac_camera.sh"
}

seal_dev() {
  echo "=== Seal dev splits through calibration_conformal (after S06) ==="
  "$PY" "${SCRIPTS}/02_validate_real_dataset.py" \
    --config "$CFG" --through calibration_conformal --seal-through calibration_conformal --strict
}

run_pipeline() {
  echo "=== Full publication pipeline ==="
  bash "${SCRIPTS}/run_edge_publication_pipeline.sh"
}

menu() {
  preflight
  echo ""
  cat <<'MENU'

=== R2 Physical Camera — what next? ===

  1) Preflight only:
       bash scripts/start_r2_physical_capture.sh preflight

  2) Prepare protocol (first time):
       bash scripts/start_r2_physical_capture.sh prepare

  3) Live shadow demo (see adapt/freeze/abstain while aiming camera):
       bash scripts/start_r2_physical_capture.sh shadow

  4) Pilot — 4 real clips (warm-up):
       bash scripts/start_r2_physical_capture.sh pilot

  5) Capture a session (interactive; ENTER per clip):
       bash scripts/start_r2_physical_capture.sh session S01
       # … S02 … S06 (dev), then seal, then S07–S10

  6) Seal dev splits (ONLY after S06, BEFORE S07):
       bash scripts/start_r2_physical_capture.sh seal

  7) Run full pipeline (after S01–S10):
       bash scripts/start_r2_physical_capture.sh pipeline

Full guide: docs/research/kbound/edge/PHYSICAL_STUDY_RUNBOOK.md
MENU
}

case "$cmd" in
  preflight) preflight ;;
  prepare) prepare_protocol ;;
  pilot) prepare_protocol; run_pilot ;;
  shadow) run_shadow ;;
  seal) seal_dev ;;
  pipeline) run_pipeline ;;
  session) run_session "${2:-}" "${3:-}" ;;
  menu|"") menu ;;
  *)
    echo "Usage: $0 {preflight|prepare|pilot|shadow|session S0x|seal|pipeline|menu}"
    exit 1
    ;;
esac
