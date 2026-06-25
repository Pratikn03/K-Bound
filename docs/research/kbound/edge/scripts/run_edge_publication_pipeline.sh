#!/usr/bin/env bash
# End-to-end physical Edge study (no --bypass-gate).
# Prerequisite: real MP4+JSON captures under artifacts_real/raw/ (S01–S10).
# See ../PHYSICAL_STUDY_RUNBOOK.md for capture commands.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
EDGE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
SCRIPTS="${EDGE}/scripts"
CFG="${EDGE}/configs/edge_real_phone_v1.yaml"
CAL="${EDGE}/configs/edge_calibration_v1.yaml"
EPOCHS="${EPOCHS:-20}"

cd "${SCRIPTS}"

echo "=== [0] Data sanity (warn if clips look like mock random noise) ==="
"${PY}" - <<'PY'
import glob, sys, cv2, numpy as np
from pathlib import Path
raw = Path(__file__).resolve().parent.parent / "artifacts_real" / "raw"
# script runs from SCRIPTS; fix path
raw = Path("..") / "artifacts_real" / "raw"
mp4s = sorted(raw.glob("S01/*.mp4"))[:3]
if not mp4s:
    print("WARN: no S01 clips found — run capture first (see PHYSICAL_STUDY_RUNBOOK.md)")
    sys.exit(0)
for p in mp4s:
    cap = cv2.VideoCapture(str(p))
    ok, f = cap.read()
    cap.release()
    if ok and f is not None and f.std() > 50 and 110 < f.mean() < 140:
        print(f"WARN: {p.name} looks like mock random noise (mean~{f.mean():.0f} std~{f.std():.0f}).")
        print("      Re-capture with: 01_capture_real_session.py (no --mock) for publication.")
        break
else:
    print("OK: sample S01 frames do not look like pure mock noise.")
PY

echo "=== [1] Validate dataset + build windows (through replication) ==="
"${PY}" 02_validate_real_dataset.py --config "${CFG}" --through replication --strict

echo "=== [2] Train source model f0 on S01/S02 (gate >= 0.80 bal-acc & macro-F1) ==="
"${PY}" 03_train_source_model.py --config "${CFG}" --epochs "${EPOCHS}"

echo "=== [3] Calibration pairs S03–S06 ==="
"${PY}" 04_generate_calibration_pairs.py --config "${CFG}"

echo "=== [4] Fit KGA benefit estimator + conformal eps ==="
"${PY}" 05_fit_kga_edge.py --config "${CFG}" --calib-config "${CAL}"

echo "=== [5] Held-out replay S07/S08 ==="
"${PY}" 06_replay_heldout.py --config "${CFG}" --calib-config "${CAL}"

echo "=== [6] Replication replay S09/S10 ==="
"${PY}" 07_replay_replication.py --config "${CFG}" --calib-config "${CAL}"

echo "=== [7] Runtime profile ==="
"${PY}" 12_profile_real_run.py --config "${CFG}"

echo "=== [8] Ablations ==="
"${PY}" 09_run_real_ablations.py --config "${CFG}"

echo "=== [9] Anti-leakage audit (strict) ==="
"${PY}" 08_audit_real_run.py --config "${CFG}" --strict

echo "=== [10] Export LaTeX camera tables ==="
"${PY}" 11_export_camera_tables.py --config "${CFG}"

echo "=== [11] Report + dashboard snapshot ==="
"${PY}" 10_make_real_report.py --config "${CFG}"
bash "${ROOT}/docs/research/kbound/scripts/build_dashboard.sh"

echo ""
echo "=== DONE ==="
echo "Check: docs/experiments/kbound/results/edge_real_phone_v1/model_card.json"
echo "       docs/experiments/kbound/results/edge_real_phone_v1/heldout_metrics.json"
echo "       docs/experiments/kbound/results/edge_real_phone_v1/camera_tables_values.tex"
echo "Dashboard study_status flips to 'verified' when source gate passes AND held-out bal-acc > 30% AND abstain < 95%."
echo "Recompile short paper: cd docs/research/kbound && pdflatex kbound_short.tex"
