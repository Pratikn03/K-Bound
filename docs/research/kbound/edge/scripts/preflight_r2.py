#!/usr/bin/env python3
"""R2 preflight — physical camera study readiness (edge_real_phone_v1).

Checks: venv deps, camera probe, protocol lock, session capture progress,
mock-vs-real clip detection. Does not capture data.

Usage:
  .venv/bin/python docs/research/kbound/edge/scripts/preflight_r2.py
  .venv/bin/python docs/research/kbound/edge/scripts/preflight_r2.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EDGE = _HERE.parent
_SRC = _EDGE / "src"
_REPO = _EDGE.parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml

SESSION_ORDER = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"]


def _load_cfg() -> dict:
    cfg_path = _EDGE / "configs" / "edge_real_phone_v1.yaml"
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _clip_looks_mock(mp4: Path) -> bool | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(mp4))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    mean, std = float(frame.mean()), float(frame.std())
    # Mock generator: uniform random 0-255 -> mean~127, std~74; pipeline uses std>50 & mean 110-140
    return std > 50 and 110 < mean < 140


def _session_progress(raw_dir: Path, session: str, checklist_csv: Path) -> dict:
    sess_dir = raw_dir / session
    expected = 0
    if checklist_csv.is_file():
        import csv
        with checklist_csv.open(encoding="utf-8") as fh:
            expected = sum(1 for _ in csv.DictReader(fh))
    captured = list(sess_dir.glob("*.mp4")) if sess_dir.is_dir() else []
    mock_n = sum(1 for p in captured if _clip_looks_mock(p) is True)
    real_n = sum(1 for p in captured if _clip_looks_mock(p) is False)
    return {
        "session": session,
        "expected_clips": expected,
        "captured_mp4": len(captured),
        "likely_mock": mock_n,
        "likely_real": real_n,
        "complete": len(captured) >= expected and expected > 0,
    }


def run_preflight(*, skip_camera: bool = False) -> dict:
    cfg = _load_cfg()
    art = _EDGE / cfg["paths"]["artifacts"]
    raw_dir = _EDGE / cfg["paths"]["raw_dir"]
    checklists = art / "checklists"

    deps: dict[str, str] = {}
    for mod in ("cv2", "torch", "numpy", "sklearn", "joblib", "yaml"):
        try:
            m = __import__(mod if mod != "yaml" else "yaml")
            deps[mod] = getattr(m, "__version__", "ok")
        except ImportError:
            deps[mod] = "MISSING"

    camera_probe: list[tuple[int, float | None]] = []
    camera_ok = False
    if not skip_camera and deps.get("cv2") != "MISSING":
        from kbound_edge.capture import list_camera_probe

        camera_probe = list_camera_probe(2)
        camera_ok = any(v is not None and v > 0.5 for _, v in camera_probe)
    elif skip_camera:
        camera_ok = False

    lock_sha = (art / "protocol_lock.sha256").read_text(encoding="utf-8").strip() if (
        art / "protocol_lock.sha256"
    ).is_file() else "MISSING"

    sessions = [_session_progress(raw_dir, s, checklists / f"{s}_checklist.csv") for s in SESSION_ORDER]
    dev_done = all(s["complete"] for s in sessions[:6])
    held_done = all(s["complete"] for s in sessions[6:8])
    repl_done = all(s["complete"] for s in sessions[8:])

    split_audit = _REPO / "experiments/kbound/results/edge_real_phone_v1/split_audit.json"
    sealed = False
    if split_audit.is_file():
        audit = json.loads(split_audit.read_text(encoding="utf-8"))
        sealed = bool(audit.get("sealed_splits", {}).get("calibration_conformal"))

    next_session = None
    for s in sessions:
        if not s["complete"]:
            next_session = s["session"]
            break

    ready_for_pipeline = dev_done and (sealed or not held_done)  # pipeline needs dev at minimum

    return {
        "protocol": cfg.get("protocol", "edge_real_phone_v1"),
        "protocol_lock_sha256": lock_sha,
        "dependencies": deps,
        "deps_ok": deps.get("cv2") != "MISSING" and deps.get("torch") != "MISSING",
        "camera_probe": [{"index": i, "motion": v} for i, v in camera_probe],
        "camera_live_ok": camera_ok,
        "raw_dir": str(raw_dir.relative_to(_REPO)),
        "sessions": sessions,
        "dev_sessions_complete": dev_done,
        "heldout_sessions_complete": held_done,
        "replication_sessions_complete": repl_done,
        "calibration_conformal_sealed": sealed,
        "next_session_to_capture": next_session,
        "ready_for_publication_pipeline": ready_for_pipeline and held_done and repl_done,
        "blockers": _blockers(deps, camera_ok, sessions, sealed),
    }


def _blockers(deps, camera_ok, sessions, sealed) -> list[str]:
    out: list[str] = []
    if deps.get("cv2") == "MISSING" or deps.get("torch") == "MISSING":
        out.append("Install torch + opencv in .venv: pip install torch torchvision opencv-python")
    if not camera_ok:
        out.append("No live camera detected — grant Camera access to Terminal/Cursor; try run_live_mac_camera.sh")
    if any(s["likely_mock"] > 0 for s in sessions):
        out.append("Mock clips detected in raw/ — archive and re-capture without --mock")
    if not sealed and any(s["session"] in ("S07", "S08") and s["captured_mp4"] > 0 for s in sessions):
        out.append("Held-out captures exist before seal — run seal after S06 only")
    if not any(s["captured_mp4"] > 0 for s in sessions):
        out.append("No real captures yet — start with pilot or S01")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--skip-camera", action="store_true", help="Skip slow OpenCV camera probe")
    args = ap.parse_args()
    report = run_preflight(skip_camera=args.skip_camera)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=== R2 Physical Camera Preflight ===")
    print(f"Protocol lock: {report['protocol_lock_sha256'][:16]}…")
    print(f"Deps OK: {report['deps_ok']}  |  Live camera: {report['camera_live_ok']}")
    print()
    print("Session progress:")
    for s in report["sessions"]:
        flag = "✓" if s["complete"] else " "
        mock = f" mock={s['likely_mock']}" if s["likely_mock"] else ""
        print(f"  [{flag}] {s['session']}: {s['captured_mp4']}/{s['expected_clips']} clips{mock}")
    print()
    if report["next_session_to_capture"]:
        print(f"Next capture: {report['next_session_to_capture']}")
    if report["blockers"]:
        print("\nBlockers:")
        for b in report["blockers"]:
            print(f"  - {b}")
    else:
        print("\nNo blockers — proceed with capture or pipeline.")
    if report["ready_for_publication_pipeline"]:
        print("\nReady for: bash docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
