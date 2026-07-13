#!/usr/bin/env python3
"""01_capture_real_session -- captures physical video windows from a camera.

Loops through the session checklist, prompts the user to place the target
object, records 40 frames, keeps frames 5--36 (32 frames), writes MP4 and
reproducibility JSON sidecar, and computes SHA-256 hashes for both.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cv2
import numpy as np

from kbound_edge.real_manifest import load_real_protocol, ProtocolError
from kbound_edge.recording import build_session_checklist, make_clip_record

def get_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(_HERE, "..", "configs", "edge_real_phone_v1.yaml"))
    ap.add_argument("--session", required=False, help="Session ID (S01--S10)")
    ap.add_argument("--phone-id", required=False, help="Phone ID (phone_a or phone_b)")
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    ap.add_argument("--pilot", action="store_true", help="Record pilot clips")
    ap.add_argument("--max-items", type=int, default=None, help="Stop after capturing N items")
    ap.add_argument("--mock", action="store_true", help="Mock camera input for testing")
    args = ap.parse_args()

    edge_dir = os.path.normpath(os.path.join(_HERE, ".."))
    cfg = load_real_protocol(os.path.abspath(args.config))
    art_dir = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["artifacts"]))

    # 1. Pilot mode handling
    if args.pilot:
        print("=== PILOT MODE ===")
        pilot_dir = os.path.join(art_dir, "pilot")
        os.makedirs(pilot_dir, exist_ok=True)
        max_items = args.max_items or 4

        # Create a simple mock/test checklist
        checklist = []
        for i in range(max_items):
            checklist.append({
                "clip_id": f"PILOT_item_{i+1:02d}",
                "session_id": "PILOT",
                "phone_id": args.phone_id or "phone_a",
                "object_id": "P01",
                "class_id": "ok",
                "shift_id": "stable",
                "repetition": i + 1,
            })
    else:
        # 2. Strict validation of session and phone
        if not args.session:
            ap.error("--session is required unless in --pilot mode")
        if not args.phone_id:
            ap.error("--phone-id is required unless in --pilot mode")

        session_id = args.session
        phone_id = args.phone_id

        sessions_cfg = cfg["sessions"]
        if session_id not in sessions_cfg:
            raise ProtocolError(f"Unknown session {session_id}")

        expected_phone = sessions_cfg[session_id]["phone_id"]
        if phone_id != expected_phone:
            raise ProtocolError(f"Incorrect phone-id: expected {expected_phone} for session {session_id}, got {phone_id}")

        # Check if the session is already sealed
        split_audit_path = os.path.normpath(
            os.path.join(edge_dir, cfg["paths"]["results_dir"], "split_audit.json")
        )
        if os.path.exists(split_audit_path):
            with open(split_audit_path) as f:
                audit = json.load(f)
            if audit.get("sealed_splits", {}).get(sessions_cfg[session_id]["split"], False):
                raise ProtocolError(f"Session {session_id} is part of a sealed split and is closed for recording.")

        checklist = build_session_checklist(cfg, session_id)
        if args.max_items:
            checklist = checklist[:args.max_items]

    # Resolve output directory
    raw_dir = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["raw_dir"]))

    # 3. Setup camera or mock camera
    cap = None
    if not args.mock:
        cap = cv2.VideoCapture(args.camera)
        # Warmup camera
        for _ in range(5):
            cap.read()

    print(f"Starting session capture. Total items to process: {len(checklist)}")

    first_res = None

    for i, item in enumerate(checklist):
        clip_id = item["clip_id"]
        sess_id = item.get("session_id", args.session)
        p_id = item.get("phone_id", args.phone_id)
        obj_id = item["object_id"]
        cls_id = item["class_id"]
        shift_id = item["shift_id"]
        rep = item["repetition"]

        # Output paths
        if args.pilot:
            out_mp4 = os.path.join(pilot_dir, f"{clip_id}.mp4")
            out_json = os.path.join(pilot_dir, f"{clip_id}.json")
        else:
            session_raw_dir = os.path.join(raw_dir, sess_id)
            os.makedirs(session_raw_dir, exist_ok=True)
            out_mp4 = os.path.join(session_raw_dir, f"{clip_id}.mp4")
            out_json = os.path.join(session_raw_dir, f"{clip_id}.json")

        # Resume without overwriting
        if os.path.exists(out_mp4) and os.path.exists(out_json):
            print(f"[{i+1}/{len(checklist)}] Skipping already captured clip: {clip_id}")
            continue

        print(f"\n[{i+1}/{len(checklist)}] NEXT CAPTURE:")
        print(f"  Clip ID:   {clip_id}")
        print(f"  Object:    {obj_id}")
        print(f"  Class:     {cls_id}")
        print(f"  Shift:     {shift_id}")
        print(f"  Repetition:{rep}")

        # Interactive prompt
        if not args.mock:
            resp = input("Place target and press ENTER to record (or Q to quit): ").strip().lower()
            if resp == 'q':
                print("Capture stopped by user.")
                break
        else:
            print("  Mocking camera capture...")
            time.sleep(0.1)

        # Capture 40 frames
        frames = []
        if args.mock:
            # Generate 40 fake frames of 224x224 RGB
            for _ in range(40):
                frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                frames.append(frame)
        else:
            for f_idx in range(40):
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Failed to read frame {f_idx} from camera")
                frames.append(frame)
                time.sleep(0.01) # Small delay to simulate ~30fps

        if len(frames) < 40:
            raise ProtocolError(f"Captured fewer than 40 frames: got {len(frames)}")

        # Keep frames 5--36 (which is index 5 to 36 inclusive, i.e., 32 frames)
        selected_frames = frames[5:37]
        assert len(selected_frames) == 32

        # Check resolution consistency
        h, w, c = selected_frames[0].shape
        if first_res is None:
            first_res = (w, h)
        else:
            if (w, h) != first_res:
                raise ProtocolError(f"Inconsistent resolution: expected {first_res}, got {(w, h)}")

        # Write MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_mp4, fourcc, 30.0, (w, h))
        for f in selected_frames:
            writer.write(f)
        writer.release()

        # Hash MP4
        mp4_sha = get_file_sha256(out_mp4)

        # Create metadata sidecar
        captured_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        clip_record = make_clip_record(
            clip_id=clip_id,
            session_id=sess_id,
            phone_id=p_id,
            object_id=obj_id,
            class_id=cls_id,
            shift_id=shift_id,
            repetition=rep,
            captured_at=captured_at,
            sha256=mp4_sha,
            frame_count=32,
            capture_mode="mock" if args.mock else "physical",
            camera_index=None if args.mock else args.camera,
        )

        # Write JSON sidecar
        with open(out_json, "w") as f:
            json.dump(clip_record, f, indent=2)

        # Hash JSON sidecar too
        json_sha = get_file_sha256(out_json)

        print(f"  [OK] Saved MP4 ({mp4_sha[:8]}) and JSON ({json_sha[:8]})")

    if cap is not None:
        cap.release()
    print("Capture session finished/paused successfully.")

if __name__ == "__main__":
    main()
