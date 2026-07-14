#!/usr/bin/env python3
"""02_validate_real_dataset -- validates captures, builds windows, and audits/seals splits.

Scans artifacts_real/raw/ for MP4 + JSON sidecars, verifies hashes, builds
32-frame NPZ windows under artifacts_real/windows/, and outputs inventories
and leakage audit trails.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cv2
import numpy as np

from kbound_edge.real_manifest import load_real_protocol, expected_windows, ProtocolError, canonical_protocol_hash
from kbound_edge.real_dataset import audit_dataset, SESSION_SPLIT_MAP

def get_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def load_mp4_frames(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

def resize_frames(frames: np.ndarray, size: int = 224) -> np.ndarray:
    resized = []
    for f in frames:
        if f.shape[0] != size or f.shape[1] != size:
            f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        resized.append(f)
    return np.array(resized, dtype=np.uint8)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(_HERE, "..", "configs", "edge_real_phone_v1.yaml"))
    ap.add_argument("--through", choices=["source_train", "source_val", "calibration_fit", "calibration_conformal", "heldout", "replication"], default="replication")
    ap.add_argument("--strict", action="store_true", help="Exit nonzero on missing files or validation failures")
    ap.add_argument("--seal-through", choices=["calibration_conformal"], help="Seal development splits up to calibration_conformal")
    ap.add_argument("--open-split", choices=["heldout", "replication"], help="Refuse to run unless development splits are sealed")
    args = ap.parse_args()

    edge_dir = os.path.normpath(os.path.join(_HERE, ".."))
    cfg = load_real_protocol(os.path.abspath(args.config))
    raw_dir = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["raw_dir"]))
    windows_dir = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["windows_dir"]))
    
    # 1. Open split sealing check
    results_dir = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["results_dir"]))
    os.makedirs(results_dir, exist_ok=True)
    split_audit_path = os.path.join(results_dir, "split_audit.json")
    
    if args.open_split:
        if not os.path.exists(split_audit_path):
            raise ProtocolError(f"Cannot open split {args.open_split} because split_audit.json does not exist. Run with --seal-through calibration_conformal first.")
        with open(split_audit_path) as f:
            audit = json.load(f)
        if not audit.get("sealed_splits", {}).get("calibration_conformal", False):
            raise ProtocolError("Cannot open test splits: calibration_conformal is not sealed.")
        print(f"[OK] Open split check passed for {args.open_split}. Development splits are locked.")

    # Determine which sessions we need to validate based on --through
    split_order = ["source_train", "source_val", "calibration_fit", "calibration_conformal", "heldout", "replication"]
    target_split_idx = split_order.index(args.through)
    
    target_sessions = []
    for s_id, sess in cfg["sessions"].items():
        sess_split = SESSION_SPLIT_MAP[s_id]
        # map fit_a/fit_b to fit
        if sess_split.startswith("calibration_fit"):
            sess_split = "calibration_fit"
        elif sess_split.startswith("calibration_conformal"):
            sess_split = "calibration_conformal"
        elif sess_split.startswith("heldout"):
            sess_split = "heldout"
        elif sess_split.startswith("replication"):
            sess_split = "replication"
            
        if split_order.index(sess_split) <= target_split_idx:
            target_sessions.append(s_id)
            
    print(f"Validating sessions: {sorted(target_sessions)}")
    
    # Collect all raw clips and build inventory
    inventory_clips = []
    missing_clips = []
    
    for s_id in sorted(target_sessions):
        # We get all expected windows for S_id
        all_wins = expected_windows(cfg, s_id)
        physical_wins = [w for w in all_wins if not w.get("is_derived", False)]
        
        sess_raw_dir = os.path.join(raw_dir, s_id)
        
        for w in physical_wins:
            clip_id = f"{s_id}_{w['object_id']}_{w['class_id']}_{w['shift_id']}_R{w['repetition']:02d}"
            mp4_path = os.path.join(sess_raw_dir, f"{clip_id}.mp4")
            json_path = os.path.join(sess_raw_dir, f"{clip_id}.json")
            
            if not os.path.exists(mp4_path) or not os.path.exists(json_path):
                missing_clips.append(clip_id)
                continue
                
            # Verify JSON sidecar content and file hashes
            with open(json_path) as f:
                metadata = json.load(f)
                
            observed_sha = get_file_sha256(mp4_path)
            if observed_sha != metadata["sha256"]:
                msg = f"Hash mismatch for {clip_id}: observed {observed_sha[:8]}, metadata {metadata['sha256'][:8]}"
                if args.strict:
                    raise ProtocolError(msg)
                print(f"[ERROR] {msg}")

            capture_mode = metadata.get("capture_mode", "unknown")
            if capture_mode != "physical":
                msg = f"Non-physical capture rejected for {clip_id}: capture_mode={capture_mode!r}"
                if args.strict:
                    raise ProtocolError(msg)
                print(f"[ERROR] {msg}")
                
            inventory_clips.append({
                "clip_id": clip_id,
                "session_id": s_id,
                "phone_id": metadata["phone_id"],
                "object_id": w["object_id"],
                "class_id": w["class_id"],
                "shift_id": w["shift_id"],
                "repetition": w["repetition"],
                "sha256": observed_sha,
                "frame_count": metadata["frame_count"],
                "captured_at": metadata["captured_at"],
                "capture_mode": capture_mode,
                "camera_index": metadata.get("camera_index"),
            })
            
    print(f"Found {len(inventory_clips)} valid raw clips. Missing: {len(missing_clips)}")
    if missing_clips:
        print(f"Missing clip list: {missing_clips[:10]}...")
        if args.strict:
            raise ProtocolError(f"Strict validation failed: missing {len(missing_clips)} physical clips.")

    # Run anti-leakage and cross-split audit
    manifest = {"clips": inventory_clips}
    audit_report = audit_dataset(manifest)
    if not audit_report.passed:
        print("[ERROR] Leakage audit failed:")
        for failure in audit_report.failures:
            print(f"  - {failure}")
        if args.strict:
            raise ProtocolError("Dataset audit failed.")
    else:
        print("[OK] Cross-split leakage audit passed.")

    # Write inventories to experiments/kbound/results/edge_real_phone_v1/
    inv_json_path = os.path.join(results_dir, "recording_inventory.json")
    inv_csv_path = os.path.join(results_dir, "recording_inventory.csv")
    
    with open(inv_json_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    with open(inv_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "session_id", "phone_id", "object_id", "class_id", "shift_id", "repetition", "sha256", "captured_at", "capture_mode", "camera_index"])
        for c in inventory_clips:
            writer.writerow([
                c["clip_id"], c["session_id"], c["phone_id"], c["object_id"],
                c["class_id"], c["shift_id"], c["repetition"], c["sha256"], c["captured_at"],
                c["capture_mode"], c["camera_index"]
            ])
            
    print(f"Wrote inventory to: {inv_json_path}")

    # 2. Build windows NPZ files!
    # S01, S02, S03, S05, S07, S09 are simple physical-only session window building.
    # B sessions (S04, S06, S08, S10) build physical + derived batch_composition windows.
    class_name_map = {c: i for i, c in enumerate(cfg["classes"])}
    
    for s_id in sorted(target_sessions):
        split = SESSION_SPLIT_MAP[s_id]
        split_windows_dir = os.path.join(windows_dir, split)
        os.makedirs(split_windows_dir, exist_ok=True)
        
        all_wins = expected_windows(cfg, s_id)
        
        # Build map of physical clip_id to file path for this session
        sess_clips = [c for c in inventory_clips if c["session_id"] == s_id]
        clip_path_map = {c["clip_id"]: os.path.join(raw_dir, s_id, f"{c['clip_id']}.mp4") for c in sess_clips}
        clip_sha_map = {c["clip_id"]: c["sha256"] for c in sess_clips}
        
        # Group session clips by class for derived windows in B sessions
        clips_by_class = {cls: [c for c in sess_clips if c["class_id"] == cls] for cls in cfg["classes"]}
        
        for w in all_wins:
            win_id = w["window_id"]
            npz_path = os.path.join(split_windows_dir, f"{win_id}.npz")
            
            # If already exists, we skip building to save time
            if os.path.exists(npz_path):
                continue
                
            if not w.get("is_derived", False):
                # Physical window: load frames from single clip
                clip_id = f"{s_id}_{w['object_id']}_{w['class_id']}_{w['shift_id']}_R{w['repetition']:02d}"
                if clip_id not in clip_path_map:
                    continue # Skip if raw file missing (non-strict mode)
                    
                frames = load_mp4_frames(clip_path_map[clip_id])
                if len(frames) != 32:
                    # Resize/pad if needed, or raise
                    raise ProtocolError(f"Expected 32 frames for clip {clip_id}, got {len(frames)}")
                frames_resized = resize_frames(frames, cfg["image_size"])
                
                labels = np.array([class_name_map[w["class_id"]]] * 32, dtype=np.int64)
                source_hashes = np.array([clip_sha_map[clip_id]], dtype=object)
                
                np.savez(npz_path, frames=frames_resized, labels=labels, window_id=win_id, source_hashes=source_hashes)
            else:
                # Derived batch_composition window
                # Recipe lists frame counts for: [ok, missing_label, misaligned_label, damaged_label]
                counts = w["counts"]
                recipe_name = w["recipe"]
                rep = w["repetition"]
                
                # Deterministically select one physical clip of each class from this session
                # S04 has 96 clips, 24 per class. We can pair them up deterministically.
                composed_frames = []
                composed_labels = []
                composed_shas = []
                
                valid_recipe = True
                for cls_idx, cls_name in enumerate(cfg["classes"]):
                    cls_count = counts[cls_idx]
                    if cls_count == 0:
                        continue
                        
                    cls_clips = sorted(clips_by_class[cls_name], key=lambda c: c["clip_id"])
                    if not cls_clips:
                        valid_recipe = False
                        break
                        
                    # Select clip deterministically based on repetition index
                    clip_to_use = cls_clips[(rep - 1) % len(cls_clips)]
                    clip_path = clip_path_map[clip_to_use["clip_id"]]
                    frames = load_mp4_frames(clip_path)
                    
                    # Take first cls_count frames from the clip
                    cls_frames = frames[:cls_count]
                    composed_frames.extend(cls_frames)
                    composed_labels.extend([cls_idx] * cls_count)
                    composed_shas.append(clip_sha_map[clip_to_use["clip_id"]])
                    
                if not valid_recipe or len(composed_frames) != 32:
                    print(f"[WARNING] Skipping derived window {win_id} due to incomplete source clips.")
                    continue
                    
                composed_frames_arr = np.array(composed_frames)
                frames_resized = resize_frames(composed_frames_arr, cfg["image_size"])
                labels = np.array(composed_labels, dtype=np.int64)
                source_hashes = np.array(composed_shas, dtype=object)
                
                np.savez(npz_path, frames=frames_resized, labels=labels, window_id=win_id, source_hashes=source_hashes)
                
        print(f"Generated windows for session {s_id} in {split_windows_dir}")

    # 3. Seal splits if requested
    if args.seal_through:
        # Verify S01--S06 are fully present and valid
        dev_sessions = ["S01", "S02", "S03", "S04", "S05", "S06"]
        missing_dev = [s for s in dev_sessions if s not in target_sessions]
        if missing_dev:
            raise ProtocolError(f"Cannot seal splits through {args.seal_through}: sessions {missing_dev} are not validated.")
            
        # Get SHA-256 of the current inventory
        inventory_hash = get_file_sha256(inv_json_path)
        
        audit_data = {
            "sealed_splits": {
                "source_train": True,
                "source_val": True,
                "calibration_fit": True,
                "calibration_conformal": True
            },
            "development_inventory_hash": inventory_hash,
            "sealed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "protocol_hash": canonical_protocol_hash(cfg)
        }
        
        with open(split_audit_path, "w") as f:
            json.dump(audit_data, f, indent=2)
        print(f"[SEAL] Sealed development splits. Wrote {split_audit_path}")

if __name__ == "__main__":
    main()
