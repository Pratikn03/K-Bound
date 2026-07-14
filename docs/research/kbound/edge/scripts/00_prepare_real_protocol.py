#!/usr/bin/env python3
"""00_prepare_real_protocol -- prepares the protocol locks and checklists.

Loads configs/edge_real_phone_v1.yaml, validates it, serializes it to the lock,
computes the canonical hash, and writes deterministic randomized checklists
for recording each session as CSV.
"""

import argparse
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from kbound_edge.real_manifest import load_real_protocol, canonical_protocol_hash
from kbound_edge.recording import build_session_checklist

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(_HERE, "..", "configs", "edge_real_phone_v1.yaml"))
    args = ap.parse_args()

    # Resolve paths relative to edge directory (parent of scripts/)
    edge_dir = os.path.normpath(os.path.join(_HERE, ".."))
    config_path = os.path.abspath(args.config)
    
    print(f"Loading protocol config: {config_path}")
    cfg = load_real_protocol(config_path)
    
    # Resolve artifacts output directories
    art_dir_rel = cfg["paths"]["artifacts"]
    art_dir = os.path.normpath(os.path.join(edge_dir, art_dir_rel))
    os.makedirs(art_dir, exist_ok=True)
    
    # Calculate canonical hash
    chash = canonical_protocol_hash(cfg)
    print(f"Canonical protocol hash: {chash}")
    
    # Write protocol_lock.json
    lock_json_path = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["protocol_lock"]))
    os.makedirs(os.path.dirname(lock_json_path), exist_ok=True)
    with open(lock_json_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote protocol lock JSON: {lock_json_path}")
    
    # Write protocol_lock.sha256
    lock_sha_path = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["protocol_lock_sha"]))
    with open(lock_sha_path, "w") as f:
        f.write(chash + "\n")
    print(f"Wrote protocol lock hash: {lock_sha_path}")
    
    # Generate checklists for all sessions S01--S10
    checklists_dir = os.path.join(art_dir, "checklists")
    os.makedirs(checklists_dir, exist_ok=True)
    
    for s_id in sorted(cfg["sessions"].keys()):
        checklist = build_session_checklist(cfg, s_id)
        csv_path = os.path.join(checklists_dir, f"{s_id}_checklist.csv")
        
        # We write columns: clip_id, object_id, class_id, shift_id, repetition
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["clip_id", "object_id", "class_id", "shift_id", "repetition"])
            for row in checklist:
                writer.writerow([
                    row["clip_id"],
                    row["object_id"],
                    row["class_id"],
                    row["shift_id"],
                    row["repetition"]
                ])
        print(f"Generated checklist for {s_id}: {csv_path} ({len(checklist)} items)")

if __name__ == "__main__":
    main()
