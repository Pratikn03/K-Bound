#!/usr/bin/env python3
"""03 -- train the FROZEN source model f0 on the source clip, then BN-recalibrate.

Supports both synthetic (default) and real_manifest protocols. In real_manifest
mode, f0 is trained on S01 physical clips and validated on S02 physical clips.
Enforces the S02 performance gate (>= 0.80 balanced acc and macro-F1).
Saves checkpoint, model card, and metadata.
"""

import argparse
import hashlib
import os
import sys
import numpy as np

import _common as C

# Import helpers from kbound_edge
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cv2
from sklearn.metrics import balanced_accuracy_score, f1_score

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
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override training epochs")
    ap.add_argument("--bypass-gate", action="store_true", help="bypass performance gate check")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    C.set_seed(cfg["seed"])

    import torch
    import hashlib
    from kbound_edge.model import build_model, train_classifier, recalibrate_bn, predict_proba, state_dict_hash, source_datasets

    protocol_name = cfg.get("protocol", "edge_label_inspection_v1")
    is_real = protocol_name == "edge_real_phone_v1"

    if not is_real:
        # --- Synthetic Mode ---
        clip = np.load(C.resolve(cfg["paths"]["source_clip"]))
        frames = list(clip["frames"])
        labels = clip["labels"]
        if (labels < 0).any():
            raise SystemExit("[03] source clip is unlabelled (labels=-1). Provide labels before training.")

        from kbound_edge.dataset import frames_to_tensor
        X_train = frames_to_tensor(frames, cfg["image_size"])
        y_train = torch.tensor(np.asarray(labels), dtype=torch.long)
        
        X_val = X_train
        y_val_np = np.asarray(labels)
    else:
        # --- Real Manifest Mode ---
        edge_dir = os.path.normpath(os.path.join(_HERE, ".."))
        results_dir = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["results_dir"]))
        inv_path = os.path.join(results_dir, "recording_inventory.json")
        
        if not os.path.exists(inv_path):
            raise SystemExit(f"[03] Recording inventory not found: {inv_path}. Run 02_validate_real_dataset.py first.")
            
        train_clips, val_clips = source_datasets(inv_path)
        if args.bypass_gate:
            print("[03] Bypassing/subsetting datasets for fast test/mock run (12 train, 6 val clips)")
            train_clips = train_clips[:12]
            val_clips = val_clips[:6]
        
        # Load physical frames and labels
        raw_dir = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["raw_dir"]))
        class_map = {name: i for i, name in enumerate(cfg["classes"])}
        
        def load_dataset(clips):
            frames = []
            labels = []
            for c in clips:
                mp4_path = os.path.join(raw_dir, c.session_id, f"{c.clip_id}.mp4")
                clip_frames = load_mp4_frames(mp4_path)
                frames.extend(list(clip_frames))
                labels.extend([class_map[c.class_id]] * len(clip_frames))
            return frames, labels
            
        train_frames, train_labels = load_dataset(train_clips)
        val_frames, val_labels = load_dataset(val_clips)
        
        from kbound_edge.dataset import frames_to_tensor
        X_train = frames_to_tensor(train_frames, cfg["image_size"])
        y_train = torch.tensor(np.asarray(train_labels), dtype=torch.long)
        
        X_val = frames_to_tensor(val_frames, cfg["image_size"])
        y_val_np = np.asarray(val_labels)

    # Build and train f0
    model = build_model(cfg["num_classes"], pretrained=False, seed=cfg["seed"], device=cfg.get("device", "cpu"))
    tr = cfg["training"]
    print(f"[03] training f0: n={len(X_train)} epochs={tr['epochs']} lr={tr['lr']} batch={tr['batch_size']}")
    train_classifier(model, X_train, y_train, epochs=tr["epochs"], lr=tr["lr"],
                     batch_size=tr["batch_size"], seed=cfg["seed"], verbose=False)
    recalibrate_bn(model, X_train, passes=tr["bn_recal_passes"])

    # Calculate metrics
    train_preds = predict_proba(model, X_train).argmax(1)
    train_acc = float((train_preds == y_train.cpu().numpy()).mean())
    
    val_preds = predict_proba(model, X_val).argmax(1)
    val_bal_acc = float(balanced_accuracy_score(y_val_np, val_preds))
    val_macro_f1 = float(f1_score(y_val_np, val_preds, average="macro"))
    
    version = state_dict_hash(model)

    out = C.resolve(cfg["paths"]["model"])
    C.ensure_parent(out)
    torch.save(model.state_dict(), out)
    
    # Save standard f0 metadata
    model_meta_path = cfg["paths"].get("model_meta", "artifacts_real/models/f0_meta.json")
    C.save_json(C.resolve(model_meta_path), {
        "model_version": version,
        "num_classes": cfg["num_classes"],
        "image_size": cfg["image_size"],
        "train_acc_eval_mode": train_acc,
        "arch": "mobilenet_v3_small",
    })
    
    print(f"[03] f0 trained: train_acc={train_acc:.3f} val_balanced_acc={val_bal_acc:.3f} val_macro_f1={val_macro_f1:.3f} model_version={version} -> {out}")

    # Enforce performance gate in real mode
    if is_real:
        card_path = os.path.join(results_dir, "model_card.json")
        protocol_lock_sha_path = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["protocol_lock_sha"]))
        
        with open(protocol_lock_sha_path) as f:
            p_hash = f.read().strip()
            
        inv_hash = get_file_sha256(inv_path)
        
        model_card = {
            "protocol_hash": p_hash,
            "data_inventory_hash": inv_hash,
            "training_command": " ".join(sys.argv),
            "metrics": {
                "train_acc": train_acc,
                "val_balanced_acc": val_bal_acc,
                "val_macro_f1": val_macro_f1
            },
            "model_version": version
        }
        C.save_json(card_path, model_card)
        print(f"[03] Wrote model card to: {card_path}")
        
        if val_bal_acc < 0.80 or val_macro_f1 < 0.80:
            if args.bypass_gate:
                print(f"[03] WARNING: Source gate failed! Balanced Acc={val_bal_acc:.3f}, Macro F1={val_macro_f1:.3f}. Bypassing for testing.")
            else:
                print(f"[03] ERROR: Source gate failed! Balanced Acc={val_bal_acc:.3f}, Macro F1={val_macro_f1:.3f}. Must be >= 0.80.")
                sys.exit(1)
        else:
            print("[03] Source gate passed successfully.")
            
    else:
        if train_acc < 0.8:
            print("[03] WARNING: low train accuracy; check BN recalibration / training settings.")

if __name__ == "__main__":
    main()
