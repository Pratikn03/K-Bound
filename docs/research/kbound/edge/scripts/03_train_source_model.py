#!/usr/bin/env python3
"""03 -- train the FROZEN source model f0 on the source clip, then BN-recalibrate.

Builds a MobileNetV3-Small with a 4-class head, trains it on the labelled source
clip, and recalibrates BatchNorm running statistics so the frozen model is a
competent EVAL-mode baseline (without this, a freshly trained small net collapses
to chance in eval mode -- the very BN mismatch Tent later targets).  Saves the
checkpoint + a meta file with the model_version hash.
"""

import argparse
import numpy as np

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    C.set_seed(cfg["seed"])

    import torch
    from kbound_edge.model import build_model, train_classifier, recalibrate_bn, predict_proba, state_dict_hash
    from kbound_edge.dataset import frames_to_tensor

    clip = np.load(C.resolve(cfg["paths"]["source_clip"]))
    frames = list(clip["frames"])
    labels = clip["labels"]
    if (labels < 0).any():
        raise SystemExit("[03] source clip is unlabelled (labels=-1). Provide labels before training.")

    X = frames_to_tensor(frames, cfg["image_size"])
    y = torch.tensor(np.asarray(labels), dtype=torch.long)

    model = build_model(cfg["num_classes"], pretrained=False, seed=cfg["seed"], device=cfg.get("device", "cpu"))
    tr = cfg["training"]
    print(f"[03] training f0: n={len(X)} epochs={tr['epochs']} lr={tr['lr']} batch={tr['batch_size']}")
    train_classifier(model, X, y, epochs=tr["epochs"], lr=tr["lr"],
                     batch_size=tr["batch_size"], seed=cfg["seed"], verbose=False)
    recalibrate_bn(model, X, passes=tr["bn_recal_passes"])

    acc = float((predict_proba(model, X).argmax(1) == np.asarray(labels)).mean())
    version = state_dict_hash(model)

    out = C.resolve(cfg["paths"]["model"])
    C.ensure_parent(out)
    torch.save(model.state_dict(), out)
    C.save_json(C.resolve(cfg["paths"]["model_meta"]), {
        "model_version": version,
        "num_classes": cfg["num_classes"],
        "image_size": cfg["image_size"],
        "train_acc_eval_mode": acc,
        "arch": "mobilenet_v3_small",
    })
    print(f"[03] f0 trained: eval-mode source acc={acc:.3f} model_version={version} -> {out}")
    if acc < 0.8:
        print("[03] WARNING: low eval-mode accuracy; check BN recalibration / training settings.")


if __name__ == "__main__":
    main()
