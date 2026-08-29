"""Train one Office-Home source checkpoint under the archived source recipe.

ResNet-50 with ImageNet V2 initialization is fine-tuned on the locked
Real_World/train split for a fixed epoch budget. Real_World/val is reported but
never used for early stopping, and target-domain labels are never loaded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oh_data as ohd  # noqa: E402


def build_resnet50(num_classes: int, device: torch.device) -> nn.Module:
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


@torch.no_grad()
def evaluate(model, paths, labels, device, transform, batch_size=128, workers=0):
    model.eval()
    dataset = ohd.ItemDataset(paths, labels, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    predictions = []
    for images, _ in loader:
        predictions.append(model(images.to(device)).argmax(1).cpu().numpy())
    prediction = np.concatenate(predictions)
    accuracy = float((prediction == labels).mean())
    recalls = [
        float((prediction[labels == cls] == cls).mean())
        for cls in np.unique(labels)
        if (labels == cls).any()
    ]
    return accuracy, float(np.mean(recalls))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = ohd.load_or_make_splits(args.data_root, args.splits)
    classes = splits["classes"]
    train_paths, train_labels = ohd.split_items(splits, ohd.SOURCE, "train")
    val_paths, val_labels = ohd.split_items(splits, ohd.SOURCE, "val")
    print(
        f"[data] source={ohd.SOURCE} train={len(train_paths)} val={len(val_paths)} "
        f"classes={len(classes)} seed={args.seed}",
        flush=True,
    )

    model = build_resnet50(len(classes), device)
    train_dataset = ohd.ItemDataset(train_paths, train_labels, ohd.train_transform())
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    started = time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_accuracy, val_balanced_accuracy = evaluate(
            model, val_paths, val_labels, device, ohd.eval_transform(), workers=args.workers
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_acc": val_accuracy,
            "val_bacc": val_balanced_accuracy,
        }
        history.append(record)
        print(
            f"  [ep {epoch}] loss={record['train_loss']:.3f} "
            f"val_acc={val_accuracy:.4f} val_bacc={val_balanced_accuracy:.4f}",
            flush=True,
        )

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"f0_resnet50_rw_seed{args.seed}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "arch": "resnet50",
            "num_classes": len(classes),
            "source": ohd.SOURCE,
            "model_seed": args.seed,
            "epochs": args.epochs,
            "recipe": {
                "initialization": "ResNet50_Weights.IMAGENET1K_V2",
                "optimizer": "AdamW",
                "lr": args.lr,
                "weight_decay": args.wd,
                "batch_size": args.bs,
                "num_workers": args.workers,
                "selection": "fixed_epoch_budget",
            },
            "val_history": history,
            "final_val_acc": history[-1]["val_acc"],
            "wall_sec": round(time.time() - started, 1),
        },
        checkpoint,
    )
    metadata = {
        "schema": "kbound_officehome_f0_v1",
        "execution_complete": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "model_seed": args.seed,
        "source_split": "Real_World/train",
        "selection_split": None,
        "report_only_split": "Real_World/val",
        "final_val_acc": history[-1]["val_acc"],
        "final_val_bacc": history[-1]["val_bacc"],
        "val_history": history,
        "wall_sec": round(time.time() - started, 1),
    }
    atomic_json(output / f"f0_meta_seed{args.seed}.json", metadata)
    print(f"[f0] saved {checkpoint} sha256={metadata['checkpoint_sha256']}", flush=True)


if __name__ == "__main__":
    main()
