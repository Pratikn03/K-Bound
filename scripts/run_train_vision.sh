#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" - <<'PY'
import csv
import json
import os
import random
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from uais_v.models.vision_resnet import VisionConfig, build_resnet_classifier

data_root = Path("data/processed/vision")
train_dir = data_root / "train"
val_dir = data_root / "val"
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff")


def has_images(directory: Path) -> bool:
    return directory.exists() and any(
        path.is_file() and path.suffix.lower() in image_extensions
        for path in directory.rglob("*")
    )


def balanced_subset(dataset, max_per_class: int, seed: int) -> Subset:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(dataset.targets):
        by_class.setdefault(int(label), []).append(idx)
    indices: list[int] = []
    for label_indices in by_class.values():
        rng.shuffle(label_indices)
        indices.extend(label_indices[:max_per_class])
    rng.shuffle(indices)
    return Subset(dataset, indices)


def evaluate(model, loader, device, positive_index: int) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    preds: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            batch_preds = probabilities.argmax(axis=1)
            labels.extend(batch_y.numpy().astype(int).tolist())
            preds.extend(batch_preds.astype(int).tolist())
            scores.extend(probabilities[:, positive_index].astype(float).tolist())
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }
    if len(set(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    return metrics


if not has_images(train_dir):
    print("Vision image data not found at", data_root, "- please add image class folders. Skipping.")
else:
    seed = 42
    torch.manual_seed(seed)
    max_per_class = int(os.environ.get("VISION_MAX_PER_CLASS", "500"))
    batch_size = int(os.environ.get("VISION_BATCH_SIZE", "32"))
    epochs = int(os.environ.get("VISION_EPOCHS", "1"))
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    transform = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    validation_root = val_dir if has_images(val_dir) else train_dir
    val_dataset = datasets.ImageFolder(validation_root, transform=transform)
    train_subset = balanced_subset(train_dataset, max_per_class=max_per_class, seed=seed)
    val_subset = balanced_subset(val_dataset, max_per_class=max_per_class, seed=seed)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_resnet_classifier(
        VisionConfig(num_classes=len(train_dataset.classes), pretrained=False)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    positive_index = 1 if len(train_dataset.classes) > 1 else 0
    metrics = evaluate(model, val_loader, device, positive_index)
    metrics.update(
        {
            "classes": train_dataset.classes,
            "device": str(device),
            "epochs": epochs,
            "max_per_class": max_per_class,
            "train_samples": len(train_subset),
            "validation_samples": len(val_subset),
            "train_loss_last": losses[-1] if losses else None,
        }
    )

    metrics_dir = Path("experiments/vision/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (metrics_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Metric", "Value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"Metric": key, "Value": value})
    print("Vision dataset:", train_dir)
    print("Vision metrics:", metrics)
PY
