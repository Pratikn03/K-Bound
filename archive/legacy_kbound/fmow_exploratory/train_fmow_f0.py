"""
train_fmow_f0.py - train a PROPER WILDS-FMoW source model f0 for K-Bound.

FMoW uses DenseNet121 as the standard backbone.
This trains a DenseNet121 ERM source model by fine-tuning from ImageNet.
Selection on id_val (in-distribution). The OOD splits are NEVER seen here.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as tvm
from torchvision import transforms
from wilds import get_dataset

NUM_CLASSES = 62

def build_model(device):
    m = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT)
    m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    return m.to(device)

def get_transforms():
    # Standard FMoW transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()

def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    repo_root = Path(__file__).resolve().parents[3]
    default_data_root = repo_root / "data"
    data_dir = Path(os.environ.get("KBOUND_DATA_ROOT", default_data_root))

    print(f"Loading FMoW from {data_dir}...")
    dataset = get_dataset(dataset="fmow", root_dir=data_dir, download=False)

    train_data = dataset.get_subset("train", transform=get_transforms())
    id_val_data = dataset.get_subset("id_val", transform=get_transforms())

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(id_val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_ckpt = out_dir / f"f0_fmow_seed{args.seed}_best.pth"
    log_file = out_dir / f"f0_fmow_seed{args.seed}_log.jsonl"

    with open(log_file, "w") as f:
        f.write(json.dumps({"event": "start", "args": vars(args)}) + "\n")

    print(f"Starting training on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        t0 = time.time()
        for batch_idx, (inputs, targets, metadata) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

        with open(log_file, "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_loss": val_loss / val_total,
                "time": elapsed
            }) + "\n")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> New best id_val accuracy! Saved to {best_ckpt.name}")

    final_hash = sha256(best_ckpt)
    print(f"Finished. Best model hash: {final_hash}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4) # WILDS standard learning rate for DenseNet on FMoW
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="experiments/kbound/checkpoints/fmow")

    args = parser.parse_args()
    train(args)
