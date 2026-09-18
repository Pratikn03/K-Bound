#!/usr/bin/env python3
"""Pilot benchmark for independent CIFAR-10 ResNet-18 source checkpoint training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "experiments/kbound/cifar"
OUTPUT_DIR = ROOT / "experiments/kbound/results/task1_independent_models_pilot"


def make_cifar_resnet18() -> nn.Module:
    m = tv.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def train_single_model(
    seed: int,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 0.1,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    output_path: Path | None = None,
) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)

    # CIFAR-10 source train / val split (45,000 train / 5,000 val)
    full_train = tv.datasets.CIFAR10(str(DATA), train=True, download=True, transform=T.ToTensor())
    generator = torch.Generator().manual_seed(42)  # Fixed source-only split across all seeds
    train_subset, val_subset = torch.utils.data.random_split(full_train, [45000, 5000], generator=generator)

    Xtr = torch.stack([full_train[i][0] for i in train_subset.indices])
    ytr = torch.tensor([full_train.targets[i] for i in train_subset.indices])

    Xval = torch.stack([full_train[i][0] for i in val_subset.indices])
    yval = torch.tensor([full_train.targets[i] for i in val_subset.indices])

    m = make_cifar_resnet18().to(device)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = epochs * (len(Xtr) // batch_size + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    aug = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])
    norm = lambda x: (x - mean) / std

    start_time = time.time()
    best_val_acc = 0.0
    best_weights = None

    for ep in range(epochs):
        ep_t0 = time.time()
        m.train()
        perm = torch.randperm(len(Xtr))
        cor = tot = 0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            xb = aug(Xtr[idx]).to(device)
            yb = ytr[idx].to(device)
            out = m(norm(xb))
            loss = F.cross_entropy(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            cor += (out.argmax(1) == yb).sum().item()
            tot += len(yb)
        if device == "mps":
            torch.mps.synchronize()
        train_acc = cor / tot

        # Validate on source val split
        m.eval()
        with torch.no_grad():
            v_preds = []
            for i in range(0, len(Xval), 512):
                v_preds.append(m(norm(Xval[i:i + 512].to(device))).argmax(1).cpu())
            val_acc = (torch.cat(v_preds) == yval).float().mean().item()

        ep_dt = time.time() - ep_t0
        print(f"Seed {seed} | Epoch {ep+1}/{epochs} ({ep_dt:.1f}s): train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in m.state_dict().items()}

    total_time = time.time() - start_time
    if output_path is not None and best_weights is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_weights, str(output_path))
        sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    else:
        sha256 = None

    return {
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "best_val_acc": best_val_acc,
        "training_time_sec": total_time,
        "checkpoint_sha256": sha256,
        "output_path": str(output_path) if output_path else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for timed pilot")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUTPUT_DIR / f"pilot_resnet18_seed{args.seed}.pt"

    print(f"Running timed pilot: training seed {args.seed} for {args.epochs} epochs...")
    res = train_single_model(seed=args.seed, epochs=args.epochs, output_path=ckpt_path)

    projected_12_ep_sec = (res["training_time_sec"] / args.epochs) * 12
    projected_5_models_hr = (projected_12_ep_sec * 5) / 3600.0

    print(f"\nPilot finished in {res['training_time_sec']:.1f}s.")
    print(f"Projected 12-epoch training per model: {projected_12_ep_sec:.1f}s ({projected_12_ep_sec/60:.1f} min)")
    print(f"Projected 5 models training total: {projected_5_models_hr:.2f} hours")

    summary = {
        "pilot_result": res,
        "projected_12_epoch_seconds_per_model": projected_12_ep_sec,
        "projected_5_models_training_hours": projected_5_models_hr,
        "full_grid_evaluation_estimate_hours": 9.2,  # 5 models * 1.84h per 432-cell Tent grid
        "projected_aggregate_compute_hours": projected_5_models_hr + 9.2,
    }

    summary_file = OUTPUT_DIR / "pilot_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_file}")


if __name__ == "__main__":
    main()
