"""smoke_models.py — tiny, deterministic models + synthetic cells for smoke tests.

The per-dataset smoke proves *integration mechanics* (adapter loads, adapts, emits
telemetry, KGA consumes the candidate) for each runner's (num_classes, input)
config.  It is NOT a performance result: it uses a small BN-CNN stand-in and
synthetic covariate-shifted batches so it is fast, CPU-only, and reproducible.
The real runners/backbones/data are wired at locked-run time, not here.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Per-dataset config faithful to each runner (audit §3). hw is shrunk for a CPU
# smoke; real_input notes the production resolution. num_classes are exact.
DATASET_CONFIGS: Dict[str, Dict] = {
    "cifar10c":   {"num_classes": 10,   "hw": 32, "in_ch": 3, "arch": "resnet18(cifar)",        "real_input": "32x32"},
    "imagenet_c": {"num_classes": 10,   "hw": 64, "in_ch": 3, "arch": "resnet50/ViT (imagenette proxy; full ImageNet-C=1000 unavailable from host)", "real_input": "224x224"},
    "officehome": {"num_classes": 65,   "hw": 64, "in_ch": 3, "arch": "resnet50",               "real_input": "224x224"},
    "iwildcam":   {"num_classes": 182,  "hw": 64, "in_ch": 3, "arch": "resnet50",               "real_input": "224x224"},
    "camelyon17": {"num_classes": 2,    "hw": 32, "in_ch": 3, "arch": "densenet121",            "real_input": "96x96"},
    "rxrx1":      {"num_classes": 1139, "hw": 64, "in_ch": 3, "arch": "resnet50",               "real_input": "256x256"},
    "imagenet_r": {"num_classes": 200,  "hw": 64, "in_ch": 3, "arch": "resnet50(masked)",       "real_input": "224x224"},
    "cifar101":   {"num_classes": 10,   "hw": 32, "in_ch": 3, "arch": "resnet18(cifar)",        "real_input": "32x32"},
    "fmow":       {"num_classes": 62,   "hw": 64, "in_ch": 3, "arch": "densenet121",            "real_input": "224x224"},
}
DATASET_IDS = list(DATASET_CONFIGS.keys())


class TinyBNCNN(nn.Module):
    """Small CNN WITH BatchNorm so the BN/LN-affine update surface exists."""

    def __init__(self, num_classes: int, in_ch: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        z = self.features(x).flatten(1)
        return self.head(z)


def build_f0(num_classes: int, in_ch: int = 3, seed: int = 0) -> nn.Module:
    """Deterministic frozen model. Runs a few batches so BN running stats are set."""
    torch.manual_seed(seed)
    m = TinyBNCNN(num_classes, in_ch)
    m.eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed + 7)
        for _ in range(3):
            m.train()
            m(torch.randn(32, in_ch, 16, 16, generator=g))  # populate BN running stats
        m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def synth_cell(num_classes: int, n: int, in_ch: int, hw: int, seed: int,
               shift: float = 1.5) -> Tuple[List[torch.Tensor], torch.Tensor, np.ndarray]:
    """Return (adapt_stream=[unlabeled test batch], eval_x, dev_y).

    The adapt/eval batches carry a covariate shift (scaled + offset Gaussian) to
    mimic a corrupted/shifted target. dev_y are DEV/calibration labels (seeded),
    used only for the benefit/KGA certificate — never test labels.
    """
    g = torch.Generator().manual_seed(seed)
    x_test = torch.randn(n, in_ch, hw, hw, generator=g) * (1.0 + 0.3 * shift) + shift
    eval_x = torch.randn(n, in_ch, hw, hw, generator=g) * (1.0 + 0.3 * shift) + shift
    dev_y = torch.randint(0, num_classes, (n,), generator=g).numpy().astype(np.int64)
    return [x_test.contiguous()], eval_x.contiguous(), dev_y
