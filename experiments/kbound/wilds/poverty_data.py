"""WILDS PovertyMap loader helpers (binned wealth for TTA-compatible classification)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T

NUM_BINS = 5
NUM_CLASSES = NUM_BINS
GROUP_FIELD = "country"
FOLD = "A"


def poverty_transform():
  # Images are pre-normalized float tensors; only spatial crop for TTA batches.
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
    ])


def _wealth_to_bins(wealth: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(wealth, edges[1:-1], right=False), 0, len(edges) - 2).astype(int)


def _fit_bin_edges(ds, n_bins: int = NUM_BINS) -> np.ndarray:
    train_idx = np.where(ds.split_array == ds.split_dict["train"])[0]
    wealth = ds.metadata_array[train_idx, ds.metadata_fields.index("y")].numpy().astype(float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(wealth, qs)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return edges


def _load_edges(cache_path: Path, ds):
    if cache_path.exists():
        obj = json.loads(cache_path.read_text())
        return np.asarray(obj["edges"], float)
    edges = _fit_bin_edges(ds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"edges": edges.tolist(), "n_bins": NUM_BINS}))
    return edges


class PovertyBinnedSubset:
    """WILDS subset with train-fixed quantile bins (TTA needs class labels)."""

    def __init__(self, wilds_subset, y_binned: np.ndarray):
        self.sub = wilds_subset
        self.y_binned = np.asarray(y_binned, int)
        self.indices = wilds_subset.indices

    def __len__(self):
        return len(self.sub)

    def __getitem__(self, i):
        x, _, md = self.sub[i]
        return x, int(self.y_binned[i]), md


def get_poverty(root: str, split: str, edges_cache: Path | None = None):
    from wilds import get_dataset

    ds = get_dataset(dataset="poverty", download=False, root_dir=root, fold=FOLD)
    raw = ds.get_subset(split, transform=poverty_transform())
    idx = np.asarray(raw.indices)
    cache = edges_cache or Path(root) / "poverty_v1.1" / "_kbound_bin_edges_foldA.json"
    edges = _load_edges(cache, ds)
    wealth = ds.metadata_array[idx, ds.metadata_fields.index("y")].numpy().astype(float)
    y = _wealth_to_bins(wealth, edges)
    md = ds.metadata_array[idx].numpy()
    country_i = ds.metadata_fields.index(GROUP_FIELD)
    groups = md[:, country_i].astype(int)
    sub = PovertyBinnedSubset(raw, y)
    return ds, sub, y, groups, edges


def make_poverty_resnet(backbone: str, device: torch.device):
    if backbone == "resnet18":
        weights = tvm.ResNet18_Weights.DEFAULT
        model = tvm.resnet18(weights=weights)
    elif backbone == "resnet50":
        weights = tvm.ResNet50_Weights.DEFAULT
        model = tvm.resnet50(weights=weights)
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    old = model.conv1
    model.conv1 = nn.Conv2d(
        8, old.out_channels, kernel_size=old.kernel_size,
        stride=old.stride, padding=old.padding, bias=False,
    )
    with torch.no_grad():
        model.conv1.weight[:, :3] = old.weight
        mean_rgb = old.weight.mean(dim=1, keepdim=True)
        model.conv1.weight[:, 3:8] = mean_rgb.expand(-1, 5, -1, -1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)
