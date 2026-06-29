"""WILDS FMoW loader helpers for K-Bound geoshift scans."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torchvision.transforms as T

NUM_CLASSES = 62
GROUP_FIELD = "region"


def image_transform(train: bool):
    if train:
        return T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(0.65, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def _patch_wilds_fmow_timestamps():
    """WILDS FMoW uses bare pd.to_datetime; pandas 2.x rejects mixed ISO ms suffixes."""
    import pandas as pd

    if getattr(pd.to_datetime, "_kbound_fmow_patched", False):
        return
    _orig = pd.to_datetime

    def _compat(arg, *args, **kwargs):
        if not args and "format" not in kwargs:
            try:
                return _orig(arg, format="ISO8601", utc=True)
            except (ValueError, TypeError):
                pass
        return _orig(arg, *args, **kwargs)

    _compat._kbound_fmow_patched = True  # type: ignore[attr-defined]
    pd.to_datetime = _compat


def get_fmow(root: str, split: str, train_tf: bool = False):
    _patch_wilds_fmow_timestamps()
    from wilds import get_dataset

    ds = get_dataset(dataset="fmow", download=False, root_dir=root)
    sub = ds.get_subset(split, transform=image_transform(train_tf))
    idx = np.asarray(sub.indices)
    y = ds.y_array[idx].numpy().astype(int)
    md = ds.metadata_array[idx].numpy()
    region_i = ds.metadata_fields.index(GROUP_FIELD)
    groups = md[:, region_i].astype(int)
    return ds, sub, y, groups


def filter_present_subset(ds, sub, y, groups):
    """Drop indices whose PNG is missing (partial download)."""
    img_dir = Path(ds.data_dir) / "images"
    if not img_dir.is_dir():
        return sub, y, groups
    present = {
        ent.name
        for ent in os.scandir(img_dir)
        if ent.is_file() and ent.name.endswith(".png") and not ent.name.startswith("._")
    }
    if not present:
        return sub, y, groups
    idx = np.asarray(sub.indices)
    keep = np.fromiter(
        (f"rgb_img_{i}.png" in present for i in idx),
        dtype=bool,
        count=len(idx),
    )
    if bool(keep.all()):
        return sub, y, groups
    sub.indices = idx[keep]
    y = y[keep]
    groups = groups[keep]
    print(f"[fmow] disk-filter kept {len(sub.indices)}/{len(idx)}", flush=True)
    return sub, y, groups
