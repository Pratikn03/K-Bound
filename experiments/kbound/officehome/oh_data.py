"""
oh_data.py - Office-Home loading, LOCKED train/val/test splits, and multi-class
CONDITION construction for the K-Bound natural-shift pipeline.

DOMAINS (4): Art, Clipart, Product, Real_World  (65 classes).
SOURCE = Real_World (closest to ImageNet); fine-tune f0 on Real_World *train*,
calibrate the certificate / tau* on Real_World *val* (in-distribution control).
TARGETS = Art, Clipart, Product; each is split val (dev / regime scan) | test
(HELD-OUT, evaluated once).  Splits are written to a JSON and reloaded so the
held-out test partition is fixed before any test evaluation.

A CONDITION = (domain, split, composition, batch_regime) x seed.  Composition
(iid / imbalanced / single_class) + small batches are the natural deployment
pathologies that make TTA help or collapse -> harmful cells come from the DATA,
never from tuned hyperparameters.  B is measured on a class-BALANCED held-out
eval pool, so collapse shows up as real accuracy loss.
INTEGRITY: labels are used only for B/oracle/eval; the routers see only Z.
"""
from __future__ import annotations
import json, os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate any truncated JPEG (avoid decoder stalls)

DOMAINS = ("Art", "Clipart", "Product", "Real_World")
SOURCE = "Real_World"
TARGETS = ("Art", "Clipart", "Product")
NUM_CLASSES = 65

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def eval_transform():
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(_MEAN, _STD)])


def train_transform():
    return T.Compose([T.Resize(256), T.RandomResizedCrop(224, scale=(0.6, 1.0)),
                      T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(_MEAN, _STD)])


def class_names(root) -> list[str]:
    d = Path(root) / SOURCE
    return sorted([c.name for c in d.iterdir() if c.is_dir()])


def scan_domain(root, domain, classes):
    cidx = {c: i for i, c in enumerate(classes)}
    items = []
    base = Path(root) / domain
    for c in classes:
        cd = base / c
        if not cd.is_dir():
            continue
        for f in sorted(cd.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and not f.name.startswith("._"):
                items.append((str(f), cidx[c]))
    return items


def make_splits(root, seed=20260615, src_train_frac=0.70, tgt_val_frac=0.50):
    """Deterministic per-domain, per-class stratified split. Returns dict + classes."""
    classes = class_names(root)
    rng = np.random.default_rng(seed)
    splits = {}
    for dom in DOMAINS:
        items = scan_domain(root, dom, classes)
        by_c = {}
        for p, y in items:
            by_c.setdefault(y, []).append(p)
        if dom == SOURCE:
            parts = {"train": [], "val": []}
            for y, paths in by_c.items():
                paths = list(paths); rng.shuffle(paths)
                k = max(1, int(round(len(paths) * src_train_frac)))
                for p in paths[:k]:
                    parts["train"].append([p, y])
                for p in paths[k:]:
                    parts["val"].append([p, y])
        else:
            parts = {"val": [], "test": []}
            for y, paths in by_c.items():
                paths = list(paths); rng.shuffle(paths)
                k = max(1, int(round(len(paths) * tgt_val_frac)))
                for p in paths[:k]:
                    parts["val"].append([p, y])
                for p in paths[k:]:
                    parts["test"].append([p, y])
        splits[dom] = parts
    return {"seed": int(seed), "src_train_frac": src_train_frac, "tgt_val_frac": tgt_val_frac,
            "classes": classes, "splits": splits}


def load_or_make_splits(root, path, **kw):
    path = Path(path)
    if path.exists():
        return json.load(open(path))
    obj = make_splits(root, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, path)
    return obj


def split_items(splits_obj, domain, split):
    items = splits_obj["splits"][domain][split]
    paths = [p for p, _ in items]
    y = np.array([int(yy) for _, yy in items], dtype=int)
    return paths, y


# ---------- image loading ----------
def _load(path, tf):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return tf(im)


class ItemDataset(torch.utils.data.Dataset):
    def __init__(self, paths, y, tf):
        self.paths = paths; self.y = y; self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return _load(self.paths[i], self.tf), int(self.y[i])


# ---------- condition construction (multi-class) ----------
def build_condition(paths, y, comp, bs, n_eval, rng, device, n_batches=3, tf=None):
    """Return (adapt_stream[list of batches], eval_x, eval_y_np) for one condition.

    eval pool is class-BALANCED (n_eval spread across present classes); the adaptation
    stream is composed per `comp` from the remaining images (label-free at use time)."""
    tf = tf or eval_transform()
    paths = list(paths); y = np.asarray(y, int)
    pos_all = np.arange(len(paths))
    classes = np.unique(y)
    per = max(1, n_eval // max(1, len(classes)))
    ev = []
    for c in classes:
        ci = pos_all[y == c]
        if len(ci):
            ev.append(rng.choice(ci, min(per, len(ci)), replace=False))
    ev = np.concatenate(ev)
    if len(ev) > n_eval:
        ev = rng.choice(ev, n_eval, replace=False)
    rng.shuffle(ev)
    remain = np.setdiff1d(pos_all, ev)
    if len(remain) == 0:
        remain = pos_all
    n_stream = max(bs, bs * n_batches)
    if comp == "iid":
        s = rng.choice(remain, n_stream, replace=len(remain) < n_stream)
    elif comp == "imbalanced":
        counts = Counter(y[remain].tolist())
        maj = counts.most_common(1)[0][0]
        mp = remain[y[remain] == maj]; op = remain[y[remain] != maj]
        nM = int(round(0.85 * n_stream))
        if len(mp) and len(op):
            s = np.concatenate([rng.choice(mp, nM, replace=len(mp) < nM),
                                rng.choice(op, n_stream - nM, replace=len(op) < (n_stream - nM))])
        else:
            s = rng.choice(remain, n_stream, replace=len(remain) < n_stream)
    elif comp == "single_class":
        counts = Counter(y[remain].tolist())
        cls = counts.most_common(1)[0][0]
        pool = remain[y[remain] == cls]
        s = rng.choice(pool, n_stream, replace=len(pool) < n_stream)
    else:
        raise ValueError(comp)
    rng.shuffle(s)

    def _stack(idxs):
        return torch.stack([_load(paths[int(i)], tf) for i in idxs]).to(device)

    stream_x = _stack(s)
    eval_x = _stack(ev)
    eval_y = y[ev].astype(int)
    stream = [stream_x[i:i + bs] for i in range(0, len(stream_x), bs)]
    return stream, eval_x, eval_y


def source_prior(splits_obj, domain=SOURCE, split="train"):
    """Source class prior (label-available on source) for label-shift candidates."""
    _, y = split_items(splits_obj, domain, split)
    p = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    p = p / max(p.sum(), 1.0)
    p[p <= 0] = 1e-6
    return p / p.sum()
