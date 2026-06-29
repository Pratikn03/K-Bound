"""
cam_data.py - WILDS Camelyon17 loading + natural-shift CONDITION construction.

Domains (official WILDS splits; f0 is trained on 'train' = centers 0,3,4):
    'test'   -> center 2  (hardest OOD hospital; the 'EXTENDED' target)
    'val'    -> center 1  (OOD hospital)
    'id_val' -> centers 0,3,4  (in-distribution control; no hospital shift)

A CONDITION = (domain, composition, batch_regime).  The adaptation stream's
composition (iid / imbalanced / single_class label-shift) and batch size are the
natural deployment pathologies that make adaptation help or (for single_class +
tiny + aggressive) collapse - so harmful cells arise from the DATA, never from
tuned hyperparameters.  Benefit B is always measured on a separate CLASS-BALANCED
held-out eval pool for the domain, so collapse shows up as real accuracy loss.

Reuses the disk-filter from run_wilds_camelyon17.py (keep only patches present on
disk) so a partially-copied dataset is handled honestly (dropped count logged).
"""
from __future__ import annotations
import os
import numpy as np
import torch

CLASSES = (0, 1)            # Camelyon17 is binary (tumor 0/1)
DOMAINS = ("test", "val", "id_val")
BATCH_REGIMES = {"large_iid": 200, "small": 16, "tiny": 8}
COMPOSITIONS = ("iid", "imbalanced", "single_class")


def load_camelyon(root, img_size=96):
    """Return (dataset, transform, keep_full_mask, n_present, n_total)."""
    from wilds import get_dataset
    import torchvision.transforms as T
    ds = get_dataset(dataset="camelyon17", download=False, root_dir=root)
    transform = T.Compose([
        T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    patches_dir = os.path.join(ds.data_dir, "patches")
    present = set()
    if os.path.isdir(patches_dir):
        for d in os.listdir(patches_dir):
            dp = os.path.join(patches_dir, d)
            if os.path.isdir(dp):
                for f in os.listdir(dp):
                    present.add(d + "/" + f)
    inp = getattr(ds, "_input_array", None)

    def _key(p):
        parts = str(p).replace("\\", "/").split("/")
        return (parts[-2] + "/" + parts[-1]) if len(parts) >= 2 else parts[-1]

    if inp is not None and present:
        keep_full = np.array([_key(p) in present for p in inp], dtype=bool)
    else:
        keep_full = None
    n_total = len(inp) if inp is not None else 0
    n_present = int(keep_full.sum()) if keep_full is not None else n_total
    return ds, transform, keep_full, n_present, n_total


def make_domain(ds, transform, keep_full, domain):
    """Return (subset, labels_np) for a domain, disk-filtered to present patches.
    labels_np[i] is the tumor label of subset[i]."""
    sub = ds.get_subset(domain, transform=transform)
    idx = np.asarray(sub.indices)
    if keep_full is not None:
        sub.indices = idx[keep_full[idx]]
    y = ds.y_array[np.asarray(sub.indices)].numpy().astype(int)
    return sub, y


def _load_x(sub, pos):
    x, _, _ = sub[int(pos)]
    return x


def build_condition(sub, y, comp, bs, n_eval, rng, device, n_batches=4, robust_tries=25):
    """Build (adapt_stream, eval_x, eval_y_np) for one condition.

    eval pool is class-BALANCED (n_eval split across classes); the adaptation stream
    is composed per `comp` from the remaining patches (label-free at use time).
    Robust to occasional unreadable/truncated PNGs: stream resamples freely; eval
    resamples WITHIN the same class to keep labels aligned.  Tensors are moved to device.
    """
    pos_all = np.arange(len(sub))
    classes = np.array([c for c in CLASSES if (y == c).any()])
    if len(classes) < 2:
        classes = np.unique(y)
    # ---- balanced held-out eval pool ----
    per = max(1, n_eval // max(1, len(classes)))
    ev = []
    for c in classes:
        ci = pos_all[y == c]
        if len(ci):
            ev.append(rng.choice(ci, min(per, len(ci)), replace=False))
    ev = np.concatenate(ev); rng.shuffle(ev)
    remain = np.setdiff1d(pos_all, ev)
    if len(remain) == 0:
        remain = pos_all
    n_stream = max(bs, bs * n_batches)
    # ---- adaptation-stream composition ----
    if comp == "iid":
        s = rng.choice(remain, n_stream, replace=len(remain) < n_stream)
    elif comp == "imbalanced":
        maj = int(rng.choice(classes))
        mp = np.intersect1d(pos_all[y == maj], remain)
        op = np.setdiff1d(remain, mp)
        if len(mp) and len(op):
            nM = int(n_stream * 0.85)
            s = np.concatenate([rng.choice(mp, nM, replace=len(mp) < nM),
                                rng.choice(op, n_stream - nM, replace=len(op) < (n_stream - nM))])
        else:
            s = rng.choice(remain, n_stream, replace=len(remain) < n_stream)
    else:  # single_class label shift (collapse-prone)
        maj = int(rng.choice(classes))
        mp = np.intersect1d(pos_all[y == maj], remain)
        pool = mp if len(mp) else remain
        s = rng.choice(pool, n_stream, replace=len(pool) < n_stream)
    rng.shuffle(s)

    def _robust_load(positions, same_class=None):
        xs, used_c = [], []
        for p in positions:
            got = False
            order = [int(p)] + [int(q) for q in rng.permutation(
                pos_all[y == same_class] if same_class is not None else pos_all)]
            for q in order[:robust_tries]:
                try:
                    xs.append(_load_x(sub, q)); used_c.append(int(y[q])); got = True; break
                except Exception:
                    continue
            if not got:
                raise RuntimeError("no readable patch in condition")
        return torch.stack(xs).to(device), np.array(used_c, dtype=int)

    stream_x, _ = _robust_load(s)
    stream = [stream_x[i:i + bs] for i in range(0, len(stream_x), bs)]
    ev_list_x, ev_list_c = [], []
    for p in ev:
        x1, c1 = _robust_load([p], same_class=int(y[p]))
        ev_list_x.append(x1); ev_list_c.append(int(c1[0]))
    eval_x = torch.cat(ev_list_x, 0).to(device)
    eval_y = np.array(ev_list_c, dtype=int)
    return stream, eval_x, eval_y
