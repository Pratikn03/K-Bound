"""GPU: build a D23-format multimodal score cache for MulSen-AD (rgb + infrared + pointcloud).

A THIRD, genuinely multi-SENSOR dataset for D23 (real RGB camera + real infrared sensor
+ real 3D point cloud), enabling a *real sensor-dropout* failure test rather than the
uniform-noise injection used for Real-IAD-D3 / MVTec-3D. Per category: one-class
PatchCore memory banks from train/good for each modality; RGB and Infrared are scored
as images; the point-cloud STL mesh is rasterized to an orthographic depth image (pure
numpy via trimesh) and scored the same way. Writes per-category caches in the
{Sval,yval,Stest,ytest,valauc} format that multimodal_reliability_test.py consumes.

    PYTHONPATH=src python src/scripts/elara_u/gpu_build_mulsen_cache.py
    PYTHONPATH=src python -m scripts.elara_u.multimodal_reliability_test \
        --cache experiments/fusion/mulsen_score_cache --glob '*.npz' --tag MulSen-AD \
        --failure dropout --out experiments/elara_u/multimodal_reliability_results_mulsen.json
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/mulsen_ad"
CACHE = ROOT / "experiments/fusion/mulsen_score_cache"
RNG = 0
MODS = {"RGB": "rgb", "Infrared": "ir", "Pointcloud": "pcd"}


def _zsig(raw, ref):
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def _load_img(p: Path):
    from PIL import Image
    im = Image.open(p)
    return im.convert("RGB") if im.mode != "RGB" else im


def _load_pcd_depth(p: Path, H=224, W=224, n=20000):
    """STL mesh -> orthographic depth image (numpy z-buffer) -> PIL RGB. No renderer needed."""
    import trimesh
    from PIL import Image
    m = trimesh.load(str(p), force="mesh")
    pts = np.asarray(m.sample(n), dtype=np.float32) if len(m.faces) else np.asarray(m.vertices, np.float32)
    pts = pts - pts.mean(0)
    s = float(np.abs(pts).max() + 1e-9)
    pts = pts / s
    ix = np.clip(((pts[:, 0] * 0.5 + 0.5) * (W - 1)).astype(int), 0, W - 1)
    iy = np.clip(((pts[:, 1] * 0.5 + 0.5) * (H - 1)).astype(int), 0, H - 1)
    depth = np.zeros((H, W), np.float32)
    np.maximum.at(depth, (iy, ix), pts[:, 2] * 0.5 + 0.5)   # nearest-to-camera z per pixel
    img = (depth * 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def _paths(cat_dir: Path, rgb_path: Path):
    """Given an RGB png path, return matched (rgb, ir, pcd) paths."""
    rel = rgb_path.relative_to(cat_dir / "RGB")
    ir = cat_dir / "Infrared" / rel
    pcd = (cat_dir / "Pointcloud" / rel).with_suffix(".stl")
    return rgb_path, ir, pcd


def _samples(cat_dir: Path):
    """train (good) + test (good+defects); returns (train_list, pool_list) of (rgb,ir,pcd,label)."""
    def collect(globpat, label_fn):
        out = []
        for rgb in sorted(glob.glob(globpat)):
            if os.path.basename(rgb).startswith("._"):
                continue
            rgb = Path(rgb)
            r, ir, pcd = _paths(cat_dir, rgb)
            if ir.exists() and pcd.exists():
                out.append((r, ir, pcd, label_fn(rgb)))
        return out
    train = collect(str(cat_dir / "RGB/train/*.png"), lambda p: 0)
    pool = []
    for sub in sorted(glob.glob(str(cat_dir / "RGB/test/*"))):
        if os.path.basename(sub).startswith("._") or not os.path.isdir(sub):
            continue
        lab = 0 if os.path.basename(sub) == "good" else 1
        pool += collect(os.path.join(sub, "*.png"), lambda p, lab=lab: lab)
    return train, pool


def build_category(cat: str, coreset=4096):
    from sklearn.metrics import roc_auc_score
    from uais.fusion.attention.realiad_3d_detector import score_one_class_patchcore
    cat_dir = RAW / cat
    train, pool = _samples(cat_dir)
    if len([s for s in pool if s[3] == 1]) < 5 or len([s for s in pool if s[3] == 0]) < 5 or len(train) < 5:
        return None
    y = np.array([s[3] for s in pool])
    idx = np.arange(len(pool)); val_idx = []
    for lab in (0, 1):
        li = idx[y == lab]; val_idx += list(li[: len(li) // 2])
    val_mask = np.zeros(len(pool), bool); val_mask[val_idx] = True
    yval, ytest = y[val_mask], y[~val_mask]
    if len(np.unique(yval)) < 2 or len(np.unique(ytest)) < 2:
        return None

    loaders = {"rgb": (_load_img, 0), "ir": (_load_img, 1), "pcd": (_load_pcd_depth, 2)}
    Sval_cols, Stest_cols, vauc = [], [], []
    for m, (load, col) in loaders.items():
        bank = [load(s[col]) for s in train]
        ref = score_one_class_patchcore(bank[: min(40, len(bank))], bank, coreset_size=coreset)
        evs = [load(s[col]) for s in pool]
        sc = _zsig(score_one_class_patchcore(bank, evs, coreset_size=coreset), ref)
        Sval_cols.append(sc[val_mask]); Stest_cols.append(sc[~val_mask])
        vauc.append(float(roc_auc_score(yval, sc[val_mask])) if len(np.unique(yval)) > 1 else 0.5)
    return np.column_stack(Sval_cols), yval, np.column_stack(Stest_cols), ytest, np.array(vauc, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--coreset", type=int, default=4096)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    cats = args.categories or sorted(p.name for p in RAW.iterdir()
                                     if p.is_dir() and not p.name.startswith("._"))
    n = 0
    for cat in cats:
        try:
            res = build_category(cat, coreset=args.coreset)
        except Exception as e:
            print(f"[{cat}] FAILED: {type(e).__name__}: {e}", flush=True); continue
        if res is None:
            print(f"[{cat}] skipped (insufficient labelled split)", flush=True); continue
        Sval, yval, Stest, ytest, vauc = res
        np.savez(CACHE / f"{cat}.npz", Sval=Sval, yval=yval, Stest=Stest, ytest=ytest, valauc=vauc)
        n += 1
        print(f"[{cat}] cached val={len(yval)} test={len(ytest)} valauc(rgb,ir,pcd)={np.round(vauc,3)}", flush=True)
    print(f"\nwrote {n} MulSen-AD category caches to {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
