"""GPU: build a D23-format multimodal score cache for MVTec-3D-AD (rgb + xyz).

A second multimodal dataset to corroborate D23 (Real-IAD-D3). Per category, builds
one-class PatchCore memory banks from train/good for two modalities -- RGB (png) and
XYZ (organized point-cloud tiff -> surface-normal image) -- and scores a labelled
val/test split drawn from validation/good + test/{good,defects}. Writes per-category
caches in the same {Sval,yval,Stest,ytest,valauc} format that
multimodal_reliability_test.py consumes:

    PYTHONPATH=src python src/scripts/elara_u/gpu_build_mvtec3d_cache.py        # build cache
    PYTHONPATH=src python src/scripts/elara_u/multimodal_reliability_test.py \
        --cache experiments/fusion/mvtec3d_score_cache --glob '*.npz' --tag MVTec-3D \
        --out experiments/elara_u/multimodal_reliability_results_mvtec3d.json   # run D23

Requires a GPU + torch (the ResNet-50 PatchCore backbone) and the MVTec-3D data under
data/raw/mvtec3d/{category}/{train,validation,test}/.../{rgb,xyz}/. Reuses the proven
score_one_class_patchcore + xyz_to_normal_image; only data loading is new. NOT run in
the dev session (no CUDA/data here) -- verify on the first GPU run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/mvtec3d"
CACHE = ROOT / "experiments/fusion/mvtec3d_score_cache"
RNG = 0


def _zsig(raw, ref):
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def _load_rgb(p: Path):
    from PIL import Image
    im = Image.open(p)
    return im.convert("RGB") if im.mode != "RGB" else im


def _load_xyz(p: Path):
    """Organized point-cloud tiff -> surface-normal image (reuses realiad detector)."""
    import tifffile
    from uais.fusion.attention.realiad_3d_detector import xyz_to_normal_image
    arr = np.asarray(tifffile.imread(p), dtype=np.float32)
    return xyz_to_normal_image(arr)


def _load_depth(p: Path):
    """xyz tiff whose 3 channels are a *replicated* single-channel depth map (3D-ADAM
    'anomalib' layout) -> contrast-stretched depth image. Surface normals are degenerate
    here (identical channels), so the geometry modality is scored as a depth image, the
    same way MVTec-3D's depth is consumed by PatchCore. Verified to carry real signal."""
    import tifffile
    from PIL import Image
    a = np.asarray(tifffile.imread(p), dtype=np.float32)
    d = a[..., 0] if a.ndim == 3 else a
    m = d > 0
    if int(m.sum()) > 10:
        lo, hi = np.percentile(d[m], 2), np.percentile(d[m], 98)
        d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    return Image.fromarray((d * 255).astype(np.uint8)).convert("RGB")


def _samples(cat_dir: Path, split: str):
    """Return list of (rgb_path, xyz_path, label) for a split. label 1 = defect."""
    out = []
    base = cat_dir / split
    if not base.exists():
        return out
    for defect_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        label = 0 if defect_dir.name == "good" else 1
        rgb_dir, xyz_dir = defect_dir / "rgb", defect_dir / "xyz"
        if not rgb_dir.exists():
            continue
        for rgb in sorted(rgb_dir.glob("*.png")):
            xyz = xyz_dir / (rgb.stem + ".tiff")
            if xyz.exists():
                out.append((rgb, xyz, label))
    return out


def build_category(cat: str, coreset=4096, cap=130, geom="normal"):
    from uais.fusion.attention.realiad_3d_detector import score_one_class_patchcore
    cat_dir = RAW / cat
    train = _samples(cat_dir, "train")            # all good
    pool = _samples(cat_dir, "validation") + _samples(cat_dir, "test")
    if len([s for s in pool if s[2] == 1]) < 5 or len([s for s in pool if s[2] == 0]) < 5:
        return None
    rng = np.random.default_rng(RNG)
    pool = [pool[i] for i in rng.permutation(len(pool))[:cap]]
    y = np.array([s[2] for s in pool])
    # stratified 50/50 val/test split
    idx = np.arange(len(pool)); val_idx = []
    for lab in (0, 1):
        li = idx[y == lab]; val_idx += list(li[: len(li) // 2])
    val_mask = np.zeros(len(pool), bool); val_mask[val_idx] = True
    yval, ytest = y[val_mask], y[~val_mask]
    if len(np.unique(yval)) < 2 or len(np.unique(ytest)) < 2:
        return None

    Sval_cols, Stest_cols, vauc = [], [], []
    from sklearn.metrics import roc_auc_score
    geom_loader = _load_depth if geom == "depth" else _load_xyz
    loaders = {"rgb": (_load_rgb, 0), "xyz": (geom_loader, 1)}
    for m, (load, col) in loaders.items():
        bank = [load(s[col]) for s in train]
        ref = score_one_class_patchcore(bank[: min(40, len(bank))], bank, coreset_size=coreset)
        evs = [load(s[col]) for s in pool]
        sc = _zsig(score_one_class_patchcore(bank, evs, coreset_size=coreset), ref)
        Sval_cols.append(sc[val_mask]); Stest_cols.append(sc[~val_mask])
        vauc.append(float(roc_auc_score(yval, sc[val_mask])) if len(np.unique(yval)) > 1 else 0.5)
    Sval = np.column_stack(Sval_cols); Stest = np.column_stack(Stest_cols)
    return Sval, yval, Stest, ytest, np.array(vauc, float)


def main():
    global RAW, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=None, help="default: all under --raw")
    ap.add_argument("--coreset", type=int, default=4096)
    ap.add_argument("--raw", default=str(RAW), help="dataset root (MVTec-3D-style train/val/test rgb+xyz)")
    ap.add_argument("--cache", default=str(CACHE), help="output score-cache dir")
    ap.add_argument("--geom", choices=["normal", "depth"], default="normal",
                    help="geometry modality: 'normal'=organized-pcd surface normals (MVTec-3D); "
                         "'depth'=contrast-stretched depth image (3D-ADAM replicated-channel tiffs)")
    args = ap.parse_args()
    RAW = Path(args.raw); CACHE = Path(args.cache)
    CACHE.mkdir(parents=True, exist_ok=True)
    cats = args.categories or sorted(p.name for p in RAW.iterdir()
                                     if p.is_dir() and not p.name.startswith("._"))
    n = 0
    for cat in cats:
        try:
            res = build_category(cat, coreset=args.coreset, geom=args.geom)
        except Exception as e:
            print(f"[{cat}] FAILED: {e}"); continue
        if res is None:
            print(f"[{cat}] skipped (insufficient labelled split)"); continue
        Sval, yval, Stest, ytest, vauc = res
        np.savez(CACHE / f"{cat}.npz", Sval=Sval, yval=yval, Stest=Stest, ytest=ytest, valauc=vauc)
        n += 1
        print(f"[{cat}] cached  val={len(yval)} test={len(ytest)} valauc(rgb,xyz)={np.round(vauc,3)}", flush=True)
    print(f"\nwrote {n} MVTec-3D category caches to {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
