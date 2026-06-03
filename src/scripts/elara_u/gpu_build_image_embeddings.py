"""GPU: extract ResNet-50 embeddings for an image anomaly category -> ADBench-format
(X, y) npz, so industrial-vision families (MVTec AD, VisA) drop into the 123-task
single-input benchmark exactly like the existing image-OOD (ADBench CV) family.

Generic by design (no fragile per-dataset parsing): you point it at a set of NORMAL
images and a set of ANOMALY images; it global-average-pools ResNet-50 features
(2048-d) and writes X[n,2048], y[n] to an .npz. Add the resulting files under
data/raw/adbench_industrial/ to gate_u_seed_eval.load_tasks (a new family) and rebuild
the score archive; honest_benchmark then evaluates them with the same detector zoo.

Examples (run on the GPU box; see research_lock/GPU_EXPERIMENTS_PROTOCOL_v1.md):
  # MVTec AD category 'bottle'
  python src/scripts/elara_u/gpu_build_image_embeddings.py \
    --normal 'data/raw/mvtec_ad/bottle/train/good/*.png' 'data/raw/mvtec_ad/bottle/test/good/*.png' \
    --anomaly 'data/raw/mvtec_ad/bottle/test/*/*.png' --exclude-anomaly-glob '*/good/*' \
    --out data/raw/adbench_industrial/mvtecad_bottle.npz
  # VisA category 'candle'
  python src/scripts/elara_u/gpu_build_image_embeddings.py \
    --normal 'data/raw/visa/candle/Data/Images/Normal/*.JPG' \
    --anomaly 'data/raw/visa/candle/Data/Images/Anomaly/*.JPG' \
    --out data/raw/adbench_industrial/visa_candle.npz

NOT run in the dev session (no CUDA/data here); verify on the first GPU run.
"""

from __future__ import annotations

import argparse
import glob
import fnmatch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _backbone():
    import torch
    from torchvision.models import ResNet50_Weights, resnet50
    w = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=w).eval()
    model.fc = torch.nn.Identity()                      # 2048-d global-avg-pool embedding
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    return model.to(dev), w.transforms(), dev


def _embed(paths, model, transform, dev, batch=32):
    import torch
    from PIL import Image
    feats = []
    with torch.no_grad():
        buf = []
        for p in paths:
            im = Image.open(p); im = im.convert("RGB") if im.mode != "RGB" else im
            buf.append(transform(im))
            if len(buf) == batch:
                feats.append(model(torch.stack(buf).to(dev)).cpu().numpy()); buf = []
        if buf:
            feats.append(model(torch.stack(buf).to(dev)).cpu().numpy())
    return np.concatenate(feats, 0) if feats else np.zeros((0, 2048), np.float32)


def _expand(globs, exclude):
    out = []
    for g in globs:
        out += [p for p in glob.glob(g, recursive=True) if "/._" not in p]
    if exclude:
        out = [p for p in out if not fnmatch.fnmatch(p, exclude)]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", nargs="+", required=True, help="glob(s) of normal images")
    ap.add_argument("--anomaly", nargs="+", required=True, help="glob(s) of anomaly images")
    ap.add_argument("--exclude-anomaly-glob", default=None, help="drop anomaly paths matching this (e.g. */good/*)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    normal = _expand(args.normal, None)
    anomaly = _expand(args.anomaly, args.exclude_anomaly_glob)
    anomaly = [p for p in anomaly if p not in set(normal)]
    if len(normal) < 20 or len(anomaly) < 5:
        raise SystemExit(f"too few images (normal={len(normal)}, anomaly={len(anomaly)})")

    model, transform, dev = _backbone()
    Xn = _embed(normal, model, transform, dev)
    Xa = _embed(anomaly, model, transform, dev)
    X = np.concatenate([Xn, Xa], 0).astype(np.float32)
    y = np.concatenate([np.zeros(len(Xn), int), np.ones(len(Xa), int)])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=y)
    print(f"wrote {out}: X{X.shape} normal={len(Xn)} anomaly={len(Xa)} rate={y.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
