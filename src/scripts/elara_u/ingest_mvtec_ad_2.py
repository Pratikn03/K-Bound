"""Ingest MVTec AD 2 into the Gate U / honest_benchmark input format (industrial family).

Converts each MVTec AD 2 category (normal vs defect) into an ADBench-style per-task
feature npz ``{X: float[n, d], y: int[n]}`` (y = 0 normal, 1 defect) -- the exact format
``gate_u_seed_eval.load_tasks`` consumes for the ``industrial`` family. Features are
ResNet-18 global-avg-pool 512-d vectors (consistent with the image-OOD family). With
``--to-archive`` it also emits additive ``experiments/elara_u/score_archive`` records
(new files only; never overwrites; never edits the manifest) so ``honest_benchmark.py``
picks the new tasks up.

WIRING: ``gate_u_seed_eval.load_tasks`` scans a FIXED set of raw subdirs and does not
auto-discover ``--out``; the wired route into the 123-task pipeline is ``--to-archive``
(emits score_archive npz that ``honest_benchmark.py`` globs). The ``{X,y}`` npz are the
right schema for any X/y consumer; to fold them into ``gate_u_seed_eval`` specifically,
point that script at ``--out`` or copy the npz into a scanned dir. See PHASE2_RUNBOOK.md.

SAFETY / TIGHT DISK: nothing downloads unless ``--download`` is passed AND the preflight
shows the data clearly fits the (nearly full) exFAT drive; otherwise the exact command is
staged in the runbook and the script exits. Torch/torchvision are lazy-imported and run
only when invoked (not on import, not under ``--dry-run``). Run heavy steps after the
RxRx1 GPU job finishes; pass ``--device mps``/``cuda`` then. Never fabricates features.

MVTec AD 2 download (license-gated; run after freeing the drive; see PHASE2_RUNBOOK.md):
  Register + download from https://www.mvtec.com/company/research/datasets/mvtec-ad-2
  Unpack to <RAW_ROOT> as <raw>/<category>/{train/good, test*/{good, <defects>}}/  (~20 GB)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data/raw/mvtec_ad_2"
OUT_DIR = ROOT / "data/raw/mvtec_ad_2_tasks"
ARCHIVE = ROOT / "experiments/elara_u/score_archive"
DOMAIN, PREFIX = "industrial", "mvtecad2_"
EST_GB = 20.0
MARGIN = 1.5
MIN_N, MIN_POS = 80, 12


def disk_free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return float("nan")


def preflight(raw_root: Path) -> bool:
    free = disk_free_gb(ROOT)
    fits = free >= EST_GB * MARGIN
    print(f"[preflight] free={free:.0f} GB  est_needed~{EST_GB:.0f} GB (x{MARGIN} margin)  "
          f"clearly_fits={fits}")
    print(f"[preflight] staged raw root: {raw_root}  present={raw_root.exists()}")
    return fits


def resnet18_features(image_paths, device="cpu", batch=64):
    """ResNet-18 (ImageNet-pretrained) 512-d pooled features. Lazy heavy import."""
    import torch
    import torchvision
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in image_paths[i:i + batch]]).to(device)
            feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, 0).astype(np.float32)


def _imgs(d: Path):
    if not d.exists():
        return []
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return [p for p in sorted(d.rglob("*")) if p.suffix.lower() in exts and not p.name.startswith("._")]


def category_paths(cat_dir: Path):
    """Return (normal_paths, defect_paths) for a MVTec-AD-2-style category."""
    normal, defect = [], []
    for split in cat_dir.iterdir() if cat_dir.exists() else []:
        if not split.is_dir() or split.name.startswith("._"):
            continue
        good = split / "good"
        if good.exists():
            normal += _imgs(good)
        for sub in split.iterdir():
            if sub.is_dir() and sub.name != "good" and not sub.name.startswith("._"):
                defect += _imgs(sub)
    return normal, defect


def to_archive(name, X, y):
    out = ARCHIVE / f"{PREFIX}{name}.npz"
    if out.exists():
        print(f"  [{name}] archive npz exists -> skip (additive; never overwrite)")
        return
    from scripts.elara_u.build_score_archive import build_task  # lazy (pyod)

    Sval, yval, Stest, ytest, det_names, vauc = build_task(X, y)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    np.savez(out, Sval=Sval, yval=yval, Stest=Stest, ytest=ytest,
             det_names=det_names, val_auc=vauc, domain=DOMAIN)
    print(f"  [{name}] wrote archive {out.name}")


def ingest(raw_root, out_dir, device, to_arch):
    cats = [p for p in sorted(raw_root.iterdir()) if p.is_dir() and not p.name.startswith("._")] \
        if raw_root.exists() else []
    if not cats:
        print(f"DATA NEEDED: no MVTec AD 2 categories under {raw_root}\n"
              "  Expected: <raw>/<category>/{train/good, test*/{good,<defects>}}/  (see header).\n"
              "  Staging only -- nothing fabricated.")
        raise SystemExit(2)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cat_dir in cats:
        normal, defect = category_paths(cat_dir)
        if len(normal) < MIN_POS or len(defect) < MIN_POS:
            print(f"[{cat_dir.name}] skipped (normal={len(normal)}, defect={len(defect)})")
            continue
        print(f"[features] {cat_dir.name}: {len(normal)} normal + {len(defect)} defect on device={device}")
        Xn = resnet18_features(normal, device=device)
        Xd = resnet18_features(defect, device=device)
        X = np.vstack([Xn, Xd])
        y = np.concatenate([np.zeros(len(Xn), int), np.ones(len(Xd), int)])
        if X.shape[0] < MIN_N or int(y.sum()) < MIN_POS:
            print(f"[{cat_dir.name}] skipped (n={X.shape[0]}, pos={int(y.sum())})")
            continue
        np.savez(out_dir / f"{cat_dir.name}.npz", X=X, y=y)
        n += 1
        print(f"[{cat_dir.name}] wrote {X.shape} pos={int(y.sum())}")
        if to_arch:
            to_archive(cat_dir.name, X, y)
    print(f"\nwrote {n} MVTec AD 2 tasks -> {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest MVTec AD 2 -> Gate U industrial tasks.")
    ap.add_argument("--raw-root", default=str(RAW_ROOT))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda (use mps/cuda AFTER RxRx1)")
    ap.add_argument("--download", action="store_true", help="download only if preflight says it clearly fits")
    ap.add_argument("--to-archive", action="store_true",
                    help="also emit additive score_archive npz so honest_benchmark picks them up")
    ap.add_argument("--dry-run", action="store_true", help="preflight only: disk + presence, no download/extract")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    fits = preflight(raw_root)
    if args.dry_run:
        print("dry-run: no download, no extraction. "
              + ("raw present -> ready to ingest." if raw_root.exists() else "raw absent -> stage download (see header / runbook)."))
        return 0
    if args.download:
        if not fits:
            print("REFUSING to download: data does not clearly fit the drive. "
                  "Free space first or stage manually (see header / PHASE2_RUNBOOK.md).")
            raise SystemExit(3)
        print("NOTE: MVTec AD 2 is license-gated; automated download is not run inline. "
              "Use the staged registration/download (header/runbook), then re-run to ingest.")
        raise SystemExit(3)
    return ingest(raw_root, Path(args.out), args.device, args.to_archive)


if __name__ == "__main__":
    raise SystemExit(main())
