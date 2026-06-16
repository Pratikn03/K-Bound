"""Ingest OpenOOD into the Gate U / honest_benchmark input format (image-OOD family).

Converts OpenOOD ID-vs-OOD image sets into ADBench-style per-task feature npz
``{X: float[n, d], y: int[n]}`` (y = 0 in-distribution, 1 OOD), the exact format
``gate_u_seed_eval.load_tasks`` consumes for the ``image_ood`` family (ResNet-18
features, like the existing ``data/raw/adbench_cv`` tasks). Optionally (--to-archive) it
also emits matching ``experiments/elara_u/score_archive/<prefix><task>.npz`` records --
additively, never overwriting an existing file and never touching the manifest -- so
``honest_benchmark.py`` (which globs the archive) picks the new tasks up.

WIRING: ``gate_u_seed_eval.load_tasks`` scans a FIXED set of raw subdirs and does not
auto-discover ``--out``; the wired route into the 123-task pipeline is ``--to-archive``
(emits score_archive npz that ``honest_benchmark.py`` globs). The ``{X,y}`` npz are the
right schema for any X/y consumer; to fold them into ``gate_u_seed_eval`` specifically,
point that script at ``--out`` or copy the npz into a scanned dir. See PHASE2_RUNBOOK.md.

SAFETY / TIGHT DISK:
  Nothing downloads unless ``--download`` is passed AND the preflight shows the data
  clearly fits the (nearly full) exFAT drive. By default the script stages the exact
  download command in the runbook and exits. Feature extraction (torch / torchvision
  ResNet-18) is lazy-imported and runs only when invoked -- not on import, and not under
  ``--dry-run``. Run the heavy steps only after the RxRx1 GPU job has finished; pass
  ``--device mps`` (or cuda) then for speed. Never fabricates features.

OpenOOD download (run after freeing the drive; see PHASE2_RUNBOOK.md):
  git clone https://github.com/Jingkang50/OpenOOD && cd OpenOOD
  python ./scripts/download/download.py --contents images --datasets default \
      --save_dir <RAW_ROOT>          # CIFAR OOD suite ~5-10 GB; ImageNet OOD ~150+ GB
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data/raw/openood"          # staged dataset root (id/ + ood/)
OUT_DIR = ROOT / "data/raw/openood_tasks"     # per-task {X,y} npz (ADBench format)
ARCHIVE = ROOT / "experiments/elara_u/score_archive"
DOMAIN, PREFIX = "image_ood", "openood_"
EST_GB = 12.0      # conservative estimate for the default CIFAR OOD suite (images)
MARGIN = 1.5       # require free >= EST_GB * MARGIN to call it "clearly fits"
MIN_N, MIN_POS = 80, 12  # load_tasks task-validity floor


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
    """ResNet-18 (ImageNet-pretrained) global-avg-pool 512-d features. Lazy heavy import."""
    import torch
    import torchvision
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch):
            ims = []
            for p in image_paths[i:i + batch]:
                im = Image.open(p)
                ims.append(tf(im.convert("RGB")))
            x = torch.stack(ims).to(device)
            feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, 0).astype(np.float32)


def _imgs(d: Path):
    if not d.exists():
        return []
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return [p for p in sorted(d.rglob("*")) if p.suffix.lower() in exts and not p.name.startswith("._")]


def discover_tasks(raw_root: Path):
    """OpenOOD staged layout -> [(task_name, id_dir, ood_dir)]. ID test images under
    <raw>/id/test (or <raw>/id), each OOD set under <raw>/ood/<name>."""
    id_dir = raw_root / "id" / "test"
    if not id_dir.exists():
        id_dir = raw_root / "id"
    ood_root = raw_root / "ood"
    tasks = []
    if ood_root.exists():
        for od in sorted(p for p in ood_root.iterdir() if p.is_dir() and not p.name.startswith("._")):
            tasks.append((od.name, id_dir, od))
    return tasks


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
    tasks = discover_tasks(raw_root)
    if not tasks:
        print(f"DATA NEEDED: no OpenOOD tasks under {raw_root}\n"
              "  Expected: <raw>/id/test/*  and  <raw>/ood/<set>/*  (see header for download).\n"
              "  Staging only -- nothing fabricated.")
        raise SystemExit(2)
    out_dir.mkdir(parents=True, exist_ok=True)
    id_paths = _imgs(tasks[0][1])
    if not id_paths:
        print(f"DATA NEEDED: no ID images under {tasks[0][1]}")
        raise SystemExit(2)
    print(f"[features] extracting ID features ({len(id_paths)} imgs) on device={device} ...")
    Xid = resnet18_features(id_paths, device=device)
    n = 0
    for name, _id_dir, ood_dir in tasks:
        ood_paths = _imgs(ood_dir)
        if len(ood_paths) < MIN_POS:
            print(f"[{name}] skipped (only {len(ood_paths)} OOD imgs)")
            continue
        Xood = resnet18_features(ood_paths, device=device)
        X = np.vstack([Xid, Xood])
        y = np.concatenate([np.zeros(len(Xid), int), np.ones(len(Xood), int)])
        if X.shape[0] < MIN_N or int(y.sum()) < MIN_POS:
            print(f"[{name}] skipped (n={X.shape[0]}, pos={int(y.sum())})")
            continue
        np.savez(out_dir / f"{name}.npz", X=X, y=y)
        n += 1
        print(f"[{name}] wrote {X.shape} pos={int(y.sum())}")
        if to_arch:
            to_archive(name, X, y)
    print(f"\nwrote {n} OpenOOD tasks -> {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest OpenOOD -> Gate U image-OOD tasks.")
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
        print("NOTE: automated OpenOOD download is intentionally not run inline; use the "
              "staged command in the header/runbook, then re-run without --download to ingest.")
        raise SystemExit(3)
    return ingest(raw_root, Path(args.out), args.device, args.to_archive)


if __name__ == "__main__":
    raise SystemExit(main())
