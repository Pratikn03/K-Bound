"""De-risk the hardest fix: measure per-modality PatchCore AUROC on a few
Real-IAD-D3 categories before committing to the full 259 GB run.

Baseline (handcrafted scorer, pooled): rgb 0.52, ps 0.55, xyz 0.48.
Target: deep PatchCore on rgb/ps and surface-normal PatchCore on xyz clears
chance per-category (especially xyz, which was below chance).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.scenario_c.prepare_realiad_d3_headroom_inputs import _load_rows  # noqa: E402
from uais.fusion.attention.realiad_3d_detector import (  # noqa: E402
    load_modality_image,
    score_one_class_patchcore,
)

MODALITIES = ("rgb", "ps", "xyz")


def _zsig(raw: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu = float(np.mean(ref))
    sd = float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def run(categories: list[str], coreset: int, max_train: int, max_eval: int) -> dict:
    rows, cats = _load_rows(ROOT / "data/raw/realiad_d3", categories)
    zip_dir = ROOT / "data/raw/realiad_d3" / "realiad_d3_raw"
    report: dict = {"categories": cats, "coreset": coreset, "per_modality": {}, "per_category": {}}

    for modality in MODALITIES:
        per_cat_auc = []
        for category in cats:
            crows = [r for r in rows if r.category == category and r.modality == modality]
            train = [r for r in crows if r.split == "train" and r.label == 0]
            ev = [r for r in crows if r.split == "test"]
            if max_train:
                train = train[:max_train]
            if max_eval:
                # stratified cap: keep both classes (plain head-truncation can
                # leave a single class and silently skip the category).
                pos = [r for r in ev if r.label == 1][: max_eval // 2]
                neg = [r for r in ev if r.label == 0][: max_eval // 2]
                ev = pos + neg
            if len(train) < 10 or len(ev) < 10 or len({r.label for r in ev}) < 2:
                continue
            t0 = time.time()
            with zipfile.ZipFile(zip_dir / f"{category}.zip") as zf:
                train_imgs = [load_modality_image(zf, r.zip_member, modality) for r in train]
                eval_imgs = [load_modality_image(zf, r.zip_member, modality) for r in ev]
            raw = score_one_class_patchcore(train_imgs, eval_imgs, coreset_size=coreset)
            ref = score_one_class_patchcore(train_imgs[: min(40, len(train_imgs))], train_imgs, coreset_size=coreset)
            scores = _zsig(raw, ref)
            y = np.array([r.label for r in ev], dtype=int)
            auc = float(roc_auc_score(y, scores))
            per_cat_auc.append(auc)
            report["per_category"].setdefault(category, {})[modality] = {
                "auc": round(auc, 4), "n_train": len(train), "n_eval": len(ev),
                "secs": round(time.time() - t0, 1),
            }
            print(f"  [{modality:>3}] {category:<28} AUROC={auc:.4f}  "
                  f"(train {len(train)}, eval {len(ev)}, {time.time()-t0:.0f}s)")
        if per_cat_auc:
            report["per_modality"][modality] = {
                "mean_auc": round(float(np.mean(per_cat_auc)), 4),
                "n_categories": len(per_cat_auc),
            }
            print(f"==> {modality}: mean within-category AUROC = {np.mean(per_cat_auc):.4f} "
                  f"over {len(per_cat_auc)} categories")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories", nargs="*", default=["ferrite_bead", "fuse_holder", "knob_cap"])
    ap.add_argument("--coreset", type=int, default=4096)
    ap.add_argument("--max-train", type=int, default=120, help="cap train-OK per category (speed)")
    ap.add_argument("--max-eval", type=int, default=160, help="cap eval per category (speed)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/fusion/realiad_d3_strong_detector_probe.json")
    args = ap.parse_args()
    print(f"=== Real-IAD-D3 strong-detector probe: {args.categories} ===")
    rep = run(args.categories, args.coreset, args.max_train, args.max_eval)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))
    print("\n=== SUMMARY (baseline handcrafted pooled: rgb 0.52 / ps 0.55 / xyz 0.48) ===")
    for m, d in rep["per_modality"].items():
        print(f"  {m}: mean within-cat AUROC {d['mean_auc']} ({d['n_categories']} cats)")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
