"""Build MVTec 3D-AD v3 fusion inputs with TRUE patch-level PatchCore scores.

Keeps the EXACT (sample_id, split, label, domain, source_path) rows of the
existing supervised-paired v2 CSV so the comparison is apples-to-apples; only
the upstream `score` and `embedding_*` columns are recomputed using the
patch-level PatchCore detector (uais.fusion.attention.patchcore_patch).

Memory bank per (category, modality) = patch embeddings of that category's
train/good images ONLY (honest one-class normal reference). Scores are
min-max normalised against the train-good score distribution per category.

Output: experiments/fusion/mvtec3d_patchcore_v3_inputs.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.decomposition import PCA  # noqa: E402

from uais.fusion.attention.patchcore_patch import (  # noqa: E402
    extract_patch_embeddings,
    greedy_coreset,
    image_anomaly_scores,
)

V2_CSV = ROOT / "experiments/fusion/mvtec3d_patchcore_supervised_paired_inputs.csv"
OUT_CSV = ROOT / "experiments/fusion/mvtec3d_patchcore_v3_inputs.csv"
DATA = ROOT / "data/raw/mvtec3d"
EMBED_DIM = 16
PATCH_GRID = 28


def _bank_paths(category: str, domain: str) -> list[Path]:
    """Train-good image paths for a category/modality (the one-class memory bank)."""
    sub = "rgb" if domain == "rgb" else "xyz"
    d = DATA / category / "train" / "good" / sub
    ext = "*.png" if sub == "rgb" else "*.tiff"
    return sorted(d.glob(ext))


def main() -> int:
    """Rebuild the MVTec 3D-AD fusion CSV with patch-level PatchCore scores,
    keeping v2 rows/splits/labels and swapping only score + embeddings."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coreset-frac", type=float, default=0.10)
    ap.add_argument("--only", default=None, help="comma-separated category subset")
    args = ap.parse_args()

    df = pd.read_csv(V2_CSV)
    cats = sorted(df["category"].unique())
    if args.only:
        cats = [c.strip() for c in args.only.split(",")]

    out_rows = []
    for cat in cats:
        for domain in ("rgb", "depth_or_xyz"):
            sub = df[(df["category"] == cat) & (df["domain"] == domain)].copy()
            if sub.empty:
                continue
            # Build the train-good memory bank for this (cat, modality).
            bank_paths = _bank_paths(cat, domain)
            if not bank_paths:
                print(f"[{cat}/{domain}] no train/good images; skipping", flush=True)
                continue
            print(f"[{cat}/{domain}] bank imgs={len(bank_paths)}  query rows={len(sub)}", flush=True)
            bank_patches, P = extract_patch_embeddings(bank_paths, patch_grid=PATCH_GRID)
            target = max(P * 4, int(args.coreset_frac * bank_patches.shape[0]))
            bank = greedy_coreset(bank_patches, target, seed=0)

            # Score every query image at its source_path.
            query_paths = [ROOT / p for p in sub["source_path"].tolist()]
            q_patches, _ = extract_patch_embeddings(query_paths, patch_grid=PATCH_GRID)
            raw_scores = image_anomaly_scores(q_patches, P, bank)

            # Normalisation reference: score a subsample of bank IMAGES against
            # the coreset to estimate the train-good score mean/std.
            n_bank_img = bank_patches.shape[0] // P
            ref_n = min(60, n_bank_img)
            ref_patches = bank_patches[: ref_n * P]
            bank_self = image_anomaly_scores(ref_patches, P, bank)
            mu = float(np.mean(bank_self))
            sd = float(np.std(bank_self) + 1e-8)
            # MONOTONIC normalisation: z-score vs train-good then logistic
            # sigmoid. This preserves the raw-score ranking exactly (so the
            # per-category AUROC equals the raw-score AUROC) while mapping to a
            # calibrated [0,1] for the fusion layer (0.5 at the train-good mean).
            z = (raw_scores - mu) / sd
            norm_scores = 1.0 / (1.0 + np.exp(-z))

            # PCA-16 embeddings: pool each query image's patches (mean) then PCA
            # fit on the bank's per-image pooled vectors.
            n_q = len(query_paths)
            q_pool = q_patches[: n_q * P].reshape(n_q, P, -1).mean(axis=1)
            n_b = bank_patches.shape[0] // P
            b_pool = bank_patches[: n_b * P].reshape(n_b, P, -1).mean(axis=1)
            k = min(EMBED_DIM, b_pool.shape[0], b_pool.shape[1])
            pca = PCA(n_components=k, random_state=0).fit(b_pool)
            emb = pca.transform(q_pool)
            # scale to [0,1] vs bank
            bemb = pca.transform(b_pool)
            elo, ehi = bemb.min(axis=0), bemb.max(axis=0)
            espan = np.where((ehi - elo) > 1e-9, ehi - elo, 1.0)
            emb = np.clip((emb - elo) / espan, 0.0, 1.0)

            sub = sub.reset_index(drop=True)
            for i, (_, row) in enumerate(sub.iterrows()):
                r = dict(row)
                r["score"] = float(norm_scores[i])
                r["raw_patchcore_score"] = float(raw_scores[i])
                for j in range(EMBED_DIM):
                    r[f"embedding_{j}"] = float(emb[i, j]) if j < k else 0.0
                out_rows.append(r)

    out = pd.DataFrame(out_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(out)} rows)")

    # Upstream sanity: per-category test AUROC (standard MVTec 3D-AD protocol),
    # averaged, plus pooled for reference.
    from sklearn.metrics import roc_auc_score
    for domain in ("rgb", "depth_or_xyz"):
        sub = out[(out["domain"] == domain) & (out["split"] == "test")]
        per_cat = []
        for _, g in sub.groupby("category"):
            if g["label"].nunique() > 1:
                per_cat.append(roc_auc_score(g["label"], g["raw_patchcore_score"]))
        if per_cat:
            print(f"  upstream {domain}: mean per-category test AUROC = {np.mean(per_cat):.4f} "
                  f"(n_cat={len(per_cat)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
