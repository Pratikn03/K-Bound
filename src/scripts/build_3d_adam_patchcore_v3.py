"""Build 3D-ADAM v3 transfer fusion inputs with TRUE patch-level PatchCore.

Held-out EXTERNAL transfer benchmark (D6). Keeps the EXACT rows of the existing
m2_external_3d_adam_sealed_inputs.csv (so the comparison to the weak-detector
-0.038 result is apples-to-apples) and only recomputes the upstream `score`
and `embedding_*` columns with the patch-level PatchCore detector.

Memory bank per (category, modality) = that category's train/good images only.
Monotonic z-sigmoid normalisation vs the train-good score distribution.

Output: experiments/fusion/m2_external_3d_adam_v3_inputs.csv
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

V2_CSV = ROOT / "experiments/fusion/m2_external_3d_adam_sealed_inputs.csv"
OUT_CSV = ROOT / "experiments/fusion/m2_external_3d_adam_v3_inputs.csv"
DATA = ROOT / "data/raw/3d_adam_anomalib"
EMBED_DIM = 16
PATCH_GRID = 28


def _bank_paths(category: str, domain: str) -> list[Path]:
    sub = "rgb" if domain == "rgb" else "xyz"
    d = DATA / category / "train" / "good" / sub
    ext = "*.png" if sub == "rgb" else "*.tiff"
    return sorted(p for p in d.glob(ext) if not p.name.startswith("._"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coreset-frac", type=float, default=0.10)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    df = pd.read_csv(V2_CSV)
    cats = sorted(df["category"].unique())
    if args.only:
        cats = [c.strip() for c in args.only.split(",")]

    # Per-category checkpoint dir so an external-drive disconnect cannot wipe
    # the whole build. Each category writes its own parquet; we assemble at end
    # and skip categories already checkpointed (resume support).
    ckpt_dir = OUT_CSV.parent / "_adam_v3_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for cat in cats:
        ck = ckpt_dir / f"{cat}.parquet"
        if ck.exists():
            print(f"[{cat}] resume: loading checkpoint", flush=True)
            out_rows.extend(pd.read_parquet(ck).to_dict("records"))
            continue
        cat_rows = []
        for domain in ("rgb", "depth_or_xyz"):
            sub = df[(df["category"] == cat) & (df["domain"] == domain)].copy()
            if sub.empty:
                continue
            bank_paths = _bank_paths(cat, domain)
            if not bank_paths:
                print(f"[{cat}/{domain}] no train/good; skip", flush=True)
                continue
            print(f"[{cat}/{domain}] bank={len(bank_paths)} query={len(sub)}", flush=True)
            bank_patches, P = extract_patch_embeddings(bank_paths, patch_grid=PATCH_GRID)
            target = max(P * 4, int(args.coreset_frac * bank_patches.shape[0]))
            bank = greedy_coreset(bank_patches, target, seed=0)

            query_paths = [ROOT / p for p in sub["source_path"].tolist()]
            q_patches, _ = extract_patch_embeddings(query_paths, patch_grid=PATCH_GRID)
            raw_scores = image_anomaly_scores(q_patches, P, bank)

            n_bank_img = bank_patches.shape[0] // P
            ref_n = min(60, n_bank_img)
            bank_self = image_anomaly_scores(bank_patches[: ref_n * P], P, bank)
            mu = float(np.mean(bank_self)); sd = float(np.std(bank_self) + 1e-8)
            z = (raw_scores - mu) / sd
            norm_scores = 1.0 / (1.0 + np.exp(-z))

            n_q = len(query_paths)
            q_pool = q_patches[: n_q * P].reshape(n_q, P, -1).mean(axis=1)
            n_b = bank_patches.shape[0] // P
            b_pool = bank_patches[: n_b * P].reshape(n_b, P, -1).mean(axis=1)
            k = min(EMBED_DIM, b_pool.shape[0], b_pool.shape[1])
            pca = PCA(n_components=k, random_state=0).fit(b_pool)
            emb = pca.transform(q_pool); bemb = pca.transform(b_pool)
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
                cat_rows.append(r)
        # Checkpoint this category before moving on.
        if cat_rows:
            pd.DataFrame(cat_rows).to_parquet(ck, index=False)
            print(f"[{cat}] checkpointed {len(cat_rows)} rows", flush=True)
            out_rows.extend(cat_rows)

    out = pd.DataFrame(out_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(out)} rows)")

    from sklearn.metrics import roc_auc_score
    for domain in ("rgb", "depth_or_xyz"):
        s = out[(out["domain"] == domain) & (out["split"] == "test")]
        per = [roc_auc_score(g["label"], g["raw_patchcore_score"])
               for _, g in s.groupby("category") if g["label"].nunique() > 1]
        if per:
            print(f"  upstream {domain}: mean per-category test AUROC = {np.mean(per):.4f} (n={len(per)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
