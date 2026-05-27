"""Phase 2.2E — build Eyecandies fusion-input CSV from per-category features.

Scoring:
- Per-(category, modality): memory bank = train good features.
- Per-sample score: cosine distance to nearest train memory-bank entry.
- Score normalisation: z-score against train scores, then sigmoid → [0,1].

Aggregates train, val, test_public, test_private into one CSV with the
schema the existing RGA pipeline expects:
  sample_id, fusion_split, category, domain, label, source_row, score,
  confidence, embedding_0..15

The `embedding_*` columns are the first 16 principal directions of the
ResNet-50 feature (deterministic via SVD per (category, modality, split=train)).
The `confidence` column is 1.0 (no per-sample confidence is available).
The `label` column is **read from test metadata.yaml only at the authorised
final-metric step** — this script writes label = NaN for now; a separate
script reads metadata.yaml right before the final ROC-AUC computation.

Wait: actually the labels are needed by the RGA pipeline at training time
for the validation-frozen head selection. BUT under one-class protocol,
train/val labels are all 0 (anomaly-free per official spec) and test
labels are the only true labels needed. So we set:
  - train: label = 0
  - val:   label = 0
  - test:  label = NaN (to be filled at the final-metric step)

Output: experiments/fusion/eyecandies_inputs.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "experiments" / "phase2" / "family_d" / "features"
OUT_CSV = ROOT / "experiments" / "fusion" / "eyecandies_inputs.csv"

SPLITS = ["train", "val", "test_public", "test_private"]
MODALITIES = ["rgb", "depth"]
EMBED_DIM = 16


def _score_modality(train_feats: np.ndarray, query_feats: np.ndarray) -> np.ndarray:
    """Cosine-distance-to-nearest-memory-bank entry, then z-norm vs train."""

    # Normalise rows
    def _norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    train_n = _norm(train_feats)
    query_n = _norm(query_feats)
    # Cosine similarity, distance = 1 - sim
    sims = query_n @ train_n.T  # [Q, T]
    dist = 1.0 - sims.max(axis=1)  # [Q]; near-zero for in-distribution
    # Calibrate against train-on-train distances (leave-one-out approx via second-best)
    train_sims = train_n @ train_n.T
    np.fill_diagonal(train_sims, -np.inf)
    train_dist = 1.0 - train_sims.max(axis=1)
    mu, sigma = float(train_dist.mean()), float(train_dist.std() + 1e-8)
    z = (dist - mu) / sigma
    # Sigmoid map to [0,1]; samples close to train get near-zero score
    score = 1.0 / (1.0 + np.exp(-z))
    return score


def _pca_directions(feats: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors (columns) computed via SVD on train."""
    # Centre
    mu = feats.mean(axis=0, keepdims=True)
    X = feats - mu
    # Truncated SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k]  # [k, D]


def _build_one_category(cat: str, writer) -> None:
    fp = FEAT_DIR / f"{cat}.npz"
    if not fp.exists():
        print(f"[{cat}] features missing; skipping")
        return
    data = np.load(fp, allow_pickle=True)

    # Score per modality
    train_rgb = data["train_rgb"]
    train_depth = data["train_depth"]
    rgb_pcs = _pca_directions(train_rgb, EMBED_DIM)
    depth_pcs = _pca_directions(train_depth, EMBED_DIM)

    train_rgb_score = _score_modality(train_rgb, train_rgb)
    train_depth_score = _score_modality(train_depth, train_depth)

    splits_data = {}
    for split in SPLITS:
        ids_key = f"{split}_sample_ids"
        if ids_key not in data.files:
            continue
        sids = data[ids_key].tolist()
        rgb = data[f"{split}_rgb"]
        depth = data[f"{split}_depth"]
        if split == "train":
            rgb_score = train_rgb_score
            depth_score = train_depth_score
        else:
            rgb_score = _score_modality(train_rgb, rgb)
            depth_score = _score_modality(train_depth, depth)
        rgb_emb = (rgb - train_rgb.mean(axis=0)) @ rgb_pcs.T
        depth_emb = (depth - train_depth.mean(axis=0)) @ depth_pcs.T
        splits_data[split] = {
            "sample_ids": sids,
            "rgb_score": rgb_score,
            "depth_score": depth_score,
            "rgb_emb": rgb_emb,
            "depth_emb": depth_emb,
        }

    # Map official splits to the fusion split column used by the runner
    fusion_split_map = {
        "train": "train",
        "val": "validation",
        "test_public": "test",
        "test_private": "test",
    }
    # Label rule:
    # - train, val: anomaly-free per official Eyecandies spec → label = 0
    # - test: deferred (must be filled at the final-metric step)
    label_rule = {
        "train": 0,
        "val": 0,
        "test_public": -1,  # placeholder; meaningless until final-metric step
        "test_private": -1,
    }

    for split, d in splits_data.items():
        fs = fusion_split_map[split]
        lbl = label_rule[split]
        for i, sid in enumerate(d["sample_ids"]):
            unique_sample = f"{cat}__{split}__{sid}"
            for mod, scr_arr, emb_arr in (
                ("rgb", d["rgb_score"], d["rgb_emb"]),
                ("depth", d["depth_score"], d["depth_emb"]),
            ):
                row = {
                    "sample_id": unique_sample,
                    "fusion_split": fs,
                    "category": cat,
                    "domain": mod,
                    "label": lbl,
                    "source_row": i,
                    "score": float(scr_arr[i]),
                    "confidence": 1.0,
                }
                for j in range(EMBED_DIM):
                    row[f"embedding_{j}"] = float(emb_arr[i, j])
                writer.writerow(row)
    print(f"[{cat}] written rows for splits: {list(splits_data.keys())}", flush=True)


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "fusion_split", "category", "domain", "label", "source_row", "score", "confidence"] + [
        f"embedding_{j}" for j in range(EMBED_DIM)
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cat in sorted(p for p in FEAT_DIR.glob("*.npz") if not p.name.startswith("._")):
            _build_one_category(cat.stem, w)
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
