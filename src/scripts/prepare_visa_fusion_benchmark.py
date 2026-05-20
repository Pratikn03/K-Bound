"""Prepare a paired VisA fusion benchmark.

VisA (Zou et al. ECCV 2022) is RGB-only. Following the LOCO-AD recipe,
we construct two co-observed domains from each image:

  rgb         - ResNet-50 penultimate features of the colour image
  edge_proxy  - ResNet-50 penultimate features of the Sobel-gradient
                magnitude image (hand-crafted structural companion).

Both domains are derived from the same single observation, so the
pairing is natural. The PatchCore-style kNN score uses the
ResNet-50 train-good memory bank for each domain.

VisA ships its own 1-class split CSV at split_csv/1cls.csv with
columns (object, split, label, image, mask). We honor that split.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class VisaObservation:
    category: str
    split: str
    label: int
    rgb_path: Path

    @property
    def sample_id(self) -> str:
        safe = f"{self.category}_{self.split}_{self.label}_{self.rgb_path.stem}"
        return f"visa_{safe}".replace("-", "_")

    @property
    def pairing_key(self) -> str:
        return f"{self.category}/{self.split}/{self.label}/{self.rgb_path.stem}"

    @property
    def defect_type(self) -> str:
        return "good" if self.label == 0 else "anomaly"


def discover_visa_observations(dataset_root: Path) -> list[VisaObservation]:
    dataset_root = Path(dataset_root)
    split_csv = dataset_root / "split_csv" / "1cls.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Expected VisA split CSV at {split_csv}")
    splits = pd.read_csv(split_csv)
    obs: list[VisaObservation] = []
    for _, row in splits.iterrows():
        rgb_path = dataset_root / str(row["image"])
        if not rgb_path.exists():
            continue
        label = 0 if str(row["label"]).lower() == "normal" else 1
        obs.append(
            VisaObservation(
                category=str(row["object"]),
                split=str(row["split"]),
                label=label,
                rgb_path=rgb_path,
            )
        )
    return obs


def _make_edge_proxy(rgb_path: Path, target_dir: Path) -> Path:
    target = target_dir / f"{rgb_path.parent.parent.parent.parent.name}_{rgb_path.parent.name}_{rgb_path.stem}.png"
    if target.exists():
        return target
    img = Image.open(rgb_path).convert("L")
    edges = img.filter(ImageFilter.FIND_EDGES).convert("RGB")
    edges.save(target)
    return target


def build_visa_fusion_frame(
    dataset_root: Path,
    *,
    embedding_dim: int = 16,
    patchcore_k: int = 3,
) -> tuple[pd.DataFrame, dict]:
    observations = discover_visa_observations(Path(dataset_root))
    if not observations:
        raise FileNotFoundError(f"No VisA observations found under {dataset_root}")

    labels = np.array([o.label for o in observations], dtype=int)
    splits = np.array([o.split for o in observations], dtype=object)
    categories = np.array([o.category for o in observations], dtype=object)
    train_mask = (splits == "train")
    normal_reference_mask = train_mask & (labels == 0)
    if not np.any(normal_reference_mask):
        raise ValueError("VisA prep requires at least one train-normal image.")

    from uais.fusion.attention.m3dm_features import (
        extract_resnet_features,
        fit_pca_projection,
        patchcore_knn_score,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        proxy_root = Path(tmpdir)
        rgb_paths = [o.rgb_path for o in observations]
        proxy_paths = [_make_edge_proxy(o.rgb_path, proxy_root) for o in observations]

        rgb_features = extract_resnet_features(rgb_paths)
        proxy_features = extract_resnet_features(proxy_paths)

    rgb_embeddings, *_ = fit_pca_projection(rgb_features, train_mask, embedding_dim)
    proxy_embeddings, *_ = fit_pca_projection(proxy_features, train_mask, embedding_dim)
    rgb_scores = patchcore_knn_score(rgb_features, normal_reference_mask, k=patchcore_k)
    proxy_scores = patchcore_knn_score(proxy_features, normal_reference_mask, k=patchcore_k)

    rows = []
    for idx, o in enumerate(observations):
        for domain, score, embedding in [
            ("rgb", rgb_scores[idx], rgb_embeddings[idx]),
            ("edge_proxy", proxy_scores[idx], proxy_embeddings[idx]),
        ]:
            row = {
                "sample_id": o.sample_id,
                "pairing_key": o.pairing_key,
                "category": o.category,
                "split": o.split,
                "defect_type": o.defect_type,
                "domain": domain,
                "label": int(o.label),
                "source_path": str(o.rgb_path),
                "score_fit_split": "train",
                "score_fit_defect_type": "good",
                "score": float(np.clip(score, 0.0, 1.0)),
                "confidence": float(np.clip(2.0 * abs(float(score) - 0.5), 0.0, 1.0)),
            }
            for emb_idx in range(embedding_dim):
                row[f"embedding_{emb_idx}"] = float(embedding[emb_idx])
            rows.append(row)

    frame = pd.DataFrame(rows)
    sample_frame = frame.groupby("sample_id").first()
    metadata = {
        "benchmark_type": "paired_visa_score_fusion",
        "natural_pairing": True,
        "pairing_unit": "single VisA image with RGB + edge-proxy co-observed domains",
        "feature_mode": "resnet50_patchcore_knn_with_edge_proxy",
        "samples": int(len(sample_frame)),
        "rows": int(len(frame)),
        "positive_fraction_actual": float(sample_frame["label"].mean()),
        "domain_order": ["rgb", "edge_proxy"],
        "embedding_dim": int(embedding_dim),
        "categories": sorted(frame["category"].unique().tolist()),
        "splits": sorted(frame["split"].unique().tolist()),
        "defect_types": sorted(frame["defect_type"].unique().tolist()),
        "score_protocol": {
            "normal_reference_split": "train",
            "normal_reference_defect_type": "good",
            "normal_reference_samples": int(normal_reference_mask.sum()),
            "score_normalization": "patchcore_knn_minmax_clipped",
            "embedding_normalization_split": "train",
        },
    }
    return frame, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/visa"))
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--patchcore-k", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/fusion/visa_fusion_inputs.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("experiments/fusion/visa_fusion_metadata.json"),
    )
    args = parser.parse_args()

    frame, metadata = build_visa_fusion_frame(
        args.dataset_root,
        embedding_dim=args.embedding_dim,
        patchcore_k=args.patchcore_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    metadata["output"] = str(args.output)
    args.metadata.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
