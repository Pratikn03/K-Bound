"""Prepare a paired MVTec LOCO-AD fusion benchmark.

MVTec LOCO-AD is RGB-only (no depth channel). We construct two
co-observed domains from each image:

  rgb         - ResNet-50 penultimate features of the colour image
  edge_proxy  - ResNet-50 penultimate features of the Sobel-gradient
                magnitude image. Acts as a hand-crafted structural
                companion to the semantic RGB stream.

Both domains are derived from the same single observation, so the
pairing is natural. The PatchCore-style kNN score uses the
ResNet-50 train-good memory bank for each domain independently.

Expected layout (matches the published LOCO-AD release):

  data/raw/mvtec_loco/<category>/
    train/good/<id>.png
    validation/good/<id>.png
    test/{good, logical_anomalies, structural_anomalies}/<id>.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class LocoObservation:
    category: str
    split: str
    defect_type: str
    stem: str
    rgb_path: Path

    @property
    def label(self) -> int:
        return 0 if self.defect_type == "good" else 1

    @property
    def sample_id(self) -> str:
        safe = "_".join([self.category, self.split, self.defect_type, self.stem])
        return f"loco_{safe}".replace("-", "_")

    @property
    def pairing_key(self) -> str:
        return f"{self.category}/{self.split}/{self.defect_type}/{self.stem}"


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _is_visible(path: Path) -> bool:
    return not path.name.startswith(".") and not path.name.startswith("._")


def discover_loco_pairs(dataset_root: Path, categories: Iterable[str] | None = None) -> list[LocoObservation]:
    dataset_root = Path(dataset_root)
    allowed = set(categories) if categories else None
    pairs: list[LocoObservation] = []
    for category_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and _is_visible(p)):
        if category_dir.name in {"ground_truth"} or (allowed and category_dir.name not in allowed):
            continue
        for split_dir in sorted(p for p in category_dir.iterdir() if p.is_dir() and _is_visible(p)):
            if split_dir.name == "ground_truth":
                continue
            for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and _is_visible(p)):
                for img in sorted(defect_dir.iterdir()):
                    if not img.is_file() or not _is_visible(img):
                        continue
                    if img.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    pairs.append(
                        LocoObservation(
                            category=category_dir.name,
                            split=split_dir.name,
                            defect_type=defect_dir.name,
                            stem=img.stem,
                            rgb_path=img,
                        )
                    )
    return pairs


def _make_edge_proxy(rgb_path: Path, target_dir: Path) -> Path:
    """Generate a Sobel-gradient magnitude image and save it as RGB."""
    target = target_dir / f"{rgb_path.parent.parent.parent.name}_{rgb_path.parent.parent.name}_{rgb_path.parent.name}_{rgb_path.stem}.png"
    if target.exists():
        return target
    img = Image.open(rgb_path).convert("L")
    edges = img.filter(ImageFilter.FIND_EDGES)
    edges = edges.convert("RGB")
    edges.save(target)
    return target


def build_loco_fusion_frame(
    dataset_root: Path,
    *,
    categories: list[str] | None = None,
    embedding_dim: int = 16,
    patchcore_k: int = 3,
) -> tuple[pd.DataFrame, dict]:
    pairs = discover_loco_pairs(Path(dataset_root), categories=categories)
    if not pairs:
        raise FileNotFoundError(f"No LOCO-AD observations found under {dataset_root}")

    labels = np.asarray([p.label for p in pairs], dtype=int)
    splits = np.asarray([p.split for p in pairs], dtype=object)
    defect_types = np.asarray([p.defect_type for p in pairs], dtype=object)
    pair_categories = np.asarray([p.category for p in pairs], dtype=object)
    train_mask = splits == "train"
    normal_reference_mask = train_mask & (defect_types == "good")
    if not np.any(normal_reference_mask):
        raise ValueError("LOCO-AD prep requires at least one train/good image.")

    # Generate edge-proxy images on disk so the existing ResNet pipeline can read them.
    import tempfile

    from uais.fusion.attention.m3dm_features import (
        extract_resnet_features,
        fit_pca_projection,
        patchcore_knn_score,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        proxy_root = Path(tmpdir)
        rgb_paths = [p.rgb_path for p in pairs]
        proxy_paths = [_make_edge_proxy(p.rgb_path, proxy_root) for p in pairs]

        rgb_features = extract_resnet_features(rgb_paths)
        proxy_features = extract_resnet_features(proxy_paths)

    rgb_embeddings, *_ = fit_pca_projection(rgb_features, train_mask, embedding_dim)
    proxy_embeddings, *_ = fit_pca_projection(proxy_features, train_mask, embedding_dim)
    rgb_scores = patchcore_knn_score(rgb_features, normal_reference_mask, k=patchcore_k)
    proxy_scores = patchcore_knn_score(proxy_features, normal_reference_mask, k=patchcore_k)

    rows = []
    for idx, pair in enumerate(pairs):
        for domain, score, embedding in [
            ("rgb", rgb_scores[idx], rgb_embeddings[idx]),
            ("edge_proxy", proxy_scores[idx], proxy_embeddings[idx]),
        ]:
            row = {
                "sample_id": pair.sample_id,
                "pairing_key": pair.pairing_key,
                "category": pair.category,
                "split": pair.split,
                "defect_type": pair.defect_type,
                "domain": domain,
                "label": pair.label,
                "source_path": str(pair.rgb_path),
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
        "benchmark_type": "paired_mvtec_loco_score_fusion",
        "natural_pairing": True,
        "pairing_unit": "single LOCO-AD image with RGB + edge-proxy co-observed domains",
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
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/mvtec_loco"))
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--patchcore-k", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/fusion/mvtec_loco_patchcore_inputs.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("experiments/fusion/mvtec_loco_patchcore_metadata.json"),
    )
    args = parser.parse_args()

    frame, metadata = build_loco_fusion_frame(
        args.dataset_root,
        categories=args.categories,
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
