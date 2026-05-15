"""Prepare a naturally paired MVTec 3D-AD score-level fusion benchmark.

Expected input layout follows the public MVTec 3D-AD structure:

  data/raw/mvtec3d/<category>/<split>/<defect_type>/rgb/<id>.png
  data/raw/mvtec3d/<category>/<split>/<defect_type>/xyz/<id>.tiff

The script emits the same long fusion schema used by the attention benchmark,
with two naturally co-observed domains per sample: ``rgb`` and
``depth_or_xyz``. Scores are lightweight normal-reference anomaly scores from
image statistics, intended as a reproducible first benchmark rather than a
replacement for specialized RGB-3D anomaly detectors.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


DOMAIN_ORDER = ["rgb", "depth_or_xyz"]
DEPTH_DIR_NAMES = ("xyz", "depth", "depth_or_xyz")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PairedObservation:
    category: str
    split: str
    defect_type: str
    stem: str
    rgb_path: Path
    depth_path: Path

    @property
    def label(self) -> int:
        return 0 if self.defect_type == "good" else 1

    @property
    def sample_id(self) -> str:
        safe = "_".join([self.category, self.split, self.defect_type, self.stem])
        return f"mvtec3d_{safe}".replace("-", "_")

    @property
    def pairing_key(self) -> str:
        return f"{self.category}/{self.split}/{self.defect_type}/{self.stem}"


def _is_visible_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(".") and not path.name.startswith("._")


def _image_files(path: Path) -> dict[str, Path]:
    if not path.exists():
        return {}
    return {
        file.stem: file
        for file in sorted(path.iterdir())
        if file.is_file()
        and not file.name.startswith(".")
        and not file.name.startswith("._")
        and file.suffix.lower() in IMAGE_SUFFIXES
    }


def _find_depth_dir(parent: Path) -> Path | None:
    for name in DEPTH_DIR_NAMES:
        candidate = parent / name
        if candidate.exists():
            return candidate
    return None


def discover_mvtec3d_pairs(dataset_root: Path, categories: Iterable[str] | None = None) -> list[PairedObservation]:
    """Discover RGB/depth pairs in an MVTec 3D-AD style directory."""
    dataset_root = Path(dataset_root)
    allowed = set(categories) if categories else None
    pairs: list[PairedObservation] = []
    for category_dir in sorted(p for p in dataset_root.iterdir() if _is_visible_dir(p)):
        if allowed is not None and category_dir.name not in allowed:
            continue
        for split_dir in sorted(p for p in category_dir.iterdir() if _is_visible_dir(p)):
            for defect_dir in sorted(p for p in split_dir.iterdir() if _is_visible_dir(p)):
                rgb_dir = defect_dir / "rgb"
                depth_dir = _find_depth_dir(defect_dir)
                if not rgb_dir.exists() or depth_dir is None:
                    continue
                rgb_files = _image_files(rgb_dir)
                depth_files = _image_files(depth_dir)
                for stem in sorted(set(rgb_files) & set(depth_files)):
                    pairs.append(
                        PairedObservation(
                            category=category_dir.name,
                            split=split_dir.name,
                            defect_type=defect_dir.name,
                            stem=stem,
                            rgb_path=rgb_files[stem],
                            depth_path=depth_files[stem],
                        )
                    )
    return pairs


def _read_image_array(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB" if image.mode not in {"L", "I;16", "F"} else "L"), dtype=np.float32)
    except UnidentifiedImageError:
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError(
                "Reading MVTec XYZ TIFF files requires tifffile. Install it with `pip install tifffile`."
            ) from exc
        return np.asarray(tifffile.imread(path), dtype=np.float32)


def _image_features(path: Path, embedding_dim: int) -> np.ndarray:
    arr = _read_image_array(path)
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1, 1)
    gray = flat.mean(axis=1)
    gray = gray[np.isfinite(gray)]
    if gray.size == 0:
        gray = np.zeros(1, dtype=np.float32)
    quantiles = np.quantile(gray, [0.05, 0.25, 0.5, 0.75, 0.95])
    values = np.array(
        [
            float(gray.mean()),
            float(gray.std()),
            float(gray.min()),
            float(gray.max()),
            *[float(q) for q in quantiles],
        ],
        dtype=np.float32,
    )
    if len(values) < embedding_dim:
        values = np.pad(values, (0, embedding_dim - len(values)))
    return values[:embedding_dim]


def _minmax(values: np.ndarray, fit_mask: np.ndarray | None = None) -> np.ndarray:
    reference = values[fit_mask] if fit_mask is not None and np.any(fit_mask) else values
    lo = np.nanmin(reference, axis=0)
    hi = np.nanmax(reference, axis=0)
    denom = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
    scaled = np.nan_to_num((values - lo) / denom, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(scaled, 0.0, 1.0)


def _normal_reference_scores(features: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    if not np.any(fit_mask):
        raise ValueError("Normal-reference scoring requires at least one fit observation.")
    reference = features[fit_mask]
    center = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    distances = np.linalg.norm((features - center) / scale, axis=1)
    reference_distances = distances[fit_mask]
    lo = float(np.min(reference_distances))
    hi = float(np.percentile(reference_distances, 95))
    if hi - lo <= 1e-9:
        hi = lo + 1.0
    return np.clip((distances - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_mvtec3d_fusion_frame(
    dataset_root: Path,
    categories: Iterable[str] | None = None,
    embedding_dim: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """Build long-format fusion rows and benchmark metadata."""
    pairs = discover_mvtec3d_pairs(Path(dataset_root), categories=categories)
    if not pairs:
        raise FileNotFoundError(f"No paired RGB/depth observations found under {dataset_root}")

    rgb_features = np.vstack([_image_features(pair.rgb_path, embedding_dim) for pair in pairs])
    depth_features = np.vstack([_image_features(pair.depth_path, embedding_dim) for pair in pairs])
    labels = np.asarray([pair.label for pair in pairs], dtype=int)
    splits = np.asarray([pair.split for pair in pairs], dtype=object)
    defect_types = np.asarray([pair.defect_type for pair in pairs], dtype=object)
    train_mask = splits == "train"
    normal_reference_mask = train_mask & (defect_types == "good")
    if not np.any(normal_reference_mask):
        raise ValueError("MVTec 3D score generation requires at least one train/good paired observation.")

    rgb_embeddings = _minmax(rgb_features, fit_mask=train_mask)
    depth_embeddings = _minmax(depth_features, fit_mask=train_mask)
    rgb_scores = _normal_reference_scores(rgb_features, normal_reference_mask)
    depth_scores = _normal_reference_scores(depth_features, normal_reference_mask)

    rows = []
    for idx, pair in enumerate(pairs):
        for domain, score, emb, source_path in [
            ("rgb", rgb_scores[idx], rgb_embeddings[idx], pair.rgb_path),
            ("depth_or_xyz", depth_scores[idx], depth_embeddings[idx], pair.depth_path),
        ]:
            row = {
                "sample_id": pair.sample_id,
                "pairing_key": pair.pairing_key,
                "category": pair.category,
                "split": pair.split,
                "defect_type": pair.defect_type,
                "domain": domain,
                "label": pair.label,
                "source_path": str(source_path),
                "score_fit_split": "train",
                "score_fit_defect_type": "good",
                "score": float(np.clip(score, 0.0, 1.0)),
                "confidence": float(np.clip(2.0 * abs(float(score) - 0.5), 0.0, 1.0)),
            }
            for emb_idx in range(embedding_dim):
                row[f"embedding_{emb_idx}"] = float(emb[emb_idx])
            rows.append(row)

    frame = pd.DataFrame(rows)
    sample_frame = frame.groupby("sample_id").first()
    metadata = {
        "benchmark_type": "naturally_paired_mvtec3d_score_fusion",
        "natural_pairing": True,
        "pairing_unit": "MVTec 3D-AD RGB/depth observation stem",
        "important_limitation": (
            "Scores are lightweight normal-reference image-statistic anomaly scores; "
            "they provide a reproducible paired fusion benchmark, not a specialized RGB-3D detector."
        ),
        "samples": int(len(sample_frame)),
        "rows": int(len(frame)),
        "positive_fraction_actual": float(sample_frame["label"].mean()),
        "domain_order": DOMAIN_ORDER,
        "embedding_dim": int(embedding_dim),
        "categories": sorted(frame["category"].unique().tolist()),
        "splits": sorted(frame["split"].unique().tolist()),
        "score_protocol": {
            "normal_reference_split": "train",
            "normal_reference_defect_type": "good",
            "normal_reference_samples": int(normal_reference_mask.sum()),
            "score_normalization": "train_good_distance_minmax_clipped",
            "embedding_normalization_split": "train",
        },
        "fusion_evaluation_protocol": (
            "Original MVTec split is preserved in the `split` column. The attention-fusion "
            "runner uses a supervised random split unless configured with an explicit split column."
        ),
        "domain_coverage": {
            domain: float(frame.loc[frame["domain"] == domain, "sample_id"].nunique() / len(sample_frame))
            for domain in DOMAIN_ORDER
        },
    }
    return frame, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare naturally paired MVTec 3D-AD fusion inputs")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/mvtec3d"))
    parser.add_argument("--output", type=Path, default=Path("experiments/fusion/mvtec3d_fusion_inputs.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("experiments/fusion/mvtec3d_fusion_metadata.json"))
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--embedding-dim", type=int, default=8)
    args = parser.parse_args()

    frame, metadata = build_mvtec3d_fusion_frame(
        args.dataset_root,
        categories=args.categories,
        embedding_dim=args.embedding_dim,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    metadata["output"] = str(args.output)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
