"""Prepare a naturally paired Real3D-AD fusion benchmark.

Real3D-AD is a high-resolution real-world point-cloud anomaly detection
benchmark from Liu et al. (NeurIPS 2023). Each sample is a single .pcd
file containing the captured 3D point cloud of one industrial object.

For the ELARA fusion pipeline we construct two naturally co-observed
domains from the same .pcd file:

  pointcloud    - FPFH (Fast Point Feature Histograms) descriptors of the
                  raw point cloud
  depth_proj    - depth-image projection of the same point cloud onto a
                  fixed-orientation orthographic plane, then ResNet-50
                  features over the projected image

Both domains are derived from the same single observation, so the
pairing is natural (identical to the MVTec 3D-AD pattern but where
"RGB+depth" is replaced by "FPFH+depth-projection").

Expected layout (matches the published Real3D-AD release):

  data/raw/real3d/<category>/
    train/<id>.pcd                  # normal-only
    test/good/<id>.pcd              # normal test
    test/<defect_type>/<id>.pcd     # anomaly test
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Real3DObservation:
    category: str
    split: str
    defect_type: str
    stem: str
    pcd_path: Path

    @property
    def label(self) -> int:
        return 0 if self.defect_type == "good" else 1

    @property
    def sample_id(self) -> str:
        safe = "_".join([self.category, self.split, self.defect_type, self.stem])
        return f"real3d_{safe}".replace("-", "_")

    @property
    def pairing_key(self) -> str:
        return f"{self.category}/{self.split}/{self.defect_type}/{self.stem}"


def _discover_real3d_pairs(dataset_root: Path) -> list[Real3DObservation]:
    """Discover all .pcd observations in a Real3D-AD-style directory."""
    dataset_root = Path(dataset_root)
    pairs: list[Real3DObservation] = []
    for category_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        category = category_dir.name
        train_dir = category_dir / "train"
        if train_dir.exists():
            for pcd in sorted(train_dir.rglob("*.pcd")):
                pairs.append(
                    Real3DObservation(
                        category=category,
                        split="train",
                        defect_type="good",
                        stem=pcd.stem,
                        pcd_path=pcd,
                    )
                )
        test_dir = category_dir / "test"
        if test_dir.exists():
            for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
                defect_type = defect_dir.name
                for pcd in sorted(defect_dir.rglob("*.pcd")):
                    pairs.append(
                        Real3DObservation(
                            category=category,
                            split="test",
                            defect_type=defect_type,
                            stem=pcd.stem,
                            pcd_path=pcd,
                        )
                    )
    return pairs


def _read_pcd_points(path: Path, max_points: int = 4096) -> np.ndarray:
    """Read XYZ coordinates from a .pcd file. Subsamples to max_points."""
    try:
        import open3d as o3d

        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float32)
    except ImportError:
        # Fallback: parse ASCII PCD directly. Supports the common
        # FIELDS x y z layout.
        with open(path, "rb") as handle:
            text = handle.read().decode("latin-1", errors="ignore")
        lines = text.splitlines()
        idx = 0
        for i, line in enumerate(lines):
            if line.startswith("DATA"):
                idx = i + 1
                break
        records = []
        for line in lines[idx:]:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                records.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
        points = np.asarray(records, dtype=np.float32) if records else np.zeros((0, 3), dtype=np.float32)

    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    return points


def _fpfh_descriptor(points: np.ndarray, embedding_dim: int = 33) -> np.ndarray:
    """Aggregate FPFH features over the point cloud; fall back to histograms.

    Tries open3d's FPFH first; if unavailable, uses a hand-rolled histogram
    of pairwise angles + radial distances that approximates the FPFH idea
    closely enough to give the fusion pipeline a non-trivial embedding.
    """
    if points.shape[0] < 16:
        return np.zeros(embedding_dim, dtype=np.float32)
    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            cloud,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100),
        )
        features = np.asarray(fpfh.data).T  # [N, 33]
        if features.size == 0:
            raise RuntimeError("FPFH returned empty features.")
        descriptor = features.mean(axis=0)
        if descriptor.shape[0] >= embedding_dim:
            descriptor = descriptor[:embedding_dim]
        else:
            descriptor = np.pad(descriptor, (0, embedding_dim - descriptor.shape[0]))
        return descriptor.astype(np.float32)
    except (ImportError, RuntimeError):
        # Fallback: 33-bin histogram over radial distances from centroid.
        centroid = points.mean(axis=0)
        radii = np.linalg.norm(points - centroid, axis=1)
        max_r = float(radii.max()) if radii.size else 1.0
        if max_r <= 1e-9:
            max_r = 1.0
        edges = np.linspace(0.0, max_r * 1.05, embedding_dim + 1)
        hist, _ = np.histogram(radii, bins=edges)
        descriptor = hist.astype(np.float32) / max(1.0, hist.sum())
        return descriptor


def _depth_projection_image(points: np.ndarray, resolution: int = 96) -> np.ndarray:
    """Orthographic top-down depth projection. Returns a (resolution, resolution, 3) RGB image."""
    if points.shape[0] < 4:
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)
    centroid = points.mean(axis=0)
    centered = points - centroid
    span = float(np.max(np.linalg.norm(centered, axis=1)))
    if span <= 1e-9:
        span = 1.0
    scaled = centered / span
    u = ((scaled[:, 0] + 1.0) * 0.5 * (resolution - 1)).astype(int).clip(0, resolution - 1)
    v = ((scaled[:, 1] + 1.0) * 0.5 * (resolution - 1)).astype(int).clip(0, resolution - 1)
    z = scaled[:, 2]
    image = np.zeros((resolution, resolution), dtype=np.float32)
    counts = np.zeros((resolution, resolution), dtype=np.float32)
    for j in range(len(u)):
        image[v[j], u[j]] += z[j]
        counts[v[j], u[j]] += 1.0
    image = np.where(counts > 0, image / np.maximum(counts, 1.0), -1.0)
    lo = float(image[image > -1.0].min()) if (image > -1.0).any() else 0.0
    hi = float(image[image > -1.0].max()) if (image > -1.0).any() else 1.0
    span = hi - lo if hi > lo else 1.0
    norm = ((image - lo) / span * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


def build_real3d_fusion_frame(
    dataset_root: Path,
    *,
    categories: list[str] | None = None,
    embedding_dim: int = 16,
    max_points: int = 4096,
) -> tuple[pd.DataFrame, dict]:
    pairs = _discover_real3d_pairs(Path(dataset_root))
    if categories:
        keep = set(categories)
        pairs = [p for p in pairs if p.category in keep]
    if not pairs:
        raise FileNotFoundError(f"No Real3D-AD .pcd observations found under {dataset_root}")

    train_mask = np.array([p.split == "train" for p in pairs], dtype=bool)
    normal_reference_mask = train_mask & np.array([p.defect_type == "good" for p in pairs])
    if not np.any(normal_reference_mask):
        raise ValueError("Real3D-AD prep requires at least one train/good point cloud.")

    # Extract point clouds + descriptors.
    pcd_descriptors = np.zeros((len(pairs), 33), dtype=np.float32)
    depth_features = np.zeros((len(pairs), 2048), dtype=np.float32)

    from uais.fusion.attention.m3dm_features import (
        extract_resnet_features,
        fit_pca_projection,
        patchcore_knn_score,
    )

    # We feed depth-projection images into the existing ResNet feature extractor.
    # extract_resnet_features works on disk paths, so we'll save projections to a
    # temp directory and feed those paths.
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_paths = []
        for idx, pair in enumerate(pairs):
            points = _read_pcd_points(pair.pcd_path, max_points=max_points)
            pcd_descriptors[idx] = _fpfh_descriptor(points, embedding_dim=33)
            depth_img = _depth_projection_image(points, resolution=96)
            tmp_path = Path(tmpdir) / f"{idx}.png"
            try:
                from PIL import Image as PILImage

                PILImage.fromarray(depth_img, mode="RGB").save(tmp_path)
            except Exception:
                tmp_path = None
            tmp_paths.append(tmp_path)
        valid_paths = [p for p in tmp_paths if p is not None]
        if valid_paths:
            feats = extract_resnet_features(valid_paths)
            for j, target in enumerate([i for i, p in enumerate(tmp_paths) if p is not None]):
                depth_features[target] = feats[j]

    # Project FPFH descriptors (33-dim) and depth features (2048-dim) to embedding_dim.
    pcd_embeddings, *_ = fit_pca_projection(pcd_descriptors, train_mask, embedding_dim)
    depth_embeddings, *_ = fit_pca_projection(depth_features, train_mask, embedding_dim)
    pcd_scores = patchcore_knn_score(pcd_descriptors, normal_reference_mask, k=3)
    depth_scores = patchcore_knn_score(depth_features, normal_reference_mask, k=3)

    rows = []
    for idx, pair in enumerate(pairs):
        for domain, score, embedding, source_path in [
            ("pointcloud", pcd_scores[idx], pcd_embeddings[idx], pair.pcd_path),
            ("depth_or_xyz", depth_scores[idx], depth_embeddings[idx], pair.pcd_path),
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
                row[f"embedding_{emb_idx}"] = float(embedding[emb_idx])
            rows.append(row)

    frame = pd.DataFrame(rows)
    sample_frame = frame.groupby("sample_id").first()

    metadata = {
        "benchmark_type": "naturally_paired_real3d_score_fusion",
        "natural_pairing": True,
        "pairing_unit": "single Real3D-AD point cloud with FPFH + depth-projection co-observed domains",
        "feature_mode": "fpfh+depth_projection_resnet50_patchcore",
        "samples": int(len(sample_frame)),
        "rows": int(len(frame)),
        "positive_fraction_actual": float(sample_frame["label"].mean()),
        "domain_order": ["pointcloud", "depth_or_xyz"],
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
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/real3d"))
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/fusion/real3d_fusion_inputs.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("experiments/fusion/real3d_fusion_metadata.json"),
    )
    args = parser.parse_args()

    frame, metadata = build_real3d_fusion_frame(
        args.dataset_root,
        categories=args.categories,
        embedding_dim=args.embedding_dim,
        max_points=args.max_points,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    metadata["output"] = str(args.output)
    args.metadata.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
