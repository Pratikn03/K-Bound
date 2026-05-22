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


def _parse_defect_from_stem(stem: str) -> str:
    """Real3D filenames encode the defect via suffix: e.g. 142_good or 142_good_cut.

    A bare ``<id>_good`` is a normal sample; ``<id>_good_<defect>`` is anomaly.
    """
    parts = stem.split("_")
    if len(parts) <= 2:
        return "good"
    return "_".join(parts[2:])


def _is_visible_pcd(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() == ".pcd"
        and not path.name.startswith(".")
        and not path.name.startswith("._")
    )


def _discover_real3d_pairs(dataset_root: Path) -> list[Real3DObservation]:
    """Discover all .pcd observations in a Real3D-AD-style directory.

    Real3D-AD's distribution layout is:
        <root>/Real3D-AD-PCD/<category>/{train,test}/<id>_<good|good_<defect>>.pcd

    The script accepts either ``data/raw/real3d`` or
    ``data/raw/real3d/Real3D-AD-PCD`` as the dataset root.
    """
    dataset_root = Path(dataset_root)
    # If the top-level only has one directory called Real3D-AD-PCD, descend.
    children = [
        p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ]
    if len(children) == 1 and children[0].name.startswith("Real3D"):
        dataset_root = children[0]

    pairs: list[Real3DObservation] = []
    for category_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        category = category_dir.name
        for split_name in ("train", "test"):
            split_dir = category_dir / split_name
            if not split_dir.exists():
                continue
            for pcd in sorted(split_dir.iterdir()):
                if not _is_visible_pcd(pcd):
                    continue
                defect_type = _parse_defect_from_stem(pcd.stem) if split_name == "test" else "good"
                pairs.append(
                    Real3DObservation(
                        category=category,
                        split=split_name,
                        defect_type=defect_type,
                        stem=pcd.stem,
                        pcd_path=pcd,
                    )
                )
    return pairs


_PCD_TYPE_MAP = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
}


def _read_pcd_binary(path: Path) -> np.ndarray:
    """Direct binary PCD parser for the common FIELDS x y z layout.

    Reads the header to derive (fields, sizes, types, counts, points,
    data mode) and decodes the binary or ASCII payload into an [N, 3]
    XYZ float32 array. Extra fields beyond x/y/z are ignored.
    """
    with open(path, "rb") as handle:
        header_bytes = b""
        while b"DATA" not in header_bytes:
            line = handle.readline()
            if not line:
                return np.zeros((0, 3), dtype=np.float32)
            header_bytes += line
        header_lines = header_bytes.decode("latin-1", errors="ignore").splitlines()
        fields: list[str] = []
        sizes: list[int] = []
        types: list[str] = []
        counts: list[int] = []
        points = 0
        data_mode = "ascii"
        for line in header_lines:
            tokens = line.strip().split()
            if not tokens:
                continue
            key = tokens[0].upper()
            if key == "FIELDS":
                fields = tokens[1:]
            elif key == "SIZE":
                sizes = [int(x) for x in tokens[1:]]
            elif key == "TYPE":
                types = tokens[1:]
            elif key == "COUNT":
                counts = [int(x) for x in tokens[1:]]
            elif key == "POINTS":
                points = int(tokens[1])
            elif key == "DATA":
                data_mode = tokens[1].lower() if len(tokens) > 1 else "ascii"

        if not fields or points <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        if not counts:
            counts = [1] * len(fields)

        try:
            xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
        except ValueError:
            return np.zeros((0, 3), dtype=np.float32)

        if data_mode == "binary":
            # Build a structured dtype matching the header.
            field_dtypes: list[tuple[str, np.dtype, int]] = []
            for f, sz, tp, ct in zip(fields, sizes, types, counts):
                dt = _PCD_TYPE_MAP.get((tp.upper(), int(sz)))
                if dt is None:
                    return np.zeros((0, 3), dtype=np.float32)
                field_dtypes.append((f, np.dtype(dt), int(ct)))
            record_dtype = np.dtype(
                [(name, dt, (ct,)) if ct > 1 else (name, dt) for name, dt, ct in field_dtypes]
            )
            raw = handle.read(points * record_dtype.itemsize)
            arr = np.frombuffer(raw, dtype=record_dtype, count=points)
            try:
                xyz = np.stack(
                    [
                        arr[fields[xi]].astype(np.float32),
                        arr[fields[yi]].astype(np.float32),
                        arr[fields[zi]].astype(np.float32),
                    ],
                    axis=1,
                )
            except Exception:
                return np.zeros((0, 3), dtype=np.float32)
            return xyz
        else:
            # ASCII fallback
            remaining = handle.read().decode("latin-1", errors="ignore")
            records = []
            for raw_line in remaining.splitlines():
                parts = raw_line.strip().split()
                if len(parts) < max(xi, yi, zi) + 1:
                    continue
                try:
                    records.append(
                        [float(parts[xi]), float(parts[yi]), float(parts[zi])]
                    )
                except ValueError:
                    continue
            return (
                np.asarray(records, dtype=np.float32)
                if records
                else np.zeros((0, 3), dtype=np.float32)
            )


def _read_pcd_points(path: Path, max_points: int = 4096) -> np.ndarray:
    """Read XYZ coordinates from a .pcd file. Subsamples to max_points."""
    points = _read_pcd_binary(path)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    return points.astype(np.float32, copy=False)


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
        # Fallback: richer geometric descriptor that augments the radial
        # histogram with PCA-based local shape statistics (eigenvalue
        # ratios = surface anisotropy / planarity / sphericity), pairwise
        # angle histograms, and curvature distribution moments. This is
        # not full FPFH but it carries non-trivial 3D shape information
        # beyond pure radial distance.
        centroid = points.mean(axis=0)
        centered = points - centroid
        radii = np.linalg.norm(centered, axis=1)
        max_r = float(radii.max()) if radii.size else 1.0
        if max_r <= 1e-9:
            max_r = 1.0

        bins_per_block = max(8, embedding_dim // 4)
        # Block 1: normalised radial distance histogram.
        edges = np.linspace(0.0, max_r * 1.05, bins_per_block + 1)
        radial_hist, _ = np.histogram(radii, bins=edges)
        radial_hist = radial_hist.astype(np.float32) / max(1.0, radial_hist.sum())

        # Block 2: pairwise angle histogram (subsample pairs for speed).
        n_pairs = min(2048, max(0, points.shape[0] - 1))
        if n_pairs >= 2:
            rng_pairs = np.random.default_rng(0)
            i_idx = rng_pairs.integers(0, points.shape[0], size=n_pairs)
            j_idx = rng_pairs.integers(0, points.shape[0], size=n_pairs)
            valid = i_idx != j_idx
            i_idx = i_idx[valid]
            j_idx = j_idx[valid]
            v1 = centered[i_idx]
            v2 = centered[j_idx]
            denom = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)).clip(1e-9)
            cosines = np.clip(np.sum(v1 * v2, axis=1) / denom, -1.0, 1.0)
            angle_edges = np.linspace(-1.0, 1.0, bins_per_block + 1)
            angle_hist, _ = np.histogram(cosines, bins=angle_edges)
            angle_hist = angle_hist.astype(np.float32) / max(1.0, angle_hist.sum())
        else:
            angle_hist = np.zeros(bins_per_block, dtype=np.float32)

        # Block 3: PCA-derived shape descriptors of the whole cloud.
        cov = np.cov(centered.T) if centered.shape[0] >= 3 else np.eye(3, dtype=np.float64)
        try:
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            eigvals = np.maximum(eigvals, 1e-12)
            eig_sum = float(eigvals.sum())
            l1, l2, l3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
            linearity = (l1 - l2) / l1
            planarity = (l2 - l3) / l1
            sphericity = l3 / l1
            anisotropy = (l1 - l3) / l1
            change_curv = l3 / max(eig_sum, 1e-12)
            eig_entropy = -float(sum((v / eig_sum) * np.log(max(v / eig_sum, 1e-12)) for v in (l1, l2, l3)))
            pca_block = np.array(
                [linearity, planarity, sphericity, anisotropy, change_curv, eig_entropy],
                dtype=np.float32,
            )
        except np.linalg.LinAlgError:
            pca_block = np.zeros(6, dtype=np.float32)

        # Block 4: distance-moment statistics (mean / std / skew-proxy / kurt-proxy).
        if radii.size >= 4:
            mu = float(radii.mean())
            sd = float(radii.std())
            sd_safe = sd if sd > 1e-9 else 1.0
            z = (radii - mu) / sd_safe
            moments = np.array(
                [mu / max_r, sd / max_r, float(np.mean(z ** 3)), float(np.mean(z ** 4))],
                dtype=np.float32,
            )
        else:
            moments = np.zeros(4, dtype=np.float32)

        descriptor = np.concatenate([radial_hist, angle_hist, pca_block, moments]).astype(np.float32)
        if descriptor.shape[0] >= embedding_dim:
            descriptor = descriptor[:embedding_dim]
        else:
            descriptor = np.pad(descriptor, (0, embedding_dim - descriptor.shape[0]))
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
    val_fraction_of_train: float = 0.15,
    supervised_paired: bool = False,
    supervised_paired_seed: int = 42,
    supervised_paired_test_fraction: float = 0.30,
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

    # Real3D-AD doesn't ship a validation split; carve one from train.
    rng = np.random.default_rng(42)
    effective_splits = np.array([p.split for p in pairs], dtype=object)
    train_indices = [i for i, p in enumerate(pairs) if p.split == "train"]
    rng.shuffle(train_indices)
    n_val = max(1, int(round(len(train_indices) * float(val_fraction_of_train))))
    for j, i in enumerate(train_indices[:n_val]):
        effective_splits[i] = "validation"

    # Supervised-paired redistribution: shuffle test rows across train/val/test
    # stratified by (category, label).
    if supervised_paired:
        rng_sp = np.random.default_rng(int(supervised_paired_seed))
        test_indices = [i for i, p in enumerate(pairs) if p.split == "test"]
        bucketed: dict[tuple, list[int]] = {}
        for idx in test_indices:
            key = (pairs[idx].category, pairs[idx].label)
            bucketed.setdefault(key, []).append(idx)
        for key, indices in bucketed.items():
            indices = list(indices)
            rng_sp.shuffle(indices)
            n = len(indices)
            n_test = max(1, int(round(n * supervised_paired_test_fraction))) if n >= 3 else 0
            remaining = n - n_test
            n_val_sp = max(1, int(round(remaining * 0.15 / 0.7))) if remaining >= 2 else 0
            for offset, idx in enumerate(indices):
                if offset < n_test:
                    effective_splits[idx] = "test"
                elif offset < n_test + n_val_sp:
                    effective_splits[idx] = "validation"
                else:
                    effective_splits[idx] = "train"

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
                "split": str(effective_splits[idx]),
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
    parser.add_argument("--val-fraction-of-train", type=float, default=0.15)
    parser.add_argument("--supervised-paired", action="store_true")
    parser.add_argument("--supervised-paired-seed", type=int, default=42)
    parser.add_argument("--supervised-paired-test-fraction", type=float, default=0.30)
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
        val_fraction_of_train=args.val_fraction_of_train,
        supervised_paired=args.supervised_paired,
        supervised_paired_seed=args.supervised_paired_seed,
        supervised_paired_test_fraction=args.supervised_paired_test_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    metadata["output"] = str(args.output)
    args.metadata.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
