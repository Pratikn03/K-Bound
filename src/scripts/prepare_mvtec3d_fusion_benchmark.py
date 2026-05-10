"""Prepare MVTec 3D-AD (bagel subset) as a two-domain fusion benchmark.

Extracts two complementary feature domains from each scan:
  rgb        — ResNet-18 penultimate-layer embedding (512-dim, pretrained ImageNet)
               anomaly score = cosine distance from mean normal training embedding
  pointcloud — Depth-map statistics (16-dim: percentiles + local variance)
               anomaly score = Mahalanobis distance from normal training distribution

Output: data/mvtec3d/bagel_fusion.csv  (or --output path)
Schema : sample_id, domain, label, score, embedding_0 .. embedding_N

Usage
-----
# After downloading and extracting the MVTec 3D-AD dataset:
python src/scripts/prepare_mvtec3d_fusion_benchmark.py \\
    --data-root /path/to/mvtec_3d_anomaly_detection \\
    --category  bagel \\
    --output    data/mvtec3d/bagel_fusion.csv

# Quick smoke-run with a randomly generated synthetic dataset:
python src/scripts/prepare_mvtec3d_fusion_benchmark.py --synthetic

Dataset download
----------------
MVTec 3D-AD is available free of charge at:
  https://www.mvtec.com/company/research/datasets/mvtec-3d-ad

After downloading, extract the ZIP.  The expected directory layout is:
  <data-root>/
    <category>/
      train/
        good/
          rgb/   *.png
          xyz/   *.tiff        (depth + normals, 3-channel float32 TIFF)
      test/
        good/
          rgb/   *.png
          xyz/   *.tiff
        <defect_class>/
          rgb/   *.png
          xyz/   *.tiff
          gt/    *.png         (ground-truth mask)

Dependencies: torch, torchvision, Pillow, numpy (+ tifffile for depth maps)
  pip install torch torchvision Pillow tifffile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy imports — fail gracefully with actionable message
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        import torchvision
        return torch, torchvision
    except ImportError:
        print("[error] torch and torchvision are required:\n"
              "  pip install torch torchvision")
        sys.exit(1)


def _require_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        print("[error] Pillow is required:\n  pip install Pillow")
        sys.exit(1)


def _require_tifffile():
    try:
        import tifffile
        return tifffile
    except ImportError:
        # Fall back to numpy for simple depth TIFFs
        return None


# ---------------------------------------------------------------------------
# RGB feature extractor
# ---------------------------------------------------------------------------

def _build_rgb_extractor():
    """Return a pretrained ResNet-18 feature extractor (penultimate layer)."""
    torch, torchvision = _require_torch()
    from torchvision import models, transforms

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()   # drop classifier head → 512-dim output
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def _extract_rgb_features(image_paths: List[Path]) -> np.ndarray:
    """Extract ResNet-18 features for a list of image paths. Returns [N, 512]."""
    torch, _ = _require_torch()
    Image = _require_pil()
    model, transform = _build_rgb_extractor()

    features = []
    with torch.no_grad():
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            x = transform(img).unsqueeze(0)   # [1, 3, 224, 224]
            feat = model(x).squeeze(0).numpy()  # [512]
            features.append(feat)
    return np.stack(features, axis=0)


# ---------------------------------------------------------------------------
# Depth-map (3D / XYZ) feature extractor
# ---------------------------------------------------------------------------

def _extract_depth_features(depth_paths: List[Path]) -> np.ndarray:
    """Extract 16-dim statistics from depth/XYZ TIFF files.

    Each TIFF is a 3-channel float32 image (X, Y, Z coordinates).
    We use the Z channel (depth) and compute robust statistics.

    Returns [N, 16].
    """
    tifffile = _require_tifffile()

    features = []
    for p in depth_paths:
        try:
            if tifffile is not None:
                data = tifffile.imread(str(p))   # [H, W, 3] float32
            else:
                data = np.array(
                    __import__("PIL").Image.open(p), dtype=np.float32
                )
            if data.ndim == 3:
                z = data[:, :, 2].ravel()   # Z channel
            else:
                z = data.ravel()
            z = z[np.isfinite(z)]
            if len(z) == 0:
                z = np.zeros(1, dtype=np.float32)

            percentiles = np.percentile(z, [5, 10, 25, 50, 75, 90, 95]).tolist()
            mean, std = float(z.mean()), float(z.std() + 1e-8)
            # Local variance: compute patch-level std and aggregate
            grid = z.reshape(-1, min(256, max(1, len(z) // 16 + 1)))
            patch_stds = grid.std(axis=1)
            p_stds = np.percentile(patch_stds, [25, 50, 75]).tolist()
            feat = [mean, std, float(z.min()), float(z.max())] + percentiles + p_stds + \
                   [float(np.median(np.abs(z - np.median(z))))]  # MAD
            features.append(feat[:16])  # cap at 16 dims
        except Exception as exc:
            print(f"[warn] Could not load depth file {p}: {exc}")
            features.append([0.0] * 16)

    arr = np.array(features, dtype=np.float32)
    # Pad to exactly 16 dims if needed
    if arr.shape[1] < 16:
        pad = np.zeros((arr.shape[0], 16 - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr[:, :16]


# ---------------------------------------------------------------------------
# Anomaly score computation
# ---------------------------------------------------------------------------

def _cosine_anomaly_score(
    features: np.ndarray,           # [N, D]
    reference_features: np.ndarray, # [N_ref, D]
) -> np.ndarray:
    """Per-sample cosine distance from the mean normal-class embedding."""
    ref_mean = reference_features.mean(axis=0, keepdims=True)   # [1, D]
    ref_norm = ref_mean / (np.linalg.norm(ref_mean) + 1e-8)
    feat_norms = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    cosine_sim = (feat_norms * ref_norm).sum(axis=1)             # [N]
    score = (1.0 - cosine_sim) / 2.0                            # map [-1,1] → [0,1]
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _mahalanobis_anomaly_score(
    features: np.ndarray,           # [N, D]
    reference_features: np.ndarray, # [N_ref, D]
) -> np.ndarray:
    """Per-sample Mahalanobis distance from the normal-class distribution, scaled to [0,1]."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(reference_features)
    prec = lw.precision_                               # [D, D]
    diff = features - reference_features.mean(axis=0)  # [N, D]
    # Batched: score_i = sqrt(diff_i^T @ prec @ diff_i)
    maha_sq = np.einsum("nd,de,ne->n", diff, prec, diff)
    maha = np.sqrt(np.clip(maha_sq, 0.0, None))
    # Normalise using training set 99th percentile to bring into [0,1]
    ref_diff = reference_features - reference_features.mean(axis=0)
    ref_maha = np.sqrt(np.clip(np.einsum("nd,de,ne->n", ref_diff, prec, ref_diff), 0.0, None))
    p99 = float(np.percentile(ref_maha, 99)) + 1e-8
    return np.clip(maha / p99, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def _collect_samples(category_root: Path) -> Tuple[List[Path], List[Path], List[int]]:
    """Return (rgb_paths, depth_paths, labels) for all test samples.

    Also returns the train/good paths for computing reference statistics.
    """
    rgb_paths, depth_paths, labels = [], [], []

    def _add_split(split_dir: Path, label: int) -> None:
        if not split_dir.exists():
            return
        rgb_dir = split_dir / "rgb"
        xyz_dir = split_dir / "xyz"
        if not rgb_dir.exists():
            rgb_dir = split_dir   # some versions put images directly
        img_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
        for img_path in img_files:
            stem = img_path.stem
            depth_candidates = list((xyz_dir if xyz_dir.exists() else split_dir).glob(f"{stem}.*"))
            if depth_candidates:
                rgb_paths.append(img_path)
                depth_paths.append(depth_candidates[0])
                labels.append(label)

    test_dir = category_root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    _add_split(test_dir / "good", label=0)
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            _add_split(defect_dir, label=1)

    return rgb_paths, depth_paths, labels


def _collect_train_normal(category_root: Path) -> Tuple[List[Path], List[Path]]:
    """Return (rgb_paths, depth_paths) for training/good normal samples."""
    train_good = category_root / "train" / "good"
    rgb_dir = train_good / "rgb"
    xyz_dir = train_good / "xyz"
    if not rgb_dir.exists():
        rgb_dir = train_good
    img_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
    rgb_paths, depth_paths = [], []
    for img_path in img_files:
        stem = img_path.stem
        depth_candidates = list((xyz_dir if xyz_dir.exists() else train_good).glob(f"{stem}.*"))
        if depth_candidates:
            rgb_paths.append(img_path)
            depth_paths.append(depth_candidates[0])
    return rgb_paths, depth_paths


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_benchmark(
    category_root: Path,
    output_path: Path,
    pca_rgb_dims: int = 32,
    seed: int = 42,
) -> pd.DataFrame:
    """Extract features, compute scores, and save fusion CSV."""
    print(f"Processing category: {category_root.name}")

    # --- Training normals (reference distribution) ---
    print("  Loading train/good samples for reference statistics ...")
    train_rgb_paths, train_depth_paths = _collect_train_normal(category_root)
    if not train_rgb_paths:
        raise FileNotFoundError(
            f"No train/good images found under {category_root}. "
            "Check --data-root and --category."
        )
    print(f"  Found {len(train_rgb_paths)} normal training samples.")

    train_rgb_feat = _extract_rgb_features(train_rgb_paths)
    train_depth_feat = _extract_depth_features(train_depth_paths)

    # PCA-reduce RGB to pca_rgb_dims for embedding column
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    n_pca = min(pca_rgb_dims, train_rgb_feat.shape[0] - 1, train_rgb_feat.shape[1])
    pca = PCA(n_components=n_pca, random_state=seed)
    rgb_scaler = StandardScaler()
    train_rgb_scaled = rgb_scaler.fit_transform(train_rgb_feat)
    pca.fit(train_rgb_scaled)

    depth_scaler = StandardScaler()
    depth_scaler.fit(train_depth_feat)

    # --- Test samples ---
    print("  Loading test samples ...")
    test_rgb_paths, test_depth_paths, labels = _collect_samples(category_root)
    if not test_rgb_paths:
        raise FileNotFoundError(f"No test samples found under {category_root / 'test'}.")
    print(f"  Found {len(test_rgb_paths)} test samples "
          f"({sum(labels)} anomalous, {len(labels) - sum(labels)} normal).")

    test_rgb_feat = _extract_rgb_features(test_rgb_paths)
    test_depth_feat = _extract_depth_features(test_depth_paths)

    # Combine train + test for embedding (fit on train only, transform all)
    all_rgb = np.concatenate([train_rgb_feat, test_rgb_feat], axis=0)
    all_rgb_scaled = rgb_scaler.transform(all_rgb)
    all_rgb_embed = pca.transform(all_rgb_scaled)
    test_rgb_embed = all_rgb_embed[len(train_rgb_feat):]

    all_depth = np.concatenate([train_depth_feat, test_depth_feat], axis=0)
    all_depth_scaled = depth_scaler.transform(all_depth)
    test_depth_scaled = all_depth_scaled[len(train_depth_feat):]
    train_depth_scaled_ref = all_depth_scaled[:len(train_depth_feat)]

    # Anomaly scores
    rgb_scores = _cosine_anomaly_score(
        rgb_scaler.transform(test_rgb_feat),
        rgb_scaler.transform(train_rgb_feat),
    )
    depth_scores = _mahalanobis_anomaly_score(train_depth_scaled_ref, train_depth_scaled_ref)
    # Recompute depth scores for test set
    depth_scores = _mahalanobis_anomaly_score(test_depth_scaled, train_depth_scaled_ref)

    # Build rows
    rows = []
    n_rgb_embed = test_rgb_embed.shape[1]
    n_depth_embed = test_depth_scaled.shape[1]

    for i, label in enumerate(labels):
        sample_id = i

        # RGB row
        rgb_row = {
            "sample_id": sample_id, "domain": "rgb",
            "label": label, "score": float(rgb_scores[i]),
            "confidence": float(abs(rgb_scores[i] - 0.5) * 2.0),
        }
        for ei in range(n_rgb_embed):
            rgb_row[f"embedding_{ei}"] = float(test_rgb_embed[i, ei])
        rows.append(rgb_row)

        # Pointcloud row
        pc_row = {
            "sample_id": sample_id, "domain": "pointcloud",
            "label": label, "score": float(depth_scores[i]),
            "confidence": float(abs(depth_scores[i] - 0.5) * 2.0),
        }
        for ei in range(n_depth_embed):
            pc_row[f"embedding_{ei}"] = float(test_depth_scaled[i, ei])
        rows.append(pc_row)

    df = pd.DataFrame(rows)
    embed_cols = sorted([c for c in df.columns if c.startswith("embedding_")])
    df = df[["sample_id", "domain", "label", "score", "confidence"] + embed_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    n_samples = df["sample_id"].nunique()
    print(f"\nSaved {len(df)} rows ({n_samples} samples × 2 domains) to {output_path}")
    print(f"  Anomaly rate: {np.mean(labels):.3f}")
    return df


# ---------------------------------------------------------------------------
# Synthetic fallback (smoke-test without real data)
# ---------------------------------------------------------------------------

def build_synthetic_benchmark(
    output_path: Path,
    n_samples: int = 120,
    n_embed_rgb: int = 16,
    n_embed_depth: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic MVTec-like benchmark for smoke-testing."""
    rng = np.random.default_rng(seed)
    n_pos = int(n_samples * 0.35)
    n_neg = n_samples - n_pos
    labels = [1] * n_pos + [0] * n_neg

    rows = []
    for i, label in enumerate(labels):
        for domain, n_embed, base_score in [
            ("rgb", n_embed_rgb, 0.6 if label else 0.2),
            ("pointcloud", n_embed_depth, 0.55 if label else 0.25),
        ]:
            noise = rng.normal(0, 0.08)
            score = float(np.clip(base_score + noise, 0.0, 1.0))
            row = {
                "sample_id": i, "domain": domain,
                "label": label, "score": score,
                "confidence": float(abs(score - 0.5) * 2.0),
            }
            embed = rng.normal(float(label) * 0.5, 0.3, size=n_embed)
            for ei, ev in enumerate(embed):
                row[f"embedding_{ei}"] = float(ev)
            rows.append(row)

    df = pd.DataFrame(rows)
    embed_cols = sorted([c for c in df.columns if c.startswith("embedding_")])
    df = df[["sample_id", "domain", "label", "score", "confidence"] + embed_cols]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic MVTec3D benchmark saved to {output_path} "
          f"({n_samples} samples, {n_pos} anomalous)")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MVTec 3D-AD (bagel) as a two-domain fusion benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Root of extracted MVTec 3D-AD dataset")
    parser.add_argument("--category", type=str, default="bagel",
                        help="Category subfolder name (default: bagel)")
    parser.add_argument("--output", type=Path,
                        default=Path("data/mvtec3d/bagel_fusion.csv"))
    parser.add_argument("--pca-rgb-dims", type=int, default=32,
                        help="PCA dimensions for RGB embedding (default: 32)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate a synthetic dataset instead (no real data needed)")
    args = parser.parse_args()

    if args.synthetic:
        build_synthetic_benchmark(output_path=args.output, seed=args.seed)
        return

    if args.data_root is None:
        print(
            "[error] --data-root is required unless --synthetic is set.\n"
            "Download MVTec 3D-AD from:\n"
            "  https://www.mvtec.com/company/research/datasets/mvtec-3d-ad\n"
            "Then run:\n"
            "  python src/scripts/prepare_mvtec3d_fusion_benchmark.py "
            "--data-root /path/to/mvtec_3d_anomaly_detection"
        )
        sys.exit(1)

    category_root = args.data_root / args.category
    if not category_root.exists():
        print(f"[error] Category directory not found: {category_root}")
        sys.exit(1)

    build_benchmark(
        category_root=category_root,
        output_path=args.output,
        pca_rgb_dims=args.pca_rgb_dims,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
