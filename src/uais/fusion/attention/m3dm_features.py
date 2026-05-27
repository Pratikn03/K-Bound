"""M3DM-style ResNet-50 penultimate features for paired RGB + depth.

Replaces the lightweight image-statistic scorer with a real pretrained
backbone, applied to both RGB and depth-as-3-channel inputs. The
output is a 2048-dim penultimate feature per image, projected to a
configurable embedding dimension via PCA fitted on the train fold.

Why this exists: the M3DM line of MVTec 3D-AD work (Wang et al. 2023,
Costanzino et al. 2023) shows that pretrained features are the
single biggest empirical lever for RGB+3D anomaly fusion. Replacing the
quantile-statistic features (8 dims) with PCA-projected ResNet-50
features (D dims) lifts the absolute performance ceiling of every
fusion method that consumes them. The cross-benchmark contrast then
re-runs at a higher and fairer per-benchmark performance basis.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

_BACKBONE = None
_DEVICE = None


def _get_backbone() -> tuple[torch.nn.Module, torch.device]:
    """Lazily load ResNet-50 with ImageNet weights; cache on first call."""
    global _BACKBONE, _DEVICE
    if _BACKBONE is None:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        # Strip the final FC layer to get the 2048-dim penultimate vector.
        model.fc = torch.nn.Identity()
        model.eval()
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = model.to(device)
        _BACKBONE = (model, weights.transforms())
        _DEVICE = device
    return _BACKBONE[0], _BACKBONE[1], _DEVICE  # type: ignore[return-value]


def _load_pil(path: Path) -> Image.Image:
    """Read an RGB or single-channel image and return a 3-channel PIL Image."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        # Depth/XYZ TIFF — use tifffile if available, otherwise PIL.
        try:
            import tifffile

            arr = tifffile.imread(path)
        except Exception:
            arr = np.asarray(Image.open(path))
        arr = np.asarray(arr, dtype=np.float32)
        # Normalize to [0, 255]; collapse extra channels to a magnitude scalar.
        if arr.ndim == 3 and arr.shape[-1] in (2, 3, 4):
            arr = np.linalg.norm(arr[..., :3], axis=-1)
        # Drop NaN/Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo < 1e-9:
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            arr = (arr - lo) / (hi - lo)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        rgb = np.stack([arr, arr, arr], axis=-1)
        return Image.fromarray(rgb, mode="RGB")
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def extract_point_cloud_statistics(path: Path) -> np.ndarray:
    """Extract PointNet++ style global geometric descriptors and shape statistics from raw XYZ.

    Returns a 12-dimensional vector:
      [linearity, planarity, sphericity, anisotropy, curvature, eigenentropy,
       mean_x, mean_y, mean_z, std_x, std_y, std_z]
    """
    try:
        import tifffile

        arr = tifffile.imread(path)
    except Exception:
        try:
            arr = np.asarray(Image.open(path))
        except Exception:
            return np.zeros(12, dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32)
    # The MVTec 3D-AD tiff contains [H, W, 3] float coordinates
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return np.zeros(12, dtype=np.float32)

    pts = arr.reshape(-1, arr.shape[-1])[..., :3]
    # Drop NaNs, Infs, and zero points (background)
    valid = np.all(np.isfinite(pts), axis=1) & (np.linalg.norm(pts, axis=1) > 1e-6)
    pts = pts[valid]

    if pts.shape[0] < 4:
        return np.zeros(12, dtype=np.float32)

    # Center of mass and std
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)

    # Global Covariance PCA
    cov = np.cov(pts.T)
    evals, _ = np.linalg.eigh(cov)
    evals = np.sort(evals)[::-1]  # descending order: l1 >= l2 >= l3
    evals = np.maximum(evals, 1e-12)

    l1, l2, l3 = evals[0], evals[1], evals[2]
    sum_l = l1 + l2 + l3
    p = evals / sum_l

    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    sphericity = l3 / l1
    anisotropy = (l1 - l3) / l1
    curvature = l3 / sum_l
    # Clamp p to avoid log(0)
    p = np.clip(p, 1e-12, 1.0)
    eigenentropy = -np.sum(p * np.log(p))

    return np.array(
        [
            linearity,
            planarity,
            sphericity,
            anisotropy,
            curvature,
            eigenentropy,
            mean[0],
            mean[1],
            mean[2],
            std[0],
            std[1],
            std[2],
        ],
        dtype=np.float32,
    )


def extract_resnet_features(
    paths: Sequence[Path],
    batch_size: int = 32,
) -> np.ndarray:
    """Return [N, 2048] or [N, 2060] float32 features.

    If paths contain XYZ tiff files, we append 12 PointNet++ style global
    geometric shape statistics, yielding 2060 dimensions.
    """
    model, transform, device = _get_backbone()
    is_depth = any(Path(p).suffix.lower() in {".tif", ".tiff"} for p in paths)
    out_dim = 2060 if is_depth else 2048
    features = np.zeros((len(paths), out_dim), dtype=np.float32)

    batch_tensors: list[torch.Tensor] = []
    batch_indices: list[int] = []

    for idx, path in enumerate(paths):
        p_path = Path(path)
        try:
            image = _load_pil(p_path)
        except Exception:
            continue

        if is_depth:
            # Extract 12 PointNet++ style global geometric descriptors
            stats_vec = extract_point_cloud_statistics(p_path)
            features[idx, 2048:] = stats_vec

        batch_tensors.append(transform(image))
        batch_indices.append(idx)
        if len(batch_tensors) >= batch_size:
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                out = model(batch).cpu().numpy()
            for j, target in enumerate(batch_indices):
                features[target, :2048] = out[j]
            batch_tensors = []
            batch_indices = []

    if batch_tensors:
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            out = model(batch).cpu().numpy()
        for j, target in enumerate(batch_indices):
            features[target, :2048] = out[j]

    return features


def fit_pca_projection(
    features: np.ndarray,
    fit_mask: np.ndarray,
    embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on the train fold; return (proj_features, mean, components, scale)."""
    fit_rows = features[fit_mask]
    if fit_rows.shape[0] < embedding_dim:
        # Degenerate — fall back to truncating the raw features.
        proj = features[:, :embedding_dim].astype(np.float32)
        mean = np.zeros(features.shape[1], dtype=np.float32)
        components = np.eye(features.shape[1], embedding_dim, dtype=np.float32).T
        scale = np.ones(embedding_dim, dtype=np.float32)
        return proj.astype(np.float32), mean, components, scale
    pca = PCA(n_components=embedding_dim, random_state=0)
    pca.fit(fit_rows)
    proj = pca.transform(features).astype(np.float32)
    # Per-component scaling to roughly [0, 1] over the fit fold for
    # downstream attention compatibility.
    fit_proj = pca.transform(fit_rows)
    lo = fit_proj.min(axis=0)
    hi = fit_proj.max(axis=0)
    span = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
    proj = (proj - lo) / span
    proj = np.clip(proj, 0.0, 1.0).astype(np.float32)
    return proj, pca.mean_.astype(np.float32), pca.components_.astype(np.float32), span.astype(np.float32)


def normal_reference_distance_score(
    features: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    """Mahalanobis-style distance from the train-good centroid, scaled to [0, 1]."""
    fit_rows = features[fit_mask]
    if fit_rows.shape[0] == 0:
        return np.zeros(features.shape[0], dtype=np.float32)
    center = fit_rows.mean(axis=0)
    scale = fit_rows.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    distances = np.linalg.norm((features - center) / scale, axis=1).astype(np.float32)
    fit_distances = distances[fit_mask]
    lo = float(np.min(fit_distances))
    hi = float(np.percentile(fit_distances, 95))
    if hi - lo <= 1e-9:
        hi = lo + 1.0
    return np.clip((distances - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def patchcore_knn_score(
    features: np.ndarray,
    fit_mask: np.ndarray,
    k: int = 3,
    coreset_size: int | None = None,
    random_state: int = 0,
) -> np.ndarray:
    """PatchCore-style kNN distance to a normal-only memory bank.

    The memory bank is the set of normal (train-good) feature rows. For each
    query feature, the score is the mean Euclidean distance to its k nearest
    bank entries. Unlike the single-centroid Mahalanobis variant, this places
    no unimodality assumption on the normal distribution, which is the
    behaviour the MVTec 3D-AD one-class protocol actually requires.

    The optional coreset subsamples the memory bank for runtime/memory savings;
    PatchCore originally uses greedy farthest-point selection but a uniform
    random subsample is sufficient for the score-fusion pipeline here.
    """
    n = features.shape[0]
    fit_rows = features[fit_mask]
    if fit_rows.shape[0] == 0:
        return np.zeros(n, dtype=np.float32)

    if coreset_size is not None and fit_rows.shape[0] > coreset_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(fit_rows.shape[0], size=coreset_size, replace=False)
        bank = fit_rows[idx]
    else:
        bank = fit_rows

    bank = bank.astype(np.float32, copy=False)
    queries = features.astype(np.float32, copy=False)
    bank_sq = np.sum(bank * bank, axis=1)
    chunk = 256
    distances = np.zeros(n, dtype=np.float32)
    k_eff = max(1, min(int(k), bank.shape[0]))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        q = queries[start:end]
        q_sq = np.sum(q * q, axis=1, keepdims=True)
        d2 = q_sq + bank_sq[None, :] - 2.0 * q @ bank.T
        d2 = np.clip(d2, 0.0, None)
        partitioned = np.partition(d2, k_eff - 1, axis=1)[:, :k_eff]
        distances[start:end] = np.sqrt(np.mean(partitioned, axis=1))

    fit_distances = distances[fit_mask]
    lo = float(np.min(fit_distances))
    hi = float(np.percentile(fit_distances, 95))
    if hi - lo <= 1e-9:
        hi = lo + 1.0
    return np.clip((distances - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


__all__ = [
    "extract_resnet_features",
    "fit_pca_projection",
    "normal_reference_distance_score",
    "patchcore_knn_score",
]
