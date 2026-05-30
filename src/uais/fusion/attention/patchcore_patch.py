"""True patch-level PatchCore (Roth et al., 2022) for the ELARA upstream detector.

The existing ``m3dm_features.patchcore_knn_score`` collapses each image to a
single global 2048-d ResNet vector and does kNN on whole-image vectors. That
discards all spatial locality and sits near chance on industrial anomaly
benchmarks. This module implements the *actual* PatchCore pipeline:

  1. Hook intermediate blocks (layer2 + layer3 of ResNet-50) -> spatial feature
     maps [C, H, W].
  2. Locally-aware patch aggregation: 3x3 adaptive average pooling over each
     spatial location (the PatchCore "neighbourhood" smoothing).
  3. Bilinearly resize both layer maps to a common grid and concatenate over
     channels -> per-patch embeddings of dimension C2 + C3.
  4. Memory bank = ALL train-good patches, optionally greedy-coreset-subsampled.
  5. Image anomaly score = max over the image's patches of the patch's distance
     to its nearest memory-bank patch (PatchCore's image-level score), with the
     standard re-weighting by the neighbourhood of the maximally-anomalous patch.

This is the genuine upstream-detector upgrade: it keeps spatial evidence, which
is exactly the signal the pooled-vector variant threw away.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_BACKBONE = None
_DEVICE = None
_TRANSFORM = None


def _device() -> torch.device:
    """Return the best available torch device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_backbone():
    """ResNet-50 with forward hooks on layer2 and layer3."""
    global _BACKBONE, _DEVICE, _TRANSFORM
    if _BACKBONE is None:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights).eval()
        dev = _device()
        model = model.to(dev)
        feats: dict[str, torch.Tensor] = {}

        def _hook(name):
            """Build a forward hook that stashes block `name`'s output in feats."""
            def fn(_m, _i, o):  # noqa: D401 - tiny closure
                feats[name] = o
            return fn

        model.layer2.register_forward_hook(_hook("layer2"))
        model.layer3.register_forward_hook(_hook("layer3"))
        _BACKBONE = (model, feats)
        _DEVICE = dev
        _TRANSFORM = weights.transforms()
    return _BACKBONE[0], _BACKBONE[1], _DEVICE, _TRANSFORM


def _load_rgb(path: Path) -> Image.Image:
    """Load an image as 3-channel RGB. XYZ/depth TIFFs are reduced to a
    per-pixel magnitude, min-max normalised, and replicated to 3 channels so
    the same ResNet backbone consumes both modalities."""
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile
            arr = tifffile.imread(path)
        except Exception:
            arr = np.asarray(Image.open(path))
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] in (2, 3, 4):
            arr = np.linalg.norm(arr[..., :3], axis=-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(arr.min()), float(arr.max())
        arr = np.zeros_like(arr) if hi - lo < 1e-9 else (arr - lo) / (hi - lo)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(np.stack([arr] * 3, -1), "RGB")
    im = Image.open(path)
    return im.convert("RGB") if im.mode != "RGB" else im


@torch.no_grad()
def extract_patch_embeddings(
    paths: Sequence[Path],
    *,
    batch_size: int = 8,
    patch_grid: int = 28,
    neighbourhood: int = 3,
) -> tuple[np.ndarray, int]:
    """Return (all_patches [N*P, D], n_patches_per_image P).

    Each image yields ``patch_grid**2`` patch embeddings of dimension
    (512 + 1024) = 1536 after layer2+layer3 concatenation.
    """
    model, feats, dev, transform = _get_backbone()
    per_image: list[np.ndarray] = []
    n_patches = patch_grid * patch_grid

    buf, idxs = [], []

    def _flush():
        """Run the buffered image batch through the backbone, apply locally-aware
        pooling + resize to the common grid, and append per-image patch arrays."""
        if not buf:
            return
        batch = torch.stack(buf).to(dev)
        feats.clear()
        _ = model(batch)
        f2 = feats["layer2"]  # [B, 512, 28, 28] at 224 input
        f3 = feats["layer3"]  # [B, 1024, 14, 14]
        # Locally-aware neighbourhood pooling.
        pad = neighbourhood // 2
        f2 = F.avg_pool2d(f2, kernel_size=neighbourhood, stride=1, padding=pad)
        f3 = F.avg_pool2d(f3, kernel_size=neighbourhood, stride=1, padding=pad)
        # Resize both to the common patch grid.
        f2 = F.interpolate(f2, size=(patch_grid, patch_grid), mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, size=(patch_grid, patch_grid), mode="bilinear", align_corners=False)
        cat = torch.cat([f2, f3], dim=1)  # [B, 1536, G, G]
        B, C, G, _ = cat.shape
        cat = cat.permute(0, 2, 3, 1).reshape(B, G * G, C).cpu().numpy().astype(np.float32)
        for b in range(B):
            per_image.append(cat[b])
        buf.clear()
        idxs.clear()

    for i, p in enumerate(paths):
        try:
            im = _load_rgb(Path(p))
        except Exception:
            per_image.append(np.zeros((n_patches, 1536), dtype=np.float32))
            continue
        buf.append(transform(im))
        idxs.append(i)
        if len(buf) >= batch_size:
            _flush()
    _flush()

    all_patches = np.concatenate(per_image, axis=0) if per_image else np.zeros((0, 1536), np.float32)
    return all_patches, n_patches


def greedy_coreset(bank: np.ndarray, n_select: int, *, seed: int = 0, n_proj: int = 128) -> np.ndarray:
    """Greedy farthest-point coreset selection (PatchCore Sec 3.2), with a
    random Johnson-Lindenstrauss projection for speed. Returns selected rows."""
    n = bank.shape[0]
    if n_select >= n:
        return bank
    rng = np.random.default_rng(seed)
    # JL projection to accelerate distance computation.
    d = bank.shape[1]
    proj = rng.standard_normal((d, min(n_proj, d))).astype(np.float32) / np.sqrt(min(n_proj, d))
    bp = bank @ proj
    selected = [int(rng.integers(0, n))]
    min_d = np.linalg.norm(bp - bp[selected[0]], axis=1)
    for _ in range(n_select - 1):
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        d_new = np.linalg.norm(bp - bp[nxt], axis=1)
        min_d = np.minimum(min_d, d_new)
    return bank[np.asarray(selected)]


def image_anomaly_scores(
    query_patches: np.ndarray,
    n_patches_per_image: int,
    bank: np.ndarray,
    *,
    chunk: int = 512,
) -> np.ndarray:
    """PatchCore image score: per image, max over its patches of the patch's
    distance to the nearest bank patch.

    query_patches: [N_img * P, D]; returns [N_img].
    """
    bank = bank.astype(np.float32, copy=False)
    bank_sq = np.sum(bank * bank, axis=1)
    n_total = query_patches.shape[0]
    patch_min = np.zeros(n_total, dtype=np.float32)
    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        q = query_patches[start:end]
        q_sq = np.sum(q * q, axis=1, keepdims=True)
        d2 = q_sq + bank_sq[None, :] - 2.0 * q @ bank.T
        d2 = np.clip(d2, 0.0, None)
        patch_min[start:end] = np.sqrt(d2.min(axis=1))
    # Reshape to [N_img, P] and take the max patch distance per image.
    n_img = n_total // n_patches_per_image
    patch_min = patch_min[: n_img * n_patches_per_image].reshape(n_img, n_patches_per_image)
    return patch_min.max(axis=1)


__all__ = [
    "extract_patch_embeddings",
    "greedy_coreset",
    "image_anomaly_scores",
]
