"""Strong per-modality detectors for Real-IAD-D3 (RGB + photometric-stereo + XYZ).

The lightweight handcrafted scorer in ``prepare_realiad_d3_headroom_inputs.py``
left every modality near chance (rgb 0.52, ps 0.55, xyz 0.48 pooled). The XYZ
collapse is the worst and the most diagnostic: that pipeline ran *image* colour/
texture statistics on a geometric coordinate map, throwing away all 3D structure.

This module replaces it with the genuine detector pattern that worked elsewhere
in the project (patch-level PatchCore over deep features), and adds the missing
geometric front-end for the point cloud:

  - RGB / PS  : deep patch features (ResNet-50 layer2+layer3) -> per-category
                PatchCore memory bank -> max-patch anomaly score.
  - XYZ       : the organized (H,W,3) point map is converted to a SURFACE-NORMAL
                image (cross product of local tangents), then fed through the same
                backbone. Normals encode local geometry (holes, dents, bumps) that
                coordinate magnitude cannot, so defects become visible to the CNN.

Scoring is one-class per category (train-OK -> bank), with monotone z-sigmoid
normalization on the train-OK score distribution (no min-max clipping).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from uais.fusion.attention.patchcore_patch import (
    _get_backbone,
    greedy_coreset,
    image_anomaly_scores,
)

__all__ = [
    "xyz_to_normal_image",
    "load_modality_image",
    "extract_patches_from_images",
    "score_one_class_patchcore",
]


def xyz_to_normal_image(xyz: np.ndarray, work_size: int = 512) -> Image.Image:
    """Convert an organized (H,W,3) XYZ point map to a 3-channel surface-normal
    image. Normals are computed from central differences of the point map
    (no neighbour search needed -- the map is already a grid), normalized, and
    mapped from [-1,1] to [0,255]. Invalid/zero-return points get a zero normal.
    """
    arr = np.asarray(xyz, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    # Work at a moderate resolution: enough to keep fine geometry, cheap to process.
    import cv2
    if max(arr.shape[:2]) > work_size:
        arr = cv2.resize(arr, (work_size, work_size), interpolation=cv2.INTER_AREA)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Local tangents via central differences along columns (u) and rows (v).
    du = np.zeros_like(arr)
    dv = np.zeros_like(arr)
    du[:, 1:-1, :] = (arr[:, 2:, :] - arr[:, :-2, :]) * 0.5
    dv[1:-1, :, :] = (arr[2:, :, :] - arr[:-2, :, :]) * 0.5
    normals = np.cross(du, dv)
    mag = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.divide(normals, mag, out=np.zeros_like(normals), where=mag > 1e-8)
    valid = (np.linalg.norm(arr, axis=2) > 1e-6)[:, :, None]
    normals = np.where(valid, normals, 0.0)
    img = ((normals * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGB")


def load_modality_image(zf: zipfile.ZipFile, member: str, modality: str) -> Image.Image:
    """Load one modality sample from a category zip into a 3-channel PIL image.
    rgb/ps -> the JPEG directly; xyz -> the surface-normal image of the TIFF."""
    data = zf.read(member)
    if modality == "xyz":
        import tifffile
        arr = tifffile.imread(io.BytesIO(data))
        return xyz_to_normal_image(np.asarray(arr, dtype=np.float32))
    im = Image.open(io.BytesIO(data))
    return im.convert("RGB") if im.mode != "RGB" else im


@torch.no_grad()
def extract_patches_from_images(
    images: Sequence[Image.Image],
    *,
    batch_size: int = 8,
    patch_grid: int = 28,
    neighbourhood: int = 3,
) -> tuple[np.ndarray, int]:
    """Patch embeddings (ResNet-50 layer2+layer3) for pre-loaded PIL images.

    Mirrors patchcore_patch.extract_patch_embeddings but consumes images directly
    so the XYZ surface-normal image (built in memory) can be scored without a
    round-trip to disk. Returns (all_patches [N*P, 1536], n_patches P).
    """
    model, feats, dev, transform = _get_backbone()
    per_image: list[np.ndarray] = []
    n_patches = patch_grid * patch_grid
    buf: list[torch.Tensor] = []

    def _flush() -> None:
        if not buf:
            return
        batch = torch.stack(buf).to(dev)
        feats.clear()
        _ = model(batch)
        f2 = feats["layer2"]
        f3 = feats["layer3"]
        pad = neighbourhood // 2
        f2 = F.avg_pool2d(f2, kernel_size=neighbourhood, stride=1, padding=pad)
        f3 = F.avg_pool2d(f3, kernel_size=neighbourhood, stride=1, padding=pad)
        f2 = F.interpolate(f2, size=(patch_grid, patch_grid), mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, size=(patch_grid, patch_grid), mode="bilinear", align_corners=False)
        cat = torch.cat([f2, f3], dim=1)
        B, C, G, _ = cat.shape
        cat = cat.permute(0, 2, 3, 1).reshape(B, G * G, C).cpu().numpy().astype(np.float32)
        for b in range(B):
            per_image.append(cat[b])
        buf.clear()

    for im in images:
        try:
            buf.append(transform(im))
        except Exception:
            per_image.append(np.zeros((n_patches, 1536), dtype=np.float32))
            continue
        if len(buf) >= batch_size:
            _flush()
    _flush()
    all_patches = np.concatenate(per_image, axis=0) if per_image else np.zeros((0, 1536), np.float32)
    return all_patches, n_patches


def score_one_class_patchcore(
    train_images: Sequence[Image.Image],
    eval_images: Sequence[Image.Image],
    *,
    coreset_size: int = 4096,
    batch_size: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """One-class PatchCore image scores for eval_images against a memory bank
    built from train_images (assumed all-normal). Returns raw max-patch distances
    (higher = more anomalous); caller applies z-sigmoid normalization."""
    train_patches, _ = extract_patches_from_images(train_images, batch_size=batch_size)
    bank = greedy_coreset(train_patches, coreset_size, seed=seed) if train_patches.shape[0] > coreset_size else train_patches
    eval_patches, n_patches = extract_patches_from_images(eval_images, batch_size=batch_size)
    return image_anomaly_scores(eval_patches, n_patches, bank)
