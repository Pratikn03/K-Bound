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
    "load_pcd_points",
    "pcd_to_geometry_image",
    "load_modality_image",
    "extract_patches_from_images",
    "score_one_class_patchcore",
]


def _parse_pcd_binary(data: bytes) -> np.ndarray:
    """Parse a binary .pcd (DATA binary) into an (N,3) float32 xyz array.

    Real-IAD-D3 stores SOME categories as binary PCD (knob_cap, fork_crimp_terminal,
    telephone_spring_switch). The ASCII-only parser silently returned empty for
    these, causing a fallback to the degenerate XYZ tiff (a placeholder-constant
    artifact that leaks the OK/NG label) -- caught by the 2026-06-02 audit. This
    parses the structured binary record so geometry comes from the real cloud."""
    idx = data.find(b"DATA binary")
    if idx < 0:
        return np.zeros((0, 3), np.float32)
    header = data[:idx].decode("ascii", "replace")
    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    npts = None
    for line in header.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "FIELDS":
            fields = p[1:]
        elif p[0] == "SIZE":
            sizes = [int(x) for x in p[1:]]
        elif p[0] == "TYPE":
            types = p[1:]
        elif p[0] == "POINTS":
            npts = int(p[1])
    if not fields or len(sizes) != len(fields) or "x" not in fields:
        return np.zeros((0, 3), np.float32)
    start = data.find(b"\n", idx) + 1
    np_type = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2",
               ("U", 4): "u4", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}
    try:
        dt = np.dtype([(f, np_type[(t, s)]) for f, t, s in zip(fields, types, sizes)])
    except KeyError:
        return np.zeros((0, 3), np.float32)
    stride_bytes = dt.itemsize
    body = data[start:]
    n_avail = len(body) // stride_bytes
    if npts:
        n_avail = min(n_avail, npts)
    if n_avail == 0:
        return np.zeros((0, 3), np.float32)
    rec = np.frombuffer(body[: n_avail * stride_bytes], dtype=dt)
    xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float32)
    return xyz


def load_pcd_points(data: bytes, stride: int = 8, max_points: int = 200_000) -> np.ndarray:
    """Parse a .pcd (FIELDS x y z [rgb]) into an (N,3) float32 array, ASCII or BINARY.

    Real-IAD-D3 stores intact full 3D in the .pcd files even where the XYZ
    *tiff* is degenerate (test-split tiffs collapse X,Y to a placeholder
    constant), so geometry MUST come from the PCD. Handles both DATA ascii and
    DATA binary encodings; for binary it parses the structured record."""
    i = data.find(b"DATA ascii")
    if i < 0:
        arr = _parse_pcd_binary(data)
        if arr.shape[0] == 0:
            return arr
        if max_points and arr.shape[0] > max_points:
            arr = arr[:: arr.shape[0] // max_points + 1]
        return arr[np.isfinite(arr).all(axis=1)]
    body = data[i:].split(b"\n", 1)[1]
    lines = body.splitlines()
    if stride > 1:
        lines = lines[::stride]
    toks = b" ".join(lines).split()
    ncol = 4 if (len(toks) % 4 == 0 and len(toks) % 3 != 0) else (4 if len(toks) >= 4 else 3)
    n = (len(toks) // ncol) * ncol
    if n == 0:
        return np.zeros((0, 3), np.float32)
    arr = np.array(toks[:n], dtype=np.float32).reshape(-1, ncol)[:, :3]
    if max_points and arr.shape[0] > max_points:
        arr = arr[:: arr.shape[0] // max_points + 1]
    return arr[np.isfinite(arr).all(axis=1)]


def pcd_to_geometry_image(points: np.ndarray, size: int = 224, seed: int = 0) -> Image.Image:
    """Project a point cloud to a 3-channel geometry image via PCA.

    Projects onto the two largest-variance principal axes (the object surface
    plane); the smallest-variance axis becomes height/relief. Rasterizes the
    max-height per cell into a relief map, then composes [grad_x, grad_y, relief]
    so the CNN sees both surface orientation and protrusion -- the signal that
    isolates dents/bumps/scratches. Pose-normalized by construction (PCA frame)."""
    import cv2
    P = np.asarray(points, np.float32)
    if P.shape[0] < 64:
        return Image.fromarray(np.zeros((size, size, 3), np.uint8), "RGB")
    P = P - P.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    sub = P[rng.integers(0, P.shape[0], min(P.shape[0], 20_000))]
    _, _, vt = np.linalg.svd(sub, full_matrices=False)
    proj = P @ vt.T  # columns: pc1, pc2, pc3 (height)
    uv, h = proj[:, :2], proj[:, 2]
    lo = uv.min(0)
    span = uv.max(0) - lo
    span[span < 1e-6] = 1.0
    ij = ((uv - lo) / span * (size - 1)).astype(np.int32).clip(0, size - 1)
    relief = np.full((size, size), np.nan, np.float32)
    order = np.argsort(h)  # write ascending so max height wins per cell
    relief[ij[order, 0], ij[order, 1]] = h[order]
    valid = ~np.isnan(relief)
    if valid.any():
        plo, phi = np.percentile(relief[valid], [2, 98])
        relief = np.clip((relief - plo) / max(phi - plo, 1e-6), 0, 1)
    relief[~valid] = 0.0
    relief = relief.astype(np.float32)
    gx = cv2.Sobel(relief, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(relief, cv2.CV_32F, 0, 1, ksize=3)
    def _n(a):
        m = np.abs(a).max()
        return (a / m * 0.5 + 0.5) if m > 1e-6 else np.full_like(a, 0.5)
    img = np.stack([_n(gx), _n(gy), relief], axis=2)
    return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8), "RGB")


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


def _pcd_member_for(xyz_member: str) -> str:
    """Derive the .pcd path from the XYZ tiff path (same sample, sibling file)."""
    return xyz_member.replace("_XYZ_", "_PCD_").replace(".tiff", ".pcd").replace(".tif", ".pcd")


def load_modality_image(zf: zipfile.ZipFile, member: str, modality: str) -> Image.Image:
    """Load one modality sample from a category zip into a 3-channel PIL image.
    rgb/ps -> the JPEG directly; xyz -> a PCA relief/geometry image from the .pcd
    point cloud (the XYZ tiff is degenerate on the test split, so geometry comes
    from the intact PCD; falls back to the tiff normal image if the PCD is absent).
    """
    if modality == "xyz":
        pcd_member = _pcd_member_for(member)
        try:
            pts = load_pcd_points(zf.read(pcd_member))
            if pts.shape[0] >= 64:
                return pcd_to_geometry_image(pts)
        except KeyError:
            pass
        import tifffile
        arr = tifffile.imread(io.BytesIO(zf.read(member)))
        return xyz_to_normal_image(np.asarray(arr, dtype=np.float32))
    data = zf.read(member)
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
