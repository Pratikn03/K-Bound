"""Regression tests for the Real-IAD-3D detector front-end.

Pins the 2026-06-02 audit fix: binary .pcd files must parse to real points (not
silently fall back to the degenerate XYZ tiff, which leaks the OK/NG label).
"""

from __future__ import annotations

import struct

import numpy as np

from uais.fusion.attention.realiad_3d_detector import (
    load_pcd_points,
    pcd_to_geometry_image,
    xyz_to_normal_image,
)


def _make_pcd(points: np.ndarray, ascii_mode: bool) -> bytes:
    n = len(points)
    header = (
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n"
        "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA %s\n"
    ) % (n, n, "ascii" if ascii_mode else "binary")
    if ascii_mode:
        body = "".join(f"{x} {y} {z} 0\n" for x, y, z in points)
        return header.encode() + body.encode()
    buf = b"".join(struct.pack("<ffff", x, y, z, 0.0) for x, y, z in points)
    return header.encode() + buf


def test_load_pcd_ascii():
    pts = np.random.default_rng(0).standard_normal((500, 3)).astype(np.float32)
    out = load_pcd_points(_make_pcd(pts, ascii_mode=True), stride=1)
    assert out.shape[0] >= 400 and out.shape[1] == 3


def test_load_pcd_binary_does_not_fall_back_to_empty():
    """The audit bug: binary PCDs returned 0 points -> degenerate-tiff fallback."""
    pts = np.random.default_rng(1).standard_normal((1000, 3)).astype(np.float32)
    out = load_pcd_points(_make_pcd(pts, ascii_mode=False))
    assert out.shape[0] >= 900, "binary PCD must parse to real points, not empty"
    assert out.shape[1] == 3
    # values round-trip (xyz preserved, not garbage)
    assert abs(float(out[:, 0].mean()) - float(pts[:, 0].mean())) < 0.1


def test_binary_and_ascii_agree():
    pts = np.random.default_rng(2).standard_normal((800, 3)).astype(np.float32)
    a = load_pcd_points(_make_pcd(pts, ascii_mode=True), stride=1)
    b = load_pcd_points(_make_pcd(pts, ascii_mode=False))
    assert abs(a[:, 2].mean() - b[:, 2].mean()) < 0.05


def test_geometry_image_shape_and_type():
    pts = np.random.default_rng(3).standard_normal((2000, 3)).astype(np.float32)
    img = pcd_to_geometry_image(pts, size=64)
    arr = np.asarray(img)
    assert arr.shape == (64, 64, 3) and arr.dtype == np.uint8


def test_xyz_normal_image_handles_degenerate():
    # constant-XY (degenerate test-tiff) must not crash; produces a valid image
    xyz = np.zeros((32, 32, 3), np.float32)
    xyz[:, :, 2] = np.linspace(0, 1, 32)[None, :]
    img = np.asarray(xyz_to_normal_image(xyz))
    assert img.shape[2] == 3
