from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_depth(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((8, 8), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_mvtec3d_builder_marks_natural_pairing_and_emits_paired_domains(tmp_path):
    root = tmp_path / "mvtec3d"
    _write_rgb(root / "bagel" / "train" / "good" / "rgb" / "000.png", 32)
    _write_depth(root / "bagel" / "train" / "good" / "xyz" / "000.tiff", 40)
    _write_rgb(root / "bagel" / "test" / "crack" / "rgb" / "001.png", 220)
    _write_depth(root / "bagel" / "test" / "crack" / "xyz" / "001.tiff", 210)

    from scripts.prepare_mvtec3d_fusion_benchmark import build_mvtec3d_fusion_frame

    frame, metadata = build_mvtec3d_fusion_frame(root, embedding_dim=8)

    assert metadata["natural_pairing"] is True
    assert metadata["benchmark_type"] == "naturally_paired_mvtec3d_score_fusion"
    assert metadata["domain_order"] == ["rgb", "depth_or_xyz"]
    assert metadata["samples"] == 2
    assert set(frame["domain"]) == {"rgb", "depth_or_xyz"}
    assert frame.groupby("sample_id")["domain"].nunique().eq(2).all()
    assert frame.groupby("sample_id")["pairing_key"].nunique().eq(1).all()
    assert set(frame.groupby("sample_id")["label"].first()) == {0, 1}
    assert frame["score"].between(0.0, 1.0).all()
    assert frame["confidence"].between(0.0, 1.0).all()
    assert {f"embedding_{idx}" for idx in range(8)}.issubset(frame.columns)
