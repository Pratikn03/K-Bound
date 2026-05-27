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


def test_mvtec3d_builder_reads_float_xyz_tiff(tmp_path):
    import tifffile

    root = tmp_path / "mvtec3d"
    _write_rgb(root / "bagel" / "train" / "good" / "rgb" / "ref.png", 40)
    _write_depth(root / "bagel" / "train" / "good" / "xyz" / "ref.tiff", 40)
    _write_rgb(root / "bagel" / "test" / "combined" / "rgb" / "000.png", 80)
    xyz_path = root / "bagel" / "test" / "combined" / "xyz" / "000.tiff"
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(xyz_path, np.zeros((8, 8, 3), dtype=np.float32))

    from scripts.prepare_mvtec3d_fusion_benchmark import build_mvtec3d_fusion_frame

    frame, _ = build_mvtec3d_fusion_frame(root, embedding_dim=8)
    assert len(frame) == 4
    assert np.isfinite(frame.filter(like="embedding_").to_numpy()).all()


def test_mvtec3d_builder_records_train_good_score_protocol(tmp_path):
    root = tmp_path / "mvtec3d"
    for idx, value in enumerate([20, 40, 60]):
        _write_rgb(root / "bagel" / "train" / "good" / "rgb" / f"{idx:03d}.png", value)
        _write_depth(root / "bagel" / "train" / "good" / "xyz" / f"{idx:03d}.tiff", value + 1)
    _write_rgb(root / "bagel" / "validation" / "good" / "rgb" / "100.png", 180)
    _write_depth(root / "bagel" / "validation" / "good" / "xyz" / "100.tiff", 181)
    _write_rgb(root / "bagel" / "test" / "crack" / "rgb" / "200.png", 220)
    _write_depth(root / "bagel" / "test" / "crack" / "xyz" / "200.tiff", 221)

    from scripts.prepare_mvtec3d_fusion_benchmark import build_mvtec3d_fusion_frame

    frame, metadata = build_mvtec3d_fusion_frame(root, embedding_dim=8)

    assert metadata["score_protocol"]["normal_reference_split"] == "train"
    assert metadata["score_protocol"]["normal_reference_defect_type"] == "good"
    assert metadata["score_protocol"]["normal_reference_samples"] == 3
    assert metadata["score_protocol"]["score_normalization"] == "train_good_distance_minmax_clipped"
    assert frame["score_fit_split"].eq("train").all()
    assert frame["score_fit_defect_type"].eq("good").all()


def test_mvtec3d_heldout_protocol_can_reserve_positive_validation_rows(tmp_path):
    root = tmp_path / "mvtec3d"
    for idx in range(4):
        _write_rgb(root / "bagel" / "train" / "good" / "rgb" / f"tr{idx}.png", 40 + idx)
        _write_depth(root / "bagel" / "train" / "good" / "xyz" / f"tr{idx}.tiff", 41 + idx)
        _write_rgb(root / "bagel" / "test" / "good" / "rgb" / f"g{idx}.png", 50 + idx)
        _write_depth(root / "bagel" / "test" / "good" / "xyz" / f"g{idx}.tiff", 51 + idx)
        _write_rgb(root / "bagel" / "test" / "crack" / "rgb" / f"c{idx}.png", 200 + idx)
        _write_depth(root / "bagel" / "test" / "crack" / "xyz" / f"c{idx}.tiff", 201 + idx)
        _write_rgb(root / "foam" / "train" / "good" / "rgb" / f"tr{idx}.png", 70 + idx)
        _write_depth(root / "foam" / "train" / "good" / "xyz" / f"tr{idx}.tiff", 71 + idx)
        _write_rgb(root / "foam" / "test" / "good" / "rgb" / f"g{idx}.png", 80 + idx)
        _write_depth(root / "foam" / "test" / "good" / "xyz" / f"g{idx}.tiff", 81 + idx)
        _write_rgb(root / "foam" / "test" / "crack" / "rgb" / f"c{idx}.png", 210 + idx)
        _write_depth(root / "foam" / "test" / "crack" / "xyz" / f"c{idx}.tiff", 211 + idx)

    from scripts.prepare_mvtec3d_fusion_benchmark import build_mvtec3d_fusion_frame

    frame, metadata = build_mvtec3d_fusion_frame(
        root,
        embedding_dim=8,
        train_categories=["bagel"],
        heldout_val_fraction=0.5,
        heldout_val_seed=3,
    )
    sample_frame = frame.groupby("sample_id").first()
    validation = sample_frame[sample_frame["split"] == "validation"]
    test = sample_frame[sample_frame["split"] == "test"]

    assert set(validation["category"]) == {"bagel"}
    assert set(validation["label"]) == {0, 1}
    assert set(test["category"]) == {"foam"}
    assert metadata["heldout_protocol"]["validation_fraction_from_train_category_test_rows"] == 0.5


def test_extract_point_cloud_statistics(tmp_path):
    import tifffile

    from uais.fusion.attention.m3dm_features import extract_point_cloud_statistics

    # Construct a synthetic plane: x from 0..7, y from 0..7, z = 2.0
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    z = np.full_like(x, 2.0, dtype=np.float32)
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

    xyz_path = tmp_path / "mock_plane.tiff"
    tifffile.imwrite(xyz_path, xyz)

    stats = extract_point_cloud_statistics(xyz_path)
    assert stats.shape == (12,)
    assert np.isfinite(stats).all()
    # Sphericity (eigenvalue ratio) for a pure 2D plane should be near 0
    assert stats[2] < 1e-3  # sphericity index
    assert abs(stats[6] - 3.5) < 1e-3  # mean_x
    assert abs(stats[7] - 3.5) < 1e-3  # mean_y
    assert abs(stats[8] - 2.0) < 1e-3  # mean_z


def test_extract_resnet_features_concatenates_point_statistics(tmp_path):
    import tifffile

    from uais.fusion.attention.m3dm_features import extract_resnet_features

    x, y = np.meshgrid(np.arange(8), np.arange(8))
    z = np.full_like(x, 2.0, dtype=np.float32)
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

    xyz_path = tmp_path / "mock_plane.tiff"
    tifffile.imwrite(xyz_path, xyz)

    # Calling extract_resnet_features on a mock tiff path should return (1, 2060)
    features = extract_resnet_features([xyz_path], batch_size=1)
    assert features.shape == (1, 2060)
    assert np.isfinite(features).all()
    # The last 12 dimensions contain the PointNet++ statistics we tested above
    assert features[0, 2048 + 2] < 1e-3  # sphericity index
    assert abs(features[0, 2048 + 8] - 2.0) < 1e-3  # mean_z
