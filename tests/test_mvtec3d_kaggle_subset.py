from scripts.download_mvtec3d_kaggle_subset import select_subset_files


def test_select_subset_files_keeps_rgb_xyz_pairs_and_limits_counts():
    files = []
    for split, defect, n in [
        ("train", "good", 5),
        ("validation", "good", 3),
        ("test", "good", 4),
        ("test", "crack", 4),
        ("test", "hole", 2),
    ]:
        for i in range(n):
            files.append(f"mvtec_3d_anomaly_detection/bagel/{split}/{defect}/rgb/{i:03d}.png")
            files.append(f"mvtec_3d_anomaly_detection/bagel/{split}/{defect}/xyz/{i:03d}.tiff")
            files.append(f"mvtec_3d_anomaly_detection/bagel/{split}/{defect}/gt/{i:03d}.png")

    selected = select_subset_files(
        files,
        category="bagel",
        max_train_good=2,
        max_validation_good=1,
        max_test_good=1,
        max_test_per_defect=2,
    )

    assert all("/gt/" not in path for path in selected)
    assert len(selected) == 16
    assert "mvtec_3d_anomaly_detection/bagel/train/good/rgb/000.png" in selected
    assert "mvtec_3d_anomaly_detection/bagel/train/good/xyz/000.tiff" in selected
    assert "mvtec_3d_anomaly_detection/bagel/test/crack/rgb/001.png" in selected
    assert "mvtec_3d_anomaly_detection/bagel/test/crack/xyz/001.tiff" in selected
