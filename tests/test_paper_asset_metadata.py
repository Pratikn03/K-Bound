from pathlib import Path


def test_fusion_configs_use_validation_threshold_selection():
    from uais.utils.config_loader import load_yaml

    repo_root = Path(__file__).resolve().parents[1]
    config_paths = [
        repo_root / "configs" / "attention_real_fusion.yaml",
        repo_root / "configs" / "attention_mvtec3d_fusion.yaml",
        repo_root / "configs" / "attention_real_fusion_hard.yaml",
    ]

    for config_path in config_paths:
        cfg = load_yaml(config_path)
        assert cfg["evaluation"]["decision_threshold"] == "val_f1"


def test_hard_real_fusion_config_targets_distinct_hard_artifacts():
    from uais.utils.config_loader import load_yaml

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(repo_root / "configs" / "attention_real_fusion_hard.yaml")

    assert cfg["data"]["path"] == "experiments/fusion/real_domain_fusion_hard_inputs.csv"
    assert cfg["output"]["model_dir"] == "models/fusion/attention_real_hard"
    assert cfg["evaluation"]["metrics_path"] == "experiments/fusion/real_hard_attention_harness_metrics.json"


def test_real_benchmark_metadata_table_includes_split_safety(tmp_path: Path):
    from scripts.generate_craf_paper_assets import write_benchmark_metadata_table

    write_benchmark_metadata_table(
        {
            "benchmark_type": "label_aligned_real_domain_score_fusion",
            "natural_pairing": False,
            "pairing_unit": "binary-label-aligned composite sample",
            "samples": 8000,
            "positive_fraction_actual": 0.301,
            "scorer_train_fraction": 1.0,
            "source_row_disjoint_splits": True,
            "fusion_split_column": "fusion_split",
            "domain_order": ["fraud", "cyber"],
        },
        tmp_path,
    )

    table = (tmp_path / "elara_benchmark_metadata.tex").read_text()
    assert "Source-row disjoint splits" in table
    assert "fusion\\_split" in table


def test_paired_benchmark_metadata_table_includes_score_protocol(tmp_path: Path):
    from scripts.generate_craf_paper_assets import write_paired_benchmark_metadata_table

    write_paired_benchmark_metadata_table(
        {
            "benchmark_type": "naturally_paired_mvtec3d_score_fusion",
            "natural_pairing": True,
            "categories": ["bagel"],
            "samples": 10,
            "positive_fraction_actual": 0.2,
            "domain_order": ["rgb", "depth_or_xyz"],
            "score_protocol": {
                "normal_reference_split": "train",
                "normal_reference_defect_type": "good",
                "normal_reference_samples": 7,
                "score_normalization": "train_good_distance_minmax_clipped",
            },
        },
        tmp_path,
    )

    table = (tmp_path / "mvtec3d_benchmark_metadata.tex").read_text()
    assert "Score fit split" in table
    assert "train/good" in table
    assert "train\\_good\\_distance\\_minmax\\_clipped" in table


def test_calibration_table_includes_cda_spearman_status(tmp_path: Path):
    from scripts.generate_craf_paper_assets import write_calibration_table

    write_calibration_table(
        {
            "table_5_calibration": {
                "static_ece": 0.1,
                "craf_ece": 0.2,
                "static_brier": 0.3,
                "craf_brier": 0.4,
            },
            "cda_validation": {
                "n_samples": 20,
                "spearman_cda_vs_ece_reliability": None,
                "spearman_cda_vs_ece_reliability_status": "undefined: fewer than three finite domains",
            },
        },
        tmp_path,
    )

    table = (tmp_path / "elara_calibration_cda.tex").read_text()
    assert "CDA/ECE Spearman status" in table
    assert "n/a (fewer than 3 domains)" in table
