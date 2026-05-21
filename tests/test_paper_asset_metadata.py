from pathlib import Path


def test_manuscript_rejects_stale_mvtec_claims():
    repo_root = Path(__file__).resolve().parents[1]
    paper = (repo_root / "docs" / "research" / "PAPER_DRAFT_v1.tex").read_text()
    thesis = (repo_root / "docs" / "research" / "THESIS_CHAPTER_v1.tex").read_text()
    readme = (repo_root / "README.md").read_text()
    combined = "\n".join([paper, thesis, readme])

    stale_claims = [
        "random forest fusion leads (clean ROC-AUC\n$0.959$)",
        "random forest fusion\nreaches $0.959$",
        "static attention dropped from $0.728$ to $0.650$, RGA from $0.668$ to\n$0.610$",
        "all deltas are negative or zero",
        "RGA gate *hurts* by $-0.040$ clean",
    ]

    for claim in stale_claims:
        assert claim not in combined

    required_claims = [
        "label-aligned stress-only benchmark",
        "canonical one-class protocol",
        "held-out-category",
        "M3DM-style",
        "diagnostic gate",
    ]
    for claim in required_claims:
        assert claim in combined


def test_manuscript_keeps_mvtec_protocol_diagnostic_claim_consistent():
    repo_root = Path(__file__).resolve().parents[1]
    paper = (repo_root / "docs" / "research" / "PAPER_DRAFT_v1.tex").read_text()
    thesis = (repo_root / "docs" / "research" / "THESIS_CHAPTER_v1.tex").read_text()
    combined = "\n".join([paper, thesis])

    contradictory_mvtec_claims = [
        "clean-and-adversarial regression on RGB/3D paired MVTec 3D-AD",
        "why the gate hurts on this benchmark",
        "below static attention by $-0.040$ at zero extra dropout",
        "Only the \\texttt{no\\_ks} variant avoids harm",
        "and a strongly negative delta",
        "clean ROC-AUC penalty",
        "overall mechanism is harmful",
        "every firing condition produces a\n\\emph{negative} ROC-AUC delta",
        "misfires on legitimate",
        "misfire pattern",
        "underperforms the static attention solution",
        "gate-hurts-on-natural-paired",
        "absolute delta remains strongly negative in every case",
        "When Reliability Gates Misfire",
    ]

    for claim in contradictory_mvtec_claims:
        assert claim not in combined


def test_enterprise_gap_closure_is_conditional_not_current_claim():
    repo_root = Path(__file__).resolve().parents[1]
    paper = (repo_root / "docs" / "research" / "PAPER_DRAFT_v1.tex").read_text()
    thesis = (repo_root / "docs" / "research" / "THESIS_CHAPTER_v1.tex").read_text()

    required_closure_dimensions = [
        "True multimodal incident detection",
        "Autonomous zero-misfire adaptation",
        "Universal system integration",
        "Deployable, auditable analyst AI",
        "Closure evidence required",
        "validate_incident_protocol",
        "CategoryAwareReliabilityEstimator",
        "calibration_monitor_report",
        "bounded_switching_certificate",
    ]
    for document in [paper, thesis]:
        for phrase in required_closure_dimensions:
            assert phrase in document

    forbidden_current_claims = [
        "ELARA is enterprise-grade",
        "ELARA is a plug-and-play",
        "ELARA is production-ready",
        "ELARA has been deployed",
        "mathematically verified tool",
        "perfectly discern",
    ]
    combined = "\n".join([paper, thesis])
    for phrase in forbidden_current_claims:
        assert phrase not in combined


def test_healthcare_gap_validation_claim_boundary_documented():
    repo_root = Path(__file__).resolve().parents[1]
    thesis = (repo_root / "docs" / "research" / "THESIS_CHAPTER_v1.tex").read_text()
    readme = (repo_root / "README.md").read_text()
    combined = "\n".join([thesis, readme])

    required = [
        "healthcare_gap_validation.json",
        "healthcare_gap1_patient_stratified_validation.json",
        "healthcare_gap2_reliability_stress_validation.json",
        "healthcare_gap4_deployment_audit_validation.json",
        "healthcare_gap_closure_results.tex",
        "healthcare_gap4_audit_results.tex",
        "healthcare_gap_closure_status.png",
        "healthcare_gap2_stress_rates.png",
        "146,688",
        "test windows are single-class positive",
        "patient-disjoint stratified replay closes Gap 1 locally",
        "Gap 2 locally",
        "natural fire rate is 0.0",
        "collapse fire rate is 1.0",
        "temporal_order_valid is false",
        "diagnostic switching certificate",
        "static loss is 0.800",
        "policy loss is 0.000",
        "Retrospective local healthcare replay",
    ]
    for phrase in required:
        assert phrase in combined

    forbidden = [
        "healthcare validation closes all four empirical gaps",
        "is clinical deployment evidence",
        "zero-misfire adaptation is proven",
    ]
    for phrase in forbidden:
        assert phrase not in combined


def test_conference_paper_does_not_promote_gridpulse_as_headline_evidence():
    repo_root = Path(__file__).resolve().parents[1]
    paper = (repo_root / "docs" / "research" / "PAPER_DRAFT_v1.tex").read_text()

    forbidden = [
        "GridPulse",
        "healthcare_paired_results",
        "clinical-vitals replay",
        "18{,}149",
        "$+0.0759$",
        "0.5131",
        "0.4372",
    ]
    for phrase in forbidden:
        assert phrase not in paper

    required = [
        "PatchCore supervised-paired",
        "public paired benchmark",
        "mvtec3d_milestone1_comparison",
    ]
    for phrase in required:
        assert phrase in paper


def test_healthcare_gap_tables_and_figures_are_materialized():
    repo_root = Path(__file__).resolve().parents[1]
    table_paths = [
        repo_root / "docs" / "research" / "tables" / "healthcare_gap_closure_results.tex",
        repo_root / "docs" / "research" / "tables" / "healthcare_gap4_audit_results.tex",
    ]
    figure_paths = [
        repo_root / "docs" / "research" / "figures" / "healthcare_gap_closure_status.png",
        repo_root / "docs" / "research" / "figures" / "healthcare_gap2_stress_rates.png",
    ]

    for path in table_paths:
        assert path.exists()
        text = path.read_text()
        assert "Gap" in text or "Audit item" in text
        assert "yes" in text or "pass" in text

    for path in figure_paths:
        assert path.exists()
        assert path.stat().st_size > 1000


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


def test_mvtec3d_sota_demarcation_assets_and_subsection_present():
    repo_root = Path(__file__).resolve().parents[1]
    table_path = repo_root / "docs" / "research" / "tables" / "mvtec3d_sota_demarcation.tex"
    figure_path = repo_root / "docs" / "research" / "figures" / "mvtec3d_sota_demarcation.png"
    paper_path = repo_root / "docs" / "research" / "PAPER_DRAFT_v1.tex"
    thesis_path = repo_root / "docs" / "research" / "THESIS_CHAPTER_v1.tex"

    assert table_path.exists(), "Emit the SOTA demarcation table before running tests."
    assert figure_path.exists(), "Emit the SOTA demarcation figure before running tests."

    paper = paper_path.read_text()
    thesis = thesis_path.read_text()

    for document in (paper, thesis):
        assert "Position Relative to Published MVTec 3D-AD State of the Art" in document
        assert "leaderboard" in document
        assert "mvtec3d_sota_demarcation.tex" in document
        for citation_key in (
            "wang2023m3dm",
            "rudolph2023ast",
            "horwitz2023btf",
            "chen2023easynet",
            "roth2022patchcore",
        ):
            assert f"\\cite{{{citation_key}}}" in document

    table_text = table_path.read_text()
    assert "leaderboard" in table_text
    assert "this work" in table_text
    assert "no localization head" in table_text
    # All five published methods must appear in the rendered table body.
    for method in ("BTF", "EasyNet", "PatchCore-3D", "AST", "M3DM"):
        assert method in table_text


def test_per_category_breakdown_assets_present():
    repo_root = Path(__file__).resolve().parents[1]
    tables_dir = repo_root / "docs" / "research" / "tables"
    expected = [
        "mvtec3d_patchcore_per_category.tex",
        "mvtec3d_patchcore_supervised_paired_per_category.tex",
        "mvtec3d_patchcore_heldout_per_category.tex",
        "mvtec_loco_patchcore_per_category.tex",
        "mvtec_loco_patchcore_supervised_paired_per_category.tex",
    ]
    for name in expected:
        path = tables_dir / name
        assert path.exists(), f"Emit the per-category table {name} before running tests."
        text = path.read_text()
        # Every per-category table must include the headline RGA+ columns
        # and at least one MVTec category.
        assert "RGA+ router" in text or "RGA+ boost" in text
        assert r"\textbf{Mean}" in text

    paper = (repo_root / "docs" / "research" / "PAPER_DRAFT_v1.tex").read_text()
    assert "Per-Category Breakdown" in paper
    assert "sec:per-category" in paper


def test_patchcore3d_baseline_results_present():
    repo_root = Path(__file__).resolve().parents[1]
    baseline_path = repo_root / "experiments" / "fusion" / "patchcore3d_baseline_results.json"
    assert baseline_path.exists(), (
        "Run src/scripts/eval_patchcore3d_baseline.py to produce the "
        "PatchCore-3D head-to-head baseline JSON before running tests."
    )
    import json
    payload = json.loads(baseline_path.read_text())
    assert payload["protocol"] == "canonical_one_class"
    assert payload["scorer"] == "patchcore_knn"
    mean = payload.get("mean_image_auroc")
    assert isinstance(mean, (int, float))
    # Sanity: re-run should be well above chance and below published headline.
    assert 0.5 < float(mean) < float(payload["published_reference"]["image_auroc"])
