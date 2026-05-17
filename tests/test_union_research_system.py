from __future__ import annotations

from pathlib import Path

from scripts.run_union_research_system import Step, build_plan, run_step


def test_union_plan_connects_legacy_new_healthcare_and_assets():
    names = [step.name for step in build_plan(mode="full", with_tests=True)]

    assert "legacy_fraud" in names
    assert "legacy_cyber" in names
    assert "legacy_behavior" in names
    assert "prep_realfusion" in names
    assert "train_realfusion" in names
    assert "prep_mvtec_m3dm" in names
    assert "train_mvtec_m3dm" in names
    assert "healthcare_gap4" in names
    assert "rebuild_paper" in names
    assert "verify_metadata_tests" in names


def test_union_plan_keeps_deprecated_dashboard_opt_in():
    default_names = [step.name for step in build_plan(mode="full")]
    explicit_names = [
        step.name
        for step in build_plan(mode="full", include_deprecated_dashboard=True)
    ]

    assert "legacy_dashboard_fusion" not in default_names
    assert "legacy_dashboard_fusion" in explicit_names


def test_m3dm_prep_matches_config_embedding_width():
    step = next(step for step in build_plan(mode="paper") if step.name == "prep_mvtec_m3dm")

    command = list(step.command)
    assert "--embedding-dim" in command
    assert command[command.index("--embedding-dim") + 1] == "16"


def test_optional_nlp_vision_are_opt_in():
    default_names = [step.name for step in build_plan(mode="legacy")]
    explicit_names = [
        step.name
        for step in build_plan(mode="legacy", include_optional_nlp_vision=True)
    ]

    assert "standalone_nlp" not in default_names
    assert "standalone_vision" not in default_names
    assert "standalone_nlp" in explicit_names
    assert "standalone_vision" in explicit_names


def test_run_step_skips_missing_required_path(tmp_path: Path):
    step = Step(
        name="missing",
        group="test",
        command=("python", "-V"),
        description="missing path check",
        required_paths=("missing/data.csv",),
    )

    result = run_step(step, logs_dir=tmp_path / "logs", dry_run=False, repo_root=tmp_path)

    assert result.status == "skipped"
    assert "missing/data.csv" in str(result.reason)


def test_run_step_dry_run_marks_existing_step_planned(tmp_path: Path):
    data = tmp_path / "data.csv"
    data.write_text("x\n", encoding="utf-8")
    step = Step(
        name="planned",
        group="test",
        command=("python", "-V"),
        description="dry run check",
        required_paths=("data.csv",),
    )

    result = run_step(step, logs_dir=tmp_path / "logs", dry_run=True, repo_root=tmp_path)

    assert result.status == "planned"


def test_run_step_recognizes_image_data_directory(tmp_path: Path):
    image_dir = tmp_path / "images" / "class_a"
    image_dir.mkdir(parents=True)
    (image_dir / "sample.jpg").write_bytes(b"fake")
    step = Step(
        name="planned_image",
        group="test",
        command=("python", "-V"),
        description="image data check",
        required_paths=("images",),
    )

    result = run_step(step, logs_dir=tmp_path / "logs", dry_run=True, repo_root=tmp_path)

    assert result.status == "planned"
