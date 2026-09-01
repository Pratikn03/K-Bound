from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from docs.research.kbound.scripts import build_cct20_release as release
from experiments.kbound.cct20.integrity import IntegrityError, stable_sha256
from experiments.kbound.cct20.protocol_seal import (
    EXPECTED_MODEL_SEEDS,
    EXPECTED_TARGET_LOCATIONS,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from experiments.kbound.cct20.two_way_inference import analyze_score_document


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _multilabel_report(correct: int, *, n_evaluation: int = 7) -> dict:
    rows = []
    for output_index in range(16):
        if output_index == 0:
            tp, fp, fn = correct, 0, n_evaluation - correct
        elif output_index == 1:
            tp, fp, fn = 0, n_evaluation - correct, 0
        else:
            tp = fp = fn = 0
        denominator = 2 * tp + fp + fn
        rows.append(
            {
                "output_index": output_index,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": 0.0 if denominator == 0 else (2.0 * tp) / denominator,
            }
        )
    return {
        "macro_f1": sum(row["f1"] for row in rows) / 16.0,
        "n_output_indicators": 16,
        "zero_denominator_convention": 0.0,
        "per_class": rows,
    }


def _strong_score() -> dict:
    cells = []
    for seed in EXPECTED_MODEL_SEEDS:
        tensor_sha = _sha(f"synthetic-tensor-{seed}")
        for location_index, location in enumerate(EXPECTED_TARGET_LOCATIONS):
            helpful = (seed + location_index) % 2 == 0
            accuracies = {
                "always_freeze": 4 / 7 if helpful else 6 / 7,
                "always_adapt": 6 / 7 if helpful else 4 / 7,
                "kga": 6 / 7,
            }
            oracle = max(accuracies.values())
            regrets = {name: oracle - value for name, value in accuracies.items()}
            cells.append(
                {
                    "checkpoint_seed": seed,
                    "checkpoint_tensor_sha256": tensor_sha,
                    "location_id": location,
                    "n_target_images": 10,
                    "n_probe_images": 3,
                    "n_evaluation_images": 7,
                    "decision": "ADAPT" if helpful else "FREEZE",
                    "set_membership_top1_accuracy": accuracies,
                    "adaptation_benefit": (accuracies["always_adapt"] - accuracies["always_freeze"]),
                    "oracle_fixed_action_accuracy": oracle,
                    "regret_to_better_fixed_action": regrets,
                    "baseline_regret_minus_kga_regret": {
                        "versus_always_adapt": (regrets["always_adapt"] - regrets["kga"]),
                        "versus_always_freeze": (regrets["always_freeze"] - regrets["kga"]),
                    },
                    "multilabel_macro_f1": {
                        name: _multilabel_report(round(value * 7)) for name, value in accuracies.items()
                    },
                }
            )
    document = {
        "schema": "kbound_cct20_set_valued_score_v1",
        "status": "ALL_LOCKED_CELLS_SCORED",
        "execution_seal_artifact_sha256": _sha("synthetic-execution-seal"),
        "target_image_count": 90,
        "probe_image_count": 27,
        "evaluation_image_count": 63,
        "benefit_sign": release.BENEFIT_SIGN,
        "primary_contrast_sign": release.PRIMARY_CONTRAST_SIGN,
        "cells": cells,
    }
    document["score_sha256"] = stable_sha256(document)
    return document


@pytest.fixture(scope="module")
def strong_bundle() -> tuple[dict, dict]:
    score = _strong_score()
    return score, analyze_score_document(score)


def _upstream() -> dict:
    return {
        "two_way_inference": {
            "path": "/synthetic/two_way_inference.json",
            "bytes": 123,
            "sha256": _sha("synthetic-inference-file"),
            "canonical_document_sha256": _sha("synthetic-inference-document"),
        }
    }


def test_release_core_exposes_locked_design_signs_and_exact_inference(strong_bundle) -> None:
    score, inference = strong_bundle
    result = release.build_release_core(
        score=score,
        inference=inference,
        upstream_artifacts=_upstream(),
    )

    assert result["schema"] == "kbound_cct20_release_manifest_v1"
    assert result["status"] == "RELEASE_COMPLETE"
    assert result["design"]["matrix_shape"] == [5, 9]
    assert result["design"]["location_cluster_count"] == 9
    assert result["sign_conventions"]["primary_contrast"] == ("baseline_regret_minus_kga_regret; positive_favors_kga")
    assert result["adaptation_effect_mix"] == {
        "helpful_cells_strictly_positive": 23,
        "neutral_cells_exactly_zero": 0,
        "harmful_cells_strictly_negative": 22,
        "mixed_helpful_and_harmful_present": True,
    }
    assert len(result["adaptation_effect_cells_by_sign"]["helpful"]) == 23
    assert len(result["adaptation_effect_cells_by_sign"]["zero"]) == 0
    assert len(result["adaptation_effect_cells_by_sign"]["harmful"]) == 22
    assert result["action_exposure"]["counts"] == {
        "ADAPT": 23,
        "FREEZE": 22,
        "ABSTAIN": 0,
    }
    assert result["false_adapt_accounting"] == {
        "event": "decision == ADAPT and adaptation_benefit <= 0",
        "unit": "checkpoint_x_location",
        "denominator_all_cells": 45,
        "adapt_count": 23,
        "false_adapt_count": 0,
        "false_adapt_rate_unconditional": 0.0,
        "false_adapt_rate_conditional": 0.0,
    }
    assert set(result["safe_utility"]) == {
        "contrast_sign",
        "frozen_noninferiority_margin",
        "versus_always_freeze",
        "versus_always_adapt",
        "passes",
    }
    assert result["secondary_outcome_reporting"]["aggregate_claim"] is None
    assert result["verdict"]["code"] == "CONFIRMATORY_STRONG_SUCCESS"
    for comparison in release.COMPARISONS:
        row = result["primary_comparisons"][comparison]
        upstream = inference["paired_two_way_product_bootstrap"]["results"][comparison]
        exact = inference["exact_nine_location_sign_flip_and_holm"][comparison]
        assert row["simultaneous_bonferroni_97_5_ci"] == upstream["simultaneous_bonferroni_97_5_ci"]
        assert row["exact_location_sign_flip_p_one_sided"] == exact["p_value_one_sided"]
        assert row["holm_adjusted_p"] == exact["holm_adjusted_p"]


def test_inference_replay_rejects_a_changed_interval(strong_bundle) -> None:
    score, inference = strong_bundle
    tampered = copy.deepcopy(inference)
    tampered["paired_two_way_product_bootstrap"]["results"]["versus_always_adapt"]["simultaneous_bonferroni_97_5_ci"][
        0
    ] += 0.001
    unsigned = dict(tampered)
    unsigned.pop("inference_sha256")
    tampered["inference_sha256"] = stable_sha256(unsigned)

    with pytest.raises(IntegrityError, match="differs from the frozen replay"):
        release._validate_inference_replay(score, tampered)


def test_release_rejects_benefit_sign_drift(strong_bundle) -> None:
    score, inference = strong_bundle
    changed = copy.deepcopy(score)
    changed["benefit_sign"] = "frozen_minus_adapted"

    with pytest.raises(IntegrityError, match="adaptation-benefit sign drift"):
        release.build_release_core(
            score=changed,
            inference=inference,
            upstream_artifacts=_upstream(),
        )


def test_score_release_contract_is_set_valued_and_population_complete() -> None:
    score = _strong_score()
    target_annotation_sha = _sha("synthetic-target-annotations")
    score.update(
        {
            "target_annotations_file_sha256": target_annotation_sha,
            "target_image_count": 23_275,
            "probe_image_count": 5_000,
            "evaluation_image_count": 18_275,
            "evaluation_prediction_row_count": 91_375,
            "truth_join_sha256": _sha("synthetic-truth-join"),
            "label_contract": {
                "primary": "top1_correct_iff_prediction_in_complete_distinct_category_set",
                "secondary": "macro_f1_over_all_16_indicators_with_top1_as_one_hot_prediction",
                "repeated_same_category": "collapsed",
                "zero_annotation_image": "experiment_failure",
            },
        }
    )
    for row in score["cells"]:
        row["probe_predictions_scored"] = False
        row["evaluation_predictions_scored"] = True

    release._validate_score_release_contract(
        score,
        target_annotations_sha256=target_annotation_sha,
    )
    score["cells"][0]["probe_predictions_scored"] = True
    with pytest.raises(IntegrityError, match="scoring scope drift"):
        release._validate_score_release_contract(
            score,
            target_annotations_sha256=target_annotation_sha,
        )


def test_score_release_contract_rejects_missing_or_inconsistent_secondary_outcome() -> None:
    score = _strong_score()
    target_annotation_sha = _sha("synthetic-target-annotations")
    score.update(
        {
            "target_annotations_file_sha256": target_annotation_sha,
            "target_image_count": 23_275,
            "probe_image_count": 5_000,
            "evaluation_image_count": 18_275,
            "evaluation_prediction_row_count": 91_375,
            "truth_join_sha256": _sha("synthetic-truth-join"),
            "label_contract": {
                "primary": "top1_correct_iff_prediction_in_complete_distinct_category_set",
                "secondary": "macro_f1_over_all_16_indicators_with_top1_as_one_hot_prediction",
                "repeated_same_category": "collapsed",
                "zero_annotation_image": "experiment_failure",
            },
        }
    )
    for row in score["cells"]:
        row["probe_predictions_scored"] = False
        row["evaluation_predictions_scored"] = True

    missing = copy.deepcopy(score)
    missing["cells"][0].pop("multilabel_macro_f1")
    with pytest.raises(IntegrityError, match="complete secondary outcome"):
        release._validate_score_release_contract(
            missing,
            target_annotations_sha256=target_annotation_sha,
        )

    inconsistent = copy.deepcopy(score)
    inconsistent["cells"][0]["multilabel_macro_f1"]["always_adapt"]["per_class"][0]["tp"] -= 1
    with pytest.raises(IntegrityError, match="does not reconcile"):
        release._validate_score_release_contract(
            inconsistent,
            target_annotations_sha256=target_annotation_sha,
        )


def test_score_global_role_counts_must_reconcile_to_every_checkpoint(strong_bundle) -> None:
    score, _ = strong_bundle
    grid = {}
    for row in score["cells"]:
        key = (row["checkpoint_seed"], row["location_id"])
        grid[key] = {
            "checkpoint_tensor_sha256": row["checkpoint_tensor_sha256"],
            "n_images": 10,
            "gate": {"decision": row["decision"]},
            "rows": [{"role": "probe" if index < 3 else "evaluation"} for index in range(10)],
        }
    release._reconcile_score_with_predictions(score, grid)

    stale_globals = copy.deepcopy(score)
    stale_globals["probe_image_count"] = 28
    stale_globals["evaluation_image_count"] = 62
    with pytest.raises(IntegrityError, match="global target/probe/evaluation counts"):
        release._reconcile_score_with_predictions(stale_globals, grid)


def test_one_shot_marker_binds_score_collection_and_all_cell_hashes(tmp_path: Path) -> None:
    execution_sha = _sha("execution")
    collection_sha = _sha("collection")
    cell_hashes = [_sha(f"cell-{index}") for index in range(45)]
    score_path = tmp_path / "score.json"
    request = {
        "execution_seal_artifact_sha256": execution_sha,
        "prediction_collection_sha256": collection_sha,
        "prediction_cell_sha256": sorted(cell_hashes),
        "output_path": str(score_path.resolve()),
        "expected_target_images": 23_275,
        "label_contract": release.SCORING_LABEL_CONTRACT,
    }
    marker = {
        "schema": "kbound_cct20_one_shot_score_marker_v1",
        "status": "SPENT_BEFORE_GROUND_TRUTH_LOAD",
        "request": request,
        "request_sha256": stable_sha256(request),
    }
    marker_path = tmp_path / "score.spent.json"
    release._exclusive_write(marker_path, release._immutable_json_payload(marker))
    release._validate_scoring_marker(
        marker,
        marker_path=marker_path,
        score_path=score_path,
        execution_artifact_sha256=execution_sha,
        prediction_collection_sha256=collection_sha,
        prediction_cell_sha256=list(reversed(cell_hashes)),
    )

    stale = copy.deepcopy(marker)
    stale["request"]["prediction_cell_sha256"] = cell_hashes[:-1]
    stale["request_sha256"] = stable_sha256(stale["request"])
    with pytest.raises(IntegrityError, match="request differs"):
        release._validate_scoring_marker(
            stale,
            marker_path=marker_path,
            score_path=score_path,
            execution_artifact_sha256=execution_sha,
            prediction_collection_sha256=collection_sha,
            prediction_cell_sha256=cell_hashes,
        )


def test_prediction_collection_requires_exact_label_free_population_replay(monkeypatch) -> None:
    collection = {"collection_sha256": _sha("collection")}
    target_manifest = {"manifest_sha256": _sha("target")}
    cells = [{"synthetic": True}]
    observed = {}

    monkeypatch.setattr(release, "validate_locked_target_population", lambda document: None)
    monkeypatch.setattr(
        release,
        "normalize_target_manifest",
        lambda document: ([{"image_id": "safe-label-free-id"}], []),
    )

    def replay(cells_arg, **kwargs):
        observed["cells"] = cells_arg
        observed.update(kwargs)
        return dict(collection)

    monkeypatch.setattr(release, "build_prediction_collection", replay)
    release._replay_prediction_collection(
        collection,
        prediction_cells=cells,
        target_manifest=target_manifest,
    )
    assert observed["cells"] == cells
    assert observed["target_index"] == [{"image_id": "safe-label-free-id"}]
    assert observed["expected_target_images"] == 23_275
    assert observed["require_replayable_probe_features"] is True

    monkeypatch.setattr(
        release,
        "build_prediction_collection",
        lambda *args, **kwargs: {"collection_sha256": _sha("incomplete")},
    )
    with pytest.raises(IntegrityError, match="full label-free target replay"):
        release._replay_prediction_collection(
            collection,
            prediction_cells=cells,
            target_manifest=target_manifest,
        )


def test_release_dependency_ledger_binds_all_138_code_and_four_data_dependencies() -> None:
    execution = {
        "dataset_dependencies": [
            {"name": name, "path": f"/data/{name}", "bytes": 1, "sha256": _sha(name)}
            for name in sorted(release.REQUIRED_DATA_DEPENDENCY_NAMES)
        ],
        "code_dependencies": [
            {"name": name, "path": f"/code/{name}", "bytes": 1, "sha256": _sha(name)}
            for name in sorted(release.REQUIRED_CODE_DEPENDENCY_NAMES)
        ],
    }
    bundle = release._execution_dependency_bundle(execution)
    assert bundle["dataset_count"] == 4
    assert bundle["code_count"] == 138
    assert bundle["total_count"] == 142
    assert len(bundle["dataset_items"]) == 4
    assert len(bundle["code_items"]) == 138


def test_publication_safety_uses_all_45_cells_and_nonpositive_boundary(strong_bundle) -> None:
    score, inference = strong_bundle
    changed = copy.deepcopy(score)
    changed["cells"][0]["adaptation_benefit"] = 0.0
    exposure = inference["action_exposure_at_checkpoint_location_unit"]

    result = release._publication_safety(changed, exposure)

    assert result["false_adapt_count"] == 1
    assert result["false_adapt_rate_unconditional"] == pytest.approx(1 / 45)
    assert result["false_adapt_rate_conditional"] == pytest.approx(1 / exposure["counts"]["ADAPT"])


def test_tex_number_has_bounded_precision_without_erasing_small_nonzero_values() -> None:
    assert release._tex_number(0.123456789123) == "0.12345679"
    assert release._tex_number(-0.005) == "-0.005"
    assert release._tex_number(-0.0) == "0"
    assert release._tex_number(1e-12) == "1.000e-12"


def test_development_trace_ledger_must_match_execution_slots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(release, "EXPECTED_DEVELOPMENT_TRACE_COUNT", 2)
    records = []
    dependencies = {}
    for index in range(2):
        path = tmp_path / f"trace-{index}.json"
        trace = {"trace_id": f"trace-{index}", "trace_sha256": _sha(f"trace-{index}")}
        receipt = write_immutable_json_with_receipt(path, trace)
        records.append(
            {
                "trace_id": trace["trace_id"],
                "trace_sha256": trace["trace_sha256"],
                "artifact_path": str(path.resolve()),
                "artifact_receipt": receipt,
            }
        )
        for name, artifact in (
            (f"development_trace_{index:02d}", path),
            (f"development_trace_receipt_{index:02d}", path.with_name(path.name + ".receipt.json")),
        ):
            dependencies[name] = {
                "name": name,
                "path": str(artifact.resolve()),
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
    bundle = release._development_trace_bundle(
        {"trace_artifacts": list(reversed(records))},
        code_dependencies=dependencies,
    )
    assert bundle["count"] == 2

    swapped = copy.deepcopy(dependencies)
    swapped["development_trace_00"], swapped["development_trace_01"] = (
        swapped["development_trace_01"],
        swapped["development_trace_00"],
    )
    with pytest.raises(IntegrityError, match="does not identify"):
        release._development_trace_bundle(
            {"trace_artifacts": records},
            code_dependencies=swapped,
        )


@pytest.mark.parametrize(
    ("strong", "expanded", "safe", "expected"),
    [
        (True, True, True, "CONFIRMATORY_STRONG_SUCCESS"),
        (True, False, True, "CONFIRMATORY_PRIMARY_SUCCESS_MIXED_EFFECTS_MISSING"),
        (False, False, True, "SAFE_UTILITY_ONLY"),
        (False, False, False, "NO_CONFIRMATORY_SUCCESS"),
    ],
)
def test_verdict_has_honest_nonpromotion_branches(
    strong: bool,
    expanded: bool,
    safe: bool,
    expected: str,
) -> None:
    inference = {
        "strong_success_checks": {
            "protocol_strong_success": strong,
            "expanded_empirical_bundle_including_mixed_effects": expanded,
        },
        "safe_utility": {"passes": safe},
    }
    verdict = release._verdict(inference)
    assert verdict["code"] == expected
    assert verdict["confirmatory_strong_claim_supported"] is expanded
    assert verdict["primary_confirmatory_claim_supported"] is strong
    if not strong:
        assert "does not" in verdict["manuscript_claim"]


def test_emit_writes_all_four_immutable_release_files(tmp_path: Path, strong_bundle) -> None:
    score, inference = strong_bundle
    core = release.build_release_core(
        score=score,
        inference=inference,
        upstream_artifacts=_upstream(),
    )
    generated = tmp_path / "paper" / "generated"
    manifest_path = tmp_path / "cct20_release_manifest.json"
    result = release.emit_release(
        core,
        release_manifest_path=manifest_path,
        generated_dir=generated,
    )

    verify_artifact_receipt(manifest_path)
    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert observed == result
    unsigned = dict(result)
    claimed = unsigned.pop("release_sha256")
    assert claimed == stable_sha256(unsigned)
    assert set(result["generated_artifacts"]) == {
        "cct20_numbers_tex",
        "cct20_primary_table_tex",
        "cct20_location_effects_tex",
    }
    for record in result["generated_artifacts"].values():
        path = Path(record["path"])
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    numbers = (generated / "cct20_numbers.tex").read_text(encoding="ascii")
    assert r"\newcommand{\CCTHelpfulCount}{23}" in numbers
    assert r"\newcommand{\CCTHarmfulCount}{22}" in numbers
    assert r"\newcommand{\CCTFalseAdaptCount}{0}" in numbers
    assert r"\newcommand{\CCTFalseAdaptRate}{0}" in numbers
    assert r"\newcommand{\CCTSafeUtilityMargin}{-0.005}" in numbers
    assert r"\newcommand{\CCTManuscriptClaim}" in numbers
    assert r"\newcommand{\CCTSecondaryMetricDisclosure}" in numbers
    primary = (generated / "cct20_primary_table.tex").read_text(encoding="ascii")
    assert "Bonferroni 97.5\\% CI" in primary
    assert "95% simultaneous Bonferroni family" in primary
    assert "Baseline regret $-$ KGA regret" in primary
    assert "Holm $p$" in primary
    locations = (generated / "cct20_location_effects.tex").read_text(encoding="ascii")
    assert "Helpful/zero/harmful" in locations
    assert "Eval. $n$/checkpoint" in locations
    assert "A/F/U (5 cells)" in locations
    assert "KGA acc. $-$ adapt acc." in locations
    assert len([line for line in locations.splitlines() if line.endswith(r"\\")]) == 10

    with pytest.raises(IntegrityError, match="already exist"):
        release.emit_release(
            core,
            release_manifest_path=manifest_path,
            generated_dir=generated,
        )


def test_emit_recovers_an_exact_read_only_partial_transaction(tmp_path: Path, strong_bundle) -> None:
    score, inference = strong_bundle
    core = release.build_release_core(
        score=score,
        inference=inference,
        upstream_artifacts=_upstream(),
    )
    generated = tmp_path / "paper" / "generated"
    partial = generated / "cct20_numbers.tex"
    release._exclusive_write(partial, release.render_numbers_tex(core).encode("ascii"))

    manifest_path = tmp_path / "cct20_release_manifest.json"
    result = release.emit_release(
        core,
        release_manifest_path=manifest_path,
        generated_dir=generated,
    )
    assert result["status"] == "RELEASE_COMPLETE"
    verify_artifact_receipt(manifest_path)
    assert partial.read_bytes() == release.render_numbers_tex(core).encode("ascii")


def test_emit_rejects_a_different_partial_transaction(
    tmp_path: Path,
    strong_bundle,
) -> None:
    score, inference = strong_bundle
    core = release.build_release_core(
        score=score,
        inference=inference,
        upstream_artifacts=_upstream(),
    )
    generated = tmp_path / "paper" / "generated"
    partial = generated / "cct20_numbers.tex"
    partial.parent.mkdir(parents=True)
    partial.write_text("% stale or fabricated output\n", encoding="ascii")
    with pytest.raises(IntegrityError, match="differs from the requested bytes"):
        release.emit_release(
            core,
            release_manifest_path=tmp_path / "cct20_release_manifest.json",
            generated_dir=generated,
        )


def test_emit_requires_canonical_release_manifest_name(tmp_path: Path, strong_bundle) -> None:
    score, inference = strong_bundle
    core = release.build_release_core(
        score=score,
        inference=inference,
        upstream_artifacts=_upstream(),
    )
    with pytest.raises(IntegrityError, match="must be named"):
        release.emit_release(
            core,
            release_manifest_path=tmp_path / "renamed.json",
            generated_dir=tmp_path / "paper" / "generated",
        )


def test_script_has_no_target_annotation_or_scorer_entrypoint() -> None:
    source = Path(release.__file__).read_text(encoding="utf-8")
    assert "trans_test_annotations.json" not in source
    assert "cct20_truth_loader" not in source
    assert "score_once(" not in source
    assert "--target-annotations" not in source


def test_publication_runbook_builds_and_seals_both_manuscript_forms() -> None:
    from docs.research.kbound.scripts.verify_release_checksums import REQUIRED_RELEASE_PATHS

    runbook = (
        release.REPOSITORY_ROOT
        / "docs/research/kbound/runbooks/release_candidate.sh"
    ).read_text(encoding="utf-8")
    renderer = (
        release.REPOSITORY_ROOT
        / "docs/research/kbound/scripts/render_pdf_pages.py"
    ).read_text(encoding="utf-8")

    assert 'BUILD_LONG_TMLR=1 PYTHON="$PY"' in runbook
    assert "tests/test_cct20_release_builder.py" in runbook
    assert "tests/test_cct20_manuscript_claim_validation.py" in runbook
    assert "tests/test_build_docx_pipeline.py" in runbook
    # The runbook and verifier share one exact-path inventory. Do not require
    # a duplicated filename list in the shell producer.
    assert '"$KB/scripts/verify_release_checksums.py" --list-required' in runbook
    for name in (
        "cct20_release_manifest.json.receipt.json",
        "cct20_numbers.tex",
        "cct20_primary_table.tex",
        "cct20_location_effects.tex",
        "kbound_short_final_draft.docx",
    ):
        prefix = "docs/research/kbound/" if name.endswith(".docx") else "docs/research/kbound/paper/generated/"
        assert prefix + name in REQUIRED_RELEASE_PATHS
    assert '("kbound_short_final_draft.pdf", "kbound_tmlr.pdf")' in renderer
