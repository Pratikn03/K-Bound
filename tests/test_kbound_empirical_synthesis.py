"""Independent hand-calculated metric cases for the retrospective synthesis."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/build_empirical_synthesis.py"


def implementation(name):
    assert SCRIPT.is_file(), "The empirical synthesis implementation is missing"
    spec = importlib.util.spec_from_file_location("empirical_synthesis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, name, None)
    assert callable(function), f"The empirical synthesis needs {name}"
    return function


def cell(identifier, benefit, action, group="g1", cluster="e1", lower=None, upper=None):
    return {
        "cell_id": identifier,
        "fold_id": "f1",
        "group": group,
        "cluster": cluster,
        "frozen_accuracy": 0.5,
        "candidate_accuracy": 0.5 + benefit,
        "action": action,
        "lower": lower,
        "upper": upper,
        "interval_status": "finite" if lower is not None else "not_applicable",
        "prediction": benefit,
    }


def test_percentage_points_weighted_loss_and_accuracy_are_distinct():
    # Incorrectly calling weighted regret an accuracy gain would fail these independent values.
    score = implementation("summarize_cells")(
        [cell("a", 0.2, "ABSTAIN"), cell("b", -0.1, "ADAPT"), cell("c", 0.1, "ADAPT")]
    )
    assert score["accuracy_percent"]["policy"] == pytest.approx(50)
    assert score["oracle_regret_pp"] == pytest.approx(10)
    assert score["weighted_loss_5_pp"] == pytest.approx(70 / 3)
    assert score["mean_positive_benefit_forgone_pp"] == pytest.approx(20 / 3)
    assert score["mean_accepted_harm_magnitude_pp"] == pytest.approx(10 / 3)


def test_zero_exposure_is_undefined_and_abstention_is_not_false_freeze():
    score = implementation("summarize_cells")([cell("a", 0.2, "ABSTAIN"), cell("b", -0.1, "FREEZE")])
    assert score["nonpositive_acceptance_conditional"] is None
    assert score["group_nonpositive_acceptance_conditional"] is None
    assert score["helpful_opportunities_missed"] == 1
    assert score["false_freeze_count"] == 0
    assert score["nonpositive_acceptance_unconditional"] == 0


def test_nonpositive_and_strictly_harmful_events_have_different_counts():
    score = implementation("summarize_cells")([cell("a", 0, "ADAPT"), cell("b", -0.1, "ADAPT")])
    assert score["nonpositive_accepted_count"] == 2
    assert score["harmful_accepted_count"] == 1
    assert score["zero_benefit_accepted_count"] == 1


def test_cell_inclusion_and_simultaneous_group_inclusion_use_different_denominators():
    score = implementation("summarize_cells")(
        [
            cell("a", 0, "ABSTAIN", lower=-0.05, upper=0.05),
            cell("b", 0.2, "ABSTAIN", lower=-0.05, upper=0.05),
            cell("c", 0, "ABSTAIN", group="g2", lower=-0.05, upper=0.05),
        ]
    )
    assert score["intervals"]["cell_inclusion"] == pytest.approx(2 / 3)
    assert score["intervals"]["simultaneous_group_inclusion"] == 0.5
    assert score["intervals"]["n_complete_finite_groups"] == 2


def test_unbounded_intervals_are_not_counted_as_observed_finite_inclusion():
    score = implementation("summarize_cells")([dict(cell("a", 0, "ABSTAIN"), interval_status="unbounded")])
    assert score["intervals"]["cell_inclusion"] is None
    assert score["intervals"]["simultaneous_group_inclusion"] is None
    assert score["intervals"]["n_unbounded_cells"] == 1


def test_equal_cell_accuracy_is_not_pooled_image_accuracy():
    score = implementation("summarize_cells")(
        [dict(cell("a", 0.1, "ADAPT"), n_evaluation_images=1), dict(cell("b", -0.1, "ADAPT"), n_evaluation_images=100)]
    )
    assert score["accuracy_percent"]["policy"] == pytest.approx(50)
    assert score["pooled_image_accuracy_percent"] is None


def test_pairing_rejects_mismatched_or_changed_outcomes():
    compare = implementation("paired_cluster_comparison")
    a = [cell("a", 0.1, "ADAPT")]
    with pytest.raises(ValueError, match="pair"):
        compare(a, [cell("b", 0.1, "FREEZE")])
    with pytest.raises(ValueError, match="outcome"):
        compare(a, [cell("a", -0.1, "FREEZE")])


def test_cluster_pairing_is_order_invariant_and_weights_clusters_equally():
    compare = implementation("paired_cluster_comparison")
    a = [cell("a", 0.1, "ADAPT"), cell("b", 0.1, "ADAPT"), cell("c", 0.3, "FREEZE", cluster="e2")]
    b = [dict(r, action="FREEZE") for r in a]
    result = compare(a, list(reversed(b)))
    assert result["n_clusters"] == 2
    assert result["mean_equal_cluster_delta_pp"] == pytest.approx(-5)
    assert result["mean_equal_cell_delta_pp"] == pytest.approx(-20 / 3)
    assert result["leave_one_cluster_out"][0]["remaining_clusters"] == 1


def test_duplicate_cells_and_nonfinite_scores_fail_closed():
    score = implementation("summarize_cells")
    a = cell("a", 0.1, "ADAPT")
    with pytest.raises(ValueError, match="duplicate"):
        score([a, a])
    with pytest.raises(ValueError, match="finite"):
        score([dict(a, candidate_accuracy=float("nan"))])


def test_matched_exposure_uses_prediction_ranking_and_preserves_exact_exposure():
    matched = implementation("matched_exposure_cells")
    rows = [
        dict(cell("a", -0.1, "ADAPT"), prediction=0.8),
        dict(cell("b", 0.2, "FREEZE"), prediction=0.9),
        dict(cell("c", 0.1, "ABSTAIN"), prediction=0.9),
    ]
    result = matched(rows)
    assert {r["cell_id"] for r in result if r["action"] == "ADAPT"} == {"b"}
    assert all(r["lower"] is None for r in result)


def test_hash_bound_input_reader_rejects_changed_bytes(tmp_path):
    reader = implementation("read_bound_input")
    p = tmp_path / "data.json"
    p.write_text('{"value": 1}')
    with pytest.raises(ValueError, match="SHA"):
        reader(tmp_path, "data.json", {}, expected_sha256="0" * 64)


def test_authority_comparison_rejects_unit_errors_and_missing_values():
    compare = implementation("verify_metric")
    assert compare(0.125, 0.125, "loss") == 1
    with pytest.raises(ValueError, match="loss"):
        compare(12.5, 0.125, "loss")
    with pytest.raises(ValueError, match="coverage"):
        compare(None, 0, "coverage")


def test_secondary_macro_f1_averages_classes_not_pooled_counts():
    score = implementation("verify_secondary_macro_f1")
    # F1s 1 and 0: macro .5; pooling counts would give 2/5 instead.
    row = {
        "macro_f1": 0.5,
        "n_output_indicators": 2,
        "zero_denominator_convention": 0,
        "per_class": [{"tp": 1, "fp": 0, "fn": 0, "f1": 1}, {"tp": 0, "fp": 3, "fn": 0, "f1": 0}],
    }
    assert score(row) == 0.5
    with pytest.raises(ValueError, match="macro"):
        score(dict(row, macro_f1=0.4))


def test_generated_outputs_are_create_only_and_never_escape_destination(tmp_path):
    write = implementation("write_outputs")
    write(tmp_path, {"empirical_example.json": b'{"a": 1}\n'})
    write(tmp_path, {"empirical_example.json": b'{"a": 1}\n'}, check=True)
    with pytest.raises(FileExistsError):
        write(tmp_path, {"empirical_example.json": b'{"a": 2}\n'})
    with pytest.raises(ValueError, match="filename"):
        write(tmp_path, {"../history.json": b"{}"})


def test_output_symlink_cannot_overwrite_a_frozen_input(tmp_path):
    write = implementation("write_outputs")
    source = tmp_path / "frozen.json"
    source.write_bytes(b"original")
    (tmp_path / "empirical_alias.json").symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        write(tmp_path, {"empirical_alias.json": b"changed"}, replace=True)
    assert source.read_bytes() == b"original"


def test_group_risk_uses_exposed_groups_and_keeps_folds_separate():
    score = implementation("summarize_cells")(
        [cell("a", -0.1, "ADAPT"), cell("b", 0.1, "ADAPT"), dict(cell("a", 0.1, "FREEZE"), fold_id="f2")]
    )
    assert score["n_groups"] == 2
    assert score["n_exposed_groups"] == 1
    assert score["group_nonpositive_acceptance_conditional"] == 1
    assert score["group_nonpositive_acceptance_unconditional"] == 0.5
    assert score["nonpositive_acceptance_conditional"] == 0.5


def test_incomplete_group_is_excluded_from_finite_simultaneous_denominator():
    score = implementation("summarize_cells")(
        [
            cell("a", 0, "ABSTAIN", lower=-0.1, upper=0.1),
            dict(cell("b", 0, "ABSTAIN"), interval_status="unbounded"),
            cell("c", 0.2, "ABSTAIN", group="g2", lower=-0.1, upper=0.1),
        ]
    )
    assert score["intervals"]["cell_inclusion"] == 0.5
    assert score["intervals"]["n_complete_finite_groups"] == 1
    assert score["intervals"]["simultaneous_group_inclusion"] == 0
