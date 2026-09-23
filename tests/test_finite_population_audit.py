"""Design-based finite-frame coverage, including dependence and empty strata."""

import copy
import itertools
from fractions import Fraction as F
from math import comb

import pytest

from kga.finite_population_audit import (
    _seal,
    assess_sample,
    draw_sample,
    make_plan,
    simplex_upper,
    transport_budget,
)


def frame(stratum="all", n=3, alpha=F(1, 5)):
    return make_plan(
        ids=["a", "b", "c"],
        frozen=[0, 0, 1],
        adapted=[1, 1, 1],
        classes=[0, 1],
        sample_size=n,
        alpha=alpha,
        predictor_identity="fixed-pair",
        stratum=stratum,
        scores=[F(1, 2)] * 3,
        bins=["x", "y", "x"],
        bin_universe=["x", "y"],
        bin_definition="fixed-output-bin-map",
    )


def draws(plan, indices, monkeypatch, tmp_path, suffix=""):
    iterator = iter(indices)
    monkeypatch.setattr("kga.finite_population_audit.secrets.randbelow", lambda _: next(iterator))
    return draw_sample(plan, tmp_path / ("draws" + suffix + ".json"))


def test_general_benefit_exhaustive_sampling_coverage(monkeypatch, tmp_path):
    plan = frame()
    labels = {"a": 1, "b": 0, "c": 1}
    covered = 0
    for j, indices in enumerate(itertools.product(range(3), repeat=3)):
        result = assess_sample(plan, draws(plan, indices, monkeypatch, tmp_path, str(j)), labels)
        lo, hi = map(F, result["benefit_interval_exact"])
        covered += lo <= 0 <= hi
        assert result["scope"] == "fixed_finite_frame_conditional_on_uniform_independent_draws"
        assert result["prospective_deployment_validated"] is False
    assert F(covered, 27) >= 1 - F(1, 5)


def test_binary_residual_coverage_and_disagreement_scaling(monkeypatch, tmp_path):
    plan = frame("disagreement", 4)
    labels = {"a": 1, "b": 1, "c": 0}
    covered = 0
    for j, indices in enumerate(itertools.product(range(2), repeat=4)):
        result = assess_sample(plan, draws(plan, indices, monkeypatch, tmp_path, str(j)), labels)
        assert F(result["disagreement_mass_exact"]) == F(2, 3)
        lo, hi = map(F, result["benefit_interval_exact"])
        covered += lo <= F(2, 3) <= hi
        assert F(result["residual_absolute_upper_exact"]) >= F(1, 2)
    assert F(covered, 16) >= F(4, 5)


def test_agreeing_frame_exact_zero_without_labels(monkeypatch, tmp_path):
    plan = make_plan(
        ids=["x"],
        frozen=[0],
        adapted=[0],
        classes=[0, 1],
        sample_size=5,
        alpha=F(1, 10),
        predictor_identity="p",
        stratum="disagreement",
    )
    draw = draws(plan, [], monkeypatch, tmp_path)
    result = assess_sample(plan, draw, {})
    assert draw["indices"] == []
    assert result["benefit_interval_exact"] == ["0", "0"]
    assert result["action"] == "ABSTAIN"
    assert result["residual_absolute_upper_exact"] is None


def test_draw_is_exclusive_and_bound_to_frame(monkeypatch, tmp_path):
    plan = frame()
    draw = draws(plan, [0, 1, 2], monkeypatch, tmp_path)
    with pytest.raises(FileExistsError):
        draw_sample(plan, tmp_path / "draws.json")
    tampered = copy.deepcopy(plan)
    tampered["adapted"][0] = 0
    with pytest.raises(ValueError, match="digest"):
        assess_sample(tampered, draw, {"a": 1, "b": 0, "c": 1})
    damaged = copy.deepcopy(draw)
    damaged["indices"][0] = 2
    with pytest.raises(ValueError, match="digest"):
        assess_sample(plan, damaged, {"a": 1, "b": 0, "c": 1})


def test_labels_only_required_for_sampled_ids(monkeypatch, tmp_path):
    plan = frame()
    draw = draws(plan, [0, 0, 0], monkeypatch, tmp_path)
    result = assess_sample(plan, draw, {"a": 1})
    assert result["draw_count"] == 3 and result["unique_labels_used"] == 1
    with pytest.raises(ValueError, match="missing"):
        assess_sample(plan, draw, {})
    with pytest.raises(ValueError, match="class"):
        assess_sample(plan, draw, {"a": 7})


def test_multiclass_disagreement_uses_paired_outcomes(monkeypatch, tmp_path):
    plan = make_plan(
        ids=["x"],
        frozen=[0],
        adapted=[1],
        classes=[0, 1, 2],
        sample_size=20,
        alpha=F(1, 10),
        predictor_identity="p",
        stratum="disagreement",
    )
    result = assess_sample(plan, draws(plan, [0] * 20, monkeypatch, tmp_path), {"x": 2})
    assert result["residual_absolute_upper_exact"] is None
    assert result["helpful_count"] == result["harmful_count"] == 0
    assert result["action"] == "ABSTAIN"


def test_simplex_bound_matches_vertex_oracle():
    assert simplex_upper([F(1, 4), F(1)], [F(1, 2), 0], [1, F(1, 2)]) == F(5, 8)
    with pytest.raises(ValueError):
        simplex_upper([0, 1], [F(3, 4)] * 2, [1, 1])


def test_transport_missing_class_and_budget_accounting(monkeypatch, tmp_path):
    plan = frame(n=3)
    source = draws(plan, [0] * 3, monkeypatch, tmp_path, "s")
    target = draws(plan, [0] * 3, monkeypatch, tmp_path, "t")
    result = transport_budget(plan, source, {"a": 0}, plan, target, {"a": 0}, alpha=F(1, 10))
    assert 0 <= F(result["rho_upper_exact"]) <= 1
    assert F(result["per_class_tv_upper_exact"]["1"]) == 1
    assert F(result["per_interval_alpha_exact"]) * result["interval_count"] == F(1, 10)
    assert result["additional_failure_budget_exact"] == "1/10"


def test_invalid_frames_reject():
    with pytest.raises(ValueError):
        make_plan(
            ids=["x", "x"],
            frozen=[0, 0],
            adapted=[1, 1],
            classes=[0, 1],
            sample_size=4,
            alpha=F(1, 10),
            predictor_identity="p",
        )
    with pytest.raises(ValueError):
        frame(n=0)


def test_set_membership_preserves_both_correct_on_disagreement(monkeypatch, tmp_path):
    plan = make_plan(
        ids=["x"],
        frozen=[0],
        adapted=[1],
        classes=[0, 1],
        sample_size=5,
        alpha=F(1, 10),
        predictor_identity="p",
        stratum="disagreement",
        label_semantics="set_membership",
        scores=[F(1, 2)],
    )
    result = assess_sample(plan, draws(plan, [0] * 5, monkeypatch, tmp_path), {"x": [0, 1]})
    assert result["helpful_count"] == result["harmful_count"] == 0
    assert result["residual_absolute_upper_exact"] is None
    assert result["action"] == "ABSTAIN"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown"),
        ("stratum", "adaptive"),
        ("label_semantics", "other"),
        ("sample_size", True),
        ("sample_size", 0),
        ("frozen", [0]),
        ("classes", [0, True]),
        ("alpha_exact", "1"),
        ("alpha_exact", "1e-2"),
        ("scores_exact", ["1/2"]),
        ("ids", ["a", "a", "b"]),
        ("bin_definition", 123),
    ],
)
def test_resealed_invalid_plan_is_not_authenticated_by_its_digest(field, value, tmp_path):
    plan = frame()
    plan[field] = value
    _seal(plan)  # Deliberately recompute the public checksum after changing the payload.
    with pytest.raises((ValueError, TypeError)):
        draw_sample(plan, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("sampling", "without_replacement"),
        ("randomness_independently_attested", True),
        ("randomness_independently_attested", 0),
        ("indices", [0, True, 2]),
        ("indices", [0, 0, 99]),
        ("indices", [0, 1]),
        ("indices", "012"),
        ("plan_sha256", "0" * 64),
    ],
)
def test_resealed_malformed_draws_are_rejected(field, value, monkeypatch, tmp_path):
    plan = frame()
    draw = draws(plan, [0, 1, 2], monkeypatch, tmp_path)
    draw[field] = value
    _seal(draw)
    with pytest.raises(ValueError):
        assess_sample(plan, draw, {"a": 1, "b": 0, "c": 1})


def test_mixed_binary_disagreement_residual_has_exact_design_coverage(monkeypatch, tmp_path):
    # Three disagreement points: two adapted-correct, one frozen-correct.
    # Scores vary over the frame, but their mean is known exactly from the plan.
    plan = make_plan(
        ids=["a", "b", "c", "d"],
        frozen=[0, 0, 0, 1],
        adapted=[1, 1, 1, 1],
        classes=[0, 1],
        sample_size=5,
        alpha=F(1, 2),
        predictor_identity="mixed",
        stratum="disagreement",
        scores=[F(1, 4), F(1, 2), F(3, 4), F(1, 2)],
    )
    labels = {"a": 1, "b": 1, "c": 0}
    benefit, residual = F(1, 4), F(1, 6)
    covered = 0
    for j, indices in enumerate(itertools.product(range(3), repeat=5)):
        result = assess_sample(plan, draws(plan, indices, monkeypatch, tmp_path, str(j)), labels)
        lo, hi = map(F, result["benefit_interval_exact"])
        residual_upper = F(result["residual_absolute_upper_exact"])
        covered += lo <= benefit <= hi and residual <= residual_upper
    assert F(covered, 3**5) >= F(1, 2)


def test_set_membership_transport_is_rejected(monkeypatch, tmp_path):
    plan = make_plan(
        ids=["a"],
        frozen=[0],
        adapted=[1],
        classes=[0, 1],
        sample_size=2,
        alpha=F(1, 10),
        predictor_identity="p",
        label_semantics="set_membership",
        bins=["x"],
        bin_universe=["x"],
        bin_definition="pair-bin",
    )
    sample = draws(plan, [0, 0], monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="single-class partition"):
        transport_budget(plan, sample, {"a": [0, 1]}, plan, sample, {"a": [0, 1]}, alpha=F(1, 10))


def test_transport_bound_exact_sampling_coverage_with_nonzero_truth(monkeypatch, tmp_path):
    # Enumerate multinomial count states, retaining their exact sequence counts.
    # Source class 0 has bins (1/2,1/2), target class 0 has (1,0).
    # Class 1 changes from x to y. Target priors (2/3,1/3) give rho=2/3.
    n, alpha, truth = 6, F(1, 2), F(2, 3)
    common = {
        "ids": ["a", "b", "c"],
        "frozen": [0, 0, 1],
        "adapted": [0, 0, 1],
        "classes": [0, 1],
        "sample_size": n,
        "alpha": alpha,
        "predictor_identity": "fixed-pair",
        "bin_universe": ["x", "y"],
        "bin_definition": "fixed-bin-map",
    }
    source = make_plan(**common, bins=["x", "y", "x"])
    target = make_plan(**common, bins=["x", "x", "y"])
    labels = {"a": 0, "b": 0, "c": 1}
    inventories = []
    for a in range(n + 1):
        for b in range(n - a + 1):
            indices = [0] * a + [1] * b + [2] * (n - a - b)
            multiplicity = comb(n, a) * comb(n - a, b)
            index = len(inventories)
            inventories.append(
                (
                    draws(source, indices, monkeypatch, tmp_path, f"s{index}"),
                    draws(target, indices, monkeypatch, tmp_path, f"t{index}"),
                    multiplicity,
                )
            )
    covered = total = nonvacuous = 0
    for source_draw, _, source_weight in inventories:
        for _, target_draw, target_weight in inventories:
            result = transport_budget(source, source_draw, labels, target, target_draw, labels, alpha=alpha)
            weight = source_weight * target_weight
            bound = F(result["rho_upper_exact"])
            covered += weight * (truth <= bound)
            nonvacuous += weight * (bound < 1)
            total += weight
    assert total == 3 ** (2 * n)
    assert nonvacuous > 0  # The test is not satisfied only by full-unit bounds.
    assert F(covered, total) >= 1 - alpha
