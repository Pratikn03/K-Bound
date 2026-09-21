from dataclasses import replace

import pytest

from kga.certificate import Certificate
from kga.policy import Decision, decide_hierarchical_candidates


def test_hierarchical_rejects_uncorrected_certificates():
    certificate = Certificate(delta_hat=0.1, epsilon=0.01, method="conformal", alpha=0.1, n=50)
    with pytest.raises(ValueError, match="alpha"):
        decide_hierarchical_candidates({"a": certificate, "b": certificate}, alpha=0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan"), float("inf")])
def test_hierarchical_rejects_invalid_requested_alpha_even_empty(alpha):
    with pytest.raises(ValueError, match="alpha"):
        decide_hierarchical_candidates({}, alpha=alpha)


def test_hierarchical_rejects_unknown_correction_even_empty():
    with pytest.raises(ValueError, match="correction"):
        decide_hierarchical_candidates({}, correction="bonferoni")


@pytest.mark.parametrize(
    "changes",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"alpha": float("nan")},
        {"epsilon": -0.1},
        {"epsilon": float("nan")},
        {"epsilon": -float("inf")},
        {"delta_hat": float("nan")},
        {"delta_hat": float("inf")},
        {"method": "evalue"},
        {"method": "unknown"},
        {"n": 0},
        {"n": -1},
        {"n": 1.5},
        {"n": True},
    ],
)
def test_hierarchical_rejects_invalid_bound_certificates(changes):
    certificate = Certificate(delta_hat=0.1, epsilon=0.01, method="conformal", alpha=0.05, n=50)
    with pytest.raises(ValueError):
        decide_hierarchical_candidates({"a": replace(certificate, **changes)}, alpha=0.1)


def test_hierarchical_infinite_radius_stays_ambiguous():
    certificate = Certificate(delta_hat=0.1, epsilon=float("inf"), method="conformal", alpha=0.05, n=1)
    assert decide_hierarchical_candidates({"a": certificate}).decision == Decision.ABSTAIN


def test_hierarchical_none_is_explicit_and_preserves_original_bounds():
    certificate = Certificate(delta_hat=0.1, epsilon=0.02, method="conformal", alpha=0.1, n=50)
    result = decide_hierarchical_candidates({"a": certificate, "b": certificate}, correction="none")
    assert result.candidate_bounds["a"] == pytest.approx((0.08, 0.12))
    assert certificate.alpha == 0.1
    assert certificate.epsilon == 0.02


def test_hierarchical_validates_unselected_candidate_and_exact_budget():
    good = Certificate(delta_hat=0.2, epsilon=0.01, method="conformal", alpha=0.05, n=50)
    invalid = replace(good, delta_hat=-0.2, alpha=0.05000000000000001)
    with pytest.raises(ValueError, match="alpha"):
        decide_hierarchical_candidates({"good": good, "unselected": invalid})


def test_decide_hierarchical_single_beneficial():
    c_norm = Certificate(delta_hat=0.08, epsilon=0.03, method="conformal", alpha=0.05, n=50)
    c_full = Certificate(delta_hat=-0.05, epsilon=0.04, method="conformal", alpha=0.05, n=50)

    res = decide_hierarchical_candidates({"norm": c_norm, "full": c_full}, alpha=0.10)
    assert res.decision == Decision.ADAPT
    assert res.selected_candidate == "norm"
    assert res.certified_lower_bound == pytest.approx(0.05)


def test_decide_hierarchical_greatest_lower_bound():
    c_norm = Certificate(delta_hat=0.08, epsilon=0.02, method="conformal", alpha=0.05, n=50)
    c_head = Certificate(delta_hat=0.10, epsilon=0.03, method="conformal", alpha=0.05, n=50)

    res = decide_hierarchical_candidates({"norm": c_norm, "head": c_head}, alpha=0.10)
    assert res.decision == Decision.ADAPT
    assert res.selected_candidate == "head"
    assert res.certified_lower_bound == pytest.approx(0.07)


def test_decide_hierarchical_all_harmful():
    c_norm = Certificate(delta_hat=-0.08, epsilon=0.02, method="conformal", alpha=0.05, n=50)
    c_head = Certificate(delta_hat=-0.10, epsilon=0.03, method="conformal", alpha=0.05, n=50)

    res = decide_hierarchical_candidates({"norm": c_norm, "head": c_head}, alpha=0.10)
    assert res.decision == Decision.FREEZE
    assert res.selected_candidate is None


def test_decide_hierarchical_ambiguous():
    c_norm = Certificate(delta_hat=0.02, epsilon=0.05, method="conformal", alpha=0.05, n=50)
    c_head = Certificate(delta_hat=-0.02, epsilon=0.05, method="conformal", alpha=0.05, n=50)

    res = decide_hierarchical_candidates({"norm": c_norm, "head": c_head}, alpha=0.10)
    assert res.decision == Decision.ABSTAIN
    assert res.selected_candidate is None


@pytest.mark.parametrize("method", ["conformal", "ebern", "hoeffding"])
def test_hierarchical_stricter_level_and_infinite_peer_are_preserved(method):
    good = Certificate(delta_hat=0.1, epsilon=0.02, method=method, alpha=0.01, n=50)
    unavailable = replace(good, epsilon=float("inf"))
    result = decide_hierarchical_candidates({"good": good, "unavailable": unavailable})
    assert result.selected_candidate == "good"
    assert result.candidate_bounds["good"] == pytest.approx((0.08, 0.12))
    assert good.alpha == 0.01
    assert good.epsilon == 0.02


@pytest.mark.parametrize("delta_hat", [-0.02, 0.02])
def test_hierarchical_strict_zero_boundaries(delta_hat):
    certificate = Certificate(delta_hat=delta_hat, epsilon=0.02, method="conformal", alpha=0.1, n=50)
    assert decide_hierarchical_candidates({"a": certificate}).decision == Decision.ABSTAIN


def test_hierarchical_empty_and_stable_tie_order():
    assert decide_hierarchical_candidates({}).decision == Decision.FREEZE
    certificate = Certificate(delta_hat=0.1, epsilon=0.02, method="conformal", alpha=0.05, n=50)
    result = decide_hierarchical_candidates({"first": certificate, "second": certificate})
    assert result.selected_candidate == "first"
