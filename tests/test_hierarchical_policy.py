import pytest
from kga.certificate import Certificate
from kga.policy import Decision, decide_hierarchical_candidates


def test_decide_hierarchical_single_beneficial():
    c_norm = Certificate(delta_hat=0.08, epsilon=0.03, method="conformal", alpha=0.1, n=50)
    c_full = Certificate(delta_hat=-0.05, epsilon=0.04, method="conformal", alpha=0.1, n=50)

    res = decide_hierarchical_candidates({"norm": c_norm, "full": c_full}, alpha=0.10)
    assert res.decision == Decision.ADAPT
    assert res.selected_candidate == "norm"
    assert res.certified_lower_bound == pytest.approx(0.05)


def test_decide_hierarchical_greatest_lower_bound():
    c_norm = Certificate(delta_hat=0.08, epsilon=0.02, method="conformal", alpha=0.1, n=50)
    c_head = Certificate(delta_hat=0.10, epsilon=0.03, method="conformal", alpha=0.1, n=50)

    res = decide_hierarchical_candidates({"norm": c_norm, "head": c_head}, alpha=0.10)
    assert res.decision == Decision.ADAPT
    assert res.selected_candidate == "head"
    assert res.certified_lower_bound == pytest.approx(0.07)


def test_decide_hierarchical_all_harmful():
    c_norm = Certificate(delta_hat=-0.08, epsilon=0.02, method="conformal", alpha=0.1, n=50)
    c_head = Certificate(delta_hat=-0.10, epsilon=0.03, method="conformal", alpha=0.1, n=50)

    res = decide_hierarchical_candidates({"norm": c_norm, "head": c_head}, alpha=0.10)
    assert res.decision == Decision.FREEZE
    assert res.selected_candidate is None


def test_decide_hierarchical_ambiguous():
    c_norm = Certificate(delta_hat=0.02, epsilon=0.05, method="conformal", alpha=0.1, n=50)
    c_head = Certificate(delta_hat=-0.02, epsilon=0.05, method="conformal", alpha=0.1, n=50)

    res = decide_hierarchical_candidates({"norm": c_norm, "head": c_head}, alpha=0.10)
    assert res.decision == Decision.ABSTAIN
    assert res.selected_candidate is None
