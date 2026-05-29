"""Deterministic checks for the Phase-4 / T3 partial-domain-failure study.

Locks in: clean quietness, the T3 mean-gate dilution boundary (soft per-domain
weighting helps at a smaller k than the hard batch gate), and the empty
all-domain-collapse extreme.
"""

from __future__ import annotations

from src.scripts.run_partial_domain_failure_study import run_study


def test_clean_quietness():
    report = run_study(seed=0)
    clean = report["rows"][0]
    assert clean["k_failed_domains"] == 0
    assert report["clean_false_fire_rate"] <= report["false_fire_budget"]
    assert abs(clean["delta_soft_vs_static"]) <= report["benefit_margin"]


def test_soft_helps_no_later_than_hard():
    report = run_study(seed=0)
    soft_k = report["first_k_with_benefit_soft"]
    hard_k = report["first_k_with_benefit_hard"]
    assert soft_k is not None and hard_k is not None
    assert soft_k <= hard_k


def test_mean_gate_dilution_region_exists():
    # There is at least one k>0 where soft weighting helps but the hard mean
    # gate does not fire (diluted) -> the T3 boundary.
    report = run_study(seed=0)
    margin = report["benefit_margin"]
    dilution = [
        r for r in report["rows"]
        if r["k_failed_domains"] > 0
        and r["delta_soft_vs_static"] > margin
        and r["gate_fire_rate"] == 0.0
    ]
    assert dilution, "expected a k where soft helps but the hard gate is diluted"


def test_all_domain_collapse_is_neutral():
    report = run_study(seed=0)
    last = report["rows"][-1]
    assert last["k_failed_domains"] == report["n_domains"]
    assert abs(last["delta_soft_vs_static"]) <= report["benefit_margin"]
