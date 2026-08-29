"""Tests for kga.assumptions.

These are mostly *negative* tests: the module's job is to refuse to certify, so the
things worth pinning down are the refusals -- unattainable alpha returning an infinite
radius, missing provenance failing closed, the gate never climbing back up the ladder,
and no code path that turns an observed hit rate into a theoretical coverage claim.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kga.assumptions import (
    CoverageClaimBasis,
    CoverageType,
    GateDecision,
    GateThresholds,
    ProtocolRecord,
    Status,
    conclusion_stability,
    conformal_radius,
    effective_units,
    evidence_support_overlap,
    leakage_audit,
    observed_coverage,
    radius_stability,
    risk_alignment_audit,
    run_gate,
)

HASH = "a" * 64


def _coverage_basis():
    return CoverageClaimBasis(
        theorem_or_result="exact split-conformal rank theorem",
        calibration_design="disjoint exchangeable residual split",
        inference_unit="domain",
        assumptions=("exchangeable domain-level residuals", "fixed estimator before calibration"),
        protocol_sha256=HASH,
        calibration_artifact_sha256="b" * 64,
        justification="The deployment unit is sampled by the locked domain-level protocol.",
    )


def _clean_record(**over):
    base = {
        "protocol": "TEST_PROTOCOL_v1",
        "dataset": "synthetic",
        "inference_unit": "domain",
        "candidate_fixed_at": "2026-01-01T00:00:00Z",
        "calibration_design_fixed_at": "2026-01-02T00:00:00Z",
        "target_evaluated_at": "2026-02-01T00:00:00Z",
        "target_labels_accessed": False,
        "target_labels_used_for_routing": False,
        "test_set_influenced_hparams": False,
        "calibration_test_separated": True,
        "protocol_lock_id": "lock-abc123",
        "failed_runs_retained": True,
    }
    base.update(over)
    return ProtocolRecord(**base)


# --------------------------------------------------------------------------- #
# conformal radius
# --------------------------------------------------------------------------- #


def test_conformal_radius_is_the_exact_rank_statistic():
    res = np.array([0.1, 0.5, 0.2, 0.4, 0.3])
    out = conformal_radius(res, alpha=0.2)
    # k = ceil(6 * 0.8) = 5 -> the 5th smallest of |residual| = 0.5
    assert out["k"] == 5
    assert out["radius"] == pytest.approx(0.5)
    assert out["level_attainable"] is True


def test_unattainable_alpha_gives_infinite_radius_not_a_finite_guess():
    # n = 5, alpha = 0.01 -> k = ceil(6 * 0.99) = 6 > 5
    out = conformal_radius(np.arange(5, dtype=float), alpha=0.01)
    assert out["level_attainable"] is False
    assert math.isinf(out["radius"])
    assert out["best_attainable_coverage"] == pytest.approx(5 / 6)


def test_empty_residuals_do_not_produce_a_radius():
    out = conformal_radius([], alpha=0.1)
    assert math.isinf(out["radius"])
    assert out["n"] == 0


# --------------------------------------------------------------------------- #
# observed coverage
# --------------------------------------------------------------------------- #


def test_observed_coverage_is_never_labelled_theoretical():
    d = np.array([0.0, 0.0, 0.0, 5.0])
    out = observed_coverage(d, d - 1, d + 1)
    assert out["coverage_type"] == CoverageType.OBSERVED_EMPIRICAL.value
    assert out["observed_coverage"] == pytest.approx(1.0)


def test_observed_coverage_counts_misses():
    d = np.array([0.0, 2.0, 0.0, 0.0])
    lo = np.full(4, -1.0)
    hi = np.full(4, 1.0)
    out = observed_coverage(d, lo, hi)
    assert out["observed_coverage"] == pytest.approx(0.75)
    lo_ci, hi_ci = out["coverage_interval_95"]
    assert 0.0 <= lo_ci <= 0.75 <= hi_ci <= 1.0


def test_clustered_interval_is_wider_than_the_iid_one_when_rows_are_correlated():
    """The whole point of A5: 200 correlated rows are not 200 draws."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 20)
    # within a group every row hits or every row misses -> n_eff is 10, not 200
    group_hit = rng.random(10) < 0.9
    hit = np.repeat(group_hit, 20)
    d = np.where(hit, 0.0, 5.0)
    iid = observed_coverage(d, np.full(200, -1.0), np.full(200, 1.0))
    clustered = observed_coverage(d, np.full(200, -1.0), np.full(200, 1.0), groups=groups)
    assert clustered["n_units"] == 10
    assert iid["n_units"] == 200
    width_iid = iid["coverage_interval_95"][1] - iid["coverage_interval_95"][0]
    width_cl = clustered["coverage_interval_95"][1] - clustered["coverage_interval_95"][0]
    assert width_cl > width_iid


def test_empty_evaluation_returns_none_not_zero():
    out = observed_coverage([], [], [])
    assert out["observed_coverage"] is None
    assert out["coverage_interval_95"] is None


# --------------------------------------------------------------------------- #
# support overlap
# --------------------------------------------------------------------------- #


def test_overlapping_evidence_passes():
    rng = np.random.default_rng(1)
    z_cal = rng.normal(size=(200, 3))
    z_dep = rng.normal(size=(100, 3))
    out = evidence_support_overlap(z_cal, z_dep)
    assert out.status in (Status.PASS.value, Status.WARNING.value)
    assert out.domain_classifier_auroc is not None
    assert out.domain_classifier_auroc < 0.75


def test_shifted_evidence_is_flagged():
    rng = np.random.default_rng(2)
    z_cal = rng.normal(size=(200, 3))
    z_dep = rng.normal(size=(100, 3)) + 8.0
    out = evidence_support_overlap(z_cal, z_dep)
    assert out.status == Status.FAIL.value
    assert out.frac_outside_envelope > 0.9
    assert out.domain_classifier_auroc > 0.9


def test_support_overlap_states_its_own_one_sidedness():
    rng = np.random.default_rng(3)
    out = evidence_support_overlap(rng.normal(size=(50, 2)), rng.normal(size=(50, 2)))
    assert any("one-sided" in n for n in out.notes)


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        evidence_support_overlap(np.zeros((10, 2)), np.zeros((10, 3)))


# --------------------------------------------------------------------------- #
# radius stability
# --------------------------------------------------------------------------- #


def test_homogeneous_units_are_stable():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(20), 10)
    res = np.abs(rng.normal(size=200))
    out = radius_stability(res, groups, alpha=0.1)
    assert out.status == Status.PASS.value
    assert out.radius_cv is not None and out.radius_cv < 0.2


def test_contamination_straddling_the_quantile_destabilises_the_radius():
    """Two bad domains out of twenty put the 90th percentile exactly on the boundary.

    Drop a contaminated domain and the radius collapses to the clean scale; drop a
    clean one and it jumps to the contaminated scale.  The promoted radius is then an
    artefact of which domain happened to be held out -- precisely the situation the
    stability check exists to catch.
    """
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(20), 10)
    res = np.abs(rng.normal(size=200)) * 0.01
    res[(groups == 0) | (groups == 1)] = 50.0
    out = radius_stability(res, groups, alpha=0.1, delta_hat=np.full(200, 1.0))
    assert out.status == Status.FAIL.value
    assert out.radius_range > 1.0
    assert out.decision_disagreement_rate is not None


def test_a_small_contaminated_fraction_does_not_move_a_loose_quantile():
    """The complementary true statement, pinned so the check is not read as magic.

    One bad domain in twenty is 5% of the rows, and a 90% quantile does not notice 5%
    contamination.  Radius stability detects dependence on the split; it is not an
    outlier detector, and this test records that boundary.
    """
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(20), 10)
    res = np.abs(rng.normal(size=200)) * 0.01
    res[groups == 0] = 50.0
    out = radius_stability(res, groups, alpha=0.1)
    assert out.status == Status.PASS.value


def test_single_unit_is_not_assessable():
    out = radius_stability(np.ones(10), np.zeros(10), alpha=0.1)
    assert out.status == Status.FAIL.value
    assert out.radius_cv is None


# --------------------------------------------------------------------------- #
# risk alignment
# --------------------------------------------------------------------------- #


def test_risk_alignment_is_always_marked_retrospective():
    out = risk_alignment_audit([0.1, -0.2, 0.3], [0.2, -0.1, 0.25])
    assert out.retrospective is True
    assert any("retrospective" in n for n in out.notes)
    assert out.sign_agreement == pytest.approx(1.0)


def test_risk_alignment_detects_sign_failure():
    out = risk_alignment_audit([0.5, 0.5, 0.5], [-0.5, -0.5, -0.5])
    assert out.sign_agreement == pytest.approx(0.0)
    assert out.mae == pytest.approx(1.0)


def test_false_adapt_by_group_uses_the_radius():
    dh = np.array([1.0, 1.0, -1.0, -1.0])
    dt = np.array([-1.0, 1.0, -1.0, 1.0])
    g = np.array(["a", "a", "b", "b"])
    out = risk_alignment_audit(dh, dt, groups=g, radius=0.5)
    # group a: dh=1 -> L=0.5>0 -> adapt; dt=-1 is harmful -> 1 of 2 false adapts
    assert out.false_adapt_by_group["a"] == pytest.approx(0.5)
    assert out.false_adapt_by_group["b"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# conclusion stability
# --------------------------------------------------------------------------- #


def test_no_alternatives_is_a_failure_not_a_pass():
    out = conclusion_stability("beats-both", {})
    assert out.status == Status.FAIL.value


def test_flip_under_an_admissible_split_fails():
    out = conclusion_stability(
        "beats-both",
        {"by seed": lambda: "beats-both", "by corruption family": lambda: "tie"},
    )
    assert out.status == Status.FAIL.value
    assert "by corruption family" in out.changed_under


def test_a_raising_alternative_counts_as_changed():
    def boom():
        raise RuntimeError("no artifact")

    out = conclusion_stability("x", {"broken": boom})
    assert out.status == Status.FAIL.value
    assert out.n_changed == 1


# --------------------------------------------------------------------------- #
# leakage audit
# --------------------------------------------------------------------------- #


def test_clean_record_passes():
    status, reasons = leakage_audit(_clean_record())
    assert status is Status.PASS
    assert reasons == []


def test_missing_lock_id_fails_closed():
    status, reasons = leakage_audit(_clean_record(protocol_lock_id=None))
    assert status is Status.FAIL
    assert any("lock identifier" in r for r in reasons)


def test_unknown_provenance_is_not_a_pass():
    status, reasons = leakage_audit(_clean_record(calibration_test_separated=None))
    assert status is Status.FAIL
    assert any("provenance unknown" in r for r in reasons)


def test_candidate_fixed_after_evaluation_is_a_violation():
    status, reasons = leakage_audit(
        _clean_record(
            candidate_fixed_at="2026-03-01T00:00:00Z",
            target_evaluated_at="2026-02-01T00:00:00Z",
        )
    )
    assert status is Status.FAIL
    assert any("A6 violated" in r for r in reasons)


def test_routing_on_target_labels_is_a_violation():
    status, reasons = leakage_audit(_clean_record(target_labels_used_for_routing=True))
    assert status is Status.FAIL
    assert any("A4 violated" in r for r in reasons)


def test_discarded_failed_runs_are_flagged():
    status, reasons = leakage_audit(_clean_record(failed_runs_retained=False))
    assert status is Status.FAIL
    assert any("failed runs" in r for r in reasons)


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def _good_gate_inputs(seed=7, n_units=40, per_unit=10):
    rng = np.random.default_rng(seed)
    n = n_units * per_unit
    groups = np.repeat(np.arange(n_units), per_unit)
    residuals = np.abs(rng.normal(scale=0.05, size=n))
    z_cal = rng.normal(size=(n, 2))
    z_dep = rng.normal(size=(200, 2))
    delta_hat = rng.normal(loc=0.3, scale=0.05, size=n)
    delta_true = delta_hat + rng.normal(scale=0.02, size=n)
    return {
        "alpha": 0.1,
        "residuals": residuals,
        "calibration_groups": groups,
        "z_cal": z_cal,
        "z_dep": z_dep,
        "delta_hat": delta_hat,
        "delta_true": delta_true,
        "interval_lower": delta_hat - 0.2,
        "interval_upper": delta_hat + 0.2,
        "evaluation_groups": groups,
        "conclusion": "no-harm",
        "alternatives": {"by seed": lambda: "no-harm", "by cell": lambda: "no-harm"},
    }


def test_a_well_provenanced_track_can_certify():
    rep = run_gate(record=_clean_record(), **_good_gate_inputs())
    assert rep.deployment_gate == GateDecision.CERTIFY.value
    assert rep.fallback_action == "adapt_freeze_abstain"


def test_certify_still_does_not_claim_theoretical_coverage_by_default():
    rep = run_gate(record=_clean_record(), **_good_gate_inputs())
    assert rep.theoretical_coverage_claimed is False
    assert rep.coverage_type == CoverageType.OBSERVED_EMPIRICAL.value
    assert any("no theoretical coverage claim" in x for x in rep.limitations)


def test_theoretical_claim_is_withdrawn_when_the_gate_does_not_certify():
    args = _good_gate_inputs()
    args["z_dep"] = np.asarray(args["z_dep"]) + 8.0  # force a support failure
    rep = run_gate(
        record=_clean_record(),
        claim_theoretical_coverage=True,
        coverage_claim_basis=_coverage_basis(),
        **args,
    )
    assert rep.deployment_gate != GateDecision.CERTIFY.value
    assert rep.theoretical_coverage_claimed is False
    assert any("withdrawn" in x for x in rep.limitations)


def test_theoretical_claim_requires_auditable_basis():
    rep = run_gate(
        record=_clean_record(),
        claim_theoretical_coverage=True,
        **_good_gate_inputs(),
    )
    assert rep.theoretical_coverage_claimed is False
    assert any("no auditable CoverageClaimBasis" in x for x in rep.limitations)


def test_valid_external_basis_can_be_recorded_but_is_not_created_by_diagnostics():
    rep = run_gate(
        record=_clean_record(),
        claim_theoretical_coverage=True,
        coverage_claim_basis=_coverage_basis(),
        **_good_gate_inputs(),
    )
    assert rep.deployment_gate == GateDecision.CERTIFY.value
    assert rep.theoretical_coverage_claimed is True
    assert rep.coverage_type == CoverageType.THEORETICAL.value
    assert rep.coverage_claim_basis["protocol_sha256"] == HASH


def test_leakage_rejects_and_nothing_downstream_can_upgrade_it():
    rep = run_gate(
        record=_clean_record(target_labels_used_for_routing=True),
        **_good_gate_inputs(),
    )
    assert rep.deployment_gate == GateDecision.REJECT.value
    assert rep.fallback_action == "none"


def test_too_few_units_downgrades_to_diagnostic_only():
    rep = run_gate(record=_clean_record(), **_good_gate_inputs(n_units=3, per_unit=5))
    assert rep.deployment_gate == GateDecision.DIAGNOSTIC_ONLY.value


def test_support_failure_restricts_to_freeze_or_abstain():
    args = _good_gate_inputs()
    args["z_dep"] = np.asarray(args["z_dep"]) + 8.0
    rep = run_gate(record=_clean_record(), **args)
    assert rep.deployment_gate == GateDecision.RESTRICTED.value
    assert rep.fallback_action == "freeze_or_abstain"


def test_unevaluated_checks_are_failures():
    rep = run_gate(record=_clean_record(), alpha=0.1)
    assert rep.support_overlap_status == Status.FAIL.value
    assert rep.radius_stability_status == Status.FAIL.value
    assert rep.conclusion_stability_status == Status.FAIL.value
    assert rep.deployment_gate in (
        GateDecision.DIAGNOSTIC_ONLY.value,
        GateDecision.REJECT.value,
    )


def test_missing_numbers_are_none_never_invented():
    rep = run_gate(record=_clean_record(), alpha=0.1)
    assert rep.observed_coverage is None
    assert rep.coverage_interval_95 is None
    assert rep.n_units is None


def test_report_round_trips_through_json():
    rep = run_gate(record=_clean_record(), **_good_gate_inputs())
    parsed = json.loads(rep.to_json())
    assert parsed["schema_version"] == "kbound-assumption-report/2"
    assert parsed["dataset"] == "synthetic"
    assert {
        "deployment_gate",
        "fallback_action",
        "coverage_type",
        "theoretical_coverage_claimed",
        "observed_coverage",
        "coverage_interval_95",
        "support_overlap_status",
        "radius_stability_status",
        "conclusion_stability_status",
        "limitations",
    } <= set(parsed)


def test_gate_is_deterministic():
    a = run_gate(record=_clean_record(), **_good_gate_inputs())
    b = run_gate(record=_clean_record(), **_good_gate_inputs())
    assert a.to_json() == b.to_json()


def test_thresholds_are_recorded_in_the_report():
    th = GateThresholds(min_effective_units=99)
    rep = run_gate(record=_clean_record(), thresholds=th, **_good_gate_inputs())
    assert rep.thresholds["min_effective_units"] == 99
    assert rep.deployment_gate == GateDecision.DIAGNOSTIC_ONLY.value


def test_effective_units_counts_groups_not_rows():
    assert effective_units(np.repeat(np.arange(5), 100), 500) == 5
    assert effective_units(None, 500) == 500
