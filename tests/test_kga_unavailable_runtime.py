"""Availability failures must not reuse or fabricate a deployment certificate.

All inputs are small synthetic fixtures. No dataset, training, or locked target
artifact is read by this suite.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kga import Certificate, Decision, FrozenLinearBenefitEstimator, KGA, decide, decide_batch, decide_kga
from kga import cli
from kga.certificate import conformal_split, evalue_anytime, split_conformal_rank_radius
from kga.routing import (
    AnytimeMulticandidatePanel,
    CandidateCertificate,
    bonferroni_multicandidate_route,
    route_panel,
)


def _estimator() -> FrozenLinearBenefitEstimator:
    return FrozenLinearBenefitEstimator(
        feature_names=("x",),
        weights=np.array([1.0]),
        intercept=0.0,
        feature_center=np.array([0.0]),
        feature_scale=np.array([1.0]),
        residuals=np.zeros(19),
        evidence_schema_version="runtime-fixture/1",
        protocol_sha256="a" * 64,
        fit_unit="development",
        calibration_unit="heldout_calibration",
    )


def _assert_no_cached_authority(gate: KGA) -> None:
    report = gate.explain()
    for key in (
        "evidence",
        "certificate",
        "decision",
        "estimator_artifact_sha256",
        "protocol_sha256",
        "evidence_schema_version",
    ):
        assert report[key] is None, key
    with pytest.raises(ValueError, match="No certificate available"):
        gate.decide()


@pytest.mark.parametrize("prior_delta", [-0.4, 0.4])
@pytest.mark.parametrize("failure", ["evidence", "estimate", "residuals", "scores", "probe", "decision"])
def test_failed_attempt_cannot_reuse_previous_commitment(prior_delta: float, failure: str) -> None:
    gate = KGA()
    scores = np.linspace(0.0, 1.0, 20)
    gate.evidence(scores, scores)
    gate.certify(delta_hat=prior_delta, calib_residuals=np.zeros(19))
    assert gate.decide() is (Decision.ADAPT if prior_delta > 0 else Decision.FREEZE)
    assert gate.last_evidence is not None

    with pytest.raises(ValueError):
        if failure == "evidence":
            gate.evidence(scores, np.array([]))
        elif failure == "estimate":
            gate.certify(delta_hat=math.nan, calib_residuals=np.zeros(19))
        elif failure == "residuals":
            gate.certify(delta_hat=prior_delta, calib_residuals=np.array([]))
        elif failure == "scores":
            gate.certify(scores=np.array([math.nan]), benefit_range=2.0)
        elif failure == "probe":
            gate.certify_probe(np.array([]), benefit_range=2.0)
        else:
            gate.decide(Certificate(prior_delta, math.nan, "conformal", 0.1, 19))
    _assert_no_cached_authority(gate)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, math.nan, "2"])
def test_invalid_probe_budget_does_not_consume_full_pool_or_reuse_commitment(budget) -> None:
    gate = KGA()
    gate.certify(delta_hat=-0.4, calib_residuals=np.zeros(19))
    assert gate.decide() is Decision.FREEZE
    with pytest.raises(ValueError, match="positive integer"):
        gate.certify_probe(np.full(30, -0.4), k=budget, benefit_range=2.0)
    _assert_no_cached_authority(gate)


@pytest.mark.parametrize("budget, expected_n", [(None, 30), (7, 7), (30, 30), (100, 30), (np.int64(7), 7)])
def test_valid_probe_budget_keeps_existing_full_pool_and_subsampling_rules(budget, expected_n: int) -> None:
    gate = KGA()
    certificate = gate.certify_probe(np.full(30, -0.4), k=budget, benefit_range=2.0)
    assert certificate.n == expected_n


@pytest.mark.parametrize("failure", ["missing", "nonfinite", "schema", "protocol", "no_evidence"])
def test_failed_frozen_estimator_attempt_clears_identity(failure: str) -> None:
    gate = KGA()
    estimator = _estimator()
    kwargs = {
        "features": {"x": -0.4},
        "evidence_schema_version": estimator.evidence_schema_version,
        "protocol_sha256": estimator.protocol_sha256,
    }
    gate.certify_evidence(estimator, **kwargs)
    assert gate.decide() is Decision.FREEZE
    assert gate.last_estimator_artifact_sha256 == estimator.artifact_sha256

    if failure == "missing":
        kwargs["features"] = {}
    elif failure == "nonfinite":
        kwargs["features"] = {"x": math.inf}
    elif failure == "schema":
        kwargs["evidence_schema_version"] = "wrong/1"
    elif failure == "protocol":
        kwargs["protocol_sha256"] = "b" * 64
    else:
        kwargs.pop("features")
    with pytest.raises(ValueError):
        gate.certify_evidence(estimator, **kwargs)
    _assert_no_cached_authority(gate)


def test_new_evidence_invalidates_previous_certificate_until_recertification() -> None:
    gate = KGA()
    gate.certify(delta_hat=0.4, calib_residuals=np.zeros(19))
    assert gate.decide() is Decision.ADAPT
    scores = np.linspace(0.0, 1.0, 20)
    evidence = gate.evidence(scores, scores)
    assert gate.last_evidence is evidence
    assert gate.last_certificate is None
    assert gate.last_decision is None
    gate.certify(delta_hat=-0.4, calib_residuals=np.zeros(19))
    assert gate.decide() is Decision.FREEZE


def test_valid_cached_estimator_identity_survives_successful_decision() -> None:
    gate = KGA()
    estimator = _estimator()
    certificate = gate.certify_evidence(
        estimator,
        protocol_sha256=estimator.protocol_sha256,
        features={"x": 0.4},
        evidence_schema_version=estimator.evidence_schema_version,
    )
    assert gate.decide() is Decision.ADAPT
    assert gate.last_certificate is certificate
    assert gate.explain()["estimator_artifact_sha256"] == estimator.artifact_sha256
    assert gate.explain()["protocol_sha256"] == estimator.protocol_sha256


def test_final_estimator_identity_failure_does_not_restore_partial_certificate() -> None:
    class BrokenIdentity(FrozenLinearBenefitEstimator):
        @property
        def artifact_sha256(self) -> str:
            raise ValueError("artifact identity unavailable")

    estimator = BrokenIdentity(**vars(_estimator()))
    gate = KGA()
    gate.certify(delta_hat=-0.4, calib_residuals=np.zeros(19))
    assert gate.decide() is Decision.FREEZE
    with pytest.raises(ValueError, match="identity unavailable"):
        gate.certify_evidence(
            estimator,
            protocol_sha256=estimator.protocol_sha256,
            features={"x": 0.4},
            evidence_schema_version=estimator.evidence_schema_version,
        )
    _assert_no_cached_authority(gate)


@pytest.mark.parametrize("features", [{}, {"x": math.nan}])
def test_custom_estimator_cannot_bypass_missing_feature_validation(features: dict) -> None:
    class PermissivePredictor(FrozenLinearBenefitEstimator):
        def predict(self, features, **kwargs) -> float:
            return 0.4

    estimator = PermissivePredictor(**vars(_estimator()))
    gate = KGA()
    with pytest.raises(ValueError, match="feature"):
        gate.certify_evidence(
            estimator,
            protocol_sha256=estimator.protocol_sha256,
            features=features,
            evidence_schema_version=estimator.evidence_schema_version,
        )
    _assert_no_cached_authority(gate)


@pytest.mark.parametrize("delta", [math.nan, math.inf, -math.inf])
def test_scalar_policy_preserves_explicit_invalid_estimate_error(delta: float) -> None:
    with pytest.raises(ValueError, match="delta_hat must be finite"):
        decide(Certificate(delta, 0.1, "conformal", 0.1, 19))


def test_batch_missing_values_abstain_without_relabeling_valid_freeze() -> None:
    delta = [0.3, -0.3, 0.0, math.nan, math.inf, -math.inf, 0.3, -0.3, -0.3]
    epsilon = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, math.nan, math.inf, -math.inf]
    assert decide_batch(delta, epsilon).tolist() == ["ADAPT", "FREEZE"] + ["ABSTAIN"] * 7


@pytest.mark.parametrize("size", range(1, 9))
def test_insufficient_conformal_calibration_abstains_for_both_signs(size: int) -> None:
    for delta in (-0.4, 0.4):
        with pytest.warns(UserWarning, match="ABSTAIN"):
            certificate = conformal_split(delta, np.zeros(size), alpha=0.1)
        assert certificate.epsilon == math.inf
        assert decide(certificate) is Decision.ABSTAIN


def test_zero_residual_rank_and_one_cell_loo_are_infeasible_not_finite() -> None:
    with pytest.warns(UserWarning, match="ABSTAIN"):
        assert split_conformal_rank_radius(np.array([]), 0.1) == math.inf
    for delta in (-0.4, 0.4):
        with pytest.warns(UserWarning, match="ABSTAIN"):
            radii, decisions = decide_kga([delta], [delta])
        assert radii.tolist() == [math.inf]
        assert decisions.tolist() == ["ABSTAIN"]


@pytest.mark.parametrize("alpha,n", [(math.nan, 19), (math.inf, 19), (0.1, 0)])
def test_invalid_certificate_level_or_sample_count_cannot_certify_freeze(alpha: float, n: int) -> None:
    with pytest.raises(ValueError):
        decide(Certificate(-0.4, 0.01, "conformal", alpha, n))


@pytest.mark.parametrize("selector", ["argmax_lcb", "first_positive"])
def test_router_never_selects_nonfinite_lcb(selector: str) -> None:
    assert bonferroni_multicandidate_route([math.inf, math.nan, -math.inf], selector=selector) is None
    assert bonferroni_multicandidate_route([math.inf, 0.2, math.nan], selector=selector) == 1
    assert bonferroni_multicandidate_route(np.array([0.0, 0.2]), selector=selector) == 1


@pytest.mark.parametrize("delta,epsilon", [(math.inf, 0.1), (math.nan, 0.1), (-0.3, math.nan), (0.3, math.inf)])
def test_unavailable_candidate_has_no_selectable_lower_bound(delta: float, epsilon: float) -> None:
    certificate = CandidateCertificate(0, delta, epsilon)
    assert not certificate.available
    assert certificate.lcb == -math.inf


def test_panel_abstains_when_every_estimate_is_unavailable() -> None:
    calibration = np.zeros((2, 19))
    result = route_panel(np.array([math.nan, math.inf]), calibration, calibration, alpha=0.1)
    assert result.decision == "abstain"
    assert result.selected is None
    assert not result.committed
    assert not result.feasible


def test_invalid_candidate_does_not_change_other_candidates_bonferroni_level() -> None:
    estimates = np.zeros((2, 19))
    truth = np.zeros((2, 19))
    estimates[0, 0] = math.nan
    result = route_panel(np.array([0.8, 0.2]), estimates, truth, alpha=0.1)
    assert result.selected == 1
    assert result.bonferroni_alpha == 0.05
    assert not result.certificates[0].available
    assert result.certificates[1].available


def test_empty_calibration_panel_abstains_without_an_imputed_estimate() -> None:
    with pytest.warns(UserWarning, match="ABSTAIN"):
        result = route_panel(np.array([0.4]), np.empty((1, 0)), np.empty((1, 0)), alpha=0.1)
    assert result.decision == "abstain"
    assert not result.feasible
    assert result.certificates[0].delta_hat == 0.4
    assert result.certificates[0].epsilon == math.inf


def test_empty_candidate_panel_has_explicit_validation_error() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        route_panel(np.array([]), np.empty((0, 19)), np.empty((0, 19)), alpha=0.1)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 2.0])
def test_invalid_anytime_step_is_atomic(value: float) -> None:
    panel = AnytimeMulticandidatePanel(2)
    reference = AnytimeMulticandidatePanel(2)
    for _ in range(25):
        assert panel.update([0.5, 0.3]) == reference.update([0.5, 0.3])
    before = [vars(process).copy() for process in panel._procs]
    before_steps = panel.steps
    with pytest.raises(ValueError, match="finite"):
        panel.update([0.5, value])
    assert panel.steps == before_steps
    assert [vars(process) for process in panel._procs] == before
    assert panel.update([0.4, 0.2]) == reference.update([0.4, 0.2])
    assert [vars(process) for process in panel._procs] == [vars(process) for process in reference._procs]


def test_cli_infinite_radius_is_abstain_and_strict_json(tmp_path, capsys) -> None:
    residuals = tmp_path / "insufficient-residuals.npy"
    np.save(residuals, np.zeros(1))
    with pytest.warns(UserWarning, match="ABSTAIN"):
        assert cli.main(["decide", "--delta-hat", "-0.4", "--calib-residuals", str(residuals)]) == 0
    text = capsys.readouterr().out
    result = json.loads(text)
    assert result["decision"] == "ABSTAIN"
    assert result["epsilon"] is None and result["lower"] is None and result["upper"] is None
    assert result["availability"] == "unavailable"
    assert result["model_action"] == "retain_frozen"
    assert result["decision_scope"] == "external_estimate_audit"
    assert result["reason"]
    assert "Infinity" not in text and "NaN" not in text
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [np.array([]), np.array([math.nan])])
def test_cli_missing_or_invalid_residuals_cannot_become_certified_freeze(value, tmp_path, capsys) -> None:
    residuals = tmp_path / "unavailable-residuals.npy"
    np.save(residuals, value)
    assert cli.main(["decide", "--delta-hat", "-0.4", "--calib-residuals", str(residuals)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "ABSTAIN"
    assert result["delta_hat"] is None and result["epsilon"] is None
    assert result["availability"] == "unavailable"
    assert result["model_action"] == "retain_frozen"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("delta", [-0.4, 0.4])
@pytest.mark.parametrize("endpoint", ["a", "b"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_evalue_nonfinite_support_cannot_certify_either_direction(delta: float, endpoint: str, value: float) -> None:
    benefits = np.full(200, delta)
    with pytest.raises(ValueError, match="finite"):
        evalue_anytime(benefits, **{endpoint: value})
    np.testing.assert_array_equal(benefits, np.full(200, delta))


@pytest.mark.parametrize("delta", [-0.4, 0.4])
@pytest.mark.parametrize("method", ["conformal", "ebern"])
def test_explain_unavailable_interval_is_strict_json_without_changing_certificate(delta: float, method: str) -> None:
    gate = KGA()
    if method == "conformal":
        with pytest.warns(UserWarning, match="ABSTAIN"):
            certificate = gate.certify(delta_hat=delta, calib_residuals=np.zeros(1))
    else:
        certificate = gate.certify(scores=np.array([delta]), benefit_range=2.0)
    assert gate.decide() is Decision.ABSTAIN
    report = gate.explain()
    text = json.dumps(report, allow_nan=False)
    assert report["certificate"]["delta_hat"] == delta
    assert report["certificate"]["epsilon"] is None
    assert report["certificate"]["lower"] is None and report["certificate"]["upper"] is None
    assert report["decision"] == "ABSTAIN"
    assert "Infinity" not in text and "NaN" not in text
    assert gate.last_certificate is certificate and certificate.epsilon == math.inf
    assert gate.decide() is Decision.ABSTAIN


def test_explain_valid_numpy_certificate_metadata_is_strict_json() -> None:
    gate = KGA()
    certificate = Certificate(np.float64(-0.4), np.float64(0.1), "conformal", np.float64(0.1), np.int64(19))
    assert gate.decide(certificate) is Decision.FREEZE
    report = gate.explain()
    assert json.loads(json.dumps(report, allow_nan=False))["certificate"]["n"] == 19
    assert report["certificate"]["epsilon"] == 0.1
    assert gate.last_certificate is certificate
