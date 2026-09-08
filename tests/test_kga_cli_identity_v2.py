"""CLI regressions for externally authorized estimator identity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kga import EVIDENCE_FEATURE_NAMES, EVIDENCE_SCHEMA_VERSION
from kga.benefit import FrozenLinearBenefitEstimator
from kga.cli import main

PROTOCOL_SHA256 = "a" * 64


def _label_free_fixture(tmp_path: Path) -> tuple[list[str], FrozenLinearBenefitEstimator]:
    rng = np.random.default_rng(20260904)
    calib_path = tmp_path / "calib.npy"
    test_path = tmp_path / "test.npy"
    estimator_path = tmp_path / "benefit-estimator.json"
    np.save(calib_path, rng.normal(size=(40, 2)))
    np.save(test_path, rng.normal(loc=0.1, size=(40, 2)))

    feature_count = len(EVIDENCE_FEATURE_NAMES)
    estimator = FrozenLinearBenefitEstimator(
        feature_names=EVIDENCE_FEATURE_NAMES,
        weights=np.zeros(feature_count),
        intercept=0.2,
        feature_center=np.zeros(feature_count),
        feature_scale=np.ones(feature_count),
        residuals=np.zeros(20),
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
        protocol_sha256=PROTOCOL_SHA256,
        fit_unit="development-domains",
        calibration_unit="calibration-domains",
    )
    estimator.write_new_json(estimator_path)
    args = [
        "decide",
        "--calib",
        str(calib_path),
        "--test",
        str(test_path),
        "--estimator-json",
        str(estimator_path),
        "--protocol-sha256",
        PROTOCOL_SHA256,
    ]
    return args, estimator


def _run_json(args: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    assert main(args) == 0
    return json.loads(capsys.readouterr().out)


def _assert_unavailable_label_free_result(result: dict[str, object]) -> None:
    assert result["decision_scope"] == "label_free_estimator"
    assert result["availability"] == "unavailable"
    assert result["decision"] == "ABSTAIN"
    assert result["model_action"] == "retain_frozen"
    assert result["delta_hat"] is None
    assert result["epsilon"] is None
    assert result["lower"] is None
    assert result["upper"] is None
    assert result["method"] == "unavailable"


def test_missing_external_identity_does_not_trust_embedded_checksum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args, _ = _label_free_fixture(tmp_path)

    result = _run_json(args, capsys)

    _assert_unavailable_label_free_result(result)
    assert "externally authorized estimator payload SHA-256 is required" in str(result["reason"])


@pytest.mark.parametrize("identity", ["", "a" * 63, "A" * 64, "g" * 64])
def test_malformed_external_identity_fails_closed_as_json(
    identity: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, _ = _label_free_fixture(tmp_path)

    result = _run_json(args + ["--estimator-payload-sha256", identity], capsys)

    _assert_unavailable_label_free_result(result)
    assert "externally authorized estimator payload SHA-256 is required" in str(result["reason"])


def test_wrong_external_identity_fails_closed_as_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args, _ = _label_free_fixture(tmp_path)

    result = _run_json(args + ["--estimator-payload-sha256", "b" * 64], capsys)

    _assert_unavailable_label_free_result(result)
    assert "payload SHA-256 mismatch" in str(result["reason"])


def test_correct_external_identity_allows_available_label_free_decision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args, estimator = _label_free_fixture(tmp_path)

    result = _run_json(args + ["--estimator-payload-sha256", estimator.payload_sha256], capsys)

    assert result["decision_scope"] == "label_free_estimator"
    assert result["availability"] == "available"
    assert result["decision"] == "ADAPT"
    assert result["model_action"] == "use_candidate"
    assert result["delta_hat"] == pytest.approx(0.2)
    assert result["epsilon"] == pytest.approx(0.0)


def test_identity_option_cannot_be_silently_ignored_by_another_convention(
    tmp_path: Path,
) -> None:
    benefits_path = tmp_path / "benefits.npy"
    np.save(benefits_path, np.full(40, 0.2))

    with pytest.raises(SystemExit, match="supply exactly one"):
        main(
            [
                "decide",
                "--benefits",
                str(benefits_path),
                "--benefit-range",
                "2.0",
                "--estimator-payload-sha256",
                "b" * 64,
            ]
        )
