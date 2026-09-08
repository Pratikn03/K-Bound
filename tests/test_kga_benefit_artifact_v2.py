from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from kga import benefit

SCHEMA_V2 = "kga-frozen-linear-benefit/2"
SCHEMA_V1 = "kga-frozen-linear-benefit/1"
PROTOCOL_SHA256 = "a" * 64


def _canonical_sha256(document: dict[str, object]) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _payload() -> dict[str, object]:
    document: dict[str, object] = {
        "schema": SCHEMA_V2,
        "feature_names": ["x", "y"],
        "weights": [0.5, -0.25],
        "intercept": 0.2,
        "feature_center": [0.0, 1.0],
        "feature_scale": [1.0, 2.0],
        "residuals": [0.01, 0.02, 0.03],
        "evidence_schema_version": "evidence/demo-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "fit_unit": "development-fit",
        "calibration_unit": "heldout-calibration",
    }
    document["payload_sha256"] = _canonical_sha256(document)
    return document


def _estimator() -> benefit.FrozenLinearBenefitEstimator:
    return benefit.FrozenLinearBenefitEstimator(
        feature_names=("x", "y"),
        weights=np.array([0.5, -0.25]),
        intercept=0.2,
        feature_center=np.array([0.0, 1.0]),
        feature_scale=np.array([1.0, 2.0]),
        residuals=np.array([0.01, 0.02, 0.03]),
        evidence_schema_version="evidence/demo-v1",
        protocol_sha256=PROTOCOL_SHA256,
        fit_unit="development-fit",
        calibration_unit="heldout-calibration",
    )


def test_v2_roundtrip_uses_an_honestly_named_required_payload_digest() -> None:
    estimator = _estimator()

    serialized = estimator.to_dict()
    loaded = benefit.FrozenLinearBenefitEstimator.from_dict(serialized)

    assert serialized["schema"] == SCHEMA_V2
    assert serialized["payload_sha256"] == estimator.payload_sha256
    assert "artifact_sha256" not in serialized
    assert loaded.payload_sha256 == estimator.payload_sha256
    # Compatibility property: this is a payload identity, never a file-byte hash.
    assert loaded.artifact_sha256 == loaded.payload_sha256


def test_missing_payload_digest_is_an_integrity_error() -> None:
    document = _payload()
    document.pop("payload_sha256")

    with pytest.raises(benefit.BenefitArtifactIntegrityError, match="payload_sha256.*required"):
        benefit.FrozenLinearBenefitEstimator.from_dict(document)


def test_schema_v1_is_rejected_without_implicit_migration() -> None:
    document = _payload()
    document["schema"] = SCHEMA_V1
    document["payload_sha256"] = _canonical_sha256(
        {key: value for key, value in document.items() if key != "payload_sha256"}
    )

    with pytest.raises(benefit.BenefitArtifactFormatError, match="schema.*1.*migration"):
        benefit.FrozenLinearBenefitEstimator.from_dict(document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("weights"), "missing.*weights"),
        (lambda value: value.__setitem__("comment", "unsigned"), "unexpected.*comment"),
    ],
)
def test_exact_schema_rejects_missing_or_extra_fields(mutation, message: str) -> None:
    document = _payload()
    mutation(document)

    with pytest.raises(benefit.BenefitArtifactFormatError, match=message):
        benefit.FrozenLinearBenefitEstimator.from_dict(document)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("feature_names", ("x", "y")),
        ("feature_names", ["x", 7]),
        ("weights", [[0.5], [-0.25]]),
        ("weights", [True, -0.25]),
        ("intercept", True),
        ("intercept", 10**1000),
        ("feature_center", [0.0, float("nan")]),
        ("feature_scale", [1.0, float("inf")]),
        ("residuals", "0.01,0.02"),
        ("evidence_schema_version", True),
        ("protocol_sha256", 7),
        ("fit_unit", ""),
        ("calibration_unit", False),
        ("payload_sha256", True),
    ],
)
def test_serialized_fields_have_exact_json_types_and_finite_scalar_arrays(
    field: str,
    invalid: object,
) -> None:
    document = _payload()
    document[field] = invalid

    with pytest.raises(benefit.BenefitArtifactError):
        benefit.FrozenLinearBenefitEstimator.from_dict(document)


@pytest.mark.parametrize(
    "malformed_digest",
    [" " + "a" * 63, "+" + "a" * 63, "-" + "a" * 63],
)
def test_rehashed_payload_cannot_smuggle_non_hex_protocol_digest(
    malformed_digest: str,
) -> None:
    document = _payload()
    document["protocol_sha256"] = malformed_digest
    document["payload_sha256"] = _canonical_sha256(
        {key: value for key, value in document.items() if key != "payload_sha256"}
    )

    with pytest.raises(benefit.BenefitArtifactFormatError, match="protocol_sha256.*lowercase"):
        benefit.FrozenLinearBenefitEstimator.from_dict(document)


def test_payload_digest_rejects_leading_whitespace_as_malformed() -> None:
    document = _payload()
    document["payload_sha256"] = " " + "a" * 63

    with pytest.raises(benefit.BenefitArtifactIntegrityError, match="payload_sha256.*lowercase"):
        benefit.FrozenLinearBenefitEstimator.from_dict(document)


def test_external_expected_digest_rejects_leading_whitespace_as_malformed() -> None:
    with pytest.raises(
        benefit.BenefitArtifactIntegrityError,
        match="externally expected payload_sha256.*lowercase",
    ):
        _estimator().require_payload_identity(" " + "a" * 63)


@pytest.mark.parametrize("field", ["weights", "intercept", "residuals", "protocol_sha256"])
def test_payload_tampering_is_detected_before_an_estimator_is_returned(field: str) -> None:
    document = _payload()
    if field == "weights":
        document[field] = [9.0, -0.25]
    elif field == "intercept":
        document[field] = -0.2
    elif field == "residuals":
        document[field] = [9.0, 9.0, 9.0]
    else:
        document[field] = "b" * 64

    with pytest.raises(benefit.BenefitArtifactIntegrityError, match="SHA-256 mismatch"):
        benefit.FrozenLinearBenefitEstimator.from_dict(document)


def test_externally_expected_identity_rejects_a_rehashed_replacement() -> None:
    original = _payload()
    replacement = copy.deepcopy(original)
    replacement["intercept"] = -0.2
    replacement["payload_sha256"] = _canonical_sha256(
        {key: value for key, value in replacement.items() if key != "payload_sha256"}
    )
    estimator = benefit.FrozenLinearBenefitEstimator.from_dict(replacement)

    with pytest.raises(benefit.BenefitArtifactIntegrityError, match="externally expected"):
        estimator.require_payload_identity(str(original["payload_sha256"]))


def test_load_json_for_deployment_requires_and_checks_external_identity(tmp_path: Path) -> None:
    path = tmp_path / "benefit.json"
    path.write_text(json.dumps(_payload()), encoding="ascii")

    loaded = benefit.FrozenLinearBenefitEstimator.load_json_for_deployment(
        path,
        expected_payload_sha256=str(_payload()["payload_sha256"]),
    )
    assert loaded.payload_sha256 == _payload()["payload_sha256"]

    with pytest.raises(benefit.BenefitArtifactIntegrityError, match="externally expected"):
        benefit.FrozenLinearBenefitEstimator.load_json_for_deployment(
            path,
            expected_payload_sha256="b" * 64,
        )


def test_load_json_rejects_duplicate_object_members(tmp_path: Path) -> None:
    document = _payload()
    raw = json.dumps(document, separators=(",", ":"))
    raw = raw.replace('"intercept":0.2', '"intercept":0.2,"intercept":-0.2', 1)
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="ascii")

    with pytest.raises(benefit.BenefitArtifactFormatError, match="duplicate.*intercept"):
        benefit.FrozenLinearBenefitEstimator.load_json(path)


def test_constructor_and_fit_outputs_are_payloads_not_deployment_attestations() -> None:
    estimator = _estimator()

    # A digest can identify this in-memory payload, but authorization must be
    # supplied independently by the deployment protocol.
    assert estimator.payload_sha256
    with pytest.raises(TypeError):
        estimator.require_payload_identity()  # type: ignore[call-arg]
