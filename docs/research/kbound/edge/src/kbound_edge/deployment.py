"""Deployment-only gate state; external authorization is never inferred from paths."""

from __future__ import annotations

from dataclasses import asdict
from numbers import Real
from typing import Any

import numpy as np

from kbound_edge.benefit_authority import EdgeBenefitArtifactError
from kbound_edge.benefit_estimator import EdgeBenefitEstimator
from kbound_edge.evidence import EDGE_EVIDENCE_NAMES
from kbound_edge.policy import Decision, kga_decide, unavailable_decision


def validated_evidence(evidence: Any) -> np.ndarray:
    """Require every ordered real feature before any coercion or model prediction."""
    if (
        type(evidence) is not np.ndarray
        or evidence.shape != (len(EDGE_EVIDENCE_NAMES),)
        or evidence.dtype.kind not in "iuf"
        or not np.isfinite(evidence).all()
    ):
        raise ValueError("EDGE_EVIDENCE_INVALID")
    return evidence


def diagnostic_number(value: Any) -> float | None:
    """Invalid optional diagnostics are absent, never invented finite values."""
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def safe_evidence_vector(extractor: Any, p0: Any, pa: Any, upd_norm: Any) -> np.ndarray | None:
    """Contain only extraction/feature failures, leaving model loading outside."""
    try:
        if diagnostic_number(upd_norm) is None:
            return None
        return validated_evidence(extractor(p0, pa, upd_norm))
    except (ValueError, TypeError, ArithmeticError, RuntimeError):
        return None


class DeploymentGate:
    """Clear prior authority before each load and expose explicit unavailable ABSTAIN."""

    def __init__(self) -> None:
        self.estimator: EdgeBenefitEstimator | None = None
        self.reason = "EDGE_AUTHORITY_INVALID"

    def invalidate(self, reason: str) -> None:
        self.estimator = None
        self.reason = reason

    def reload(self, path: str, **authority: Any) -> None:
        self.invalidate("EDGE_AUTHORITY_INVALID")
        try:
            self.estimator = EdgeBenefitEstimator.load(path, **authority)
        except EdgeBenefitArtifactError as exc:
            self.invalidate(exc.code)

    @property
    def eps(self) -> float | None:
        if self.estimator is None or self.estimator.decision_metadata is None:
            return None
        return self.estimator.decision_metadata.eps

    def decide(self, evidence: Any) -> Decision:
        try:
            evidence = validated_evidence(evidence)
        except ValueError:
            self.invalidate("EDGE_EVIDENCE_INVALID")
            return unavailable_decision(self.reason)
        est = self.estimator
        if est is None or est.decision_metadata is None or est.authority_receipt is None:
            return unavailable_decision(self.reason)
        try:
            decision = kga_decide(est.predict_one(evidence), est.decision_metadata.eps)
        except (ValueError, RuntimeError, ArithmeticError):
            self.invalidate("EDGE_PREDICTION_INVALID")
            return unavailable_decision(self.reason)
        receipt = est.authority_receipt
        decision.provenance = {
            "authority_sha256": receipt.authority_sha256,
            "estimator_sha256": receipt.estimator_sha256,
            "metadata_sha256": receipt.metadata_sha256,
            "evidence_schema_version": receipt.evidence_schema_version,
            **asdict(receipt.identities),
        }
        return decision


def decide_estimator(estimator: Any, evidence: Any, eps: float | None) -> Decision:
    """Deployment gates own their sealed radius; in-memory training retains its API."""
    if isinstance(estimator, DeploymentGate):
        return estimator.decide(evidence)
    try:
        evidence = validated_evidence(evidence)
    except ValueError:
        return unavailable_decision("EDGE_EVIDENCE_INVALID")
    if estimator is None or eps is None:
        return unavailable_decision("EDGE_AUTHORITY_INVALID")
    return kga_decide(estimator.predict_one(evidence), eps)
