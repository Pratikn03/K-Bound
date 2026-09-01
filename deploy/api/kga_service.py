"""Availability-aware KGA decision service shared by HTTP routes and callers.

Score evidence alone is diagnostic. This service's ``full`` certificates are
labelled or externally estimated benefit audits, not a schema-bound label-free
deployment estimator. An unavailable certificate always retains the frozen
predictor without calling that fallback a certified FREEZE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np

from kga import KGA
from kga._validation import as_float_array
from kga.certificate import Certificate
from kga.evidence import Evidence
from kga.policy import Decision, decide

CertMode = Literal["proxy", "full"]
DecisionScope = Literal["evidence_only", "paired_benefit_audit", "external_estimate_audit"]
_Predictor = TypeVar("_Predictor")


@dataclass(frozen=True)
class KGAServiceResult:
    """Decision semantics and execution fallback, kept explicitly separate."""

    decision: Decision
    certificate: Certificate | None
    evidence: Evidence | None
    availability: Literal["available", "unavailable"]
    reason: str | None
    decision_scope: DecisionScope

    @property
    def model_action(self) -> Literal["use_candidate", "retain_frozen"]:
        if self.availability == "available" and self.decision is Decision.ADAPT and self.certificate is not None:
            try:
                if decide(self.certificate) is Decision.ADAPT:
                    return "use_candidate"
            except (ValueError, TypeError, OverflowError):
                pass
        return "retain_frozen"

    def select_predictor(self, frozen: _Predictor, candidate: _Predictor) -> _Predictor:
        """Return the selected object without modifying either predictor.

        On ABSTAIN/unavailable, ``frozen`` is returned by identity. Selection
        does not convert an audit into a deployment-valid coverage guarantee.
        """
        return candidate if self.model_action == "use_candidate" else frozen


def assess_kga_decision(
    calib_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    alpha: float = 0.1,
    cert_mode: CertMode = "proxy",
    benefit_scores: list[float] | None = None,
    calib_residuals: list[float] | None = None,
    delta_hat: float | None = None,
    method: str = "ebern",
    benefit_range: float | None = None,
) -> KGAServiceResult:
    """Return an auditable decision or ABSTAIN/unavailable with a reason.

    ``proxy`` preserves the legacy mode name but supplies evidence only; no
    benefit estimate or radius is manufactured from the score distribution.
    ``full`` requires paired benefits, or an explicit benefit point estimate
    together with held-out residuals. It remains an audit-only interface.
    """
    evidence = None
    scope: DecisionScope = "evidence_only"
    if cert_mode == "full":
        scope = "paired_benefit_audit" if benefit_scores is not None else "external_estimate_audit"
    try:
        kga = KGA(alpha=alpha, method=method)
        if cert_mode not in ("proxy", "full"):
            raise ValueError(f"unknown cert_mode: {cert_mode!r}")
        evidence = kga.evidence(calib_scores, test_scores)
        if evidence.n_calib < 2 or evidence.n_test < 2:
            raise ValueError("at least two calibration and two evaluation score rows are required")
        if not np.isfinite(evidence.to_vector()).all():
            evidence = None
            raise ValueError("computed evidence features are nonfinite")
        if cert_mode == "proxy":
            return KGAServiceResult(
                Decision.ABSTAIN,
                None,
                evidence,
                "unavailable",
                "score evidence alone supplies no frozen benefit estimator or calibrated benefit interval",
                scope,
            )
        if benefit_scores is not None:
            if calib_residuals is not None or delta_hat is not None:
                raise ValueError("supply paired benefits or delta_hat plus residuals, not both conventions")
            certificate = kga.certify(
                scores=as_float_array(benefit_scores),
                method=method,
                benefit_range=benefit_range,
            )
        elif calib_residuals is not None and delta_hat is not None:
            certificate = kga.certify(
                delta_hat=delta_hat,
                calib_residuals=as_float_array(calib_residuals),
            )
        else:
            raise ValueError("full mode requires paired benefits or an explicit delta_hat with held-out residuals")
        decision = kga.decide(certificate)
        if not np.isfinite(certificate.epsilon):
            return KGAServiceResult(
                Decision.ABSTAIN,
                certificate,
                evidence,
                "unavailable",
                "insufficient calibration or evaluation information for a finite radius",
                scope,
            )
        reason = "benefit interval does not support a strict decision" if decision is Decision.ABSTAIN else None
        return KGAServiceResult(decision, certificate, evidence, "available", reason, scope)
    except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
        return KGAServiceResult(Decision.ABSTAIN, None, evidence, "unavailable", str(exc), scope)


def perform_kga_decide(
    calib_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    alpha: float = 0.1,
    cert_mode: CertMode = "proxy",
    benefit_scores: list[float] | None = None,
    calib_residuals: list[float] | None = None,
    delta_hat: float | None = None,
    method: str = "ebern",
    benefit_range: float | None = None,
) -> tuple[Decision, Certificate | None, Evidence | None]:
    """Compatibility three-item result; unavailable artifacts are ``None``.

    Use :func:`assess_kga_decision` for the availability reason and explicit
    frozen-predictor fallback. Valid full-audit certificates are unchanged.
    """
    result = assess_kga_decision(
        calib_scores,
        test_scores,
        alpha=alpha,
        cert_mode=cert_mode,
        benefit_scores=benefit_scores,
        calib_residuals=calib_residuals,
        delta_hat=delta_hat,
        method=method,
        benefit_range=benefit_range,
    )
    return result.decision, result.certificate, result.evidence
