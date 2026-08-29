"""KGA decision core — shared by HTTP routes and unit tests."""

from __future__ import annotations

from typing import Literal

import numpy as np

from kga import KGA
from kga.certificate import Certificate, conformal_split
from kga.evidence import Evidence, compute_evidence
from kga.policy import Decision, decide

CertMode = Literal["proxy", "full"]


def perform_kga_decide(
    calib_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    alpha: float = 0.1,
    cert_mode: CertMode = "proxy",
    benefit_scores: list[float] | None = None,
    calib_residuals: list[float] | None = None,
    method: str = "ebern",
    benefit_range: float | None = None,
) -> tuple[Decision, Certificate, Evidence]:
    """Run evidence + certificate + trichotomy.

    ``proxy`` (default): conservative label-free certificate for score-only APIs.
    ``full``: paper-style certificate when paired ``benefit_scores`` or
    ``calib_residuals`` (+ optional explicit risks) are supplied.
    """
    evidence = compute_evidence(calib_scores, test_scores)
    kga = KGA(alpha=alpha, method=method)

    if cert_mode == "full":
        if benefit_scores is not None:
            certificate = kga.certify(
                scores=np.asarray(benefit_scores, dtype=float),
                method=method,
                benefit_range=benefit_range,
            )
        elif calib_residuals is not None:
            certificate = kga.certify(
                delta_hat=0.0,
                calib_residuals=np.asarray(calib_residuals, dtype=float),
            )
        else:
            raise ValueError("cert_mode='full' requires benefit_scores or calib_residuals")
    else:
        residual_proxy = np.abs(calib_scores - float(np.median(calib_scores)))
        certificate = conformal_split(0.0, residual_proxy, alpha=alpha)

    decision = decide(certificate, alpha=alpha)
    return decision, certificate, evidence
