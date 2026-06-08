"""FastAPI routes serving the KGA (Knowability-Guided Adaptation) certificate.

Exposes the label-free K-Bound decision over HTTP:

* ``POST /decide``      -- given calibration and test detector scores (and an
                          optional ``alpha``), return the ADAPT/FREEZE/ABSTAIN
                          decision, the certificate ``delta_hat +/- epsilon``,
                          and the label-free evidence ``Z``.  Auth-protected.
* ``GET  /kga/health``  -- liveness probe for the KGA subsystem (no auth).

The decision math lives entirely in the importable :mod:`kga` package; this
module is a thin, validated transport layer over it, following the request /
response and validation style of :mod:`deploy.api.main`.

Because labels are unavailable at request time, the benefit point estimate is
reported conservatively as ``delta_hat = 0`` and the certificate radius is the
split-conformal radius of the calibration scores' own dispersion (a deterministic,
label-free residual proxy).  Callers with a real per-sample benefit estimate
should use the :class:`kga.KGA` API directly server-side.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from kga import __version__ as kga_version
from kga.certificate import conformal_split
from kga.evidence import compute_evidence
from kga.policy import decide

from .auth import authenticate

# Bound request sizes so a single call cannot exhaust memory (mirrors the
# MAX_* limits in deploy/api/main.py).
MAX_KGA_SCORES = 200_000

router = APIRouter()


class KGADecideRequest(BaseModel):
    """Request body for ``POST /decide``.

    Attributes
    ----------
    calib_scores : list[float]
        Detector scores on the calibration / source split.
    test_scores : list[float]
        Detector scores on the unlabelled test / target split.
    alpha : float
        Miscoverage level in ``(0, 1)``; bounds the false-adapt probability.
    """

    calib_scores: list[float] = Field(
        ...,
        min_length=2,
        max_length=MAX_KGA_SCORES,
        description="Calibration detector scores (label-free).",
    )
    test_scores: list[float] = Field(
        ...,
        min_length=2,
        max_length=MAX_KGA_SCORES,
        description="Unlabelled test detector scores.",
    )
    alpha: float = Field(
        0.1,
        gt=0.0,
        lt=1.0,
        description="Miscoverage level in (0, 1). Default 0.1.",
    )

    @field_validator("calib_scores", "test_scores")
    @classmethod
    def _validate_finite(cls, v: list[float]) -> list[float]:
        if any(not np.isfinite(x) for x in v):
            raise ValueError("scores must be finite numbers")
        return v


class KGAEvidenceModel(BaseModel):
    """Label-free evidence ``Z`` summary returned to the caller."""

    ks_mean: float
    ks_max: float
    disagree: float
    entropy_shift: float
    conf_shift: float
    ess_frac: float
    n_calib: int
    n_test: int
    n_detectors: int


class KGADecideResponse(BaseModel):
    """Response body for ``POST /decide``."""

    decision: str = Field(..., description="One of ADAPT, FREEZE, ABSTAIN.")
    delta_hat: float = Field(..., description="Estimated benefit of adapting over freezing.")
    epsilon: float = Field(..., description="Certificate radius at the requested alpha.")
    method: str = Field(..., description="Certificate estimator identifier.")
    evidence: KGAEvidenceModel


@router.get("/kga/health")
async def kga_health() -> dict:
    """Liveness probe for the KGA subsystem."""
    return {"status": "ok", "component": "kga", "version": kga_version}


@router.post("/decide", response_model=KGADecideResponse)
async def kga_decide(
    req: KGADecideRequest,
    authenticated: bool = Depends(authenticate),
) -> KGADecideResponse:
    """Decide ADAPT/FREEZE/ABSTAIN from label-free calibration and test scores."""
    try:
        calib = np.asarray(req.calib_scores, dtype=float)
        test = np.asarray(req.test_scores, dtype=float)
        evidence = compute_evidence(calib, test)

        # Conservative, label-free certificate: point estimate 0 with a
        # split-conformal radius from the calibration dispersion (deterministic).
        residual_proxy = np.abs(calib - float(np.median(calib)))
        certificate = conformal_split(0.0, residual_proxy, alpha=req.alpha)
        decision = decide(certificate, alpha=req.alpha)

        return KGADecideResponse(
            decision=decision.value,
            delta_hat=certificate.delta_hat,
            epsilon=certificate.epsilon,
            method=certificate.method,
            evidence=KGAEvidenceModel(
                ks_mean=evidence.ks_mean,
                ks_max=evidence.ks_max,
                disagree=evidence.disagree,
                entropy_shift=evidence.entropy_shift,
                conf_shift=evidence.conf_shift,
                ess_frac=evidence.ess_frac,
                n_calib=evidence.n_calib,
                n_test=evidence.n_test,
                n_detectors=evidence.n_detectors,
            ),
        )
    except ValueError as err:
        # Input that passes pydantic but fails the evidence/certificate
        # preconditions (e.g. all-identical scores) -> 400.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except Exception as err:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="KGA decision failed",
        ) from err
