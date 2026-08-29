"""FastAPI routes serving the KGA (Knowability-Guided Adaptation) certificate.

* ``GET  /kga/health``  — liveness probe (no auth).
* ``POST /decide``      — ADAPT/FREEZE/ABSTAIN from label-free scores (auth).

``cert_mode``:
  * ``proxy`` (default) — score-only conservative certificate (deployment API).
  * ``full`` — paper-style certificate when ``benefit_scores`` or ``calib_residuals`` supplied.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator

from kga import __version__ as kga_version

from .auth import authenticate
from .kga_service import perform_kga_decide

MAX_KGA_SCORES = 200_000

router = APIRouter()


class KGADecideRequest(BaseModel):
    calib_scores: list[float] = Field(..., min_length=2, max_length=MAX_KGA_SCORES)
    test_scores: list[float] = Field(..., min_length=2, max_length=MAX_KGA_SCORES)
    alpha: float = Field(0.1, gt=0.0, lt=1.0)
    cert_mode: Literal["proxy", "full"] = Field(
        "proxy",
        description="proxy=score-only API cert; full=paper cert with benefit/residual inputs.",
    )
    benefit_scores: list[float] | None = Field(
        None,
        description="Paired per-sample benefits for cert_mode=full (Theorem 3 path).",
    )
    calib_residuals: list[float] | None = Field(
        None,
        description="Held-out |Delta_hat - Delta| residuals for conformal full cert.",
    )
    method: str = Field("ebern", description="Batch estimator for full+benefit_scores.")
    benefit_range: float | None = Field(
        None,
        gt=0.0,
        description="A-priori support width for ebern/hoeffding paired benefits.",
    )

    @field_validator("calib_scores", "test_scores", "benefit_scores", "calib_residuals")
    @classmethod
    def _validate_finite(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if any(not np.isfinite(x) for x in v):
            raise ValueError("scores must be finite numbers")
        return v

    @model_validator(mode="after")
    def _full_mode_inputs(self) -> KGADecideRequest:
        if self.cert_mode == "full" and self.benefit_scores is None and self.calib_residuals is None:
            raise ValueError("cert_mode='full' requires benefit_scores or calib_residuals")
        if (
            self.cert_mode == "full"
            and self.benefit_scores is not None
            and self.method in {"ebern", "hoeffding"}
            and self.benefit_range is None
        ):
            raise ValueError("ebern/hoeffding benefit_scores require an a-priori benefit_range")
        return self


class KGAEvidenceModel(BaseModel):
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
    decision: str
    delta_hat: float
    epsilon: float | None
    radius_feasible: bool
    method: str
    cert_mode: str
    evidence: KGAEvidenceModel


@router.get("/kga/health")
async def kga_health() -> dict:
    return {"status": "ok", "component": "kga", "version": kga_version}


@router.post("/decide", response_model=KGADecideResponse)
async def kga_decide(
    req: KGADecideRequest,
    authenticated: bool = Depends(authenticate),
) -> KGADecideResponse:
    try:
        decision, certificate, evidence = perform_kga_decide(
            np.asarray(req.calib_scores, dtype=float),
            np.asarray(req.test_scores, dtype=float),
            alpha=req.alpha,
            cert_mode=req.cert_mode,
            benefit_scores=req.benefit_scores,
            calib_residuals=req.calib_residuals,
            method=req.method,
            benefit_range=req.benefit_range,
        )
        radius_feasible = bool(np.isfinite(certificate.epsilon))
        return KGADecideResponse(
            decision=decision.value,
            delta_hat=certificate.delta_hat,
            epsilon=certificate.epsilon if radius_feasible else None,
            radius_feasible=radius_feasible,
            method=certificate.method,
            cert_mode=req.cert_mode,
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except Exception as err:  # pragma: no cover
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="KGA decision failed",
        ) from err
