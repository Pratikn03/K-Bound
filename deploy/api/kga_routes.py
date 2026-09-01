"""FastAPI routes serving the KGA (Knowability-Guided Adaptation) certificate.

* ``GET  /kga/health``  — liveness probe (no auth).
* ``POST /decide``      — diagnostic or benefit-audit decisions (auth).

``cert_mode``:
  * ``proxy`` (default) — score evidence only; ABSTAIN, no benefit certificate.
  * ``full`` — paired-benefit or explicit-estimate audit, not a label-free
    deployment certificate. Missing evidence retains the frozen predictor.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, field_validator, model_validator

from kga import __version__ as kga_version
from kga._validation import as_float_array

from .auth import authenticate
from .kga_service import assess_kga_decision

MAX_KGA_SCORES = 200_000


class _KGAValidationRoute(APIRoute):
    """Keep malformed requests as JSON-safe 422 errors, never certificates."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def validated(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                # The default handler echoes raw input, which may itself be
                # NaN/Infinity and fail strict JSON encoding. Do not echo it.
                errors = [{key: error[key] for key in ("type", "loc", "msg") if key in error} for error in exc.errors()]
                return JSONResponse(status_code=422, content={"detail": errors})

        return validated


router = APIRouter(route_class=_KGAValidationRoute)


class KGADecideRequest(BaseModel):
    calib_scores: list[float] = Field(..., min_length=2, max_length=MAX_KGA_SCORES)
    test_scores: list[float] = Field(..., min_length=2, max_length=MAX_KGA_SCORES)
    alpha: float = Field(0.1, gt=0.0, lt=1.0)
    cert_mode: Literal["proxy", "full"] = Field(
        "proxy",
        description="proxy=diagnostic scores only; full=paired-benefit or explicit-estimate audit.",
    )
    benefit_scores: list[float] | None = Field(
        None,
        description="Paired per-sample benefits for the cert_mode=full audit certificate.",
    )
    calib_residuals: list[float] | None = Field(
        None,
        description="Held-out |Delta_hat - Delta| residuals for conformal full cert.",
    )
    delta_hat: float | None = Field(
        None,
        description="Explicit benefit estimate required with calib_residuals; no implicit zero estimate.",
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

    @field_validator("delta_hat", "benefit_range")
    @classmethod
    def _validate_finite_scalar(cls, value: float | None) -> float | None:
        if value is not None and not np.isfinite(value):
            raise ValueError("benefit estimates and support widths must be finite")
        return value

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
    delta_hat: float | None
    epsilon: float | None
    radius_feasible: bool
    method: str
    cert_mode: str
    evidence: KGAEvidenceModel | None
    availability: Literal["available", "unavailable"]
    reason: str | None
    model_action: Literal["use_candidate", "retain_frozen"]
    decision_scope: Literal["evidence_only", "paired_benefit_audit", "external_estimate_audit"]


@router.get("/kga/health")
async def kga_health() -> dict:
    return {"status": "ok", "component": "kga", "version": kga_version}


@router.post("/decide", response_model=KGADecideResponse)
async def kga_decide(
    req: KGADecideRequest,
    authenticated: bool = Depends(authenticate),
) -> KGADecideResponse:
    try:
        result = assess_kga_decision(
            as_float_array(req.calib_scores),
            as_float_array(req.test_scores),
            alpha=req.alpha,
            cert_mode=req.cert_mode,
            benefit_scores=req.benefit_scores,
            calib_residuals=req.calib_residuals,
            delta_hat=req.delta_hat,
            method=req.method,
            benefit_range=req.benefit_range,
        )
        certificate, evidence = result.certificate, result.evidence
        radius_feasible = certificate is not None and bool(np.isfinite(certificate.epsilon))
        return KGADecideResponse(
            decision=result.decision.value,
            delta_hat=certificate.delta_hat if certificate is not None else None,
            epsilon=certificate.epsilon if certificate is not None and radius_feasible else None,
            radius_feasible=radius_feasible,
            method=certificate.method if certificate is not None else "unavailable",
            cert_mode=req.cert_mode,
            evidence=(
                KGAEvidenceModel(
                    ks_mean=evidence.ks_mean,
                    ks_max=evidence.ks_max,
                    disagree=evidence.disagree,
                    entropy_shift=evidence.entropy_shift,
                    conf_shift=evidence.conf_shift,
                    ess_frac=evidence.ess_frac,
                    n_calib=evidence.n_calib,
                    n_test=evidence.n_test,
                    n_detectors=evidence.n_detectors,
                )
                if evidence is not None
                else None
            ),
            availability=result.availability,
            reason=result.reason,
            model_action=result.model_action,
            decision_scope=result.decision_scope,
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except Exception as err:  # pragma: no cover
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="KGA decision failed",
        ) from err
