"""Pydantic schemas for the UAIS local inference API."""

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    domain: str = Field(..., description="fraud | cyber | behavior")
    samples: list[dict[str, Any]]


class PredictResponse(BaseModel):
    domain: str
    predictions: list[float]


__all__ = ["PredictRequest", "PredictResponse"]
