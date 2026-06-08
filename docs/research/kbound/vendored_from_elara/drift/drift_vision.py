"""Stub for vision drift detection."""

from __future__ import annotations


def drift_report_vision(ref_embeddings, cur_embeddings) -> dict[str, float]:
    # TODO: implement real drift detection (e.g., FID, embedding shift)
    return {"embedding_shift": 0.0}


__all__ = ["drift_report_vision"]
