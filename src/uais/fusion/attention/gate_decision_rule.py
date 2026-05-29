"""Coherence-certified gate decision rule for Reliability-Gated Attention.

Motivation
----------
ELARA's empirical signature is a *cross-benchmark contrast*: the same
validation-derived KS-drift gate **helps** under coherent score collapse
(Family B, label-aligned stress) but **hurts or is null** under naturally
paired data with legitimate inter-category heterogeneity (Family D Eyecandies,
MVTec~3D-AD). The mechanism that drives both behaviours is identical, so the
open question the paper raises is:

    *When should a validation-derived drift gate be trusted to switch?*

This module answers that question with a rule that is cheap to evaluate, has a
finite-sample guarantee, and predicts the observed contrast:

    switch  iff  (drift is coherent across the batch)
            and  (a bounded switching certificate holds on calibration data)

The first condition separates a *systematic* collapse (every sample's evidence
degrades the same way -> the per-sample reliability signal is tightly clustered)
from *legitimate heterogeneity* (a category mixture -> the reliability signal is
dispersed). The second condition is the existing
``uais.utils.metrics.bounded_switching_certificate`` finite-sample condition.

The rule is deliberately conservative: if either condition fails it keeps the
static path, which is exactly the behaviour that avoids the Family-D / MVTec
regression while preserving the Family-B gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uais.utils.metrics import bounded_switching_certificate

__all__ = [
    "per_sample_mean_reliability",
    "drift_coherence",
    "GateDecision",
    "decide_switch",
]


def per_sample_mean_reliability(
    reliability_weights: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Mean reliability over each sample's *present* domains.

    Parameters
    ----------
    reliability_weights : [N, D] float in [0, 1]
        Per-(sample, domain) reliability weights.
    masks : [N, D] bool
        ``True`` marks a missing domain for that sample.

    Returns
    -------
    [N] float
        Per-sample mean reliability over present domains; ``nan`` for samples
        with no present domain.
    """
    weights = np.asarray(reliability_weights, dtype=float)
    masks = np.asarray(masks, dtype=bool)
    if weights.shape != masks.shape:
        raise ValueError("reliability_weights and masks must have the same shape.")
    if weights.ndim != 2:
        raise ValueError("reliability_weights must be a 2-D [N, D] array.")

    present = ~masks
    present_counts = present.sum(axis=1)
    masked_weights = np.where(present, weights, 0.0)
    sums = masked_weights.sum(axis=1)
    out = np.full(weights.shape[0], np.nan, dtype=float)
    nonempty = present_counts > 0
    out[nonempty] = sums[nonempty] / present_counts[nonempty]
    return out


def drift_coherence(
    reliability_weights: np.ndarray,
    masks: np.ndarray,
) -> float:
    """Coherence of the drift signal across a batch, in [0, 1].

    Coherence is high when the per-sample mean reliability is tightly
    clustered (a systematic, batch-wide shift) and low when it is dispersed
    (a heterogeneous mixture). It is defined as ``1 - min(1, 2 * std(r))``
    where ``r`` is the per-sample mean reliability over present domains.
    The factor of two maps the maximal dispersion of a [0, 1] variable
    (std ~ 0.5) to coherence ~ 0.

    Returns ``nan`` when fewer than two samples carry a present domain.
    """
    r = per_sample_mean_reliability(reliability_weights, masks)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    return float(np.clip(1.0 - 2.0 * float(np.std(r)), 0.0, 1.0))


@dataclass
class GateDecision:
    """Outcome of the coherence-certified switch rule for one batch."""

    switch_allowed: bool
    coherence: float
    coherence_ok: bool
    certified: bool
    fire_rate: float
    decisions: np.ndarray = field(repr=False)
    coherence_min: float = 0.0
    margin_epsilon: float = 0.0
    certificate: dict | None = None

    def as_dict(self) -> dict:
        return {
            "switch_allowed": bool(self.switch_allowed),
            "coherence": float(self.coherence),
            "coherence_ok": bool(self.coherence_ok),
            "certified": bool(self.certified),
            "fire_rate": float(self.fire_rate),
            "coherence_min": float(self.coherence_min),
            "margin_epsilon": float(self.margin_epsilon),
            "n_switched": int(np.asarray(self.decisions, dtype=bool).sum()),
            "certificate": self.certificate,
        }


def decide_switch(
    reliability_weights: np.ndarray,
    masks: np.ndarray,
    tau: float,
    *,
    coherence_min: float = 0.5,
    calibration_static_loss: np.ndarray | None = None,
    calibration_reliability_loss: np.ndarray | None = None,
    calibration_fire_decisions: np.ndarray | None = None,
    margin_epsilon: float = 0.0,
) -> GateDecision:
    """Decide, per sample, whether to switch to the reliability path.

    A sample's candidate fire decision is ``r_i < tau`` (the standard gate).
    The batch-level switch is *allowed* only when both hold:

    1. ``drift_coherence(...) >= coherence_min`` — the shift is systematic
       rather than heterogeneous; and
    2. a bounded switching certificate computed on the provided calibration
       losses is ``certified`` (skipped, treated as satisfied, when no
       calibration data is supplied).

    When the switch is not allowed, no sample is routed through the
    reliability path (``decisions`` is all ``False``), which preserves the
    static prediction and avoids the legitimate-heterogeneity regression.

    Parameters
    ----------
    reliability_weights, masks : see :func:`per_sample_mean_reliability`.
    tau : float
        Per-sample gate threshold; fire when mean reliability ``< tau``.
    coherence_min : float
        Minimum coherence required to allow switching.
    calibration_* : optional 1-D arrays
        Per-sample static loss, reliability-path loss, and fire decisions on a
        labelled calibration set (e.g. validation with injected stress). When
        provided they are passed to
        :func:`uais.utils.metrics.bounded_switching_certificate`.
    margin_epsilon : float
        Margin for the switching certificate.
    """
    if not 0.0 <= float(tau) <= 1.0:
        raise ValueError("tau must lie in [0, 1].")

    r = per_sample_mean_reliability(reliability_weights, masks)
    candidate_fire = np.zeros(r.shape[0], dtype=bool)
    finite = np.isfinite(r)
    candidate_fire[finite] = r[finite] < float(tau)

    coherence = drift_coherence(reliability_weights, masks)
    coherence_ok = bool(np.isfinite(coherence) and coherence >= float(coherence_min))

    certificate: dict | None = None
    have_calibration = (
        calibration_static_loss is not None
        and calibration_reliability_loss is not None
        and calibration_fire_decisions is not None
    )
    if have_calibration:
        certificate = bounded_switching_certificate(
            calibration_static_loss,
            calibration_reliability_loss,
            calibration_fire_decisions,
            margin_epsilon=margin_epsilon,
        )
        certified = bool(certificate["certified"])
    else:
        certified = True  # no calibration evidence available -> do not block on it

    switch_allowed = bool(coherence_ok and certified)
    decisions = candidate_fire if switch_allowed else np.zeros_like(candidate_fire)

    return GateDecision(
        switch_allowed=switch_allowed,
        coherence=float(coherence) if np.isfinite(coherence) else float("nan"),
        coherence_ok=coherence_ok,
        certified=certified,
        fire_rate=float(decisions.mean()) if decisions.size else 0.0,
        decisions=decisions,
        coherence_min=float(coherence_min),
        margin_epsilon=float(margin_epsilon),
        certificate=certificate,
    )
