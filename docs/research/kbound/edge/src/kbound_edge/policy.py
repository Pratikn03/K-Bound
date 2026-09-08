"""kbound_edge.policy -- kga_decide + the policy-comparison family.

``kga_decide`` is a conditional interval gate used on the online path. It REUSES the
published decision rule :func:`kbound.certificate.decide` (imported via
:mod:`kbound_edge._bridge`) and simply packages the result with the
supplied benefit interval ``[lower, upper] = [Bhat - eps, Bhat + eps]`` and a
human-readable reason:

    lower > 0           -> ADAPT    (positive direction under interval coverage)
    upper < 0           -> FREEZE   (negative direction under interval coverage)
    interval spans 0    -> ABSTAIN  (insufficient evidence)

This module validates numeric inputs, not their calibration, sampling, or
artifact provenance. A caller-supplied radius is not itself a coverage proof.

The same module exposes the full family of comparison policies required for the
ablation table (always-freeze, always-adapt, confidence gate, entropy gate,
KGA-no-radius, KGA-full) behind a common :class:`PolicyContext` so a runner can
evaluate any of them on identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kbound_edge._bridge import decide as _certificate_decide


@dataclass
class Decision:
    """An interval-conditional decision; coverage remains a caller premise."""

    decision: str  # 'adapt' | 'freeze' | 'abstain'
    bhat: float | None  # predicted benefit
    eps: float | None  # conformal radius
    lower: float | None  # Bhat - eps
    upper: float | None  # Bhat + eps
    reason: str
    availability: str = "available"
    certificate_status: str = "interval_conditional"
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "bhat": self.bhat,
            "eps": self.eps,
            "lower": self.lower,
            "upper": self.upper,
            "reason": self.reason,
            "availability": self.availability,
            "certificate_status": self.certificate_status,
            "model_action": "retain_frozen",
            **{key: self.provenance.get(key) for key in (
                "authority_sha256", "estimator_sha256", "metadata_sha256", "protocol_sha256",
                "frozen_model_sha256", "candidate_sha256", "fit_sha256", "calibration_sha256",
                "runtime_versions", "evidence_schema_version",
            )},
        }


def unavailable_decision(reason: str) -> Decision:
    return Decision("abstain", None, None, None, None, reason,
                    availability="unavailable", certificate_status="not_issued")


def kga_decide(bhat: float, eps: float) -> Decision:
    """Validate supplied interval arithmetic, then report its conditional action.

    Malformed inputs raise ValueError before a decision record is constructed;
    they are never converted into invented finite certificates. Numeric validity
    does not establish coverage or an authorized estimator identity.

    Parameters
    ----------
    bhat : float
        Predicted benefit from the (calibration-fit) benefit estimator.
    eps : float
        Split-conformal radius from the (calibration-conformal) split.

    Returns
    -------
    Decision
    """
    # Validate before float coercion can erase boolean or masked inputs.
    decision = _certificate_decide(bhat, eps)
    bhat = float(bhat)
    eps = float(eps)
    lower = bhat - eps
    upper = bhat + eps
    if decision == "adapt":
        reason = f"supplied lower bound {lower:+.4f} > 0: ADAPT conditional on valid interval coverage"
    elif decision == "freeze":
        reason = f"supplied upper bound {upper:+.4f} < 0: FREEZE conditional on valid interval coverage"
    else:
        reason = (
            f"interval [{lower:+.4f}, {upper:+.4f}] spans 0: no strict direction supported -- abstain (keep frozen)"
        )
    return Decision(decision=decision, bhat=bhat, eps=eps, lower=lower, upper=upper, reason=reason)


# ---------------------------------------------------------------------------
# Policy-comparison family (for the ablation table)
# ---------------------------------------------------------------------------


@dataclass
class PolicyContext:
    """Everything a comparison policy may read for one window.

    ``bhat`` / ``eps`` come from the benefit estimator + conformal radius;
    ``evidence`` is the name->value label-free evidence dict (so the simple gates
    can read e.g. ``post_conf`` and ``entropy_drop``).
    """

    bhat: float | None
    eps: float | None
    evidence: dict[str, float] = field(default_factory=dict)
    conf_tau: float = 0.50  # threshold for the confidence gate
    entropy_tau: float = 0.05  # threshold for the entropy gate


def _always_freeze(ctx: PolicyContext) -> str:
    return "freeze"


def _always_adapt(ctx: PolicyContext) -> str:
    return "adapt"


def _confidence_gate(ctx: PolicyContext) -> str:
    # Adapt when the candidate is confident enough, else stay frozen.
    return "adapt" if ctx.evidence.get("post_conf", 0.0) >= ctx.conf_tau else "freeze"


def _entropy_gate(ctx: PolicyContext) -> str:
    # Adapt when adaptation reduced entropy by at least entropy_tau, else freeze.
    return "adapt" if ctx.evidence.get("entropy_drop", 0.0) >= ctx.entropy_tau else "freeze"


def _kga_no_radius(ctx: PolicyContext) -> str:
    # KGA with eps forced to 0: pure sign of the predicted benefit (no abstain
    # region except the measure-zero boundary).  This is the "trust the
    # estimator, ignore uncertainty" ablation.
    return "abstain" if ctx.bhat is None or ctx.eps is None else _certificate_decide(ctx.bhat, 0.0)


def _kga_full(ctx: PolicyContext) -> str:
    # Conditional rule; this does not verify the radius's provenance.
    return "abstain" if ctx.bhat is None or ctx.eps is None else _certificate_decide(ctx.bhat, ctx.eps)


#: Registry of comparison policies.  Keys are stable identifiers used in reports.
POLICIES: dict[str, Callable[[PolicyContext], str]] = {
    "always_freeze": _always_freeze,
    "always_adapt": _always_adapt,
    "confidence_gate": _confidence_gate,
    "entropy_gate": _entropy_gate,
    "kga_no_radius": _kga_no_radius,
    "kga_full": _kga_full,
}


def apply_policy(name: str, ctx: PolicyContext) -> str:
    """Evaluate a named comparison policy on a window context."""
    if name not in POLICIES:
        raise KeyError(f"unknown policy '{name}'; choices: {sorted(POLICIES)}")
    return POLICIES[name](ctx)
