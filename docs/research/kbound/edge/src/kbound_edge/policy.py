"""kbound_edge.policy -- kga_decide + the policy-comparison family.

``kga_decide`` is the certified gate used on the online path.  It REUSES the
published decision rule :func:`kbound.certificate.decide` (imported via
:mod:`kbound_edge._bridge`) and simply packages the result with the certified
benefit interval ``[lower, upper] = [Bhat - eps, Bhat + eps]`` and a
human-readable reason:

    lower > 0           -> ADAPT    (benefit certified positive)
    upper < 0           -> FREEZE   (harm certified)
    interval spans 0    -> ABSTAIN  (insufficient evidence)

The same module exposes the full family of comparison policies required for the
ablation table (always-freeze, always-adapt, confidence gate, entropy gate,
KGA-no-radius, KGA-full) behind a common :class:`PolicyContext` so a runner can
evaluate any of them on identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from kbound_edge._bridge import decide as _certificate_decide


@dataclass
class Decision:
    """A certified adapt/freeze/abstain decision with its benefit interval."""

    decision: str          # 'adapt' | 'freeze' | 'abstain'
    bhat: float            # predicted benefit
    eps: float             # conformal radius
    lower: float           # Bhat - eps
    upper: float           # Bhat + eps
    reason: str

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "bhat": float(self.bhat),
            "eps": float(self.eps),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "reason": self.reason,
        }


def kga_decide(bhat: float, eps: float) -> Decision:
    """Certified gate: reuse :func:`kbound.certificate.decide` and report the interval.

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
    bhat = float(bhat)
    eps = float(eps)
    decision = _certificate_decide(bhat, eps)  # reused verbatim
    lower = bhat - eps
    upper = bhat + eps
    if decision == "adapt":
        reason = f"lower bound {lower:+.4f} > 0: benefit certified positive (alpha-level)"
    elif decision == "freeze":
        reason = f"upper bound {upper:+.4f} < 0: harm certified -- candidate rejected"
    else:
        reason = (
            f"interval [{lower:+.4f}, {upper:+.4f}] spans 0: "
            "insufficient evidence to certify -- abstain (keep frozen)"
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

    bhat: float
    eps: float
    evidence: Dict[str, float] = field(default_factory=dict)
    conf_tau: float = 0.50       # threshold for the confidence gate
    entropy_tau: float = 0.05    # threshold for the entropy gate


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
    return _certificate_decide(ctx.bhat, 0.0)


def _kga_full(ctx: PolicyContext) -> str:
    # The certified gate: three-way with the conformal radius.
    return _certificate_decide(ctx.bhat, ctx.eps)


#: Registry of comparison policies.  Keys are stable identifiers used in reports.
POLICIES: Dict[str, Callable[[PolicyContext], str]] = {
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
