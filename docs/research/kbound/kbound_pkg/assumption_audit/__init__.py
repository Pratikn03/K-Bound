"""K-Bound assumption audit — falsification-oriented pre-deployment checks.

Does NOT verify exchangeability or risk alignment in full generality.
Produces warnings and recommended safe actions only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class AuditReport:
    assumption_status: str  # supported_by_protocol | not_falsified | warning | unresolved | falsified
    support_distance: float
    calibration_warning: float | None
    risk_alignment_warning: float
    recommended_safe_action: str  # this diagnostic-only API returns abstain
    guarantee_wording: str  # does_not_apply | unresolved; never applies
    details: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def feature_range_violation(z_deploy: np.ndarray, z_calib: np.ndarray) -> float:
    """Fraction of deployment features outside calibration [p5,p95] envelope."""
    z_deploy = np.asarray(z_deploy, dtype=float)
    z_calib = np.asarray(z_calib, dtype=float)
    if z_deploy.ndim == 1:
        z_deploy = z_deploy.reshape(1, -1)
    if z_calib.ndim == 1:
        z_calib = z_calib.reshape(1, -1)
    if any(a.ndim != 2 or not a.size or not np.all(np.isfinite(a)) for a in (z_deploy, z_calib)):
        raise ValueError("Calibration and deployment evidence must be nonempty finite matrices")
    if z_deploy.shape[1] != z_calib.shape[1]:
        raise ValueError("Calibration and deployment evidence must share a feature schema")
    lo = np.percentile(z_calib, 5, axis=0)
    hi = np.percentile(z_calib, 95, axis=0)
    out = (z_deploy < lo) | (z_deploy > hi)
    return float(out.mean())


def residual_drift_score(residuals_a: np.ndarray, residuals_b: np.ndarray) -> float:
    """Normalized mean absolute shift between residual distributions."""
    a = np.asarray(residuals_a, dtype=float)
    b = np.asarray(residuals_b, dtype=float)
    if any(x.ndim != 1 or not x.size or not np.all(np.isfinite(x)) for x in (a, b)):
        raise ValueError("Residual samples must be nonempty finite vectors")
    return float(abs(a.mean() - b.mean()) / (a.std() + b.std() + 1e-8))


def run_audit(
    z_deploy: np.ndarray,
    z_calib: np.ndarray,
    residuals_calib: np.ndarray | None = None,
    residuals_deploy: np.ndarray | None = None,
    *,
    range_warn: float = 0.25,
    drift_warn: float = 0.5,
) -> AuditReport:
    """Run diagnostics, without promoting their outcome to a coverage premise.

    Residual drift, when supplied, is a retrospective labelled diagnostic. Passing
    either diagnostic establishes neither exchangeability nor benefit sign. The
    safe recommendation is therefore always abstention (retain the frozen model).
    ``does_not_apply`` withholds this audit's guarantee language after a warning;
    it does not logically prove every possible coverage premise false.
    """
    if not np.isfinite(range_warn) or not 0 < range_warn <= 1:
        raise ValueError("range_warn must lie in (0, 1]")
    if not np.isfinite(drift_warn) or drift_warn <= 0:
        raise ValueError("drift_warn must be finite and positive")
    if (residuals_calib is None) != (residuals_deploy is None):
        raise ValueError("Supply both residual samples or neither")
    support_dist = feature_range_violation(z_deploy, z_calib)
    cal_warn = None
    if residuals_calib is not None and residuals_deploy is not None:
        cal_warn = residual_drift_score(residuals_calib, residuals_deploy)

    risk_warn = support_dist if cal_warn is None else max(support_dist, cal_warn)
    status = "not_falsified"
    guarantee = "unresolved"
    action = "abstain"

    if support_dist >= range_warn or (cal_warn is not None and cal_warn >= drift_warn):
        status = "warning"
        guarantee = "does_not_apply"
        action = "abstain"

    return AuditReport(
        assumption_status=status,
        support_distance=support_dist,
        calibration_warning=cal_warn,
        risk_alignment_warning=risk_warn,
        recommended_safe_action=action,
        guarantee_wording=guarantee,
        details=(
            f"feature_range_violation={support_dist:.3f}; "
            + ("residual_drift=not evaluated" if cal_warn is None else f"residual_drift={cal_warn:.3f}")
            + "; diagnostics cannot establish coverage or authorize ADAPT"
        ),
    )


def run_stress_suite(
    conditions: Sequence[dict],
) -> list[dict]:
    """Evaluate pre-registered stress conditions (for offline audit reports)."""
    out = []
    for cond in conditions:
        z_cal = np.asarray(cond.get("z_calib", []), dtype=float)
        z_dep = np.asarray(cond.get("z_deploy", []), dtype=float)
        rep = run_audit(
            z_dep,
            z_cal,
            cond.get("residuals_calib"),
            cond.get("residuals_deploy"),
        )
        out.append({"condition_id": cond.get("id"), **rep.to_dict()})
    return out
