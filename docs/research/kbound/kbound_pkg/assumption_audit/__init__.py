"""K-Bound assumption audit — falsification-oriented pre-deployment checks.

Does NOT verify exchangeability or risk alignment in full generality.
Produces warnings and recommended safe actions only.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class AuditReport:
    assumption_status: str  # supported_by_protocol | not_falsified | warning | unresolved | falsified
    support_distance: float
    calibration_warning: float
    risk_alignment_warning: float
    recommended_safe_action: str  # adapt | freeze | abstain
    guarantee_wording: str  # applies | does_not_apply | unresolved
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
    lo = np.percentile(z_calib, 5, axis=0)
    hi = np.percentile(z_calib, 95, axis=0)
    out = (z_deploy < lo) | (z_deploy > hi)
    return float(out.mean())


def residual_drift_score(residuals_a: np.ndarray, residuals_b: np.ndarray) -> float:
    """Normalized mean absolute shift between residual distributions."""
    a = np.asarray(residuals_a, dtype=float)
    b = np.asarray(residuals_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 1.0
    return float(abs(a.mean() - b.mean()) / (a.std() + b.std() + 1e-8))


def run_audit(
    z_deploy: np.ndarray,
    z_calib: np.ndarray,
    residuals_calib: Optional[np.ndarray] = None,
    residuals_deploy: Optional[np.ndarray] = None,
    *,
    range_warn: float = 0.25,
    drift_warn: float = 0.5,
) -> AuditReport:
    """Run pre-registered audit bundle on label-free deployment evidence."""
    support_dist = feature_range_violation(z_deploy, z_calib)
    cal_warn = 0.0
    if residuals_calib is not None and residuals_deploy is not None:
        cal_warn = residual_drift_score(residuals_calib, residuals_deploy)

    risk_warn = max(support_dist, cal_warn)
    status = "not_falsified"
    guarantee = "applies"
    action = "adapt"  # default; overridden by warnings

    if support_dist >= range_warn or cal_warn >= drift_warn:
        status = "warning"
        guarantee = "does_not_apply"
        action = "abstain"
    if support_dist >= 2 * range_warn:
        status = "warning"
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
            f"residual_drift={cal_warn:.3f}"
        ),
    )


def run_stress_suite(
    conditions: Sequence[Dict],
) -> List[Dict]:
    """Evaluate pre-registered stress conditions (for offline audit reports)."""
    out = []
    for cond in conditions:
        z_cal = np.asarray(cond.get("z_calib", []), dtype=float)
        z_dep = np.asarray(cond.get("z_deploy", []), dtype=float)
        rep = run_audit(
            z_dep, z_cal,
            cond.get("residuals_calib"),
            cond.get("residuals_deploy"),
        )
        out.append({"condition_id": cond.get("id"), **rep.to_dict()})
    return out
