"""kbound_edge.conformal -- Split-conformal residual radius (alpha = 0.10).

This module owns ONLY the finite-sample radius computation and the strict
fit/conformal separation.  The three-way decision rule itself is NOT
re-implemented here -- callers combine the radius with the reused
:func:`kbound.certificate.decide`.

Split-conformal protocol (enforced by the API shape)
----------------------------------------------------
1. The benefit estimator is fit on the calibration-FIT split only.
2. Residuals are taken from the calibration-CONFORMAL split only:
       r_i = | estimator.predict(Z_conf_i) - B_conf_i |.
3. The radius is the conservative order statistic
       eps = r_( k ),   k = ceil( (n+1) * (1 - alpha) ),   (1-indexed)
   which gives finite-sample marginal coverage >= 1 - alpha for an exchangeable
   fresh point (Vovk; Lei et al. 2018).  If k > n the radius is +inf (cannot
   certify with so few conformal points).

alpha is fixed at 0.10 throughout the edge layer (``ALPHA``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# population (1-alpha)-quantile radius from the paper, re-exported for comparison
from kbound_edge._bridge import conformal_radius as population_conformal_radius

#: Fixed miscoverage level for the entire edge layer.
ALPHA: float = 0.10


def conservative_conformal_radius(residuals, alpha: float = ALPHA) -> float:
    """Finite-sample split-conformal radius = conservative residual order statistic.

    eps = r_(k) with k = ceil((n+1)*(1-alpha)), residuals sorted ascending.

    Parameters
    ----------
    residuals : array-like of shape (n,)
        Absolute calibration-conformal residuals |Bhat_i - B_i|.
    alpha : float, default=0.10
        Miscoverage level in (0, 1).

    Returns
    -------
    eps : float
        The conservative radius (``math.inf`` if n is too small for the level).
    """
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("residuals must be a non-empty 1-D array")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    r = np.sort(r)
    n = r.size
    k = math.ceil((n + 1) * (1.0 - alpha))  # 1-indexed rank
    if k > n:
        return math.inf
    return float(r[k - 1])


@dataclass
class ConformalRadius:
    """Result of a split-conformal calibration."""

    eps: float
    n_conformal: int
    alpha: float
    method: str  # "conservative" or "population"
    residuals: np.ndarray

    def as_dict(self) -> dict:
        return {
            "eps": float(self.eps),
            "n_conformal": int(self.n_conformal),
            "alpha": float(self.alpha),
            "method": self.method,
        }


def calibrate_conformal(
    estimator,
    Z_conf: np.ndarray,
    B_conf: np.ndarray,
    alpha: float = ALPHA,
    conservative: bool = True,
) -> ConformalRadius:
    """Compute the conformal radius from the CONFORMAL split only.

    The estimator must already be fit (on the calibration-FIT split).  This
    function never sees the fit split: it calls ``estimator.predict`` on
    ``Z_conf`` and forms residuals against ``B_conf`` only -- structurally
    guaranteeing the split-conformal separation.

    Parameters
    ----------
    estimator : object with a ``.predict(Z) -> np.ndarray`` method
        A benefit estimator already fit on the calibration-fit split.
    Z_conf : np.ndarray of shape (n_conf, d)
        Evidence features of the calibration-conformal split.
    B_conf : np.ndarray of shape (n_conf,)
        True benefits of the calibration-conformal split.
    alpha : float, default=0.10
    conservative : bool, default=True
        If True use the finite-sample order-statistic radius; if False use the
        paper's population (1-alpha)-quantile (:func:`kbound.certificate.conformal_radius`).

    Returns
    -------
    ConformalRadius
    """
    Z_conf = np.asarray(Z_conf, dtype=float)
    B_conf = np.asarray(B_conf, dtype=float)
    if Z_conf.ndim != 2 or B_conf.ndim != 1 or len(Z_conf) != len(B_conf):
        raise ValueError("Z_conf must be (n,d) and B_conf must be (n,) with matching n")
    if len(B_conf) == 0:
        raise ValueError("conformal split must be non-empty")

    B_pred = np.asarray(estimator.predict(Z_conf), dtype=float)
    residuals = np.abs(B_pred - B_conf)

    if conservative:
        eps = conservative_conformal_radius(residuals, alpha=alpha)
        method = "conservative"
    else:
        eps = float(population_conformal_radius(residuals, alpha=alpha))
        method = "population"

    return ConformalRadius(
        eps=eps,
        n_conformal=len(residuals),
        alpha=alpha,
        method=method,
        residuals=residuals,
    )


class RealCertificateResult:
    """Result object carrying fitted estimator, conformal radius, and split provenance."""
    def __init__(self, fit_sessions, conformal_sessions, fit_source_hashes, conformal_source_hashes, estimator, conformal_radius):
        self.fit_sessions = fit_sessions
        self.conformal_sessions = conformal_sessions
        self.fit_source_hashes = fit_source_hashes
        self.conformal_source_hashes = conformal_source_hashes
        self.estimator = estimator
        self.conformal_radius = conformal_radius


def fit_real_certificate(bundle: dict, estimator_kwargs: dict = None, alpha: float = ALPHA, conservative: bool = True) -> RealCertificateResult:
    """Fit benefit estimator on fit split and calibrate radius on conformal split."""
    from kbound_edge.benefit_estimator import EdgeBenefitEstimator

    fit_data = bundle["fit"]
    conf_data = bundle["conformal"]

    if estimator_kwargs is None:
        estimator_kwargs = {"random_state": 0}

    est = EdgeBenefitEstimator(**estimator_kwargs)
    est.fit(fit_data["Z"], fit_data["B"])

    cr = calibrate_conformal(est, conf_data["Z"], conf_data["B"], alpha=alpha, conservative=conservative)

    return RealCertificateResult(
        fit_sessions=fit_data["sessions"],
        conformal_sessions=conf_data["sessions"],
        fit_source_hashes=fit_data["source_hashes"],
        conformal_source_hashes=conf_data["source_hashes"],
        estimator=est,
        conformal_radius=cr
    )
