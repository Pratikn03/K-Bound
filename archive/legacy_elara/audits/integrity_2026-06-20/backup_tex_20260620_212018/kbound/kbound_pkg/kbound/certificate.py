"""kbound.certificate -- Finite-sample conformal + empirical-Bernstein certificates.

FROZEN SNAPSHOT. The canonical, maintained implementation of these certificates
lives in the top-level ``kga`` package (``kga/certificate.py``). This file is a
frozen copy bundled with the K-Bound paper reproduction package and is kept
byte-stable so published numbers reproduce. Do NOT edit the math here to "fix" or
extend it -- edit ``kga/certificate.py`` and, if the reproduction package is
re-released, re-vendor from there. ``empirical_bernstein_lcb`` below is
numerically identical to ``kga.certificate.empirical_bernstein``
(verified; see ``docs/research/kbound/ELARA_KGA_MERGE_PLAN.md``).

Theorem thm:cert (K-Bound paper):
    Given calibration residuals r_i = |Bhat_i - B_i|, the split-conformal
    radius eps = quantile(r, 1 - alpha) guarantees that a new prediction Bhat
    deviates from the true B by at most eps with probability >= 1 - alpha over
    the random calibration split.

Decision rule (Proposition):
    ADAPT   if  Bhat - eps > 0   (certified beneficial)
    FREEZE  if  Bhat + eps < 0   (certified harmful)
    ABSTAIN otherwise            (uncertainty too large)
"""

from __future__ import annotations

import math
import numpy as np


def conformal_radius(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal radius from calibration residuals.

    Implements Thm thm:cert: eps = quantile(|Bhat - B|, 1 - alpha).
    The resulting interval [Bhat - eps, Bhat + eps] covers the true B with
    probability >= 1 - alpha over the random calibration split.

    Parameters
    ----------
    residuals : array-like of shape (n,)
        Absolute prediction errors |Bhat_i - B_i| on the calibration set.
    alpha : float, default=0.1
        Miscoverage level in (0, 1).  Typical value 0.10.

    Returns
    -------
    eps : float
        The (1 - alpha)-quantile of the residuals.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.abs(np.random.default_rng(0).standard_normal(200))
    >>> eps = conformal_radius(r, alpha=0.10)
    >>> assert eps > 0
    """
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 1 or len(r) == 0:
        raise ValueError("residuals must be a non-empty 1-D array")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(np.quantile(r, 1.0 - alpha))


def empirical_bernstein_lcb(x: np.ndarray, alpha: float = 0.1) -> float:
    """Maurer-Pontil empirical-Bernstein lower confidence bound on E[X].

    Implements the one-sided LCB from Maurer & Pontil (2009):

        lcb = mean(x) - sqrt(2 * Var_hat(x) * log(2/alpha) / n)
                      - 7 * log(2/alpha) / (3 * (n - 1))

    This is used in the batch certificate (Thm thm:cert) to certify sign(Delta)
    from a fixed sample of paired benefits X_i = loss(f0) - loss(fa).

    Parameters
    ----------
    x : array-like of shape (n,)
        Sample values in a bounded interval.
    alpha : float, default=0.1
        Confidence level: P(E[X] >= lcb) >= 1 - alpha.

    Returns
    -------
    lcb : float
        Lower confidence bound.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.default_rng(1).uniform(0.1, 0.5, size=100)
    >>> lcb = empirical_bernstein_lcb(x, alpha=0.05)
    >>> assert lcb < np.mean(x)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x must be a 1-D array with at least 2 elements")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(x)
    mean = float(x.mean())
    var_hat = float(x.var(ddof=1))
    # Maurer-Pontil (2009) bound
    first_term = math.sqrt(2.0 * var_hat * math.log(2.0 / alpha) / n)
    second_term = 7.0 * math.log(2.0 / alpha) / (3.0 * (n - 1))
    return mean - first_term - second_term


def decide(Bhat: float, eps: float) -> str:
    """Certificate decision from predicted benefit and conformal radius.

    Implements the three-way split decision (Proposition, K-Bound paper):

        ADAPT   if  Bhat - eps > 0
        FREEZE  if  Bhat + eps < 0
        ABSTAIN otherwise

    Parameters
    ----------
    Bhat : float
        Predicted benefit (e.g. from leave-one-out GBR).
    eps : float
        Conformal radius from :func:`conformal_radius`.

    Returns
    -------
    decision : str
        One of ``'adapt'``, ``'freeze'``, ``'abstain'``.

    Examples
    --------
    >>> decide(0.15, 0.05)
    'adapt'
    >>> decide(-0.15, 0.05)
    'freeze'
    >>> decide(0.03, 0.10)
    'abstain'
    """
    if Bhat - eps > 0:
        return "adapt"
    if Bhat + eps < 0:
        return "freeze"
    return "abstain"
