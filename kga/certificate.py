"""kga.certificate -- finite-sample certificates Delta_hat +/- epsilon for KGA.

A *certificate* answers the question "by how much does adapting beat freezing,
and how sure are we?".  Concretely it carries

    Delta_hat   estimated benefit  Delta = R(f0) - R(fa)  (>0 means adapt helps),
    epsilon     a one-sided/symmetric confidence radius at level ``alpha``,

so that, with probability at least ``1 - alpha``, the true benefit obeys
``Delta >= Delta_hat - epsilon`` (and, for the two-sided estimators, also
``Delta <= Delta_hat + epsilon``).  Feeding the certificate to
:func:`kga.policy.decide` yields the ADAPT/FREEZE/ABSTAIN trichotomy with the
false-adapt ``<= alpha`` guarantee of Theorem 3.

This module provides four estimators, each mirroring a piece of the paper's
canonical code:

* :func:`empirical_bernstein` -- Maurer & Pontil (2009) empirical-Bernstein LCB,
  the batch Theorem 3 certificate, identical to
  ``vendored_from_elara/certification/switching_certificate.py::
  empirical_bernstein_lcb`` and ``kbound_pkg/kbound/certificate.py``.
* :func:`hoeffding` -- the distribution-free Hoeffding LCB (looser baseline).
* :func:`conformal_split` -- the exact split-conformal order-statistic radius
  over ``|Delta_hat - Delta|`` calibration residuals, the
  cross-task estimator used in ``knowability_experiment.py`` /
  ``mixed_regime_experiment.py``.
* :func:`evalue_anytime` -- the anytime-valid testing-by-betting e-process
  (Ville's inequality), Theorem 3b, mirroring
  ``experiments/kbound/theory_validation/val_thm3_evalue.py`` and
  ``kbound_pkg/kbound/eprocess.py``.

All estimators are pure ``numpy``/``math``, deterministic, and torch-free.

Provenance / attribution
------------------------
The empirical-Bernstein switching certificate (:func:`empirical_bernstein`,
Maurer & Pontil 2009) is **shared with the companion ELARA work**: the identical
Maurer-Pontil lower-confidence-bound underlies ELARA's Phase-2
``switching_certificate``, which now *delegates* to this ``kga`` implementation as
the single source of truth.  It is reproduced in the K-Bound tree so the package
is fully self-contained, and the shared lineage is acknowledged openly rather than
hidden.  K-Bound imports nothing from ``src/elara``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Allowed estimator identifiers (the ``method`` field of a :class:`Certificate`).
METHODS = ("ebern", "hoeffding", "conformal", "evalue")


@dataclass(frozen=True)
class Certificate:
    """A finite-sample certificate for the benefit of adapting over freezing.

    Attributes
    ----------
    delta_hat : float
        Point estimate of the benefit ``Delta = R(f0) - R(fa)``.  Positive means
        adapting reduces risk (helps); negative means it hurts.
    epsilon : float
        Confidence radius at level ``alpha``.  The certified lower bound is
        ``delta_hat - epsilon`` and the certified upper bound is
        ``delta_hat + epsilon``.
    method : str
        Estimator identifier, one of :data:`METHODS`.
    alpha : float
        Miscoverage level in ``(0, 1)`` at which ``epsilon`` was computed.
    n : int
        Effective sample size the certificate was built from (number of paired
        benefits, calibration residuals, or observed e-process steps).
    """

    delta_hat: float
    epsilon: float
    method: str
    alpha: float
    n: int

    @property
    def lower(self) -> float:
        """Certified lower bound ``delta_hat - epsilon`` on the true benefit."""
        return self.delta_hat - self.epsilon

    @property
    def upper(self) -> float:
        """Certified upper bound ``delta_hat + epsilon`` on the true benefit."""
        return self.delta_hat + self.epsilon


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _as_1d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


# ---------------------------------------------------------------------------
# (1) Empirical-Bernstein  (batch Theorem 3 -- the default certificate)
# ---------------------------------------------------------------------------
# PROVENANCE (integrity pass 2026-06-20): the empirical-Bernstein certificate below
# is shared with the ELARA companion work (same Maurer-Pontil 2009 LCB). ELARA's
# certification.switching_certificate delegates to this function; it is vendored here
# so the K-Bound package runs with zero dependency on src/elara. Honest attribution.
def empirical_bernstein(
    paired_benefits: np.ndarray,
    *,
    alpha: float = 0.1,
    benefit_range: float | None = None,
) -> Certificate:
    """Maurer-Pontil (2009) empirical-Bernstein certificate (batch Theorem 3).

    For i.i.d. paired benefits ``X_i = loss(f0_i) - loss(fa_i)`` in ``[a, b]``
    with empirical mean ``mu_hat`` and unbiased sample variance ``V_hat``, with
    probability at least ``1 - alpha``::

        E[X] >= mu_hat - sqrt( 2 V_hat ln(2/alpha) / n )
                       - 7 R ln(2/alpha) / ( 3 (n - 1) ),     R = b - a.

    The estimate is ``delta_hat = mu_hat`` and the radius is the sum of the two
    subtracted terms, so ``delta_hat - epsilon`` reproduces the LCB above.  This
    is tighter than Hoeffding whenever ``V_hat << R^2`` (the usual regime, where
    most paired benefits agree in sign), and is deterministic, streamable, and
    finite-sample valid.

    Parameters
    ----------
    paired_benefits : array-like of shape (n,)
        Per-sample benefits ``X_i`` in a bounded interval.
    alpha : float, default=0.1
        Miscoverage level.  ``P(E[X] >= delta_hat - epsilon) >= 1 - alpha``.
    benefit_range : float, optional
        The range ``R = b - a``.  If ``None`` it is estimated as
        ``max - min`` of the observed benefits.  For ``|p - y|`` paired losses
        the exact range is ``2.0`` and should be passed explicitly.

    Returns
    -------
    Certificate
        With ``method='ebern'``.

    References
    ----------
    Maurer & Pontil, "Empirical Bernstein Bounds and Sample-Variance
    Penalization" (COLT 2009).  Mirrors
    ``switching_certificate.py::empirical_bernstein_lcb``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.full(500, 0.2)
    >>> cert = empirical_bernstein(x, alpha=0.05, benefit_range=2.0)
    >>> cert.lower < cert.delta_hat
    True
    """
    _check_alpha(alpha)
    arr = _as_1d(paired_benefits, "paired_benefits")
    n = arr.size
    mean = float(arr.mean())
    if n < 2:
        # No variance estimate; degenerate radius (infinite -> always ABSTAIN).
        return Certificate(delta_hat=mean, epsilon=float("inf"), method="ebern", alpha=alpha, n=n)
    var = float(arr.var(ddof=1))
    if benefit_range is not None:
        rng = float(benefit_range)
    else:
        rng = float(arr.max() - arr.min())
    rng = max(rng, 1e-12)

    ln_term = math.log(2.0 / alpha)
    var_term = math.sqrt(2.0 * var * ln_term / n)
    range_term = 7.0 * rng * ln_term / (3.0 * (n - 1))
    epsilon = var_term + range_term
    return Certificate(delta_hat=mean, epsilon=epsilon, method="ebern", alpha=alpha, n=n)


# ---------------------------------------------------------------------------
# (2) Hoeffding  (distribution-free baseline)
# ---------------------------------------------------------------------------
def hoeffding(
    paired_benefits: np.ndarray,
    *,
    alpha: float = 0.1,
    benefit_range: float | None = None,
) -> Certificate:
    """Hoeffding certificate -- distribution-free LCB on the mean benefit.

    For i.i.d. ``X_i`` in ``[a, b]``, Hoeffding's inequality gives, with
    probability at least ``1 - alpha`` (one-sided)::

        E[X] >= mu_hat - R sqrt( ln(1/alpha) / (2 n) ),     R = b - a.

    This variance-free radius is always at least as large as the
    empirical-Bernstein one in the small-variance regime, so it is provided
    mainly as a conservative baseline / sanity comparator.

    Parameters
    ----------
    paired_benefits : array-like of shape (n,)
    alpha : float, default=0.1
    benefit_range : float, optional
        Range ``R = b - a``.  Estimated from the data if ``None``; pass ``2.0``
        for ``|p - y|`` paired losses.

    Returns
    -------
    Certificate
        With ``method='hoeffding'``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0.0, 0.4, 200)
    >>> cert = hoeffding(x, alpha=0.1, benefit_range=2.0)
    >>> cert.epsilon > 0
    True
    """
    _check_alpha(alpha)
    arr = _as_1d(paired_benefits, "paired_benefits")
    n = arr.size
    mean = float(arr.mean())
    if benefit_range is not None:
        rng = float(benefit_range)
    else:
        rng = float(arr.max() - arr.min())
    rng = max(rng, 1e-12)
    epsilon = rng * math.sqrt(math.log(1.0 / alpha) / (2.0 * n))
    return Certificate(delta_hat=mean, epsilon=epsilon, method="hoeffding", alpha=alpha, n=n)


# ---------------------------------------------------------------------------
# (3) Split-conformal  (cross-task estimator used in the main experiments)
# ---------------------------------------------------------------------------
def split_conformal_rank_radius(calib_residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Return the exact finite-sample split-conformal residual radius.

    For sorted residuals ``r_(1) <= ... <= r_(n)``, this uses
    ``k = min(n, ceil((n + 1) * (1 - alpha)))`` and returns ``r_(k)``.
    Unlike ``numpy.quantile``'s default interpolation, this is an observed
    order statistic and matches the finite-sample rank argument.
    """
    _check_alpha(alpha)
    arr = _as_1d(calib_residuals, "calib_residuals")
    if np.any(arr < 0.0):
        raise ValueError("calib_residuals must be non-negative (they are |Delta_hat - Delta|)")
    n = arr.size
    k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
    return float(np.sort(arr)[k - 1])


def conformal_split(
    delta_hat: float,
    calib_residuals: np.ndarray,
    *,
    alpha: float = 0.1,
) -> Certificate:
    """Split-conformal certificate around a benefit point estimate.

    Given calibration residuals ``r_i = |Delta_hat_i - Delta_i|`` from held-out
    tasks/instances, the radius

        k = min(n, ceil((n + 1) * (1 - alpha)))
        epsilon = r_(k)

    guarantees that a fresh ``Delta_hat`` deviates from the true ``Delta`` by at
    most ``epsilon`` with probability at least ``1 - alpha`` over the exchangeable
    calibration split (split-conformal coverage). The returned radius is an
    observed residual order statistic; it does not use interpolated quantiles.

    Parameters
    ----------
    delta_hat : float
        Point estimate of the benefit for the query instance (e.g. from a
        leave-one-out / cross-fitted benefit regressor).
    calib_residuals : array-like of shape (n,)
        Absolute prediction errors ``|Delta_hat_i - Delta_i|`` on the
        calibration set.  Must be non-negative.
    alpha : float, default=0.1
        Miscoverage level.

    Returns
    -------
    Certificate
        With ``method='conformal'`` and ``n`` = number of calibration residuals.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.abs(np.random.default_rng(0).standard_normal(200))
    >>> cert = conformal_split(0.2, r, alpha=0.1)
    >>> k = min(len(r), int(np.ceil((len(r) + 1) * 0.9)))
    >>> cert.epsilon == float(np.sort(r)[k - 1])
    True
    """
    _check_alpha(alpha)
    if not math.isfinite(float(delta_hat)):
        raise ValueError(f"delta_hat must be finite, got {delta_hat}")
    arr = _as_1d(calib_residuals, "calib_residuals")
    epsilon = split_conformal_rank_radius(arr, alpha)
    return Certificate(delta_hat=float(delta_hat), epsilon=epsilon, method="conformal", alpha=alpha, n=arr.size)


# ---------------------------------------------------------------------------
# (4) Anytime-valid e-value  (Theorem 3b -- Ville / testing-by-betting)
# ---------------------------------------------------------------------------
def evalue_anytime(
    paired_benefits: np.ndarray,
    *,
    alpha: float = 0.1,
    a: float = -1.0,
    b: float = 1.0,
    bet_cap_frac: float = 0.5,
    prior_var: float = 0.25,
    prior_weight: float = 1.0,
) -> Certificate:
    """Anytime-valid e-value certificate (Theorem 3b, testing-by-betting).

    Runs the two one-sided betting e-processes of
    ``val_thm3_evalue.py`` over the supplied benefit stream::

        E_t^+ = prod_{i<=t} (1 + lam_i  X_i)       tests H0 : Delta <= 0,
        E_t^- = prod_{i<=t} (1 + nu_i (-X_i))      tests H0': Delta >= 0,

    with predictable truncated-aGRAPA bets ``lam_i = clip(mu_hat/sigma2_hat, 0,
    bet_cap_frac/|a|)`` (and symmetrically ``nu_i``).  Each process is a
    nonnegative supermartingale under its null, so by **Ville's inequality**
    ``P(exists t : E_t >= 1/alpha) <= alpha`` -- simultaneously over all sample
    sizes, with no multiplicity correction for repeated looks.

    The returned certificate encodes the *current* anytime decision in the same
    ``delta_hat +/- epsilon`` form consumed by :func:`kga.policy.decide`:

    * if ``E_t^+ >= 1/alpha`` (reject ``Delta <= 0``) we emit a positive
      certified lower bound, so ``decide`` returns ADAPT;
    * if ``E_t^- >= 1/alpha`` (reject ``Delta >= 0``) we emit a negative
      certified upper bound, so ``decide`` returns FREEZE;
    * otherwise the radius covers the running mean and ``decide`` ABSTAINs,
      i.e. "keep sampling".

    ``delta_hat`` is the running sample mean of the benefits (a consistent point
    estimate of ``Delta``); only the *sign of the certified bound* carries the
    anytime guarantee.

    Parameters
    ----------
    paired_benefits : array-like of shape (n,)
        Ordered stream of per-sample benefits ``X_i`` in ``[a, b]``.
    alpha : float, default=0.1
        Anytime type-I error budget.
    a, b : float, default=-1.0, 1.0
        Lower/upper bounds of the benefit range.  ``a`` must be ``< 0`` and
        ``b`` must be ``> 0`` for the two one-sided tests to be well posed.
    bet_cap_frac : float, default=0.5
        Bet cap: ``lam in [0, bet_cap_frac/|a|]``; ``< 1`` guarantees the wealth
        factors stay strictly positive.
    prior_var, prior_weight : float
        Prior pseudo-variance and pseudo-count for the running second moment.

    Returns
    -------
    Certificate
        With ``method='evalue'`` and ``n`` = stream length.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.full(2000, 0.3)            # strongly positive benefit
    >>> cert = evalue_anytime(x, alpha=0.1)
    >>> cert.lower > 0                    # ADAPT certified
    True
    """
    _check_alpha(alpha)
    if a >= 0.0:
        raise ValueError(f"a must be < 0 for the one-sided test on H0: Delta<=0; got {a}")
    if b <= 0.0:
        raise ValueError(f"b must be > 0 for the one-sided test on H0': Delta>=0; got {b}")
    arr = _as_1d(paired_benefits, "paired_benefits")
    n = arr.size

    lam_max_plus = bet_cap_frac / (-a)
    lam_max_minus = bet_cap_frac / b
    log_thr = math.log(1.0 / alpha)

    s1 = 0.0  # running sum of X_j (for E^+)
    s1m = 0.0  # running sum of -X_j (for E^-)
    s2 = prior_weight * prior_var  # running second moment + prior (E^+)
    s2m = prior_weight * prior_var  # running second moment + prior (E^-)
    cnt = 0.0
    cnt_var = float(prior_weight)

    log_w_plus = 0.0
    log_w_minus = 0.0

    for x in arr:
        x_clamp = float(max(a, min(b, x)))
        # Predictable bets from PAST statistics only.
        mu_hat = s1 / cnt if cnt > 0 else 0.0
        mu_hat_m = s1m / cnt if cnt > 0 else 0.0
        sig2 = s2 / cnt_var
        sig2m = s2m / cnt_var
        lam_plus = float(np.clip(mu_hat / sig2 if sig2 > 0 else 0.0, 0.0, lam_max_plus))
        lam_minus = float(np.clip(mu_hat_m / sig2m if sig2m > 0 else 0.0, 0.0, lam_max_minus))

        log_w_plus += math.log(max(1.0 + lam_plus * x_clamp, 1e-300))
        log_w_minus += math.log(max(1.0 + lam_minus * (-x_clamp), 1e-300))

        # Update predictable stats AFTER betting (so they remain F_{i-1}-measurable).
        s1 += x_clamp
        s1m += -x_clamp
        s2 += x_clamp**2
        s2m += x_clamp**2
        cnt += 1.0
        cnt_var += 1.0

    mean = float(arr.mean())
    # Encode the *current* anytime decision in delta_hat +/- epsilon form, so the
    # same kga.policy.decide trichotomy reads it. With epsilon == 0 the decision
    # reduces to sign(delta_hat); we set delta_hat to a certified-sign margin.
    if log_w_plus >= log_thr:
        # E_t^+ crossed: reject H0: Delta <= 0  =>  certified positive lower
        # bound  =>  ADAPT. The running mean may not yet be positive, but the
        # e-process certifies Delta > 0 at anytime level alpha; report a strictly
        # positive certified margin (use the mean if it is already positive).
        delta_hat = mean if mean > 0.0 else 1e-12
        return Certificate(delta_hat=delta_hat, epsilon=0.0, method="evalue", alpha=alpha, n=n)
    if log_w_minus >= log_thr:
        # E_t^- crossed: reject H0': Delta >= 0  =>  certified negative upper
        # bound  =>  FREEZE.
        delta_hat = mean if mean < 0.0 else -1e-12
        return Certificate(delta_hat=delta_hat, epsilon=0.0, method="evalue", alpha=alpha, n=n)

    # Neither threshold crossed: ABSTAIN ("keep sampling"). The radius brackets
    # zero around the running mean so that delta_hat - epsilon <= 0 <= delta_hat
    # + epsilon.
    epsilon = abs(mean) + 1e-12
    return Certificate(delta_hat=mean, epsilon=epsilon, method="evalue", alpha=alpha, n=n)
