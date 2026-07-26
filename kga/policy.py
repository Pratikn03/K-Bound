"""kga.policy -- Trichotomy decision rule with the false-adapt <= alpha guarantee.

The Knowability-Guided Adaptation (KGA) gate turns a finite-sample certificate
``Delta_hat +/- epsilon`` into one of three actions:

* ``ADAPT``   -- the certified lower bound is strictly positive,
                 ``Delta_hat - epsilon > 0`` (adapting is provably beneficial);
* ``FREEZE``  -- the certified upper bound is strictly negative,
                 ``Delta_hat + epsilon < 0`` (adapting is provably harmful);
* ``ABSTAIN`` -- the certificate brackets zero, so the sign of the true benefit
                 ``Delta = R(f0) - R(fa)`` is not knowable at level ``alpha``.

This is exactly the rule implemented in every K-Bound experiment script (e.g.
``src/scripts/kbound/knowability_experiment.py`` lines 120-121,
``mixed_regime_experiment.py`` line 94) and in
``docs/research/kbound/kbound_pkg/kbound/certificate.py::decide``.

False-adapt guarantee (Theorem 3, ``thm:cert`` in
``docs/research/kbound/kbound.tex``).
    The certificate is constructed so that the bound ``Delta >= Delta_hat -
    epsilon`` holds with probability at least ``1 - alpha`` (one-sided).  The
    ADAPT branch fires only when ``Delta_hat - epsilon > 0``, which on the
    good event implies ``Delta > 0``.  Therefore

        P( ADAPT  and  Delta <= 0 )  <=  alpha,

    i.e. the probability of a *harmful* adaptation ("false adapt") is bounded by
    ``alpha``.  The symmetric statement bounds the false-freeze probability.
    The anytime e-value certificate (Theorem 3b, the testing-by-betting variant
    validated in ``experiments/kbound/theory_validation/val_thm3_evalue.py``)
    upgrades this to hold *simultaneously over all sample sizes* via Ville's
    inequality.

Note that the guarantee is inherited entirely from the certificate radius
``epsilon`` -- the decision function itself is a deterministic threshold and
introduces no additional error.

Which function should I call?
-----------------------------
* :func:`decide`      -- one :class:`~kga.certificate.Certificate` -> one
  :class:`Decision`.  The primitive.
* :func:`decide_batch` -- vectorised :func:`decide` over arrays of
  ``delta_hat`` / ``epsilon`` (``epsilon`` may be scalar or per-cell).
* :func:`decide_kga`  -- **the canonical end-to-end rule**: stored per-cell
  benefit estimates and realised benefits in, decisions out, with the
  leave-one-out-of-pool exact-rank conformal radius.  Every driver in this
  repository must route through this function (fix-queue item 15); the seven
  copy-pasted ``decide_kga`` forks that produced the published numbers used an
  interpolated ``np.quantile`` over a pool that included the scored cell, and
  they are gone.

Tie-breaking
------------
Every comparison is **strict**, and this is deliberate: it is the same
convention as the knowability frontier, where a cell is committed only when
``|M| > beta`` and ``|M| == beta`` is unknowable
(``experiments/kbound/theory_validation/val_frontier.py:67,72``;
``val_benefit_frontier.py:50`` takes the ``abs(M) <= beta`` branch as
"unknowable").  So a certificate whose lower bound is exactly zero ABSTAINs, and
a certificate with ``epsilon = +inf`` (the small-``n`` infeasible conformal case)
always ABSTAINs.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from kga.certificate import Certificate


class Decision(str, Enum):
    """Trichotomy outcome of the KGA gate.

    Subclasses :class:`str` so that members compare equal to their string value
    (``Decision.ADAPT == "ADAPT"``) and serialise cleanly to JSON.
    """

    ADAPT = "ADAPT"
    FREEZE = "FREEZE"
    ABSTAIN = "ABSTAIN"


def decide(certificate: Certificate, alpha: float | None = None) -> Decision:
    """Apply the K-Bound trichotomy to a finite-sample certificate.

    Parameters
    ----------
    certificate : Certificate
        A certificate carrying ``delta_hat`` (estimated benefit of adapting over
        freezing) and ``epsilon`` (the one-sided/symmetric confidence radius at
        level ``certificate.alpha``).
    alpha : float, optional
        If given, asserts that the certificate was computed at this level.  The
        decision rule itself does not re-use ``alpha`` (the guarantee is baked
        into ``epsilon``); this argument exists only to make the operating level
        explicit at the call site and to catch accidental mismatches.

    Returns
    -------
    Decision
        ``Decision.ADAPT``  if ``delta_hat - epsilon > 0``,
        ``Decision.FREEZE`` if ``delta_hat + epsilon < 0``,
        ``Decision.ABSTAIN`` otherwise.

    Raises
    ------
    ValueError
        If ``epsilon`` is negative, if ``delta_hat``/``epsilon`` are not finite,
        or if ``alpha`` is supplied and disagrees with ``certificate.alpha``.

    Notes
    -----
    Boundaries are decided with *strict* inequalities, so a certificate whose
    lower bound is exactly zero (``delta_hat == epsilon``) ABSTAINs rather than
    ADAPTs.  This is the conservative choice and matches the reference scripts.

    Examples
    --------
    >>> from kga.certificate import Certificate
    >>> decide(Certificate(delta_hat=0.15, epsilon=0.05, method="ebern",
    ...                     alpha=0.1, n=100))
    <Decision.ADAPT: 'ADAPT'>
    >>> decide(Certificate(delta_hat=-0.15, epsilon=0.05, method="ebern",
    ...                     alpha=0.1, n=100))
    <Decision.FREEZE: 'FREEZE'>
    >>> decide(Certificate(delta_hat=0.03, epsilon=0.10, method="ebern",
    ...                     alpha=0.1, n=100))
    <Decision.ABSTAIN: 'ABSTAIN'>
    """
    delta_hat = float(certificate.delta_hat)
    epsilon = float(certificate.epsilon)
    if not math.isfinite(delta_hat):
        raise ValueError(f"delta_hat must be finite, got {delta_hat}")
    if math.isnan(epsilon):
        raise ValueError("epsilon must not be NaN")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if alpha is not None and abs(alpha - float(certificate.alpha)) > 1e-12:
        raise ValueError(
            f"alpha={alpha} does not match certificate.alpha={certificate.alpha}; "
            "compute the certificate at the level you intend to decide at"
        )

    # An infinite radius is the maximally-uncertain certificate (e.g. a single
    # sample with no variance estimate): it brackets all of R, so ABSTAIN.
    if math.isinf(epsilon):
        return Decision.ABSTAIN
    if delta_hat - epsilon > 0.0:
        return Decision.ADAPT
    if delta_hat + epsilon < 0.0:
        return Decision.FREEZE
    return Decision.ABSTAIN


# ---------------------------------------------------------------------------
# The canonical K-Bound decision path (fix-queue item 15)
# ---------------------------------------------------------------------------
#: Calibration conventions accepted by :func:`decide_kga`.
CALIBRATIONS = ("loo", "in_pool")


def decide_batch(delta_hat, epsilon, *, alpha: float = 0.1) -> np.ndarray:
    """Vectorised :func:`decide`: arrays in, array of decision strings out.

    Parameters
    ----------
    delta_hat : array-like
        Per-cell benefit point estimates ``Delta_hat_i``.
    epsilon : float or array-like
        Confidence radius.  A scalar is broadcast; a per-cell array (what
        :func:`kga.certificate.conformal_radii_loo` returns) is used elementwise.
        Non-finite entries yield ABSTAIN.
    alpha : float, default=0.1
        The level ``epsilon`` was computed at.  Recorded on the intermediate
        certificates so a mismatched level is caught, exactly as in
        :func:`decide`.

    Returns
    -------
    numpy.ndarray of str
        ``'ADAPT'`` / ``'FREEZE'`` / ``'ABSTAIN'``, same shape as ``delta_hat``.
        Plain ``str`` rather than :class:`Decision` because that is what every
        artifact schema in this repository stores.

    Examples
    --------
    >>> decide_batch([0.2, -0.2, 0.0], 0.1).tolist()
    ['ADAPT', 'FREEZE', 'ABSTAIN']
    >>> decide_batch([0.2], float('inf')).tolist()
    ['ABSTAIN']
    """
    from kga.certificate import Certificate

    bh = np.asarray(delta_hat, dtype=float)
    eps = np.broadcast_to(np.asarray(epsilon, dtype=float), bh.shape)
    flat = np.empty(bh.size, dtype=object)
    for i in range(bh.size):
        cert = Certificate(
            delta_hat=float(bh.flat[i]),
            epsilon=float(eps.flat[i]),
            method="conformal",
            alpha=float(alpha),
            n=int(bh.size),
        )
        flat[i] = decide(cert, alpha=alpha).value
    return flat.reshape(bh.shape)


def decide_kga(
    delta_hat,
    benefit,
    *,
    alpha: float = 0.1,
    calibration: str = "loo",
):
    """THE canonical K-Bound rule: stored ``b_hat``/``B`` in, decisions out.

    This is the single entry point every driver, re-scoring script and table
    generator in the repository must call (fix-queue item 15).  It implements
    exactly one radius rule.  It has exactly one switch, ``calibration``, and
    that switch is **not** a statistical option: its non-default value replays an
    archived pre-fix artifact and is forbidden for a new number (see below).
    There is no clamp mode and no interpolated mode, at any setting:

    1. **Radius** -- the exact split-conformal *rank* quantile
       ``eps = r_(k)``, ``k = ceil((n + 1)(1 - alpha))``, over the absolute
       residuals ``r_i = |delta_hat_i - benefit_i|``
       (:func:`kga.certificate.split_conformal_rank_radius`).  Interpolated
       ``np.quantile`` is never used.  When ``k > n`` the radius is ``+inf``
       and every cell ABSTAINs (fix-queue item 25).
    2. **Pool** -- cell ``i``'s radius is calibrated on the *other* ``n - 1``
       residuals only (:func:`kga.certificate.conformal_radii_loo`), so
       ``eps`` is not a function of the label of the cell it is used to protect
       (fix-queue item 4).
    3. **Trichotomy** -- ADAPT iff ``delta_hat_i - eps_i > 0``; FREEZE iff
       ``delta_hat_i + eps_i < 0``; otherwise ABSTAIN.  Strict inequalities, so
       a lower bound of exactly zero ABSTAINs, matching the ``|M| > beta``
       commitment rule of the knowability frontier.

    Implements ``thm:certificate`` (Theorem 3): on the ``1 - alpha`` coverage
    event the ADAPT branch fires only when ``Delta > 0``, hence
    ``Pr[ADAPT and Delta <= 0] <= alpha``.

    Parameters
    ----------
    delta_hat : array-like of shape (n,)
        Per-cell benefit point estimates, e.g. the leave-one-cell-out
        gradient-boosted ``b_hat`` stored in the per-condition dumps.
    benefit : array-like of shape (n,)
        The realised per-cell benefits ``B_i = acc(f_a) - acc(f_0)`` used to form
        calibration residuals.  Used for the radius only, never for the decision.
    alpha : float, default=0.1
        Miscoverage level.
    calibration : {'loo', 'in_pool'}, default='loo'
        ``'loo'`` is the rule.  ``'in_pool'`` reproduces the archived, leaky
        pre-fix radius (one radius over all ``n`` residuals, including the scored
        cell's own) and exists only so a historical artifact can be replayed;
        it must not be used for a new number.  It changes the *pool*, never the
        rank rule.

    There is deliberately no ``on_infeasible`` argument.  When the calibration
    pool is too small for ``alpha`` (``k > n``, i.e. ``n <= 8`` at
    ``alpha = 0.10``; under ``'loo'`` that means ``n <= 9`` cells) the radius is
    ``+inf``, a :class:`UserWarning` is emitted, and every cell ABSTAINs.  The
    superseded clamped radius is not reachable from here at any setting.

    Returns
    -------
    (epsilon, decisions) : (numpy.ndarray, numpy.ndarray)
        ``epsilon`` has one radius **per cell** (it is not a scalar under
        ``'loo'``); ``decisions`` are ``'ADAPT'``/``'FREEZE'``/``'ABSTAIN'``
        strings.  A caller that needs one number for a table must report
        ``float(np.mean(epsilon))`` and label it a mean.

    Raises
    ------
    ValueError
        If the two arrays differ in length, or ``calibration`` is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> b = np.linspace(-0.2, 0.2, 40)
    >>> eps, dec = decide_kga(b, b)          # perfect estimator -> zero residuals
    >>> set(dec.tolist()) <= {'ADAPT', 'FREEZE', 'ABSTAIN'}
    True
    """
    from kga.certificate import conformal_radii_loo, split_conformal_rank_radius

    if calibration not in CALIBRATIONS:
        raise ValueError(f"calibration must be one of {sorted(CALIBRATIONS)}, got {calibration!r}")
    bh = np.asarray(delta_hat, dtype=float).ravel()
    bt = np.asarray(benefit, dtype=float).ravel()
    if bh.shape != bt.shape:
        raise ValueError(f"delta_hat and benefit must have the same length, got {bh.size} and {bt.size}")
    residuals = np.abs(bh - bt)
    if calibration == "loo":
        eps = conformal_radii_loo(residuals, alpha)
    else:
        eps = np.full(bh.size, split_conformal_rank_radius(residuals, alpha), dtype=float)
    return eps, decide_batch(bh, eps, alpha=alpha)
