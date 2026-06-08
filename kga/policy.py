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
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING

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
