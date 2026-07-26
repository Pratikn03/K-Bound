"""kga.kga -- the KGA facade: Knowability-Guided Adaptation in one object.

``KGA`` ties together the three K-Bound stages into a small, documented API:

    1. ``.evidence(calib, test)``      -> label-free evidence ``Z``  (kga.evidence)
    2. ``.certify(...)``               -> certificate ``Delta_hat +/- epsilon``
                                          (kga.certificate)
    3. ``.decide(...)``                -> ADAPT / FREEZE / ABSTAIN  (kga.policy)

with ``.explain()`` returning every intermediate quantity for auditing.

The object holds only an operating level ``alpha`` and a default certificate
``method``; it carries no learned state, so a single instance is reusable and
thread-safe.  Everything is pure ``numpy``/``scipy`` -- no torch.

Example
-------
>>> import numpy as np
>>> from kga import KGA
>>> rng = np.random.default_rng(0)
>>> kga = KGA(alpha=0.1, method="ebern")
>>> # Strongly positive paired benefits -> ADAPT is certifiable.
>>> benefits = rng.normal(0.3, 0.1, size=400)
>>> cert = kga.certify(scores=benefits, benefit_range=2.0)
>>> kga.decide(cert).value in {"ADAPT", "FREEZE", "ABSTAIN"}
True

``benefit_range`` is required for the ``ebern``/``hoeffding`` conventions: it is
the *a priori* support width ``b - a``, and estimating it from the sample voids
the Maurer-Pontil / Hoeffding guarantee (panel findings F1-12, F2-12).  For
paired 0/1 losses it is ``2.0``.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from kga.certificate import (
    Certificate,
    conformal_split,
    empirical_bernstein,
    evalue_anytime,
    hoeffding,
)
from kga.evidence import Evidence, compute_evidence
from kga.policy import Decision, decide

#: Mapping from the public ``method`` name to the batch certificate estimator.
_BATCH_ESTIMATORS = {
    "ebern": empirical_bernstein,
    "hoeffding": hoeffding,
    "evalue": evalue_anytime,
}


class KGA:
    """Knowability-Guided Adaptation gate.

    Parameters
    ----------
    alpha : float, default=0.1
        Operating miscoverage level.  Bounds the false-adapt (and false-freeze)
        probability via Theorem 3 -- see :mod:`kga.policy`.
    method : str, default='ebern'
        Default certificate estimator when paired benefit *scores* are supplied
        to :meth:`certify`.  One of ``'ebern'`` (empirical-Bernstein, the batch
        Theorem 3 default), ``'hoeffding'``, or ``'evalue'`` (anytime Theorem 3b).
        When two scalar risks ``adapt_risk``/``freeze_risk`` are supplied with
        calibration residuals, the split-conformal estimator is used instead.

    Attributes
    ----------
    alpha : float
    method : str
    last_evidence : Optional[Evidence]
        The most recent :class:`Evidence` produced by :meth:`evidence`.
    last_certificate : Optional[Certificate]
        The most recent :class:`Certificate` produced by :meth:`certify`.
    last_decision : Optional[Decision]
        The most recent :class:`Decision` produced by :meth:`decide`.
    """

    def __init__(self, alpha: float = 0.1, method: str = "ebern") -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if method not in _BATCH_ESTIMATORS:
            raise ValueError(f"method must be one of {sorted(_BATCH_ESTIMATORS)}, got {method!r}")
        self.alpha = float(alpha)
        self.method = method
        self.last_evidence: Evidence | None = None
        self.last_certificate: Certificate | None = None
        self.last_decision: Decision | None = None

    # ------------------------------------------------------------------
    # Stage 1: label-free evidence
    # ------------------------------------------------------------------
    def evidence(self, calib: np.ndarray, test: np.ndarray, **kwargs: object) -> Evidence:
        """Compute the label-free evidence ``Z`` (delegates to :func:`compute_evidence`).

        Parameters
        ----------
        calib, test : array-like
            Calibration and unlabelled test detector scores; see
            :func:`kga.evidence.compute_evidence`.
        **kwargs
            Forwarded to :func:`compute_evidence` (e.g. ``bins``, ``extra``).

        Returns
        -------
        Evidence
        """
        ev = compute_evidence(calib, test, **kwargs)  # type: ignore[arg-type]
        self.last_evidence = ev
        return ev

    # ------------------------------------------------------------------
    # Stage 2: finite-sample certificate
    # ------------------------------------------------------------------
    def certify(
        self,
        adapt_risk: float | None = None,
        freeze_risk: float | None = None,
        *,
        scores: np.ndarray | None = None,
        calib_residuals: np.ndarray | None = None,
        delta_hat: float | None = None,
        method: str | None = None,
        benefit_range: float | None = None,
    ) -> Certificate:
        """Build a certificate ``Delta_hat +/- epsilon`` for adapting vs freezing.

        Three calling conventions are supported (pick exactly one):

        1. **Paired benefits** -- pass ``scores`` = per-sample benefits
           ``X_i = loss(f0_i) - loss(fa_i)`` (positive means adapt helps).  A
           batch estimator (``method`` or ``self.method``) produces the
           certificate.  This is the Theorem 3 / 3b path.
        2. **Two scalar risks + calibration residuals** -- pass ``adapt_risk``
           and ``freeze_risk`` (so ``delta_hat = freeze_risk - adapt_risk``) and
           ``calib_residuals`` = held-out ``|Delta_hat_i - Delta_i|`` errors.
           The split-conformal radius is used (Theorem 3, cross-task estimator).
        3. **Explicit point estimate + calibration residuals** -- pass
           ``delta_hat`` directly together with ``calib_residuals`` for the
           split-conformal radius.

        Parameters
        ----------
        adapt_risk, freeze_risk : float, optional
            Estimated risks of the adapted and frozen predictors.  Convention 2.
            ``delta_hat = R(f0) - R(fa) = freeze_risk - adapt_risk``.
        scores : array-like, optional
            Per-sample paired benefits.  Convention 1.
        calib_residuals : array-like, optional
            Calibration residuals for the split-conformal radius (conventions 2
            and 3).
        delta_hat : float, optional
            Explicit benefit point estimate.  Convention 3.
        method : str, optional
            Override the batch estimator for convention 1.
        benefit_range : float, optional
            *A priori* range ``R = b - a`` of the paired benefits.  **Required**
            for ``ebern``/``hoeffding`` (convention 1) -- there is no safe
            data-estimated default, because substituting the observed
            ``max - min`` makes the deviation term data-dependent and voids the
            finite-sample guarantee (panel findings F1-12 / F2-12).  Pass ``2.0``
            for ``|p - y|`` paired 0/1 losses.  Ignored by ``evalue`` and by the
            conformal conventions.

        Returns
        -------
        Certificate

        Raises
        ------
        ValueError
            If the supplied arguments do not match exactly one convention, or if
            ``benefit_range`` is missing for ``ebern``/``hoeffding``.
        """
        # Convention 1 (paired benefits) takes precedence when `scores` is given.
        if scores is not None:
            if method is None:
                method = self.method
            if method not in _BATCH_ESTIMATORS:
                raise ValueError(f"method must be one of {sorted(_BATCH_ESTIMATORS)}, got {method!r}")
            estimator = _BATCH_ESTIMATORS[method]
            if method in ("ebern", "hoeffding"):
                cert = estimator(  # type: ignore[operator]
                    np.asarray(scores), alpha=self.alpha, benefit_range=benefit_range
                )
            else:  # evalue
                cert = estimator(np.asarray(scores), alpha=self.alpha)  # type: ignore[operator]
            self.last_certificate = cert
            return cast(Certificate, cert)

        # Conformal conventions (2 or 3): need a point estimate + residuals.
        if delta_hat is None:
            if adapt_risk is None or freeze_risk is None:
                raise ValueError(
                    "Provide either `scores` (paired benefits), or `delta_hat` + "
                    "`calib_residuals`, or both `adapt_risk` and `freeze_risk` + "
                    "`calib_residuals`."
                )
            delta_hat = float(freeze_risk) - float(adapt_risk)
        if calib_residuals is None:
            raise ValueError(
                "`calib_residuals` is required for the conformal certificate "
                "(pass held-out |Delta_hat - Delta| residuals)."
            )
        cert = conformal_split(float(delta_hat), np.asarray(calib_residuals), alpha=self.alpha)
        self.last_certificate = cert
        return cert

    def certify_probe(
        self,
        benefits: np.ndarray,
        *,
        k: int | None = None,
        seed: int = 0,
        method: str | None = None,
        benefit_range: float | None = None,
    ) -> Certificate:
        """Target-label-light certificate from a labeled micro-probe.

        Uses per-sample paired benefits (positive means adapt/fuse helps) on a
        held-out probe set.  When ``k`` is smaller than the supplied pool,
        ``k`` benefits are drawn without replacement (deterministic given
        ``seed``) to simulate a fixed-size deployment probe.

        Parameters
        ----------
        benefits : array-like
            Per-sample benefits ``X_i = loss(f0_i) - loss(fa_i)`` or
            placement-benefit differences for AUROC routing.
        k : int, optional
            Probe size.  ``None`` or ``k >= len(benefits)`` uses the full pool.
        seed : int
            RNG seed for subsampling when ``k < len(benefits)``.
        method, benefit_range
            Forwarded to the batch estimator (default ``self.method``).
            ``benefit_range`` is required for ``ebern``/``hoeffding``; see
            :meth:`certify`.

        Returns
        -------
        Certificate
        """
        pool = np.asarray(benefits, dtype=float).ravel()
        if pool.size == 0:
            raise ValueError("benefits must be non-empty")
        if not np.all(np.isfinite(pool)):
            raise ValueError("benefits must be finite")
        if k is not None and 0 < k < pool.size:
            rng = np.random.default_rng(seed)
            idx = rng.choice(pool.size, size=k, replace=False)
            pool = pool[idx]
        if method is None:
            method = self.method
        if method not in _BATCH_ESTIMATORS:
            raise ValueError(f"method must be one of {sorted(_BATCH_ESTIMATORS)}, got {method!r}")
        estimator = _BATCH_ESTIMATORS[method]
        if method in ("ebern", "hoeffding"):
            cert = estimator(pool, alpha=self.alpha, benefit_range=benefit_range)  # type: ignore[operator]
        else:
            cert = estimator(pool, alpha=self.alpha)  # type: ignore[operator]
        self.last_certificate = cert
        return cast(Certificate, cert)

    # ------------------------------------------------------------------
    # Stage 3: trichotomy decision
    # ------------------------------------------------------------------
    def decide(self, certificate: Certificate | None = None) -> Decision:
        """Apply the trichotomy to a certificate (default: the last one built).

        Parameters
        ----------
        certificate : Certificate, optional
            The certificate to decide on.  If ``None``, uses
            :attr:`last_certificate` (and raises if there is none).

        Returns
        -------
        Decision
            ADAPT / FREEZE / ABSTAIN, with the false-adapt ``<= alpha`` guarantee
            of Theorem 3 (see :func:`kga.policy.decide`).
        """
        if certificate is None:
            certificate = self.last_certificate
        if certificate is None:
            raise ValueError("No certificate available; call certify(...) first or pass one.")
        dec = decide(certificate, alpha=self.alpha)
        self.last_decision = dec
        return dec

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def explain(self) -> dict:
        """Return all intermediate quantities of the last gate evaluation.

        Returns
        -------
        dict
            Keys: ``alpha``, ``method``, ``evidence`` (or ``None``),
            ``certificate`` (or ``None``) with ``delta_hat``/``epsilon``/
            ``lower``/``upper``/``method``/``n``, and ``decision`` (or ``None``).
            All values are JSON-serialisable.
        """
        ev = self.last_evidence
        cert = self.last_certificate
        dec = self.last_decision
        evidence_dict = None
        if ev is not None:
            evidence_dict = {
                "ks_mean": ev.ks_mean,
                "ks_max": ev.ks_max,
                "disagree": ev.disagree,
                "entropy_shift": ev.entropy_shift,
                "conf_shift": ev.conf_shift,
                "calib_entropy": ev.calib_entropy,
                "test_entropy": ev.test_entropy,
                "calib_conf": ev.calib_conf,
                "test_conf": ev.test_conf,
                "ess": ev.ess,
                "ess_frac": ev.ess_frac,
                "n_calib": ev.n_calib,
                "n_test": ev.n_test,
                "n_detectors": ev.n_detectors,
            }
        certificate_dict = None
        if cert is not None:
            certificate_dict = {
                "delta_hat": cert.delta_hat,
                "epsilon": cert.epsilon,
                "lower": cert.lower,
                "upper": cert.upper,
                "method": cert.method,
                "alpha": cert.alpha,
                "n": cert.n,
            }
        return {
            "alpha": self.alpha,
            "method": self.method,
            "evidence": evidence_dict,
            "certificate": certificate_dict,
            "decision": dec.value if dec is not None else None,
        }
