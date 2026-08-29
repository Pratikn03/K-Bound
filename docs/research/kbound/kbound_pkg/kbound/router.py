"""kbound.router -- Benefit router: leave-one-out GBR + split-conformal decisions.

Mirrors the ``decide_kga`` function in cifar_tent_mps_v2.py with an
object-oriented interface.

The router trains a GradientBoostingRegressor to predict per-condition benefit B
from label-free evidence features Z using leave-one-out cross-validation.
The conformal radius uses the exact finite-sample residual rank. If the
calibration pool is too small for the requested level, it is +inf and forces
ABSTAIN rather than silently using an under-covering clamped rank.

Decision rule (identical to cifar_tent_mps_v2.py):
    ADAPT   if  Bhat_i - eps > 0
    FREEZE  if  Bhat_i + eps < 0
    ABSTAIN otherwise
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from kbound.certificate import conformal_radius, decide


class BenefitRouter:
    """Leave-one-out gradient-boosted benefit estimator + split-conformal gate.

    Implements the full KGA machinery from cifar_tent_mps_v2.py:
        1. Fit a GBR on (Z, B) with leave-one-out cross-val to get Bhat.
        2. Compute eps = conformal_radius(|Bhat - B|, alpha).
        3. Decide ADAPT / FREEZE / ABSTAIN per condition.

    Parameters
    ----------
    n_estimators : int, default=250
    max_depth : int, default=2
    learning_rate : float, default=0.05
    subsample : float, default=0.8
    random_state : int, default=0

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> Z = rng.standard_normal((40, 11))
    >>> B = rng.uniform(-0.2, 0.4, 40)
    >>> router = BenefitRouter()
    >>> Bhat, eps, decs = router.decide_all(Z, B, alpha=0.10)
    >>> assert len(decs) == 40
    """

    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 2,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self._model: GradientBoostingRegressor | None = None

    def fit(self, Z: np.ndarray, B: np.ndarray) -> "BenefitRouter":
        """Fit a GBR on all (Z, B) pairs (not LOO -- used to get a final model).

        For leave-one-out predictions use :meth:`leave_one_out`.

        Parameters
        ----------
        Z : array-like of shape (n, d)
            Evidence feature matrix.
        B : array-like of shape (n,)
            True benefit values.

        Returns
        -------
        self
        """
        Z = np.asarray(Z, dtype=float)
        B = np.asarray(B, dtype=float)
        self._model = self._make_gbr().fit(Z, B)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Predict benefit for new conditions using the fitted GBR.

        Parameters
        ----------
        Z : array-like of shape (n, d)
        Returns
        -------
        Bhat : np.ndarray of shape (n,)
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        return self._model.predict(np.asarray(Z, dtype=float))

    def leave_one_out(self, Z: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Leave-one-out cross-validated benefit predictions.

        For each condition i, fits a GBR on all other conditions and predicts
        condition i.  Mirrors ``decide_kga`` in cifar_tent_mps_v2.py exactly.

        Parameters
        ----------
        Z : array-like of shape (n, d)
        B : array-like of shape (n,)

        Returns
        -------
        Bhat : np.ndarray of shape (n,)
            LOO predictions.
        """
        Z = np.asarray(Z, dtype=float)
        B = np.asarray(B, dtype=float)
        N = len(B)
        if N < 2:
            raise ValueError("Need at least 2 conditions for LOO")
        Bhat = np.zeros(N)
        for i in range(N):
            tr = np.arange(N) != i
            m = self._make_gbr().fit(Z[tr], B[tr])
            Bhat[i] = m.predict(Z[i : i + 1])[0]
        return Bhat

    def decide_all(
        self,
        Z: np.ndarray,
        B: np.ndarray,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Full KGA pipeline: LOO GBR -> conformal eps -> decisions.

        Identical to ``decide_kga`` in cifar_tent_mps_v2.py.

        Parameters
        ----------
        Z : array-like of shape (n, d)
        B : array-like of shape (n,)
        alpha : float, default=0.1
            Miscoverage level.

        Returns
        -------
        Bhat : np.ndarray of shape (n,)
        eps : float
        decisions : np.ndarray of shape (n,) with values 'adapt','freeze','abstain'
        """
        Z = np.asarray(Z, dtype=float)
        B = np.asarray(B, dtype=float)
        Bhat = self.leave_one_out(Z, B)
        residuals = np.abs(Bhat - B)
        eps = conformal_radius(residuals, alpha=alpha)
        decisions = np.array([decide(bh, eps) for bh in Bhat])
        return Bhat, eps, decisions

    def _make_gbr(self) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state,
        )
