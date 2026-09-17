"""kbound_edge.benefit_estimator -- HistGradientBoostingRegressor benefit model.

Predicts the per-window adaptation benefit ``B`` (e.g. accuracy of the adapted
candidate minus accuracy of the frozen model on the same window) from the
label-free evidence vector ``Z``.

The estimator is fit ONLY on the calibration-FIT split.  The conformal radius is
then computed on the held-out calibration-CONFORMAL split (see
:mod:`kbound_edge.conformal`) -- this module deliberately knows nothing about
conformal so the split cannot accidentally leak.

joblib is used for persistence so the fitted estimator + conformal radius can be
saved by ``scripts/05_fit_kga_edge.py`` and loaded by the replay/shadow runners.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class EdgeBenefitEstimator:
    """Gradient-boosted regressor B_hat = f(Z) for the edge benefit signal.

    Parameters mirror a small, low-variance configuration suitable for the
    modest number of calibration conditions typical of an inspection setup.

    Parameters
    ----------
    max_iter : int, default=200
    learning_rate : float, default=0.05
    max_depth : int, default=3
    l2_regularization : float, default=0.0
    min_samples_leaf : int, default=5
    random_state : int, default=0
    """

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        l2_regularization: float = 0.0,
        min_samples_leaf: int = 5,
        random_state: int = 0,
    ) -> None:
        self.params = dict(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            l2_regularization=l2_regularization,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            early_stopping=False,
        )
        self._model: Optional[HistGradientBoostingRegressor] = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, Z: np.ndarray, B: np.ndarray) -> "EdgeBenefitEstimator":
        """Fit on the calibration-FIT split only.

        Parameters
        ----------
        Z : np.ndarray of shape (n_fit, d)
        B : np.ndarray of shape (n_fit,)
        """
        Z = np.asarray(Z, dtype=float)
        B = np.asarray(B, dtype=float)
        if Z.ndim != 2 or B.ndim != 1 or len(Z) != len(B):
            raise ValueError("Z must be (n,d), B must be (n,), with matching n")
        self._model = HistGradientBoostingRegressor(**self.params).fit(Z, B)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Predict benefit for one or more evidence rows."""
        if self._model is None:
            raise RuntimeError("EdgeBenefitEstimator.fit() must be called before predict()")
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[None, :]
        return self._model.predict(Z)

    def predict_one(self, z: np.ndarray) -> float:
        """Predict benefit for a single evidence vector -> float."""
        return float(self.predict(np.asarray(z, dtype=float)[None, :])[0])

    # -- persistence -----------------------------------------------------------
    def save(self, path: str) -> None:
        import joblib

        if self._model is None:
            raise RuntimeError("Nothing to save: estimator is not fitted")
        joblib.dump({"params": self.params, "model": self._model}, path)

    @classmethod
    def load(cls, path: str) -> "EdgeBenefitEstimator":
        import joblib

        blob = joblib.load(path)
        obj = cls()
        obj.params = blob["params"]
        obj._model = blob["model"]
        return obj
