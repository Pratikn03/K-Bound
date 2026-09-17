"""Arm B (WIN_HUNT_v4) — CrossFitBenefitEstimator.

Frozen per research_lock/WIN_HUNT_v4_PROTOCOL.yaml (arm_B_estimator_v2). Pure
numpy/sklearn, CPU. The hyperparameter grid is DECLARED here (module constants)
BEFORE scoring; there is NO search — the "grid" is a 3-config ensemble averaged
with EQUAL weights, so nothing is selected at scoring time.

Method:
  * K-fold (K=5; GroupKFold by seed when groups given) cross-fit. In each fold an
    equal-weight average of the 3 GradientBoostingRegressors below predicts the
    held-out benefit B̂ (out-of-fold / OOF).
  * OOF residuals r = B - B̂ feed BOTH
      (i)  an isotonic residual-SCALE model s(z) = Iso(|B̂|) -> E|r|, floored at
           iso_eps (monotone, label-free), and
      (ii) a rank-corrected (1-alpha) quantile of the SCALE-NORMALIZED score
           u = |r| / s (locally-adaptive split conformal, Lei et al. style).
  * radius(z) = max( q_{1-alpha}(u) * s(z), global_floor ).
  * decision (interval clears zero): ADAPT iff B̂-radius>0, FREEZE iff
    B̂+radius<0, else ABSTAIN.

Anti-leakage (spy-fold): any feature column CONSTANT within every cross-fit group
is a fold-membership indicator (carries no cross-group signal); it is stripped
before fitting, so OOF predictions are invariant to it — the arm-B spy-fold bar.
"""
from __future__ import annotations

import os
import sys

import numpy as np

try:  # heavy imports guarded so tau validators need no sklearn to import this
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold, KFold
except Exception:  # pragma: no cover
    GradientBoostingRegressor = IsotonicRegression = None
    GroupKFold = KFold = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radius_v2 import rank_quantile  # noqa: E402

# FROZEN declared grid (no search over it). INCUMBENT_CFG == radius_v2.GBR_CFG.
INCUMBENT_CFG = dict(n_estimators=250, max_depth=2, learning_rate=0.05,
                     subsample=0.8, random_state=0)
DEEPER_CFG = dict(n_estimators=400, max_depth=3, learning_rate=0.05,
                  subsample=0.8, random_state=0)
SHALLOW_CFG = dict(n_estimators=600, max_depth=1, learning_rate=0.05,
                   subsample=0.8, random_state=0)
GRID = (INCUMBENT_CFG, DEEPER_CFG, SHALLOW_CFG)


def _ensemble_fit(Z, B):
    return [GradientBoostingRegressor(**cfg).fit(Z, B) for cfg in GRID]


def _ensemble_pred(models, Z):
    return np.mean([m.predict(Z) for m in models], axis=0)


class CrossFitBenefitEstimator:
    """Cross-fitted 3-config GBR ensemble + isotonic-normalized conformal radius."""

    def __init__(self, alpha: float = 0.10, n_splits: int = 5,
                 floor: float = 1e-3, iso_eps: float = 1e-4):
        self.alpha = float(alpha)
        self.n_splits = int(n_splits)
        self.floor = float(floor)
        self.iso_eps = float(iso_eps)

    # -- fold-membership spy guard -------------------------------------------
    @staticmethod
    def _strip_group_leak(Z, groups):
        """Keep only columns that VARY within some group (drop fold-index spies)."""
        if Z.ndim != 2 or Z.shape[1] == 0:
            return Z, np.arange(Z.shape[1] if Z.ndim == 2 else 0)
        if groups is None:
            return Z, np.arange(Z.shape[1])
        keep = []
        for j in range(Z.shape[1]):
            varies = False
            for g in np.unique(groups):
                col = Z[groups == g, j]
                if col.size and (col.max() - col.min()) > 1e-12:
                    varies = True
                    break
            if varies:
                keep.append(j)
        return Z[:, keep], np.array(keep, dtype=int)

    def _splits(self, n, groups):
        if groups is not None and GroupKFold is not None:
            k = min(self.n_splits, len(np.unique(groups)))
            return list(GroupKFold(n_splits=max(k, 2)).split(
                np.zeros(n), groups=groups))
        return list(KFold(n_splits=min(self.n_splits, n), shuffle=True,
                          random_state=0).split(np.zeros(n)))

    def fit(self, Z, B, groups=None):
        Z = np.asarray(Z, float)
        B = np.asarray(B, float)
        groups = None if groups is None else np.asarray(groups)
        Zc, self.keep_cols_ = self._strip_group_leak(Z, groups)
        self.n_features_in_ = Z.shape[1]
        n = len(B)
        oof = np.full(n, np.nan)
        for tr, te in self._splits(n, groups):
            models = _ensemble_fit(Zc[tr], B[tr])
            oof[te] = _ensemble_pred(models, Zc[te])
        miss = np.isnan(oof)
        if miss.any():  # points never held out (tiny group): self-consistent fill
            models = _ensemble_fit(Zc, B)
            oof[miss] = _ensemble_pred(models, Zc[miss])
        self.oof_bhat_ = oof
        self.oof_resid_ = B - oof
        # isotonic residual-scale: |B̂| -> E|resid|, floored at iso_eps
        self.iso_ = IsotonicRegression(increasing=True, out_of_bounds="clip")
        self.iso_.fit(np.abs(oof), np.abs(self.oof_resid_))
        s_oof = np.maximum(self.iso_.predict(np.abs(oof)), self.iso_eps)
        u = np.abs(self.oof_resid_) / s_oof
        self.q_ = float(rank_quantile(u, 1.0 - self.alpha))  # rank-corrected
        self.models_ = _ensemble_fit(Zc, B)  # full-data ensemble for new points
        return self

    def _prep(self, Z):
        Z = np.asarray(Z, float)
        return Z[:, self.keep_cols_] if Z.ndim == 2 else Z

    def predict(self, Z):
        return _ensemble_pred(self.models_, self._prep(Z))

    def scale(self, bhat):
        return np.maximum(self.iso_.predict(np.abs(np.asarray(bhat, float))),
                          self.iso_eps)

    def radius(self, Z=None, bhat=None):
        if bhat is None:
            bhat = self.predict(Z)
        return np.maximum(self.q_ * self.scale(bhat), self.floor)

    def _decide(self, bhat, rad):
        out = np.zeros(np.shape(bhat), dtype=int)
        out[bhat - rad > 0] = 1
        out[bhat + rad < 0] = -1
        return out

    def decide(self, Z=None, bhat=None):
        if bhat is None:
            bhat = self.predict(Z)
        rad = self.radius(bhat=bhat)
        return self._decide(bhat, rad), bhat, rad

    # OOF helpers (honest cross-fit; used by validators / reruns) -------------
    def oof_radius(self):
        return np.maximum(self.q_ * self.scale(self.oof_bhat_), self.floor)

    def oof_decide(self):
        rad = self.oof_radius()
        return self._decide(self.oof_bhat_, rad), self.oof_bhat_, rad
