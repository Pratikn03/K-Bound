"""Arm A — Jackknife+ conformal radius for the KGA benefit certificate.

Frozen per research_lock/WIN_HUNT_v4_PROTOCOL.yaml (arm_A_jackknife_plus_radius).
Pure numpy/sklearn, CPU. Implements the jackknife+ predictive interval of
Barber, Candes, Ramdas & Tibshirani (2021, "Predictive inference with the
jackknife+", Ann. Statist. 49(1)) for the per-cell benefit B = a_adapted - a0.

Given calibration pairs {(Z_i, B_i)}_{i=1..n} and a base regressor mu, fit the n
leave-one-out models mu_{-i}, form LOO residuals R_i = |mu_{-i}(Z_i) - B_i|, and
for a test point z return the jackknife+ interval

    [ q^-_alpha{ mu_{-i}(z) - R_i },  q^+_alpha{ mu_{-i}(z) + R_i } ]

where (rank-corrected order statistics, NO interpolation):
    q^-_alpha{v} = floor(alpha (n+1))-th smallest of v   (-inf if that rank < 1)
    q^+_alpha{v} = ceil((1-alpha)(n+1))-th smallest of v  (+inf if that rank > n)

Finite-sample, distribution-free two-sided coverage:  P(B in interval) >= 1-2alpha.

Decision (trichotomy on the benefit interval [lo, hi]):
    ADAPT   iff lo > 0        (whole interval clears zero from above)
    FREEZE  iff hi < 0        (whole interval clears zero from below)
    ABSTAIN otherwise

fit_grouped implements the leave-one-seed variant (CV+ with folds = seed groups),
matching the existing leave-one-seed cross-fitting (radius_v2.crossfit_oof): one
model per held-out group, residuals scored out-of-fold, and the CV+ interval built
from every (fold-model, residual) pair.

Usage (library):
    from radius_jackknife_plus import JackknifePlusGate
    gate = JackknifePlusGate(alpha=0.05).fit(Z_cal, B_cal)      # leave-one-out
    decision, lo, hi = gate.decide(z)
    gate = JackknifePlusGate(alpha=0.05).fit_grouped(Z, B, seeds)  # leave-one-seed

Self-check:
    .venv/bin/python docs/research/kbound/gapclose_wave5/radius_jackknife_plus.py
"""
from __future__ import annotations

import numpy as np

try:  # heavy import kept optional (mirrors radius_v2.py)
    from sklearn.ensemble import GradientBoostingRegressor
except Exception:  # pragma: no cover
    GradientBoostingRegressor = None

# EXACT KGA GBR config (matches eps_recal_camelyon.py / radius_v2.GBR_CFG).
GBR_CFG = dict(n_estimators=250, max_depth=2, learning_rate=0.05,
               subsample=0.8, random_state=0)

DECISION_CODE = {"ADAPT": 1, "FREEZE": -1, "ABSTAIN": 0}  # for logged scoring


def _default_learner():
    if GradientBoostingRegressor is None:  # pragma: no cover
        raise ImportError("sklearn GradientBoostingRegressor unavailable")
    return GradientBoostingRegressor(**GBR_CFG)


# ------------------------------------------------------------ order statistics
def q_minus(v: np.ndarray, alpha: float) -> float:
    """Jackknife+ lower order statistic: floor(alpha(n+1))-th smallest of v.

    Returns -inf when that rank falls below 1 (interval unbounded below), as in
    Barber et al. (2021). No np.quantile interpolation.
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    k = int(np.floor(alpha * (n + 1)))
    if k < 1:
        return -np.inf
    k = min(k, n)
    return float(np.partition(v, k - 1)[k - 1])


def q_plus(v: np.ndarray, alpha: float) -> float:
    """Jackknife+ upper order statistic: ceil((1-alpha)(n+1))-th smallest of v.

    Returns +inf when that rank exceeds n (interval unbounded above).
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    if k > n:
        return np.inf
    k = max(k, 1)
    return float(np.partition(v, k - 1)[k - 1])


# ------------------------------------------------------------------- the gate
class JackknifePlusGate:
    """Jackknife+ (leave-one-out) / CV+ (leave-one-group) benefit gate.

    Parameters
    ----------
    alpha : per-direction miscoverage; the interval has coverage >= 1 - 2 alpha.
    make_learner : zero-arg callable returning a FRESH unfitted regressor.
        Default = GradientBoostingRegressor(**GBR_CFG) (the KGA GBR).
    """

    def __init__(self, alpha: float = 0.05, make_learner=None):
        self.alpha = float(alpha)
        self.make_learner = make_learner or _default_learner
        self._fitted = False
        self._grouped = False

    # -- fitting ------------------------------------------------------------
    def fit(self, Z, B):
        """Leave-one-out jackknife+. Stores the n LOO models and LOO residuals."""
        Z = np.asarray(Z, dtype=float)
        B = np.asarray(B, dtype=float)
        n = B.shape[0]
        if n < 2:
            raise ValueError(f"jackknife+ needs n>=2 calibration points; got {n}")
        self._models = []
        self._resid = np.empty(n)
        idx = np.arange(n)
        for i in range(n):
            tr = idx != i
            m = self.make_learner().fit(Z[tr], B[tr])
            self._models.append(m)
            self._resid[i] = abs(float(m.predict(Z[i:i + 1])[0]) - float(B[i]))
        self._n = n
        self._fitted, self._grouped = True, False
        return self

    def fit_grouped(self, Z, B, groups):
        """Leave-one-seed CV+ (folds = groups). One model per group; residuals
        scored out-of-fold. Matches the existing leave-one-seed cross-fitting."""
        Z = np.asarray(Z, dtype=float)
        B = np.asarray(B, dtype=float)
        groups = np.asarray(groups)
        gids = np.unique(groups)
        if gids.size < 2:
            raise ValueError(f"CV+ needs >=2 groups; got {gids.size}")
        self._group_ids = gids
        self._row_group = groups
        self._models = {}
        self._resid = np.empty(B.shape[0])
        for g in gids:
            tr, te = groups != g, groups == g
            m = self.make_learner().fit(Z[tr], B[tr])
            self._models[g] = m
            self._resid[te] = np.abs(m.predict(Z[te]) - B[te])
        self._n = int(B.shape[0])
        self._fitted, self._grouped = True, True
        return self

    # -- prediction ---------------------------------------------------------
    def _loo_predict(self, z: np.ndarray) -> np.ndarray:
        """mu_{-i}(z) aligned with self._resid (per-point LOO or per-group CV+)."""
        z = np.asarray(z, dtype=float).reshape(1, -1)
        if self._grouped:
            pred = {g: float(self._models[g].predict(z)[0]) for g in self._group_ids}
            return np.array([pred[g] for g in self._row_group])
        return np.array([float(m.predict(z)[0]) for m in self._models])

    def interval(self, z):
        """(lo, hi): jackknife+ / CV+ benefit interval at test point z."""
        if not self._fitted:
            raise RuntimeError("call fit(...) or fit_grouped(...) first")
        mu = self._loo_predict(z)
        return (q_minus(mu - self._resid, self.alpha),
                q_plus(mu + self._resid, self.alpha))

    def decide(self, z):
        """(decision, lo, hi) with decision in {'ADAPT','FREEZE','ABSTAIN'}."""
        lo, hi = self.interval(z)
        if lo > 0.0:
            d = "ADAPT"
        elif hi < 0.0:
            d = "FREEZE"
        else:
            d = "ABSTAIN"
        return d, lo, hi

    def decide_batch(self, Zt):
        """Vectorized over rows of Zt -> (codes[int], lo[float], hi[float])."""
        Zt = np.asarray(Zt, dtype=float)
        codes = np.empty(Zt.shape[0], dtype=int)
        los = np.empty(Zt.shape[0])
        his = np.empty(Zt.shape[0])
        for j in range(Zt.shape[0]):
            d, lo, hi = self.decide(Zt[j])
            codes[j], los[j], his[j] = DECISION_CODE[d], lo, hi
        return codes, los, his


# ------------------------------------------------------------------- selfcheck
def _selfcheck() -> int:
    rng = np.random.default_rng(0)
    from sklearn.tree import DecisionTreeRegressor
    mk = lambda: DecisionTreeRegressor(max_depth=3, random_state=0)
    cov = 0
    reps = 200
    for _ in range(reps):
        Z = rng.uniform(-1, 1, (50, 2))
        B = np.sin(2 * Z[:, 0]) + rng.normal(0, 0.2, 50)
        z = rng.uniform(-1, 1, (1, 2))
        bt = float(np.sin(2 * z[0, 0]) + rng.normal(0, 0.2))
        lo, hi = JackknifePlusGate(0.05, mk).fit(Z, B).interval(z[0])
        cov += int(lo <= bt <= hi)
    c = cov / reps
    ok = c >= 0.90
    print(f"[radius_jackknife_plus selfcheck] coverage={c:.3f} (>=0.90) -> "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())
