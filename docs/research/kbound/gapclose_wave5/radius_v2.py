"""Gap A — de-biased, drift-robust conformal radius for the KGA certificate.

Frozen per PROTOCOL_GAPCLOSE_WAVE5_v1.md. Pure numpy/sklearn, CPU.

Key idea: the published radius is the Q90 of |B - Bhat| — a SYMMETRIC quantile that
pays systematic model bias b0 in both directions (eps ~ |b0| + z*sigma). Signed
asymmetric conformal quantiles absorb b0 as a recentering (width ~ z*sigma), while
per-direction validity (false-adapt / false-freeze <= alpha) is preserved by
split-conformal on the signed residual. Optional layers: DML-style cross-fitted
ridge orthogonalization, Mondrian bins, likelihood-ratio weighted quantiles.
"""
from __future__ import annotations

import numpy as np

try:  # heavy import kept optional so tau/evidence validators don't need sklearn
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
except Exception:  # pragma: no cover
    GradientBoostingRegressor = LogisticRegression = Ridge = None

GBR_CFG = dict(n_estimators=250, max_depth=2, learning_rate=0.05,
               subsample=0.8, random_state=0)  # EXACT eps_recal_camelyon.py config
Z80 = 1.2815515655446004  # central-80% half-width of a standard normal
Z90 = 1.6448536269514722  # legacy Q90(|noise|) convention used in the diag


# ---------------------------------------------------------------- quantiles
def rank_quantile(x: np.ndarray, q: float, w: np.ndarray | None = None) -> float:
    """Finite-sample rank-corrected empirical quantile (conformal convention).

    Unweighted: the ceil((n+1)q)-th order statistic (clipped). Weighted: the
    weighted-CDF generalization of the same +1 correction (Tibshirani et al.).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if w is None:
        k = int(np.ceil((n + 1) * q))
        k = min(max(k, 1), n)
        return float(np.sort(x)[k - 1])
    w = np.asarray(w, dtype=float)
    w = w / (w.sum() + 1.0)  # +1 mass reserved for the test point (conformal)
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    idx = np.searchsorted(cw, q, side="left")
    idx = min(idx, n - 1)
    return float(x[order][idx])


def crossfit_oof(Z: np.ndarray, B: np.ndarray, groups: np.ndarray,
                 second_stage: bool = False) -> np.ndarray:
    """Leave-one-group-out cross-fitted predictions B̂(Z) (+ optional DML ridge stage)."""
    Bhat = np.full(B.shape, np.nan)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        m = GradientBoostingRegressor(**GBR_CFG).fit(Z[tr], B[tr])
        pred = m.predict(Z[te])
        if second_stage:
            r_tr = B[tr] - m.predict(Z[tr])
            rr = Ridge(alpha=1.0).fit(Z[tr], r_tr)
            pred = pred + rr.predict(Z[te])
        Bhat[te] = pred
    return Bhat


def lr_weights(Z_cal: np.ndarray, Z_eval: np.ndarray):
    """Label-free likelihood-ratio weights w(z) = p_eval(z)/p_cal(z) via logistic
    regression. Returns (w_cal, w_test): weights at calibration AND test points
    (proper weighted conformal needs the test point's own weight)."""
    X = np.vstack([Z_cal, Z_eval])
    y = np.r_[np.zeros(len(Z_cal)), np.ones(len(Z_eval))]
    clf = LogisticRegression(max_iter=2000, C=10.0).fit(X, y)

    def ratio(Z):
        p = np.clip(clf.predict_proba(Z)[:, 1], 1e-4, 1 - 1e-4)
        return p / (1 - p)

    w_cal, w_te = ratio(Z_cal), ratio(Z_eval)
    norm = w_cal.mean()
    return (np.clip(w_cal / norm, 0.02, 50.0),
            np.clip(w_te / norm, 0.02, 50.0))


def weighted_bounds_per_test(resid_cal: np.ndarray, w_cal: np.ndarray,
                             w_test: np.ndarray, alpha: float):
    """Proper weighted split-conformal signed bounds (Tibshirani et al. 2019).

    Lower bound: alpha-quantile of the weighted residual law with the TEST
    point's mass w_test[j] placed at -inf (worst case); upper bound: at +inf.
    Returns (q_lo, q_hi) arrays of shape (n_test,).
    """
    order = np.argsort(resid_cal)
    r = resid_cal[order]
    w = w_cal[order]
    cw = np.cumsum(w)
    total = cw[-1]
    q_lo = np.empty(w_test.shape)
    q_hi = np.empty(w_test.shape)
    for j, wt in enumerate(w_test):
        tot = total + wt
        # lower: test mass at -inf counts below every candidate
        k = np.searchsorted(wt + cw, alpha * tot, side="left")
        q_lo[j] = r[min(k, len(r) - 1)] if k < len(r) else r[-1]
        if wt >= alpha * tot:  # test mass alone exceeds alpha -> no finite bound
            q_lo[j] = r[0] - 10.0 * (r[-1] - r[0] + 1e-6)
        # upper: test mass at +inf
        k2 = np.searchsorted(cw, (1 - alpha) * tot, side="left")
        q_hi[j] = r[min(k2, len(r) - 1)] if k2 < len(r) else \
            r[-1] + 10.0 * (r[-1] - r[0] + 1e-6)
    return q_lo, q_hi


# ---------------------------------------------------------------- certificate
def signed_bounds(resid_cal: np.ndarray, alpha: float,
                  w: np.ndarray | None = None) -> tuple[float, float]:
    """(q_lo, q_hi): P(B >= Bhat + q_lo) >= 1-alpha and P(B <= Bhat + q_hi) >= 1-alpha."""
    return (rank_quantile(resid_cal, alpha, w),
            rank_quantile(resid_cal, 1.0 - alpha, w))


def decide(Bhat: np.ndarray, q_lo: float, q_hi: float) -> np.ndarray:
    """Per-cell trichotomy. 1 = ADAPT, -1 = FREEZE, 0 = ABSTAIN."""
    out = np.zeros(Bhat.shape, dtype=int)
    out[Bhat + q_lo > 0] = 1
    out[Bhat + q_hi < 0] = -1
    return out


def mondrian_bounds(Bhat_cal: np.ndarray, resid_cal: np.ndarray, Bhat_te: np.ndarray,
                    alpha: float, n_bins: int = 3, min_bin: int = 30):
    """Per-bin signed bounds on terciles of B̂ (falls back to global if bins thin)."""
    qs = np.quantile(Bhat_cal, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    lo = np.empty(Bhat_te.shape)
    hi = np.empty(Bhat_te.shape)
    g_lo, g_hi = signed_bounds(resid_cal, alpha)
    for b in range(n_bins):
        m_cal = (Bhat_cal >= qs[b]) & (Bhat_cal < qs[b + 1])
        m_te = (Bhat_te >= qs[b]) & (Bhat_te < qs[b + 1])
        if m_cal.sum() >= min_bin:
            l, h = signed_bounds(resid_cal[m_cal], alpha)
        else:
            l, h = g_lo, g_hi
        lo[m_te], hi[m_te] = l, h
    return lo, hi


# ---------------------------------------------------------------- evaluation
def evaluate_variant(Z, B, groups, alpha=0.10, variant="V1",
                     sigma_meas: np.ndarray | None = None,
                     weight_fn=None) -> dict:
    """Cross-fitted (leave-one-group-out) evaluation of one radius variant.

    For each held-out group g: fit estimator + calibrate quantiles on the OOF
    residuals of the OTHER groups (their own inner cross-fit), decide on g.
    """
    Z, B, groups = np.asarray(Z, float), np.asarray(B, float), np.asarray(groups)
    second = variant in ("V2", "V3")
    Bhat = crossfit_oof(Z, B, groups, second_stage=second)
    resid = B - Bhat

    fa_n = fa_d = 0
    cov_lo = cov_hi = n_te = 0
    widths, dec_all = [], np.zeros(B.shape, int)
    cov_lo_by_group = {}
    for g in np.unique(groups):
        cal, te = groups != g, groups == g
        if variant == "V3":
            lo_arr, hi_arr = mondrian_bounds(Bhat[cal], resid[cal], Bhat[te], alpha)
        elif variant == "V4":  # proper weighted conformal, per-test bounds
            wf = weight_fn if weight_fn else lr_weights
            w_cal, w_te = wf(Z[cal], Z[te])
            lo_arr, hi_arr = weighted_bounds_per_test(resid[cal], w_cal,
                                                      w_te, alpha)
        elif variant == "V0":  # published baseline: symmetric |resid| Q90
            eps = rank_quantile(np.abs(resid[cal]), 1 - alpha)
            lo_arr = np.full(te.sum(), -eps)
            hi_arr = np.full(te.sum(), eps)
        else:  # V1 / V2: signed asymmetric
            l, h = signed_bounds(resid[cal], alpha)
            lo_arr = np.full(te.sum(), l)
            hi_arr = np.full(te.sum(), h)
        d = np.zeros(te.sum(), int)
        d[Bhat[te] + lo_arr > 0] = 1
        d[Bhat[te] + hi_arr < 0] = -1
        dec_all[te] = d
        fa_n += int(((d == 1) & (B[te] <= 0)).sum())
        fa_d += int(te.sum())
        cov_lo += int((B[te] >= Bhat[te] + lo_arr).sum())
        cov_hi += int((B[te] <= Bhat[te] + hi_arr).sum())
        cov_lo_by_group[str(g)] = float((B[te] >= Bhat[te] + lo_arr).mean())
        n_te += int(te.sum())
        widths.append(0.5 * (hi_arr - lo_arr))
    w_eff = float(np.mean(np.concatenate(widths)))

    out = dict(variant=variant, alpha=alpha, w_eff=w_eff,
               fa_emp=fa_n / fa_d, fa_mc_se=float(np.sqrt(alpha * (1 - alpha) / fa_d)),
               cov_lo=cov_lo / n_te, cov_hi=cov_hi / n_te,
               adapt_rate=float((dec_all == 1).mean()),
               freeze_rate=float((dec_all == -1).mean()),
               abstain_rate=float((dec_all == 0).mean()),
               cov_lo_by_group=cov_lo_by_group,
               mean_bias=float(resid.mean()), resid_std=float(resid.std()))
    if sigma_meas is not None:
        sm = float(np.mean(sigma_meas))
        out.update(w_meas=Z80 * sm, ratio80=w_eff / (Z80 * sm),
                   eps_meas_legacy=Z90 * sm, ratio_legacy=w_eff / (Z90 * sm))
    return out
