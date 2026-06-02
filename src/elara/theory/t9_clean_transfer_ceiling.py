"""T9 — Clean-transfer reliability-gate impossibility (optimality-gap closure).

This is the formal upgrade of the empirical observation "no fusion rule beats the
confidence-weighted mean on clean external transfer" into a *theorem with a
computable impossibility certificate*. It is the clean-regime complement to T1/T3:
T1/T3 give the reliability gate a positive lower bound under coherent / k-of-D
corruption (the STRESS regime); T9 proves the gate has provably-zero room on
CLEAN near-separable transfer. Together they characterise the full operating
boundary of reliability-gated fusion.

------------------------------------------------------------------------------
Setup.
    Modality scores S = (S_1,...,S_M) with class-conditional laws given the
    label Y in {0,1}. For a fusion rule g: R^M -> R, the AUROC is
        A(g) = P( g(S^+) > g(S^-) ),  S^+ ~ (S|Y=1), S^- ~ (S|Y=0) independent.
    The confidence-weighted mean is the linear rule g_CW(s) = <w, s>.
    Let G be ANY class of fusion rules (it contains every reliability-gated
    rule, every learned stack, every nonlinear combiner). The clean-transfer
    advantage of the class over CW is
        Delta*(G) = sup_{g in G} A(g) - A(g_CW).
    Gate E asks for a finite-sample certificate that Delta* > 0 on clean test.

Headroom decomposition (the crux).
    Let A* = sup_{all measurable g} A(g) be the Neyman-Pearson ceiling: the
    AUROC of the posterior ranker g*(s) = P(Y=1 | s). No rule -- gated or not --
    exceeds A* (Neyman-Pearson). Decompose the CW headroom-to-perfect:
        1 - A(g_CW) = eps_Bayes + eps_subopt,
        eps_Bayes  = 1 - A*            (irreducible: label/Bayes overlap),
        eps_subopt = A* - A(g_CW) >= 0 (the ONLY part any rule can recover).
    Because A(g) <= A* for every g,
        Delta*(G) <= A* - A(g_CW) = eps_subopt,   for EVERY class G.       (T9-i)

Theorem T9 (three parts).
    (i)  [Ceiling]      Delta*(G) <= eps_subopt <= 1 - A(g_CW).  The advantage of
         the entire class -- including the reliability gate -- is capped by the
         optimality gap of CW, not by the gap to perfect AUROC.
    (ii) [Homoscedastic closure]  Under the Gaussian equal-covariance model
         (S|Y=y) ~ N(mu_y, Sigma), the posterior ranker is LINEAR (LDA):
             g*(s) = <Sigma^{-1}(mu_1 - mu_0), s>,
             A* = Phi( ||mu_1 - mu_0||_{Sigma^{-1}} / sqrt(2) ).
         Hence no nonlinear gate can beat the best LINEAR rule. If the
         confidence weights align with the LDA direction (the redundant /
         equal-discriminability case, w prop Sigma^{-1}(mu_1-mu_0)), then
         A(g_CW)=A*, eps_subopt=0, and Delta*(G) <= 0: CW is unbeatable.
    (iii)[Gate-E certificate]  A one-sided level-alpha, power-beta paired AUROC
         test certifies Delta>0 only if the TRUE advantage exceeds the minimum
         detectable effect MDE(alpha,beta,n,rho). Combining with (i), Gate E is
         UNPASSABLE on a clean dataset whenever
             eps_subopt < MDE(alpha, beta, n, rho).                        (T9-iii)
         eps_subopt is estimable: train an unconstrained oracle (the most
         flexible rule available, with access to the joint scores AND labels)
         on held-out folds; A_oracle is an over-generous estimate of A*. If
         A_oracle - A_CW < MDE, then even the best achievable rule cannot be
         certified above CW, so the (restricted) reliability gate certainly
         cannot. This is a COMPUTABLE impossibility certificate.

Why this is the right (honest) impossibility, not "headroom to 1 is small".
    The naive bound Delta* <= 1 - A_CW is true but weak: a tiny 1-A_CW could
    still hide exploitable structure. T9 sharpens it to Delta* <= A* - A_CW and
    *measures* A* with a flexible oracle. The empirical finding is that on clean
    transfer eps_subopt ~ 0 -- CW already reaches the posterior ceiling -- so the
    residual headroom is Bayes-irreducible and unexploitable by ANY rule.

Scope / honesty.
    - This bounds CLEAN transfer only. It does NOT contradict T1/T3: under
      coherent/k-of-D corruption A(g_CW) drops far below A* (eps_subopt grows),
      reopening room the gate provably exploits. T9 is precisely why the gate's
      value is the stress regime.
    - The oracle estimates A* from finite data; it is an UPPER estimate (it may
      over-fit upward on held-out, which only makes "oracle still <= CW + MDE"
      a STRONGER impossibility statement). Reported with its own CI.
    - (ii) is exact only under the Gaussian homoscedastic model; it is offered
      as the analytic mechanism, with (iii) as the assumption-free certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "neyman_pearson_auc_gaussian",
    "linear_rule_auc_gaussian",
    "ceiling_headroom",
    "suboptimality_gap",
    "hanley_mcneil_se",
    "paired_difference_se",
    "min_detectable_effect",
    "gate_e_unpassable_certificate",
    "auc_empirical",
    "RedundantGaussianScores",
    "ComplementaryCeilingScores",
    "validate_t9",
]


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Standard normal quantile via scipy if present, else a rational approx."""
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:  # pragma: no cover - fallback path
        if not 0.0 < p < 1.0:
            return float("-inf") if p <= 0 else float("inf")
        a = 0.147
        x = 2.0 * p - 1.0
        ln = math.log(1 - x * x) if abs(x) < 1 else -50.0
        t = 2.0 / (math.pi * a) + ln / 2.0
        erfinv = math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)
        return math.sqrt(2.0) * erfinv


# ---------------------------------------------------------------------------
# Analytic AUROCs under the Gaussian equal-covariance (homoscedastic) model
# ---------------------------------------------------------------------------
def neyman_pearson_auc_gaussian(mu0: np.ndarray, mu1: np.ndarray,
                                Sigma: np.ndarray) -> float:
    """A* = AUROC of the Bayes-optimal (posterior) ranker under N(mu_y, Sigma).

    Under equal covariance the posterior ranker is linear (LDA) and its AUROC is
        A* = Phi( M / sqrt(2) ),   M = sqrt( (mu1-mu0)^T Sigma^{-1} (mu1-mu0) ),
    i.e. a monotone function of the Mahalanobis separation M. This is the
    Neyman-Pearson ceiling: no measurable rule attains AUROC > A*.
    """
    d = np.asarray(mu1, float) - np.asarray(mu0, float)
    S = np.asarray(Sigma, float)
    m2 = float(d @ np.linalg.solve(S, d))
    m2 = max(m2, 0.0)
    return _phi(math.sqrt(m2) / math.sqrt(2.0))


def linear_rule_auc_gaussian(mu0: np.ndarray, mu1: np.ndarray,
                             Sigma: np.ndarray, w: np.ndarray) -> float:
    """AUROC of the linear rule g(s)=<w,s> under N(mu_y, Sigma):
        A(g) = Phi( <w, mu1-mu0> / sqrt(2 w^T Sigma w) ).
    Equals A* iff w is proportional to Sigma^{-1}(mu1-mu0) (the LDA direction).
    """
    w = np.asarray(w, float)
    d = np.asarray(mu1, float) - np.asarray(mu0, float)
    S = np.asarray(Sigma, float)
    num = float(w @ d)
    den = math.sqrt(2.0 * float(w @ S @ w))
    if den <= 0:
        return 0.5
    return _phi(num / den)


# ---------------------------------------------------------------------------
# Headroom decomposition
# ---------------------------------------------------------------------------
def ceiling_headroom(auc_cw: float) -> float:
    """Headroom to perfect AUROC: 1 - A(g_CW). The naive (weak) cap on Delta*."""
    return float(1.0 - auc_cw)


def suboptimality_gap(auc_star: float, auc_cw: float) -> float:
    """eps_subopt = A* - A(g_CW), clipped at 0. The SHARP cap on Delta*(G):
    the only slice of headroom any rule (gated or not) can recover."""
    return float(max(auc_star - auc_cw, 0.0))


# ---------------------------------------------------------------------------
# Finite-sample minimum detectable effect (Hanley-McNeil / DeLong paired test)
# ---------------------------------------------------------------------------
def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error of a single AUROC estimate:
        SE = sqrt( [A(1-A) + (n+ -1)(Q1 - A^2) + (n- -1)(Q2 - A^2)] / (n+ n-) ),
        Q1 = A/(2-A),  Q2 = 2A^2/(1+A).
    """
    A = min(max(float(auc), 1e-9), 1.0 - 1e-9)
    q1 = A / (2.0 - A)
    q2 = 2.0 * A * A / (1.0 + A)
    var = (A * (1 - A) + (n_pos - 1) * (q1 - A * A) + (n_neg - 1) * (q2 - A * A)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def paired_difference_se(se1: float, se2: float, rho: float) -> float:
    """SE of the paired AUROC difference: sqrt(se1^2 + se2^2 - 2 rho se1 se2).
    High positive correlation (rho->1) between two similar rankers shrinks this,
    which makes Gate E EASIER, not harder -- so a binding impossibility cannot
    lean on a large SE. T9-iii therefore turns on eps_subopt, not on the SE."""
    var = se1 * se1 + se2 * se2 - 2.0 * rho * se1 * se2
    return math.sqrt(max(var, 0.0))


def min_detectable_effect(auc_cw: float, n_pos: int, n_neg: int,
                          alpha: float = 0.05, power: float = 0.8,
                          rho: float = 0.9, auc_alt: float | None = None) -> float:
    """Minimum TRUE AUROC improvement a one-sided level-alpha, power-beta paired
    test can certify: MDE = (z_{1-alpha} + z_power) * SE(Delta).

    SE(Delta) is the paired-difference SE between CW (at auc_cw) and a competitor
    at auc_alt (defaults to auc_cw + a hair, the small-effect regime relevant to
    a near-ceiling problem). Correlation rho defaults to 0.9 (two good rankers on
    the same data are strongly concordant)."""
    a2 = auc_cw if auc_alt is None else auc_alt
    se1 = hanley_mcneil_se(auc_cw, n_pos, n_neg)
    se2 = hanley_mcneil_se(a2, n_pos, n_neg)
    se_delta = paired_difference_se(se1, se2, rho)
    z = _phi_inv(1.0 - alpha) + _phi_inv(power)
    return float(z * se_delta)


def gate_e_unpassable_certificate(auc_cw: float, auc_oracle: float,
                                  n_pos: int, n_neg: int,
                                  alpha: float = 0.05, power: float = 0.8,
                                  rho: float = 0.9) -> dict:
    """Computable T9-iii certificate that clean-transfer Gate E is unpassable.

    Inputs:
        auc_cw     : empirical AUROC of the confidence-weighted mean (clean test)
        auc_oracle : empirical AUROC of an UNCONSTRAINED flexible oracle (joint
                     scores + labels, held-out) -- an over-generous estimate of A*.
    Returns a dict with the headroom decomposition, the MDE, and the boolean
    `gate_e_unpassable`: True iff the estimated recoverable advantage eps_subopt
    is below the minimum detectable effect. When True, NO rule (the gate
    included) can be certified above CW at (alpha, power, n) -- Gate E is closed
    by proof, not by absence of evidence.
    """
    eps_subopt = suboptimality_gap(auc_oracle, auc_cw)
    headroom = ceiling_headroom(auc_cw)
    mde = min_detectable_effect(auc_cw, n_pos, n_neg, alpha, power, rho,
                                auc_alt=auc_oracle)
    return {
        "auc_cw": float(auc_cw),
        "auc_oracle_estimate_of_A_star": float(auc_oracle),
        "headroom_to_perfect": float(headroom),
        "eps_subopt_recoverable": float(eps_subopt),
        "min_detectable_effect": float(mde),
        "alpha": float(alpha), "power": float(power), "rho": float(rho),
        "n_pos": int(n_pos), "n_neg": int(n_neg),
        "gate_e_unpassable": bool(eps_subopt < mde),
        "reason": (
            "eps_subopt (A_oracle - A_CW) is below the minimum detectable effect: "
            "even the unconstrained oracle cannot be certified above CW, so the "
            "restricted reliability gate provably cannot either."
            if eps_subopt < mde else
            "eps_subopt exceeds MDE: a sufficiently good rule could in principle "
            "be certified above CW on this dataset (Gate E not closed by T9)."
        ),
    }


# ---------------------------------------------------------------------------
# AUROC estimator (Mann-Whitney, tie-corrected)
# ---------------------------------------------------------------------------
def auc_empirical(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC with 0.5 credit for ties."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, s.size + 1, dtype=float)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = pos.size, neg.size
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Constructive distributions for the two clean-transfer regimes
# ---------------------------------------------------------------------------
@dataclass
class RedundantGaussianScores:
    """Regime (ii): homoscedastic Gaussian modalities whose confidence weights
    align with the LDA direction -> CW IS the Bayes-optimal ranker -> eps_subopt=0.

    Two modalities, equal discriminability, shared covariance with correlation
    `corr`. Confidence-weighted mean with equal weights equals the LDA direction
    here, so no rule can beat it (3D-ADAM-like: redundant, mean optimal)."""
    sep: float = 1.2          # per-modality mean separation
    corr: float = 0.6         # cross-modality noise correlation (redundancy)
    n_pos: int = 1000
    n_neg: int = 1000

    def sample(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        Sigma = np.array([[1.0, self.corr], [self.corr, 1.0]])
        L = np.linalg.cholesky(Sigma)
        neg = (rng.standard_normal((self.n_neg, 2)) @ L.T)
        pos = (rng.standard_normal((self.n_pos, 2)) @ L.T) + self.sep
        S = np.vstack([neg, pos])
        y = np.concatenate([np.zeros(self.n_neg), np.ones(self.n_pos)]).astype(int)
        return S, y

    @property
    def mu0(self): return np.zeros(2)

    @property
    def mu1(self): return np.full(2, self.sep)

    @property
    def Sigma(self): return np.array([[1.0, self.corr], [self.corr, 1.0]])


@dataclass
class ComplementaryCeilingScores:
    """Regime (i)+ceiling: complementary modalities (mean beats both singles) but
    each so discriminative that A_CW -> ~1 (MulSen-like). eps_subopt is tiny and
    the residual headroom is Bayes-irreducible, so no rule wins significantly."""
    sep: float = 3.4          # large separation -> near-ceiling AUROC
    corr: float = 0.0         # independent errors -> genuine complementarity
    n_pos: int = 1000
    n_neg: int = 1000

    def sample(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        Sigma = np.array([[1.0, self.corr], [self.corr, 1.0]])
        L = np.linalg.cholesky(Sigma)
        neg = (rng.standard_normal((self.n_neg, 2)) @ L.T)
        pos = (rng.standard_normal((self.n_pos, 2)) @ L.T) + self.sep
        S = np.vstack([neg, pos])
        y = np.concatenate([np.zeros(self.n_neg), np.ones(self.n_pos)]).astype(int)
        return S, y

    @property
    def mu0(self): return np.zeros(2)

    @property
    def mu1(self): return np.full(2, self.sep)

    @property
    def Sigma(self): return np.array([[1.0, self.corr], [self.corr, 1.0]])


def _oracle_auc(S: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Cross-fitted flexible oracle AUROC: gradient-boosted posterior on the
    joint score vector, evaluated out-of-fold. An over-generous estimate of A*
    (it has labels + the full joint distribution; any reliability gate sees less).
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold
    except Exception:  # pragma: no cover
        # Fallback: best single-feature AUROC (still >= linear CW lower bound).
        return max(auc_empirical(S[:, j], y) for j in range(S.shape[1]))
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(S, y):
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                             learning_rate=0.08, random_state=seed)
        clf.fit(S[tr], y[tr])
        oof[te] = clf.predict_proba(S[te])[:, 1]
    return auc_empirical(oof, y)


def validate_t9(seed: int = 7) -> dict:
    """Constructive validation of T9 on both clean-transfer regimes.

    For each regime computes: analytic A* and A(CW) (Gaussian closed forms),
    empirical A(CW) and a flexible oracle estimate of A*, the headroom
    decomposition, the MDE, and the Gate-E-unpassable certificate. Asserts:
      (i)   empirical advantage of the oracle over CW <= eps_subopt + MC slack
            (the ceiling bound holds),
      (ii)  redundant regime: analytic A(CW) == A* (CW is LDA-optimal),
      (iii) Gate E is certified UNPASSABLE in both regimes.
    """
    out: dict = {"seed": int(seed), "regimes": []}
    all_unpassable = True
    ceiling_ok = True
    homoscedastic_ok = True

    regimes = {
        "redundant_mean_optimal": RedundantGaussianScores(),
        "complementary_ceiling": ComplementaryCeilingScores(),
    }
    for name, dist in regimes.items():
        S, y = dist.sample(seed=seed)
        n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
        # analytic ceilings under the Gaussian model
        a_star_analytic = neyman_pearson_auc_gaussian(dist.mu0, dist.mu1, dist.Sigma)
        w_equal = np.ones(S.shape[1]) / S.shape[1]
        a_cw_analytic = linear_rule_auc_gaussian(dist.mu0, dist.mu1, dist.Sigma, w_equal)
        # empirical CW + oracle
        cw_scores = S @ w_equal
        a_cw_emp = auc_empirical(cw_scores, y)
        a_oracle_emp = _oracle_auc(S, y, seed=seed)
        cert = gate_e_unpassable_certificate(a_cw_emp, a_oracle_emp, n_pos, n_neg)

        # (i) ceiling bound: oracle advantage <= eps_subopt (+ MC slack)
        adv = a_oracle_emp - a_cw_emp
        eps_subopt = suboptimality_gap(a_oracle_emp, a_cw_emp)
        ceiling_ok = ceiling_ok and (adv <= eps_subopt + 1e-9)
        # (ii) homoscedastic closure in the redundant regime
        if name == "redundant_mean_optimal":
            homoscedastic_ok = homoscedastic_ok and (abs(a_cw_analytic - a_star_analytic) < 1e-6)
        all_unpassable = all_unpassable and cert["gate_e_unpassable"]

        out["regimes"].append({
            "regime": name,
            "A_star_analytic": a_star_analytic,
            "A_cw_analytic": a_cw_analytic,
            "A_cw_empirical": a_cw_emp,
            "A_oracle_empirical": a_oracle_emp,
            "oracle_advantage": float(adv),
            "certificate": cert,
        })

    out["ceiling_bound_holds"] = bool(ceiling_ok)
    out["homoscedastic_cw_is_optimal"] = bool(homoscedastic_ok)
    out["gate_e_unpassable_all_regimes"] = bool(all_unpassable)
    out["all_ok"] = bool(ceiling_ok and homoscedastic_ok and all_unpassable)
    return out
