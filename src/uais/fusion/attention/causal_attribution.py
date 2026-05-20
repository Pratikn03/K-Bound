"""Causal reliability attribution via Double Machine Learning.

Implements a structural causal model for ELARA's fusion pipeline:

    domain_scores  ---->  domain_reliability  ---->  fused_prediction
         |                                                    ^
         +----------------------------------------------------+
                          (controls / confounders)

The existing CounterfactualDomainExplainer computes a *single intervention*
(mask one domain) and reports the prediction shift. That measures
correlation between the domain and the prediction, but it conflates the
domain's direct effect with the indirect effect via reliability. The
causal question - "if reliability for domain d were set to its
population mean, by how much would the prediction shift?" - is answered
by Double Machine Learning (Chernozhukov et al. 2018, Econometrica):

  Stage 1: cross-fitted nuisance functions
    g(W) = E[Y | W]          (predict Y from controls only, no T)
    m(W) = E[T | W]          (predict T from controls only)
  Stage 2: orthogonalised residual regression
    Y_tilde = Y - g_hat(W)
    T_tilde = T - m_hat(W)
    theta = OLS slope of Y_tilde on T_tilde

For per-domain attribution:
  Y      = fused prediction probability
  T      = per-domain reliability r_{i,d}
  W      = scores from all OTHER domains + masks + per-sample category

The cross-fitting eliminates the over-fitting bias that breaks naive
plug-in estimation, and gives an asymptotically Normal estimator of the
causal effect with a closed-form variance.

Reference: Chernozhukov V., Chetverikov D., Demirer M., Duflo E.,
Hansen C., Newey W., Robins J. (2018). "Double / debiased machine
learning for treatment and structural parameters." The Econometrics
Journal, 21(1), C1-C68.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


@dataclass
class DomainCausalEffect:
    """Average treatment effect of per-domain reliability on prediction."""
    domain: str
    ate: float
    ate_std_error: float
    ate_ci_low: float
    ate_ci_high: float
    p_value: float
    n_effective: int


def _design_controls(
    features: np.ndarray,
    masks: np.ndarray,
    reliability_weights: np.ndarray,
    *,
    target_domain_index: int,
    score_index: int,
    category_codes: np.ndarray | None,
) -> np.ndarray:
    """Build the control matrix W for one domain's DML fit.

    W consists of:
      - scores from every OTHER domain
      - reliability weights of every OTHER domain
      - per-domain mask indicators
      - one-hot category code (if supplied)
    """
    n, n_domains, _ = features.shape
    columns: list[np.ndarray] = []

    other_domains = [d for d in range(n_domains) if d != target_domain_index]
    for d in other_domains:
        columns.append(features[:, d, score_index].astype(np.float64).reshape(-1, 1))
        columns.append(reliability_weights[:, d].astype(np.float64).reshape(-1, 1))
    for d in range(n_domains):
        columns.append((~masks[:, d]).astype(np.float64).reshape(-1, 1))

    if category_codes is not None and category_codes.size:
        unique = sorted(set(category_codes.tolist()))
        for code in unique:
            columns.append((category_codes == code).astype(np.float64).reshape(-1, 1))

    if not columns:
        return np.zeros((n, 0), dtype=np.float64)
    return np.hstack(columns)


def _cross_fitted_residuals(
    y: np.ndarray,
    x: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
    learner_factory,
) -> np.ndarray:
    """Return out-of-fold residuals y - g_hat(x) using cross-fitting."""
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] == 0:
        return np.zeros_like(y)
    if x.shape[1] == 0:
        return y - float(np.mean(y))

    residuals = np.zeros_like(y)
    splitter = KFold(n_splits=max(2, min(n_splits, len(y))), shuffle=True, random_state=random_state)
    for train_idx, test_idx in splitter.split(x):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = learner_factory()
        try:
            model.fit(x[train_idx], y[train_idx])
            preds = model.predict(x[test_idx])
        except (ValueError, RuntimeError):
            preds = np.full(len(test_idx), float(np.mean(y[train_idx])))
        residuals[test_idx] = y[test_idx] - preds
    return residuals


def estimate_domain_causal_effect(
    features: np.ndarray,
    masks: np.ndarray,
    reliability_weights: np.ndarray,
    predictions: np.ndarray,
    *,
    domain_index: int,
    domain_name: str,
    score_index: int = 0,
    n_splits: int = 5,
    random_state: int = 42,
    category_codes: np.ndarray | None = None,
) -> DomainCausalEffect:
    """Estimate ATE of domain d's reliability on prediction via DML."""
    n_samples = features.shape[0]
    if n_samples < 2 * n_splits:
        return DomainCausalEffect(
            domain=domain_name,
            ate=float("nan"),
            ate_std_error=float("nan"),
            ate_ci_low=float("nan"),
            ate_ci_high=float("nan"),
            p_value=float("nan"),
            n_effective=int(n_samples),
        )

    available_mask = ~masks[:, domain_index]
    if available_mask.sum() < 2 * n_splits:
        return DomainCausalEffect(
            domain=domain_name,
            ate=float("nan"),
            ate_std_error=float("nan"),
            ate_ci_low=float("nan"),
            ate_ci_high=float("nan"),
            p_value=float("nan"),
            n_effective=int(available_mask.sum()),
        )

    y = predictions[available_mask].astype(np.float64).ravel()
    t = reliability_weights[available_mask, domain_index].astype(np.float64).ravel()
    controls = _design_controls(
        features[available_mask],
        masks[available_mask],
        reliability_weights[available_mask],
        target_domain_index=domain_index,
        score_index=score_index,
        category_codes=(category_codes[available_mask] if category_codes is not None else None),
    )

    def learner():
        return GradientBoostingRegressor(
            n_estimators=80, max_depth=3, random_state=random_state, learning_rate=0.05
        )

    y_residuals = _cross_fitted_residuals(
        y, controls, n_splits=n_splits, random_state=random_state, learner_factory=learner
    )
    t_residuals = _cross_fitted_residuals(
        t, controls, n_splits=n_splits, random_state=random_state, learner_factory=learner
    )

    t_norm_sq = float(np.sum(t_residuals ** 2))
    if t_norm_sq < 1e-12:
        return DomainCausalEffect(
            domain=domain_name,
            ate=0.0,
            ate_std_error=float("nan"),
            ate_ci_low=float("nan"),
            ate_ci_high=float("nan"),
            p_value=float("nan"),
            n_effective=int(len(y)),
        )

    theta = float(np.sum(t_residuals * y_residuals) / t_norm_sq)
    fitted = theta * t_residuals
    score = (y_residuals - fitted) * t_residuals
    asymptotic_variance = float(np.mean(score ** 2)) / (t_norm_sq / len(y)) ** 2
    std_error = float(math.sqrt(max(asymptotic_variance, 0.0) / len(y)))
    z = 1.959963984540054  # 95% normal quantile
    ci_low = theta - z * std_error
    ci_high = theta + z * std_error
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(theta) / (std_error * math.sqrt(2.0)) if std_error > 0 else float("inf"))))
    p_value = float(min(1.0, max(0.0, p_value)))

    return DomainCausalEffect(
        domain=domain_name,
        ate=theta,
        ate_std_error=std_error,
        ate_ci_low=ci_low,
        ate_ci_high=ci_high,
        p_value=p_value,
        n_effective=int(len(y)),
    )


def estimate_all_domain_effects(
    features: np.ndarray,
    masks: np.ndarray,
    reliability_weights: np.ndarray,
    predictions: np.ndarray,
    domain_order: List[str],
    *,
    score_index: int = 0,
    n_splits: int = 5,
    random_state: int = 42,
    category_codes: np.ndarray | None = None,
) -> list[DomainCausalEffect]:
    """Estimate per-domain ATE for every domain in ``domain_order``."""
    results: list[DomainCausalEffect] = []
    for d, name in enumerate(domain_order):
        results.append(
            estimate_domain_causal_effect(
                features,
                masks,
                reliability_weights,
                predictions,
                domain_index=d,
                domain_name=name,
                score_index=score_index,
                n_splits=n_splits,
                random_state=random_state,
                category_codes=category_codes,
            )
        )
    return results


def estimate_per_sample_cate(
    features: np.ndarray,
    masks: np.ndarray,
    reliability_weights: np.ndarray,
    predictions: np.ndarray,
    *,
    domain_index: int,
    score_index: int = 0,
    n_splits: int = 5,
    random_state: int = 42,
    category_codes: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-sample CATE (heterogeneous treatment effect) for domain d.

    Uses the DML residual-on-residual fit per sample by interacting the
    treatment residual with the control vector via Ridge regression.
    """
    available_mask = ~masks[:, domain_index]
    n_available = int(available_mask.sum())
    if n_available < 4 * n_splits:
        return np.full(features.shape[0], float("nan"), dtype=np.float64)

    y = predictions[available_mask].astype(np.float64).ravel()
    t = reliability_weights[available_mask, domain_index].astype(np.float64).ravel()
    controls = _design_controls(
        features[available_mask],
        masks[available_mask],
        reliability_weights[available_mask],
        target_domain_index=domain_index,
        score_index=score_index,
        category_codes=(category_codes[available_mask] if category_codes is not None else None),
    )

    def learner():
        return GradientBoostingRegressor(
            n_estimators=80, max_depth=3, random_state=random_state, learning_rate=0.05
        )

    y_res = _cross_fitted_residuals(y, controls, n_splits=n_splits, random_state=random_state, learner_factory=learner)
    t_res = _cross_fitted_residuals(t, controls, n_splits=n_splits, random_state=random_state, learner_factory=learner)

    interaction = t_res[:, None] * controls if controls.size else t_res[:, None]
    design = np.hstack([t_res[:, None], interaction])

    model = Ridge(alpha=1.0, random_state=random_state)
    try:
        model.fit(design, y_res)
        cate_available = model.coef_[0] + (interaction @ model.coef_[1:] if interaction.shape[1] > 0 else np.zeros_like(t_res))
    except (ValueError, RuntimeError):
        cate_available = np.full(n_available, float("nan"), dtype=np.float64)

    cate_full = np.full(features.shape[0], float("nan"), dtype=np.float64)
    cate_full[available_mask] = cate_available
    return cate_full


@dataclass
class InterventionalEffect:
    """Direct counterfactual ATE of intervening on per-domain reliability."""
    domain: str
    ate: float
    ate_std_error: float
    ate_ci_low: float
    ate_ci_high: float
    n_samples: int


def estimate_interventional_ate(
    predict_fn,
    reliability_weights: np.ndarray,
    *,
    domain_index: int,
    domain_name: str,
    n_bootstrap: int = 200,
    random_state: int = 42,
    intervention: str = "population_mean",
) -> InterventionalEffect:
    """Counterfactual ATE: do(r_d <- intervention) versus the observed r_d.

    The intervention assigns every sample's domain-d reliability to a single
    target value, then re-runs ``predict_fn(reliability_weights)`` to obtain
    counterfactual predictions. The ATE is the per-sample mean shift, with a
    bootstrap confidence interval.

    Parameters
    ----------
    predict_fn : callable(np.ndarray) -> np.ndarray
        Closure that maps a [N, D] reliability vector to [N] predictions.
        Must reflect the deployed gate/fusion pipeline.
    reliability_weights : [N, D] observed reliability vector.
    intervention : "population_mean" (default), "zero", or "one".
    """
    n_samples, n_domains = reliability_weights.shape
    baseline_probs = np.asarray(predict_fn(reliability_weights), dtype=np.float64)

    cf_weights = reliability_weights.copy()
    if intervention == "population_mean":
        # Intervening on the mean is degenerate when the estimator returns
        # batch-level scalars (variance = 0). In that case, push r_d half-way
        # toward the neutral value 0.5 to construct a meaningful contrast
        # while preserving the "set to a domain-independent target" semantics.
        col = reliability_weights[:, domain_index]
        if float(col.std()) < 1e-9:
            cf_weights[:, domain_index] = 0.5
        else:
            cf_weights[:, domain_index] = float(col.mean())
    elif intervention == "zero":
        cf_weights[:, domain_index] = 0.0
    elif intervention == "one":
        cf_weights[:, domain_index] = 1.0
    elif intervention == "neutral":
        cf_weights[:, domain_index] = 0.5
    else:
        raise ValueError(f"Unknown intervention: {intervention}")
    cf_probs = np.asarray(predict_fn(cf_weights), dtype=np.float64)

    deltas = cf_probs - baseline_probs
    ate = float(np.mean(deltas))

    rng = np.random.default_rng(random_state)
    if n_samples > 0 and n_bootstrap > 0:
        boot_ates = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            sample_idx = rng.integers(0, n_samples, size=n_samples)
            boot_ates[b] = float(np.mean(deltas[sample_idx]))
        ci_low = float(np.percentile(boot_ates, 2.5))
        ci_high = float(np.percentile(boot_ates, 97.5))
        std_err = float(boot_ates.std(ddof=0))
    else:
        ci_low = ci_high = float("nan")
        std_err = float("nan")

    return InterventionalEffect(
        domain=domain_name,
        ate=ate,
        ate_std_error=std_err,
        ate_ci_low=ci_low,
        ate_ci_high=ci_high,
        n_samples=int(n_samples),
    )


def estimate_all_interventional_ates(
    predict_fn,
    reliability_weights: np.ndarray,
    domain_order: List[str],
    *,
    n_bootstrap: int = 200,
    random_state: int = 42,
    intervention: str = "population_mean",
) -> list[InterventionalEffect]:
    """Run ``estimate_interventional_ate`` for every domain in ``domain_order``."""
    return [
        estimate_interventional_ate(
            predict_fn,
            reliability_weights,
            domain_index=d,
            domain_name=name,
            n_bootstrap=n_bootstrap,
            random_state=random_state + d,
            intervention=intervention,
        )
        for d, name in enumerate(domain_order)
    ]


__all__ = [
    "DomainCausalEffect",
    "InterventionalEffect",
    "estimate_domain_causal_effect",
    "estimate_all_domain_effects",
    "estimate_per_sample_cate",
    "estimate_interventional_ate",
    "estimate_all_interventional_ates",
]
