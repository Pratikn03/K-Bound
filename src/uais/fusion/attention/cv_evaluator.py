"""Unified 5-fold cross-validation evaluator for ALL baseline families.

Addresses reviewer comments:
  - "Add 5-fold cross-validation"
  - "Confidence intervals for reported metrics"
  - "Statistical comparison tests (e.g., DeLong test)"
  - "Different algorithm families are compared without a fully unified framework.
     → Dimensionality reduction must be applied consistently, and all models
       must be evaluated under the same conditions."

Every supervised + unsupervised baseline goes through the SAME
StratifiedKFold split, the SAME dimensionality reduction (if any), and the
SAME metric pipeline.  This gives a leakage-free, paired comparison that the
DeLong test can be applied to.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from uais.fusion.attention.dim_reduction import DimReducer, NoOpReducer, make_reducer
from uais.fusion.attention.leakage_guard import (
    assert_no_oversampling_in_test,
    check_train_test_contamination,
    flag_suspicious_metrics,
)
from uais.utils.metrics import (
    aggregate_cv_metrics,
    bootstrap_metric_ci,
    classification_metrics,
)

try:
    from uais.utils.stats import delong_roc_test
except ImportError:  # pragma: no cover
    delong_roc_test = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reducer-aware feature flattening
# ---------------------------------------------------------------------------

def _flatten_with_mask(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    n, d, f = features.shape
    flat = features.copy().astype(np.float32)
    for di in range(d):
        flat[:, di, :][masks[:, di]] = 0.5
    flat = flat.reshape(n, d * f)
    indicators = masks.astype(np.float32)
    return np.concatenate([flat, indicators], axis=1)


def _apply_reducer_3d(
    features: np.ndarray,
    masks: np.ndarray,
    reducer: DimReducer,
) -> np.ndarray:
    """Apply a DR transform to [N, D, F] features after flattening.

    Returns [N, K] reduced features.  Mask information is preserved by the
    flattened input via missingness indicators.
    """
    return reducer.transform(_flatten_with_mask(features, masks))


# ---------------------------------------------------------------------------
# Model factory protocol — caller supplies fit/predict wrappers
# ---------------------------------------------------------------------------

@dataclass
class BaselineSpec:
    """Specification for one baseline in the unified CV runner.

    Fields
    ------
    name           : key used in the results dict
    make           : zero-arg callable returning a fresh model instance
    fit_signature  : "fit(features, masks, labels)" or
                     "fit(features_2d, labels)" — affects how features
                     are passed; the DR transform is applied transparently
                     either way.
    is_unsupervised: if True, fit() filters to normal-only training data
                     and labels are not passed
    needs_3d       : if True, model expects [N, D, F]+masks API; otherwise
                     model receives a flat [N, K] reduced matrix and labels
    predict_method : "predict_proba" or "score_samples"
    """
    name: str
    make: Callable[[], Any]
    needs_3d: bool = True
    is_unsupervised: bool = False
    predict_method: str = "predict_proba"


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------

def _fit_predict_one(
    spec: BaselineSpec,
    features_3d: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    reducer: DimReducer,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Fit a single model on the training fold (with DR fit on train only)
    and return (test_labels, test_probs, hyperparameters)."""
    # Fit reducer on training only (no leakage)
    if not isinstance(reducer, NoOpReducer):
        reducer.fit(_flatten_with_mask(features_3d[train_idx], masks[train_idx]))

    model = spec.make()

    if spec.needs_3d:
        # Multi-modal model — keeps the [N, D, F] view + masks
        if spec.is_unsupervised:
            model.fit(features_3d[train_idx], masks[train_idx], labels[train_idx])
        else:
            model.fit(features_3d[train_idx], masks[train_idx], labels[train_idx])
        probs = getattr(model, spec.predict_method)(
            features_3d[test_idx], masks[test_idx]
        )
    else:
        # Flat model — receives reduced 2D matrix
        X_tr = _apply_reducer_3d(features_3d[train_idx], masks[train_idx], reducer)
        X_te = _apply_reducer_3d(features_3d[test_idx], masks[test_idx], reducer)
        if spec.is_unsupervised:
            # Filter to normal-only training samples
            normal_mask = labels[train_idx] == 0
            model.fit(X_tr[normal_mask])
        else:
            model.fit(X_tr, labels[train_idx])

        method = getattr(model, spec.predict_method)
        out = method(X_te)
        # sklearn predict_proba → [N, 2]; score_samples → [N]
        if out.ndim == 2 and out.shape[1] == 2:
            probs = out[:, 1]
        else:
            probs = out

    hp: Dict = {}
    if hasattr(model, "get_hyperparameters"):
        try:
            hp = model.get_hyperparameters()
        except Exception:
            pass
    if hasattr(model, "get_params") and not hp:
        try:
            hp = model.get_params()
        except Exception:
            pass
    return labels[test_idx], np.asarray(probs).astype(np.float32), hp


# ---------------------------------------------------------------------------
# Main CV runner
# ---------------------------------------------------------------------------

@dataclass
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    bootstrap_resamples: int = 1000
    bootstrap_alpha: float = 0.05
    reducer_name: str = "none"
    reducer_kwargs: Dict = field(default_factory=dict)
    leakage_warn_auc: float = 0.99
    leakage_warn_f1: float = 0.99


def cross_validate_baselines(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    specs: List[BaselineSpec],
    config: Optional[CVConfig] = None,
) -> Dict[str, Any]:
    """Run StratifiedKFold across all specs with a shared DR transform.

    Returns
    -------
    {
        "config": ...,
        "per_baseline": {
            name: {
                "per_fold": [{metric: value, ...}, ...],
                "aggregate": {"roc_auc": {"mean":..,"std":..,"lo":..,"hi":..,"n":k}, ...},
                "bootstrap_ci": {"roc_auc": {"mean":..,"lo":..,"hi":..,"std":..}, ...},
                "hyperparameters": {...},  # from first fold
                "leakage_warnings": [...],
            },
        },
        "delong_pvalues": {
            (baseline_a, baseline_b): float,
            ...
        },
        "contamination_check": [{n_duplicate_rows: ..., fraction_test_in_train: ...}, ...],
        "test_oversampling_check": "ok" | "<error>",
    }
    """
    cfg = config or CVConfig()
    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=cfg.random_state,
    )

    # Per-baseline accumulators
    per_baseline: Dict[str, Dict] = {
        s.name: {
            "per_fold": [], "hyperparameters": None,
            "y_true_concat": [], "y_prob_concat": [],
            "leakage_warnings": [],
        }
        for s in specs
    }
    contamination_reports: List[Dict] = []
    test_oversampling_status = "ok"

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"=== Fold {fold_i + 1}/{cfg.n_splits} ===")

        # Leakage check 1: no train rows in test
        flat_tr = _flatten_with_mask(features[train_idx], masks[train_idx])
        flat_te = _flatten_with_mask(features[test_idx], masks[test_idx])
        contamination_reports.append(check_train_test_contamination(flat_tr, flat_te))

        # Leakage check 2: no duplicate rows in test set itself
        try:
            assert_no_oversampling_in_test(flat_te)
        except AssertionError as exc:
            test_oversampling_status = str(exc)

        # Build a fresh reducer per fold (so DR is fit on training-only data)
        reducer = make_reducer(cfg.reducer_name, **cfg.reducer_kwargs)

        for spec in specs:
            try:
                y_true, y_prob, hp = _fit_predict_one(
                    spec, features, masks, labels,
                    train_idx, test_idx, reducer,
                )
                m = classification_metrics(y_true, y_prob)
                m["fold"] = fold_i + 1
                per_baseline[spec.name]["per_fold"].append(m)
                per_baseline[spec.name]["y_true_concat"].append(y_true)
                per_baseline[spec.name]["y_prob_concat"].append(y_prob)
                if per_baseline[spec.name]["hyperparameters"] is None:
                    per_baseline[spec.name]["hyperparameters"] = hp

                # Per-fold leakage flag
                warns = flag_suspicious_metrics(
                    m, auc_threshold=cfg.leakage_warn_auc,
                    f1_threshold=cfg.leakage_warn_f1,
                )
                if warns:
                    per_baseline[spec.name]["leakage_warnings"].append(
                        {"fold": fold_i + 1, "warnings": warns}
                    )
            except Exception as exc:
                logger.warning(f"{spec.name} failed on fold {fold_i + 1}: {exc}")
                per_baseline[spec.name]["per_fold"].append({
                    "fold": fold_i + 1, "error": f"{type(exc).__name__}: {exc}",
                })

    # Aggregate across folds + compute bootstrap CIs on pooled predictions
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    pooled_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, data in per_baseline.items():
        data["aggregate"] = aggregate_cv_metrics(data["per_fold"])
        if data["y_true_concat"]:
            y_true = np.concatenate(data["y_true_concat"])
            y_prob = np.concatenate(data["y_prob_concat"])
            pooled_predictions[name] = (y_true, y_prob)
            data["bootstrap_ci"] = {
                "roc_auc": bootstrap_metric_ci(
                    y_true, y_prob, roc_auc_score,
                    n_resamples=cfg.bootstrap_resamples,
                    alpha=cfg.bootstrap_alpha,
                    random_state=cfg.random_state,
                ),
                "pr_auc": bootstrap_metric_ci(
                    y_true, y_prob, average_precision_score,
                    n_resamples=cfg.bootstrap_resamples,
                    alpha=cfg.bootstrap_alpha,
                    random_state=cfg.random_state,
                ),
                "f1": bootstrap_metric_ci(
                    y_true, (y_prob >= 0.5).astype(int),
                    lambda yt, yp: f1_score(yt, yp, zero_division=0),
                    n_resamples=cfg.bootstrap_resamples,
                    alpha=cfg.bootstrap_alpha,
                    random_state=cfg.random_state,
                ),
            }
        data.pop("y_true_concat", None)
        data.pop("y_prob_concat", None)

    # Pairwise DeLong on pooled across-fold predictions
    delong_pairs = pairwise_delong_from_predictions(pooled_predictions)

    contamination_summary = {
        "per_fold": contamination_reports,
        "max_duplicates": max((r["n_duplicate_rows"] for r in contamination_reports),
                              default=0),
    }

    return {
        "config": {
            "n_splits": cfg.n_splits,
            "random_state": cfg.random_state,
            "reducer": cfg.reducer_name,
            "reducer_kwargs": cfg.reducer_kwargs,
            "bootstrap_resamples": cfg.bootstrap_resamples,
            "leakage_warn_auc": cfg.leakage_warn_auc,
        },
        "per_baseline": per_baseline,
        "pairwise_delong_pvalues": delong_pairs,
        "contamination_check": contamination_summary,
        "test_oversampling_check": test_oversampling_status,
    }


# ---------------------------------------------------------------------------
# Pairwise DeLong wrapper
# ---------------------------------------------------------------------------

def pairwise_delong_from_predictions(
    predictions_by_name: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """Compute DeLong p-values for every pair of (y_true, y_prob) arrays.

    Args
    ----
    predictions_by_name : {name: (y_true, y_prob)} for pooled CV predictions.

    Returns a flat dict keyed by "name_a__vs__name_b".
    """
    if delong_roc_test is None:
        return {}
    out: Dict[str, float] = {}
    names = list(predictions_by_name.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            y_true_a, y_prob_a = predictions_by_name[a]
            y_true_b, y_prob_b = predictions_by_name[b]
            if not np.array_equal(y_true_a, y_true_b):
                continue
            try:
                p = float(delong_roc_test(y_true_a, y_prob_a, y_prob_b))
                out[f"{a}__vs__{b}"] = p
            except Exception:
                continue
    return out


__all__ = [
    "BaselineSpec", "CVConfig",
    "cross_validate_baselines",
    "pairwise_delong_from_predictions",
]
