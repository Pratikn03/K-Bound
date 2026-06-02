"""Shared metric utilities for fraud / anomaly models."""

from __future__ import annotations

import numpy as np
from sklearn import metrics


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # Clip so a probability of exactly 1.0 (digitize -> bin n_bins) folds into the
    # top bin instead of being silently dropped (audit fix: biased ECE on saturated
    # scores, which feed the reliability weight).
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        bin_acc = np.mean(y_true[mask])
        bin_conf = np.mean(y_prob[mask])
        ece += np.abs(bin_acc - bin_conf) * np.mean(mask)
    return float(ece)


def detection_rate_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.01) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    negatives = y_prob[y_true == 0]
    if negatives.size == 0:
        return {"tpr_at_fpr": float("nan"), "threshold_at_fpr": float("nan"), "fpr": float("nan")}
    threshold = float(np.quantile(negatives, 1 - target_fpr))
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tpr_at_fpr": float(tpr), "threshold_at_fpr": float(threshold), "fpr": float(fpr)}


def _compute_from_pred_and_prob(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=float)
    if np.any((y_true != 0) & (y_true != 1)):
        y_true = np.where(y_true > 0, 1, 0)
    scores: dict[str, float] = {}
    try:
        scores["roc_auc"] = metrics.roc_auc_score(y_true, y_prob)
    except ValueError:
        scores["roc_auc"] = float("nan")
    try:
        scores["pr_auc"] = metrics.average_precision_score(y_true, y_prob)
    except ValueError:
        scores["pr_auc"] = float("nan")

    scores["f1"] = metrics.f1_score(y_true, y_pred, zero_division=0)
    scores["precision"] = metrics.precision_score(y_true, y_pred, zero_division=0)
    scores["recall"] = metrics.recall_score(y_true, y_pred, zero_division=0)
    scores["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    scores["balanced_accuracy"] = metrics.balanced_accuracy_score(y_true, y_pred)
    try:
        scores["mcc"] = metrics.matthews_corrcoef(y_true, y_pred)
    except ValueError:
        scores["mcc"] = float("nan")
    try:
        scores["cohen_kappa"] = metrics.cohen_kappa_score(y_true, y_pred)
    except ValueError:
        scores["cohen_kappa"] = float("nan")
    scores["brier"] = brier_score(y_true, y_prob)
    scores["ece"] = expected_calibration_error(y_true, y_prob)

    # Confusion matrix entries — addresses reviewer comment on imbalanced metrics
    try:
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        scores["tp"] = int(tp)
        scores["fp"] = int(fp)
        scores["tn"] = int(tn)
        scores["fn"] = int(fn)
    except ValueError:
        scores["tp"] = scores["fp"] = scores["tn"] = scores["fn"] = 0

    scores.update(detection_rate_at_fpr(y_true, y_prob, target_fpr=0.01))
    return scores


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    scores = _compute_from_pred_and_prob(y_true, y_pred, y_prob)
    scores["decision_threshold"] = float(threshold)
    return scores


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix (2x2 for binary classification)."""
    return metrics.confusion_matrix(y_true, y_pred)


def classification_metrics(
    y_true: np.ndarray,
    y_pred_or_prob: np.ndarray,
    y_prob: np.ndarray | float | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Backward-compatible wrapper for mixed call signatures."""
    y_true = np.asarray(y_true)
    if y_prob is None:
        return compute_classification_metrics(y_true, y_pred_or_prob, threshold=threshold)
    if np.isscalar(y_prob):
        return compute_classification_metrics(y_true, y_pred_or_prob, threshold=float(y_prob))
    y_pred = np.asarray(y_pred_or_prob)
    y_prob_arr = np.asarray(y_prob)
    return _compute_from_pred_and_prob(y_true, y_pred, y_prob_arr)


def anomaly_metrics(y_true, scores, threshold=None, contamination: float = 0.05):
    # Simple anomaly metric: threshold by quantile if none provided
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if threshold is None:
        threshold = np.quantile(scores, 1 - contamination)
    preds = (scores >= threshold).astype(int)
    out = classification_metrics(y_true, preds, threshold=0.5)
    out["threshold"] = float(threshold)
    return out


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find probability threshold that maximizes F1 on given labels."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    finite = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[finite]
    y_prob = y_prob[finite]
    thresholds = np.unique(y_prob)
    if len(thresholds) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        f1 = metrics.f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


def select_decision_threshold(y_true: np.ndarray, y_prob: np.ndarray, strategy: str | None = "fixed_0p5") -> float:
    """Select a decision threshold from validation labels/probabilities only."""
    normalized = (strategy or "fixed_0p5").strip().lower()
    if normalized in {"fixed", "fixed_0p5", "fixed_0.5", "0.5"}:
        return 0.5
    if normalized in {"val_f1", "best_f1"}:
        return best_f1_threshold(y_true, y_prob)
    raise ValueError(f"Unknown decision threshold strategy: {strategy}")


def calibration_monitor_report(
    reference_y: np.ndarray,
    reference_prob: np.ndarray,
    current_y: np.ndarray,
    current_prob: np.ndarray,
    max_ece_delta: float = 0.05,
    max_brier_delta: float = 0.05,
    n_bins: int = 10,
) -> dict[str, object]:
    """Compare deployment calibration against a reference window.

    Returns a compact alert report suitable for deployment dashboards or
    runbooks. Labels may be delayed in real systems; when they are available,
    this function makes calibration drift explicit and auditable.
    """
    reference_y = np.asarray(reference_y, dtype=float)
    reference_prob = np.asarray(reference_prob, dtype=float)
    current_y = np.asarray(current_y, dtype=float)
    current_prob = np.asarray(current_prob, dtype=float)

    ref = {
        "ece": expected_calibration_error(reference_y, reference_prob, n_bins=n_bins),
        "brier": brier_score(reference_y, reference_prob),
    }
    cur = {
        "ece": expected_calibration_error(current_y, current_prob, n_bins=n_bins),
        "brier": brier_score(current_y, current_prob),
    }
    deltas = {
        "ece_delta": float(cur["ece"] - ref["ece"]),
        "brier_delta": float(cur["brier"] - ref["brier"]),
    }
    reasons = []
    if deltas["ece_delta"] > max_ece_delta:
        reasons.append("ece_delta")
    if deltas["brier_delta"] > max_brier_delta:
        reasons.append("brier_delta")
    return {
        "alert": bool(reasons),
        "reasons": reasons,
        "reference": ref,
        "current": cur,
        "deltas": deltas,
        "thresholds": {
            "max_ece_delta": float(max_ece_delta),
            "max_brier_delta": float(max_brier_delta),
        },
    }


def bounded_switching_certificate(
    static_loss: np.ndarray,
    reliability_loss: np.ndarray,
    fire_decisions: np.ndarray,
    margin_epsilon: float = 0.0,
) -> dict[str, float | bool]:
    """Empirical certificate for when a reliability switch is preferable.

    If the gate fires only on samples where the reliability path has positive
    average loss advantage over static attention by more than ``margin_epsilon``,
    and the realized gated policy loss is lower than static loss, the certificate
    is marked true. This is not a proof for new populations; it is the auditable
    finite-sample condition the formal switching-rule analysis can cite.
    """
    static_loss = np.asarray(static_loss, dtype=float)
    reliability_loss = np.asarray(reliability_loss, dtype=float)
    fire_decisions = np.asarray(fire_decisions, dtype=bool)
    if static_loss.shape != reliability_loss.shape or static_loss.shape != fire_decisions.shape:
        raise ValueError("static_loss, reliability_loss, and fire_decisions must have the same shape.")
    if static_loss.ndim != 1:
        raise ValueError("switching certificate inputs must be one-dimensional.")

    policy_loss_values = np.where(fire_decisions, reliability_loss, static_loss)
    fired = fire_decisions
    fired_advantage = float(np.mean(static_loss[fired] - reliability_loss[fired])) if fired.any() else 0.0
    static_mean = float(np.mean(static_loss))
    reliability_mean = float(np.mean(reliability_loss))
    policy_mean = float(np.mean(policy_loss_values))
    certified = bool(fired.any() and fired_advantage > margin_epsilon and policy_mean < static_mean)
    return {
        "certified": certified,
        "static_loss": static_mean,
        "reliability_loss": reliability_mean,
        "policy_loss": policy_mean,
        "fired_advantage": fired_advantage,
        "fire_rate": float(np.mean(fire_decisions)),
        "margin_epsilon": float(margin_epsilon),
    }


def reliability_degradation_auc(
    noise_levels: np.ndarray,
    roc_auc_values: np.ndarray,
) -> float:
    """Area under the performance-vs-drift curve (trapezoidal rule).

    Higher value means the model degrades more slowly as noise/drift increases.
    Used to compare CRAF vs static fusion under simulated domain shift.
    """
    noise_levels = np.asarray(noise_levels, dtype=float)
    roc_auc_values = np.asarray(roc_auc_values, dtype=float)
    valid = np.isfinite(roc_auc_values)
    if valid.sum() < 2:
        return float("nan")
    return float(np.trapezoid(roc_auc_values[valid], noise_levels[valid]))


# ---------------------------------------------------------------------------
# Statistical helpers for benchmark reporting (addresses reviewer comments
# on missing CIs, statistical comparisons, and leakage detection)
# ---------------------------------------------------------------------------


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap a 1-alpha confidence interval for any (y_true, y_score) metric.

    Returns {"mean", "lo", "hi", "std"}.  Use for ROC-AUC, PR-AUC, F1, etc.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            samples.append(float(metric_fn(y_true[idx], y_score[idx])))
        except Exception:
            continue
    if not samples:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "std": float("nan")}
    arr = np.asarray(samples)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(arr.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "std": float(arr.std()),
    }


def aggregate_cv_metrics(per_fold: list) -> dict[str, dict[str, float]]:
    """Aggregate per-fold metric dicts into mean/std/95% CI across folds.

    Input : list of dicts, e.g. [{"roc_auc": 0.81, "f1": 0.66, ...}, ...]
    Output: {"roc_auc": {"mean": ..., "std": ..., "lo": ..., "hi": ...}, ...}
    """
    if not per_fold:
        return {}
    keys = set().union(*per_fold)
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = []
        for fold in per_fold:
            v = fold.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                vals.append(fv)
        if not vals:
            out[k] = {"mean": float("nan"), "std": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
            continue
        arr = np.asarray(vals)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        # Normal approximation 95% CI for the mean across folds
        if len(arr) > 1:
            se = std / np.sqrt(len(arr))
            lo, hi = mean - 1.96 * se, mean + 1.96 * se
        else:
            lo = hi = mean
        out[k] = {"mean": mean, "std": std, "lo": float(lo), "hi": float(hi), "n": len(arr)}
    return out


def flag_suspicious_performance(
    metrics_dict: dict[str, float], auc_threshold: float = 0.99, f1_threshold: float = 0.99
) -> dict[str, bool]:
    """Detect suspiciously-perfect metric values that often indicate leakage.

    Addresses reviewer comment: 'Unrealistic AUC = 1.000 ... possible data leakage
    or contamination'.  Returns a dict of boolean flags for each metric exceeding
    the leakage-suspicion threshold.
    """
    flags = {}
    for key, thr in (
        ("roc_auc", auc_threshold),
        ("pr_auc", auc_threshold),
        ("f1", f1_threshold),
        ("accuracy", f1_threshold),
    ):
        v = metrics_dict.get(key, float("nan"))
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v >= thr:
            flags[f"{key}_suspicious"] = True
    return flags


__all__ = [
    "brier_score",
    "expected_calibration_error",
    "detection_rate_at_fpr",
    "compute_classification_metrics",
    "classification_metrics",
    "compute_confusion_matrix",
    "anomaly_metrics",
    "best_f1_threshold",
    "select_decision_threshold",
    "reliability_degradation_auc",
    "bootstrap_metric_ci",
    "aggregate_cv_metrics",
    "flag_suspicious_performance",
]
