"""Result aggregation helpers for multi-seed UAV experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_METRICS = ("roc_auc", "pr_auc", "f1", "ece", "brier", "accuracy")


def _finite_values(values: Iterable[Any]) -> list[float]:
    finite = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            finite.append(number)
    return finite


def summarize_values(values: Iterable[Any], alpha: float = 0.05) -> dict[str, Any]:
    """Summarize finite values with mean, std, and percentile CI."""
    finite = _finite_values(values)
    if not finite:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None, "n": 0}

    arr = np.asarray(finite, dtype=float)
    if len(arr) == 1:
        ci_low = ci_high = float(arr[0])
    else:
        ci_low = float(np.percentile(arr, 100 * alpha / 2))
        ci_high = float(np.percentile(arr, 100 * (1 - alpha / 2)))

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(len(arr)),
    }


def summarize_seed_metric_rows(
    rows: Sequence[dict[str, Any]],
    methods: Sequence[str],
    metrics: Sequence[str] = DEFAULT_METRICS,
    alpha: float = 0.05,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Summarize nested per-seed method metric rows.

    Input rows match ``table_1_clean_performance`` where each method contains a
    metric dictionary. Output is method -> metric -> summary.
    """
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for method in methods:
        method_summary = {}
        for metric in metrics:
            method_summary[metric] = summarize_values(
                row.get(method, {}).get(metric) for row in rows
            )
        summary[method] = method_summary
    return summary


def aggregate_stress_rows(
    rows: Sequence[dict[str, Any]],
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Group per-seed stress rows and flatten metric summaries for tables."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    output = []
    for key_tuple in sorted(grouped, key=lambda item: tuple(str(x) for x in item)):
        group_rows = grouped[key_tuple]
        out = {key: value for key, value in zip(group_keys, key_tuple)}
        seeds = {row.get("seed") for row in group_rows if row.get("seed") is not None}
        out["n_seeds"] = len(seeds) if seeds else len(group_rows)

        for metric in metric_keys:
            metric_summary = summarize_values((row.get(metric) for row in group_rows), alpha=alpha)
            out[metric] = metric_summary["mean"]
            out[f"{metric}_std"] = metric_summary["std"]
            out[f"{metric}_ci_low"] = metric_summary["ci_low"]
            out[f"{metric}_ci_high"] = metric_summary["ci_high"]

        if {"static_auc", "craf_auc"}.issubset(metric_keys):
            deltas = [
                float(row["craf_auc"]) - float(row["static_auc"])
                for row in group_rows
                if row.get("static_auc") is not None
                and row.get("craf_auc") is not None
                and np.isfinite(float(row["static_auc"]))
                and np.isfinite(float(row["craf_auc"]))
            ]
            delta_summary = summarize_values(deltas, alpha=alpha)
            out["delta_auc"] = delta_summary["mean"]
            out["delta_auc_std"] = delta_summary["std"]
            out["delta_auc_ci_low"] = delta_summary["ci_low"]
            out["delta_auc_ci_high"] = delta_summary["ci_high"]

        output.append(out)
    return output


__all__ = [
    "aggregate_stress_rows",
    "summarize_seed_metric_rows",
    "summarize_values",
]
