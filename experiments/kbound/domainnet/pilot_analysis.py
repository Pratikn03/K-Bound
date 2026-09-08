"""Post-seal descriptive pilot analysis, never an outcome-access authority.

The caller must verify all check actions/prediction seals before supplying labels.
These fixed development criteria do not establish independent-environment coverage,
population safety, or eligibility for the five-model confirmatory cohort.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from kga.certificate import Certificate
from kga.evaluation import evaluate_actions
from kga.policy import decide


@dataclass(frozen=True)
class ScoredCell:
    cell_id: str
    labels: tuple[int, ...]
    frozen_predictions: tuple[int, ...]
    candidate_predictions: tuple[int, ...]
    delta_hat: float
    epsilon: float | None
    action: str
    no_radius_action: str


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite nonboolean number")
    return float(value)


def _direction(delta_hat: float, epsilon: float | None) -> str:
    if epsilon is None:
        return "ABSTAIN"
    return decide(Certificate(delta_hat=delta_hat, epsilon=epsilon, method="conformal", alpha=0.1, n=1)).value


def _validate(row: ScoredCell) -> None:
    if (
        not isinstance(row, ScoredCell)
        or not isinstance(row.cell_id, str)
        or not re.fullmatch(r"DEV_check:(0|[1-9][0-9]*)", row.cell_id)
    ):
        raise ValueError("only uniquely identified DEV_check cells are permitted")
    for vector in (row.labels, row.frozen_predictions, row.candidate_predictions):
        if type(vector) is not tuple or any(type(x) is not int or not 0 <= x < 126 for x in vector):
            raise ValueError("labels and predictions must be immutable integer tuples in 0..125")
        if len(vector) != len(row.labels) or len(vector) < 128:
            raise ValueError("each complete E window requires at least 128 paired images")
    delta_hat = _number(row.delta_hat, "delta_hat")
    epsilon = None if row.epsilon is None else _number(row.epsilon, "epsilon")
    if epsilon is not None and epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if row.action != _direction(delta_hat, epsilon):
        raise ValueError("sealed action disagrees with the declared strict interval")
    if row.no_radius_action != _direction(delta_hat, 0.0):
        raise ValueError("no-radius action disagrees with the same point predictor")


def summarize_check(rows: list[ScoredCell]) -> dict[str, Any]:
    """Validate and summarize original sealed actions; no refitting or tuning."""
    if type(rows) is not list or not rows:
        raise ValueError("nonempty scored-cell list required")
    for row in rows:
        _validate(row)
    if len({row.cell_id for row in rows}) != len(rows):
        raise ValueError("duplicate cell IDs")
    deltas = [
        (
            sum(p == y for p, y in zip(r.candidate_predictions, r.labels, strict=True))
            - sum(p == y for p, y in zip(r.frozen_predictions, r.labels, strict=True))
        )
        / len(r.labels)
        for r in rows
    ]
    kga = evaluate_actions([r.action for r in rows], deltas)
    no_radius = evaluate_actions([r.no_radius_action for r in rows], deltas)
    n = len(rows)
    helpful = sum(d > 0 for d in deltas)
    harmful = sum(d < 0 for d in deltas)
    headroom = min(kga.summary.mean_always_adapt_regret, kga.summary.mean_always_freeze_regret)
    loo_adapt, loo_freeze = [], []
    if n > 1:
        for i in range(n):
            kept = kga.records[:i] + kga.records[i + 1 :]
            loo_adapt.append(math.fsum(r.always_adapt_regret - r.regret for r in kept) / (n - 1))
            loo_freeze.append(math.fsum(r.always_freeze_regret - r.regret for r in kept) / (n - 1))
    inclusion = [
        None if r.epsilon is None else abs(r.delta_hat - d) <= r.epsilon for r, d in zip(rows, deltas, strict=True)
    ]
    unavailable = sum(x is None for x in inclusion)
    counts = [0] * 126
    correct = {name: [0] * 126 for name in ("frozen", "candidate", "served", "no_radius")}
    image_weighted_regret = {"kga": 0.0, "always_adapt": 0.0, "always_freeze": 0.0, "no_radius": 0.0}
    for row, contribution, ablation in zip(rows, kga.records, no_radius.records, strict=True):
        for y, f, a in zip(row.labels, row.frozen_predictions, row.candidate_predictions, strict=True):
            counts[y] += 1
            correct["frozen"][y] += f == y
            correct["candidate"][y] += a == y
            correct["served"][y] += (a if row.action == "ADAPT" else f) == y
            correct["no_radius"][y] += (a if row.no_radius_action == "ADAPT" else f) == y
        for name, value in (
            ("kga", contribution.regret),
            ("always_adapt", contribution.always_adapt_regret),
            ("always_freeze", contribution.always_freeze_regret),
            ("no_radius", ablation.regret),
        ):
            image_weighted_regret[name] += len(row.labels) * value
    total = sum(counts)
    present = [label for label, count in enumerate(counts) if count]
    opportunity = helpful / n >= 0.15 and harmful / n >= 0.15 and headroom >= 0.0025
    checks = {
        "complete_check_geometry": n >= 20,
        "opportunity": opportunity,
        "adapt_exposure": kga.summary.adapt_rate >= 0.1,
        "freeze_exposure": kga.summary.freeze_rate >= 0.1,
        "commitment_exposure": kga.summary.commitment_rate >= 0.3,
        "observed_false_adapt": kga.summary.false_adapt_rate_all <= 0.05,
        "observed_false_freeze": kga.summary.false_freeze_rate_all <= 0.05,
        "mean_gap_vs_adapt": kga.summary.always_adapt_minus_kga_regret > 0,
        "mean_gap_vs_freeze": kga.summary.always_freeze_minus_kga_regret > 0,
        "leave_one_out_vs_adapt": bool(loo_adapt) and min(loo_adapt) > 0,
        "leave_one_out_vs_freeze": bool(loo_freeze) and min(loo_freeze) > 0,
    }
    status = (
        "INCONCLUSIVE_DATA_GEOMETRY"
        if n < 20
        else "NO_DEMONSTRATED_OPPORTUNITY"
        if not opportunity
        else "GO_FURTHER_DEVELOPMENT"
        if all(checks.values())
        else "INCONCLUSIVE_GATE"
    )
    return {
        "schema": "kbound-domainnet-dev-pilot-analysis/1",
        "status": status,
        "description": "Descriptive one-checkpoint painting DEV screen, not independent-environment inference.",
        "eligible_for_confirmatory": False,
        "population_certificate": False,
        "checks": checks,
        "n_cells": n,
        "helpful_count": helpful,
        "harmful_count": harmful,
        "tie_count": n - helpful - harmful,
        "oracle_headroom": headroom,
        "cell_ids": [r.cell_id for r in rows],
        "delta_cells": deltas,
        "sample_counts": [len(r.labels) for r in rows],
        "cell_interval_inclusion": None if unavailable else sum(x is True for x in inclusion) / n,
        "interval_inclusion_by_cell": inclusion,
        "unavailable_intervals": unavailable,
        "kga": asdict(kga.summary),
        "no_radius": asdict(no_radius.summary),
        "minimum_leave_one_out_gaps": {
            "vs_adapt": min(loo_adapt) if loo_adapt else None,
            "vs_freeze": min(loo_freeze) if loo_freeze else None,
        },
        "sample_weighted": {
            "n_images": total,
            **{name + "_accuracy": sum(values) / total for name, values in correct.items()},
            "regret": {name: value / total for name, value in image_weighted_regret.items()},
        },
        "macro_recall": {
            "present_classes": len(present),
            "absent_classes": 126 - len(present),
            "counts_by_class": counts,
            **{
                name: math.fsum(values[label] / counts[label] for label in present) / len(present)
                for name, values in correct.items()
            },
        },
    }


def source_readiness(epochs: list[dict[str, Any]]) -> dict[str, Any]:
    """Readiness from the complete final-epoch trace, never best-epoch selection."""
    if type(epochs) is not list or len(epochs) != 20:
        raise ValueError("complete 20-epoch trace required")
    losses = []
    for i, row in enumerate(epochs, 1):
        for key, expected in (("epoch", i), ("processed_count", 16811), ("source_monitor_count", 1892)):
            if type(row.get(key)) is not int or row[key] != expected:
                raise ValueError("source epoch identity/count mismatch")
        loss = _number(row["mean_loss"], "mean_loss")
        accuracy = _number(row["source_monitor_accuracy"], "source_monitor_accuracy")
        if loss < 0 or not 0 <= accuracy <= 1:
            raise ValueError("source loss or accuracy out of range")
        losses.append(loss)
    first = math.fsum(losses[:5]) / 5
    last = math.fsum(losses[-5:]) / 5
    ready = accuracy >= 0.05 and last < first
    return {
        "ready": ready,
        "status": "SOURCE_READY_FOR_DEV_PILOT" if ready else "INCONCLUSIVE_SOURCE_LEARNING",
        "final_monitor_accuracy": accuracy,
        "first_five_mean_loss": first,
        "last_five_mean_loss": last,
        "selection": "final_epoch_only",
        "eligible_for_confirmatory": False,
    }
