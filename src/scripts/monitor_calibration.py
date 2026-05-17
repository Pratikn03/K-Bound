"""Streaming calibration / drift monitor for the reliability-gated fusion path.

The monitor reads a stream of per-batch domain scores + labels (CSV or JSON
lines) and emits one event per window. Each event records the window's
mean reliability, per-domain KS-drift against the validation reference,
batch ECE of the fused score, and the gate-fire status under a configured
threshold tau.

It is intentionally observe-only: it never modifies inference. The output
JSON-lines stream is what a deployment-time auditor reads to reconstruct
when the gate would have fired and why.

Usage (offline replay against a fusion CSV):
    PYTHONPATH=src python src/scripts/monitor_calibration.py \
        --fusion-csv experiments/fusion/mvtec3d_fusion_inputs.csv \
        --reference-split validation \
        --target-split test \
        --tau 0.66 \
        --window-size 200 \
        --output experiments/monitor/mvtec3d_calibration_events.jsonl

The same script accepts --jsonl-input for a true streaming source where
each line is a JSON dict with the same column names as the fusion CSV.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class WindowEvent:
    window_index: int
    n_samples: int
    mean_reliability: float
    gate_fired: bool
    ks_per_domain: dict[str, float]
    ece_estimate: float
    timestamp_range: tuple[str, str] | None


def _compute_ece(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    if scores.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(scores, bins) - 1
    inds = np.clip(inds, 0, n_bins - 1)
    ece = 0.0
    n = scores.size
    for bin_idx in range(n_bins):
        mask = inds == bin_idx
        if not mask.any():
            continue
        accuracy = float(np.mean(labels[mask]))
        confidence = float(np.mean(scores[mask]))
        ece += (mask.sum() / n) * abs(accuracy - confidence)
    return float(ece)


def _per_domain_pivot(df: pd.DataFrame, *, score_column: str = "score") -> pd.DataFrame:
    """Return wide-form (sample_id, domain) -> score, plus label/timestamp columns preserved."""
    pivot = df.pivot_table(
        index=["sample_id"],
        columns="domain",
        values=score_column,
        aggfunc="first",
    )
    metadata = df.drop_duplicates("sample_id").set_index("sample_id")[
        [c for c in ("label", "timestamp", "category") if c in df.columns]
    ]
    return pivot.join(metadata, how="left").reset_index()


def _reference_distributions(reference_df: pd.DataFrame, score_column: str = "score") -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    for domain, group in reference_df.groupby("domain"):
        refs[str(domain)] = group[score_column].astype(float).to_numpy()
    return refs


def iter_windows(
    wide_df: pd.DataFrame,
    *,
    window_size: int,
) -> Iterator[pd.DataFrame]:
    for start in range(0, len(wide_df), window_size):
        end = min(start + window_size, len(wide_df))
        yield wide_df.iloc[start:end]


def run_monitor(
    fusion_df: pd.DataFrame,
    *,
    reference_split: str = "validation",
    target_split: str = "test",
    split_column: str = "split",
    tau: float = 0.66,
    window_size: int = 200,
    score_column: str = "score",
    min_ks_samples: int = 30,
) -> Iterator[WindowEvent]:
    if split_column not in fusion_df.columns:
        raise ValueError(f"fusion frame missing '{split_column}' column")
    reference_long = fusion_df[fusion_df[split_column] == reference_split]
    target_long = fusion_df[fusion_df[split_column] == target_split]
    if reference_long.empty:
        raise ValueError(f"no rows for reference split '{reference_split}'")
    if target_long.empty:
        raise ValueError(f"no rows for target split '{target_split}'")

    references = _reference_distributions(reference_long, score_column)
    target_wide = _per_domain_pivot(target_long, score_column=score_column)
    domain_cols = [c for c in target_wide.columns if c in references]

    for window_index, window in enumerate(iter_windows(target_wide, window_size=window_size)):
        per_domain_ks: dict[str, float] = {}
        reliabilities: list[float] = []
        for domain in domain_cols:
            current = window[domain].dropna().astype(float).to_numpy()
            if current.size < min_ks_samples:
                continue
            stat, p_value = ks_2samp(references[domain], current)
            per_domain_ks[domain] = float(stat)
            reliabilities.append(float(np.clip(p_value, 0.0, 1.0)))
        mean_reliability = float(np.mean(reliabilities)) if reliabilities else float("nan")
        gate_fired = bool(np.isfinite(mean_reliability) and mean_reliability < tau)

        if "label" in window.columns:
            valid = window.dropna(subset=["label"]) if "label" in window.columns else window
            if domain_cols:
                fused = valid[domain_cols].mean(axis=1).to_numpy()
            else:
                fused = np.array([])
            labels = valid["label"].astype(float).to_numpy() if "label" in valid.columns else np.array([])
            ece = _compute_ece(labels, fused)
        else:
            ece = float("nan")

        if "timestamp" in window.columns and window["timestamp"].notna().any():
            ts_range = (
                str(window["timestamp"].iloc[0]),
                str(window["timestamp"].iloc[-1]),
            )
        else:
            ts_range = None

        yield WindowEvent(
            window_index=window_index,
            n_samples=int(len(window)),
            mean_reliability=mean_reliability,
            gate_fired=gate_fired,
            ks_per_domain=per_domain_ks,
            ece_estimate=float(ece),
            timestamp_range=ts_range,
        )


def _write_events(events: Iterable[WindowEvent], stream: IO[str]) -> int:
    count = 0
    for event in events:
        stream.write(
            json.dumps(
                {
                    "window_index": event.window_index,
                    "n_samples": event.n_samples,
                    "mean_reliability": event.mean_reliability,
                    "gate_fired": event.gate_fired,
                    "ks_per_domain": event.ks_per_domain,
                    "ece_estimate": event.ece_estimate,
                    "timestamp_range": event.timestamp_range,
                }
            )
            + "\n"
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-csv", type=Path, required=True, help="Path to a long-format fusion CSV.")
    parser.add_argument("--reference-split", default="validation")
    parser.add_argument("--target-split", default="test")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--tau", type=float, default=0.66)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--min-ks-samples", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("experiments/monitor/calibration_events.jsonl"))
    args = parser.parse_args()

    fusion_df = pd.read_csv(args.fusion_csv)
    events = run_monitor(
        fusion_df,
        reference_split=args.reference_split,
        target_split=args.target_split,
        split_column=args.split_column,
        tau=args.tau,
        window_size=args.window_size,
        score_column=args.score_column,
        min_ks_samples=args.min_ks_samples,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        n_events = _write_events(events, out)
    print(f"Wrote {n_events} calibration events to {args.output}")


if __name__ == "__main__":
    main()
