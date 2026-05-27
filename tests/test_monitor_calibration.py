from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.monitor_calibration import run_monitor


def _make_fusion_frame(
    *,
    n_samples_per_split: int = 240,
    domains: tuple[str, ...] = ("d0", "d1"),
    drift: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for split, mean in (("validation", 0.30), ("test", 0.30 + (0.45 if drift else 0.0))):
        scores = {domain: np.clip(rng.normal(mean, 0.05, size=n_samples_per_split), 0.0, 1.0) for domain in domains}
        labels = rng.integers(0, 2, size=n_samples_per_split)
        for idx in range(n_samples_per_split):
            for domain in domains:
                rows.append(
                    {
                        "sample_id": f"{split}_{idx}",
                        "domain": domain,
                        "score": float(scores[domain][idx]),
                        "label": int(labels[idx]),
                        "split": split,
                    }
                )
    return pd.DataFrame(rows)


def test_monitor_emits_one_event_per_window():
    fusion = _make_fusion_frame(n_samples_per_split=200, drift=False)
    events = list(
        run_monitor(
            fusion,
            tau=0.66,
            window_size=50,
            split_column="split",
            min_ks_samples=20,
        )
    )
    assert len(events) == 4
    for event in events:
        assert event.n_samples == 50
        assert set(event.ks_per_domain) == {"d0", "d1"}


def test_monitor_gate_does_not_fire_without_drift():
    fusion = _make_fusion_frame(n_samples_per_split=200, drift=False)
    events = list(run_monitor(fusion, tau=0.66, window_size=200, split_column="split", min_ks_samples=30))
    assert len(events) == 1
    assert events[0].gate_fired is False
    assert events[0].mean_reliability > 0.66


def test_monitor_gate_fires_under_strong_drift():
    fusion = _make_fusion_frame(n_samples_per_split=200, drift=True)
    events = list(run_monitor(fusion, tau=0.66, window_size=200, split_column="split", min_ks_samples=30))
    assert len(events) == 1
    assert events[0].gate_fired is True
    assert events[0].mean_reliability < 0.05


def test_monitor_raises_when_reference_split_missing():
    fusion = _make_fusion_frame()
    fusion = fusion[fusion["split"] != "validation"]
    with pytest.raises(ValueError, match="reference split"):
        list(run_monitor(fusion, split_column="split"))


def test_monitor_writes_jsonl(tmp_path):
    fusion = _make_fusion_frame(n_samples_per_split=200, drift=True)
    out_path = tmp_path / "events.jsonl"
    with out_path.open("w", encoding="utf-8") as stream:
        for event in run_monitor(fusion, tau=0.66, window_size=50, split_column="split", min_ks_samples=20):
            stream.write(
                json.dumps(
                    {
                        "window_index": event.window_index,
                        "gate_fired": event.gate_fired,
                        "mean_reliability": event.mean_reliability,
                        "ks_per_domain": event.ks_per_domain,
                    }
                )
                + "\n"
            )
    parsed = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(parsed) == 4
    assert all(record["gate_fired"] for record in parsed)
