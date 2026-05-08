from __future__ import annotations

import numpy as np


def test_gate_stats_reports_sample_weighted_adaptation():
    from src.scripts.run_breakthrough_experiment import _gate_decision_stats

    weights = np.array(
        [
            [0.9, 0.2, 0.0],
            [0.8, 0.1, 0.4],
        ],
        dtype=np.float32,
    )
    masks = np.array(
        [
            [False, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )

    stats = _gate_decision_stats(weights, masks, threshold=0.7)

    assert stats["adapted"] is True
    assert stats["n_samples"] == 2
    assert stats["n_present"] == 4
    assert np.isclose(stats["mean_reliability"], 0.575)


def test_reliability_component_weights_disable_and_renormalize():
    from src.scripts.run_breakthrough_experiment import _component_weights

    cfg = {"ece_weight": 0.45, "ks_weight": 0.35, "sharpness_weight": 0.20}

    weights = _component_weights(cfg, disabled=("ece",))

    assert weights == {
        "ece_weight": 0.0,
        "ks_weight": 0.35 / 0.55,
        "sharpness_weight": 0.20 / 0.55,
    }


def test_tau_sweep_rows_aggregate_with_gate_metrics():
    from uais.utils.result_aggregation import aggregate_stress_rows

    rows = [
        {
            "seed": 1,
            "condition": "clean",
            "tau": 0.5,
            "static_auc": 0.80,
            "craf_auc": 0.82,
            "adaptation_rate": 0.25,
            "mean_reliability": 0.70,
        },
        {
            "seed": 2,
            "condition": "clean",
            "tau": 0.5,
            "static_auc": 0.84,
            "craf_auc": 0.85,
            "adaptation_rate": 0.50,
            "mean_reliability": 0.72,
        },
    ]

    summary = aggregate_stress_rows(
        rows,
        group_keys=("condition", "tau"),
        metric_keys=("static_auc", "craf_auc", "adaptation_rate", "mean_reliability"),
    )

    clean = summary[0]
    assert clean["n_seeds"] == 2
    assert np.isclose(clean["static_auc"], 0.82)
    assert np.isclose(clean["craf_auc"], 0.835)
    assert np.isclose(clean["delta_auc"], 0.015)
    assert np.isclose(clean["adaptation_rate"], 0.375)
