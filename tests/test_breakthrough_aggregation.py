import numpy as np


def test_summarize_seed_metric_rows_reports_mean_std_and_ci():
    from uais.utils.result_aggregation import summarize_seed_metric_rows

    rows = [
        {
            "seed": 1,
            "static_attention": {"roc_auc": 0.80, "pr_auc": 0.70, "f1": 0.60},
            "craf_attention": {"roc_auc": 0.84, "pr_auc": 0.74, "f1": 0.64},
        },
        {
            "seed": 2,
            "static_attention": {"roc_auc": 0.82, "pr_auc": 0.72, "f1": 0.62},
            "craf_attention": {"roc_auc": 0.86, "pr_auc": 0.76, "f1": 0.66},
        },
        {
            "seed": 3,
            "static_attention": {"roc_auc": 0.81, "pr_auc": 0.71, "f1": 0.61},
            "craf_attention": {"roc_auc": 0.85, "pr_auc": 0.75, "f1": 0.65},
        },
    ]

    summary = summarize_seed_metric_rows(rows, methods=("static_attention", "craf_attention"))

    assert summary["static_attention"]["roc_auc"]["n"] == 3
    assert np.isclose(summary["static_attention"]["roc_auc"]["mean"], 0.81)
    assert summary["static_attention"]["roc_auc"]["ci_low"] <= 0.81
    assert summary["static_attention"]["roc_auc"]["ci_high"] >= 0.81
    assert np.isclose(summary["craf_attention"]["f1"]["mean"], 0.65)


def test_aggregate_stress_rows_groups_scenarios_across_seeds():
    from uais.utils.result_aggregation import aggregate_stress_rows

    rows = [
        {"seed": 1, "attack": "zero_attack", "target_domain": "all", "static_auc": 0.70, "craf_auc": 0.76},
        {"seed": 2, "attack": "zero_attack", "target_domain": "all", "static_auc": 0.72, "craf_auc": 0.77},
        {"seed": 1, "attack": "max_attack", "target_domain": "all", "static_auc": 0.73, "craf_auc": 0.75},
    ]

    summary = aggregate_stress_rows(
        rows,
        group_keys=("attack", "target_domain"),
        metric_keys=("static_auc", "craf_auc"),
    )

    zero = next(row for row in summary if row["attack"] == "zero_attack")
    assert zero["n_seeds"] == 2
    assert np.isclose(zero["static_auc"], 0.71)
    assert np.isclose(zero["craf_auc"], 0.765)
    assert np.isclose(zero["delta_auc"], 0.055)
    assert "craf_auc_ci_low" in zero
    assert "craf_auc_ci_high" in zero


def test_metrics_from_validation_threshold_records_threshold_source():
    from scripts.run_breakthrough_experiment import _metrics_from_validation_threshold

    val_labels = np.array([0, 0, 1, 1], dtype=int)
    val_probs = np.array([0.1, 0.2, 0.35, 0.4], dtype=float)
    test_labels = np.array([0, 1, 1], dtype=int)
    test_probs = np.array([0.2, 0.34, 0.38], dtype=float)

    metrics = _metrics_from_validation_threshold(
        test_labels,
        test_probs,
        val_labels=val_labels,
        val_probs=val_probs,
        strategy="val_f1",
    )

    assert metrics["decision_threshold"] == np.float64(0.35)
    assert metrics["threshold_strategy"] == "val_f1"
    assert metrics["f1"] == np.float64(2 / 3)
