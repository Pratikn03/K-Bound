import numpy as np
import pytest

from uais.utils.stats import delong_roc_test
from uais.utils.metrics import (
    bounded_switching_certificate,
    calibration_monitor_report,
    reliability_degradation_auc,
    select_decision_threshold,
)


def test_delong_roc_test_ignores_non_finite_scores():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    y_score_a = np.array([0.1, 0.8, np.nan, 0.7, 0.4, 0.6, 0.3, 0.9], dtype=float)
    y_score_b = np.array([0.2, 0.5, 0.4, 0.8, 0.7, np.nan, 0.3, 0.6], dtype=float)

    p_value = delong_roc_test(y_true, y_score_a, y_score_b)

    assert np.isfinite(p_value)
    assert 0.0 <= p_value <= 1.0


def test_delong_roc_test_filters_non_finite_rows_pairwise():
    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    y_score_a = np.array([0.1, 0.9, np.nan, 0.8, 0.2, 0.7], dtype=float)
    y_score_b = np.array([0.1, 0.9, 0.99, 0.8, 0.2, 0.7], dtype=float)

    p_value = delong_roc_test(y_true, y_score_a, y_score_b)

    assert p_value == pytest.approx(1.0)


def test_reliability_degradation_auc_uses_finite_points():
    noise = np.array([0.0, 0.1, 0.2, 0.3], dtype=float)
    auc = np.array([0.9, np.nan, 0.8, 0.7], dtype=float)

    area = reliability_degradation_auc(noise, auc)

    assert np.isclose(area, 0.245)


def test_select_decision_threshold_uses_validation_f1_without_test_labels():
    y_val = np.array([0, 0, 1, 1], dtype=int)
    p_val = np.array([0.10, 0.20, 0.35, 0.40], dtype=float)

    threshold = select_decision_threshold(y_val, p_val, strategy="val_f1")

    assert threshold == pytest.approx(0.35)


def test_select_decision_threshold_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown decision threshold strategy"):
        select_decision_threshold(np.array([0, 1]), np.array([0.2, 0.8]), strategy="test_f1")


def test_calibration_monitor_report_flags_ece_regression():
    reference_y = np.array([0, 0, 1, 1], dtype=float)
    reference_p = np.array([0.05, 0.10, 0.90, 0.95], dtype=float)
    current_y = np.array([0, 0, 1, 1], dtype=float)
    current_p = np.array([0.90, 0.90, 0.10, 0.10], dtype=float)

    report = calibration_monitor_report(
        reference_y,
        reference_p,
        current_y,
        current_p,
        max_ece_delta=0.10,
        max_brier_delta=0.10,
    )

    assert report["alert"] is True
    assert "ece_delta" in report["reasons"]
    assert report["current"]["ece"] > report["reference"]["ece"]


def test_bounded_switching_certificate_requires_positive_margin():
    static_loss = np.array([0.10, 0.30, 0.20, 0.40])
    reliability_loss = np.array([0.05, 0.10, 0.25, 0.35])
    fire = np.array([True, True, False, False])

    cert = bounded_switching_certificate(
        static_loss,
        reliability_loss,
        fire,
        margin_epsilon=0.02,
    )

    assert cert["certified"] is True
    assert cert["fired_advantage"] > 0.02
    assert cert["policy_loss"] < cert["static_loss"]
