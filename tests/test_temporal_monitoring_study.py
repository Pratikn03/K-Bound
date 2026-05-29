"""Deterministic checks for the Phase-10 / P6 temporal-monitoring study.

These lock in the deployment-style finding: a label-free drift statistic gates a
safe fallback so the acted policy keeps the upside of reliability gating where it
is safe (clean + in-distribution failure) and sheds the downside where it is not
(transfer drift), with no constant false alarms on clean periods.
"""

from __future__ import annotations

import numpy as np

from src.scripts.run_temporal_monitoring_study import run_study


def test_clean_windows_do_not_false_alarm():
    report = run_study(seed=0)
    assert report["clean_false_alarm_rate"] <= report["false_fire_budget"] + 1e-9


def test_all_transfer_drift_windows_detected():
    report = run_study(seed=0)
    assert report["drift_detection_rate"] == 1.0


def test_in_distribution_failure_gates_and_helps():
    report = run_study(seed=0)
    fail = [r for r in report["rows"] if r["regime"] == "IN_DIST_FAILURE"]
    assert fail, "expected in-distribution-failure windows"
    # not alerted, gate engaged, and gating helps on average
    assert all(not r["alert"] for r in fail)
    assert all(r["gate_state"] == "allow" for r in fail)
    assert np.mean([r["auc_gated"] for r in fail]) >= np.mean([r["auc_static"] for r in fail])


def test_policy_abstains_and_avoids_harm_under_drift():
    report = run_study(seed=0)
    drift = [r for r in report["rows"] if r["regime"] == "TRANSFER_DRIFT"]
    assert drift, "expected transfer-drift windows"
    # on every drift window the certificate is invalidated and we fall back
    assert all(r["alert"] for r in drift)
    assert all(r["gate_state"] == "abstain" for r in drift)
    assert all(r["fallback_state"] == "static_fallback" for r in drift)
    # at least one drift window is one where always-gating would have hurt
    assert any(r["auc_gated"] < r["auc_static"] for r in drift)
    # and the policy is no worse than always-gating across the drift region
    assert report["drift_mean_auc_acted_policy"] >= report["drift_mean_auc_always_gated"] - 1e-9


def test_acted_policy_dominates_both_fixed_policies():
    report = run_study(seed=0)
    assert report["mean_auc_acted_policy"] >= report["mean_auc_always_static"] - 1e-9
    assert report["mean_auc_acted_policy"] >= report["mean_auc_always_gated"] - 1e-9


def test_certificate_invalidation_implies_fallback():
    report = run_study(seed=0)
    for r in report["rows"]:
        if r["certificate_state"] == "INVALID":
            assert r["gate_state"] == "abstain"
            assert r["fallback_state"] == "static_fallback"
        else:
            assert r["fallback_state"] == "none"


def test_study_is_labelled_exploratory():
    report = run_study(seed=0)
    assert "EXPLORATORY" in report["label"].upper()
