"""Phase 1.A — gating tests for the canonical label/metric semantics audit.

These tests assert that the audit script's output remains consistent with
the verdict METRICS_VALID_BUT_MISINTERPRETED. If the underlying canonical
JSONs change in a way that breaks the verdict, the tests fail and the
manuscript prose must be re-audited before any canonical PR-AUC / ECE /
Brier value is promoted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

AUDIT_JSON = Path("experiments/audit/canonical_label_semantics.json")


@pytest.fixture(scope="module")
def report() -> dict:
    if not AUDIT_JSON.exists():
        pytest.skip(f"audit not yet run: {AUDIT_JSON}")
    return json.loads(AUDIT_JSON.read_text())


def test_verdict_classified(report):
    assert report["verdict"] in {
        "METRICS_VALID_BUT_MISINTERPRETED",
        "LABEL_SEMANTICS_BUG",
        "SCORE_ORIENTATION_BUG",
        "METRIC_IMPLEMENTATION_BUG",
        "STALE_ARTIFACT_LINKAGE",
        "RAW_PREDICTIONS_MISSING_AND_RERUN_REQUIRED",
        "MULTIPLE_CAUSES",
    }


def test_verdict_is_misinterpretation_not_bug(report):
    """Phase 1.A's audit confirmed no code bug. If this changes, the
    manuscript reframing must be revisited."""
    assert report["verdict"] == "METRICS_VALID_BUT_MISINTERPRETED", (
        f"audit verdict changed to {report['verdict']!r}; if a bug now exists "
        f"the canonical prose in the paper / thesis must be re-audited before commit."
    )


def test_three_canonical_cells_present(report):
    benchmarks = sorted((c["benchmark"], c["protocol"]) for c in report["cells"])
    assert ("MVTec 3D-AD", "PatchCore canonical one-class") in benchmarks
    assert ("MVTec LOCO-AD", "PatchCore canonical one-class") in benchmarks
    assert ("VisA", "RGB+edge canonical one-class") in benchmarks


def test_label_semantics_anomaly_eq_1_in_every_canonical_cell(report):
    for cell in report["cells"]:
        sem = cell["label_definition"].get("inferred_label_semantics", "")
        assert "label_eq_1_means_anomaly" in sem, (
            f"{cell['benchmark']} reports {sem!r}; canonical one-class label "
            f"semantics must be anomaly=1 (train/val all-normal, test contains anomalies)."
        )


def test_canonical_train_val_prevalence_is_zero(report):
    for cell in report["cells"]:
        lab = cell["label_definition"]
        train_keys = [k for k in lab if k.startswith("split_train_prevalence")]
        val_keys = [k for k in lab if k.startswith("split_validation_prevalence")]
        for k in train_keys + val_keys:
            assert lab[k] == 0.0 or lab[k] is None, (
                f"{cell['benchmark']} split prevalence {k}={lab[k]} should be 0.0 under canonical one-class."
            )


def test_canonical_test_prevalence_is_high(report):
    """Canonical one-class places ALL anomalies in test → test prevalence is high."""
    for cell in report["cells"]:
        repro = cell.get("artifact_reproduction", {})
        prev = repro.get("test_fold_prevalence")
        if prev is None:
            continue
        assert prev > 0.30, (
            f"{cell['benchmark']} canonical test-fold prevalence is {prev:.4f}; "
            f"expected >0.30 under one-class protocol. Verify the input CSV split."
        )


def test_rga_boosted_pr_auc_matches_test_prevalence(report):
    """rga_boosted_fusion collapses to a constant predictor under one-class
    (its `selected_candidate = 'constant'` fallback). On a high-prevalence
    test fold, PR-AUC of a constant predictor equals the prevalence."""
    for cell in report["cells"]:
        repro = cell.get("artifact_reproduction", {})
        match_map = repro.get("per_method_pr_auc_vs_test_prevalence", {})
        m = match_map.get("rga_boosted_fusion")
        if m is None:
            continue
        assert m["matches_prevalence_within_0_005"] is True, (
            f"{cell['benchmark']} rga_boosted_fusion PR-AUC = {m['reported_pr_auc']} "
            f"vs test prevalence = {m['test_fold_prevalence']}. If they no longer "
            f"match within 0.005, the constant-predictor explanation has broken "
            f"and the audit must be re-run."
        )


def test_polarity_diagnostic_log_exists():
    p = Path("experiments/audit/polarity_diagnostic_log.csv")
    assert p.exists(), f"polarity diagnostic log missing: {p}"


def test_polarity_log_primary_metrics_do_not_use_flip():
    import csv
    p = Path("experiments/audit/polarity_diagnostic_log.csv")
    if not p.exists():
        pytest.skip("polarity log not yet generated")
    with p.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows, "polarity diagnostic log is empty"
    for r in rows:
        flag = (r.get("primary_metrics_use_flip") or "").strip().lower()
        assert flag in {"false", "0"}, (
            f"primary_metrics_use_flip must be False for every row (Phase 1.F lock); "
            f"row: {r}"
        )
