"""A label-free diagnostic cannot authorize an adaptation or prove coverage."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "docs/research/kbound/kbound_pkg/assumption_audit/__init__.py"
SPEC = importlib.util.spec_from_file_location("legacy_assumption_audit_scope", MODULE)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_matching_label_free_evidence_cannot_justify_opposite_benefit_worlds():
    # In a binary disagreement cell f0=0 and fa=1, these same model outputs
    # are compatible with benefit +1 when Y=1 or -1 when Y=0. Neither world
    # supplies a coverage premise to this diagnostic-only API.
    evidence = np.tile([0.0, 1.0, 0.75], (40, 1))
    for _benefit in (-1.0, 1.0):
        report = audit.run_audit(evidence, evidence)
        assert report.assumption_status == "not_falsified"
        assert report.guarantee_wording == "unresolved"
        assert report.recommended_safe_action == "abstain"
        assert report.calibration_warning is None
        assert "not evaluated" in report.details


def test_observed_residual_agreement_still_does_not_prove_target_coverage():
    evidence = np.zeros((40, 2))
    report = audit.run_audit(evidence, evidence, np.ones(40), np.ones(40))
    assert report.calibration_warning == 0.0
    assert report.guarantee_wording == "unresolved"
    assert report.recommended_safe_action == "abstain"


def test_warning_withholds_certificate_and_never_authorizes_adapt():
    report = audit.run_audit(np.ones((40, 2)), np.zeros((40, 2)))
    assert report.assumption_status == "warning"
    assert report.guarantee_wording == "does_not_apply"
    assert report.recommended_safe_action == "abstain"


@pytest.mark.parametrize("bad", [np.empty((0, 2)), [[np.nan, 0]], [[np.inf, 0]]])
def test_missing_or_nonfinite_evidence_cannot_look_not_falsified(bad):
    with pytest.raises(ValueError):
        audit.run_audit(bad, np.zeros((40, 2)))


def test_incomplete_residual_pair_is_rejected_instead_of_reported_zero():
    with pytest.raises(ValueError, match="both"):
        audit.run_audit(np.zeros((40, 2)), np.zeros((40, 2)), np.ones(40))


@pytest.mark.parametrize("threshold", [0, -1, np.nan, np.inf])
def test_invalid_warning_threshold_cannot_silence_a_diagnostic(threshold):
    with pytest.raises(ValueError):
        audit.run_audit(np.zeros((40, 2)), np.zeros((40, 2)), drift_warn=threshold)


def test_corrected_replay_preserves_historical_authorities_and_its_own_output(tmp_path):
    originals = [
        REPO / "docs/research/kbound/results/assumption_audit_v1.json",
        REPO / "docs/research/kbound/reports/assumption_audit_v1.md",
    ]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    command = [
        sys.executable,
        str(REPO / "docs/research/kbound/scripts/run_assumption_audit_v1.py"),
        "--output-dir",
        str(tmp_path),
    ]
    first = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    report = json.loads((tmp_path / "assumption_audit_v2.json").read_text())
    assert report["status"] == "posthoc_diagnostic_replay_not_new_statistical_validation"
    assert all(row["guarantee_wording"] != "applies" for row in report["stress_suite"])
    assert all(row["recommended_safe_action"] == "abstain" for row in report["stress_suite"])
    assert any("does not construct opposite-benefit" in row for row in report["limitations"])
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    second = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert second.returncode != 0
    assert "existing evidence is preserved" in second.stderr
