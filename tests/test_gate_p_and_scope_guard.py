"""Tests for the Gate P production audit + the serving scope guard."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_scope_guard_advisory_mode_flags_disagreement(monkeypatch):
    # Fix-queue item 30 (reproducibility hygiene): these two tests used to mutate
    # os.environ in place. A failure between set and pop leaked
    # UAIS_SCOPE_REFERENCE into every subsequent test in the session, so the
    # suite's result depended on collection order. monkeypatch restores the
    # environment even when the assertion fails.
    monkeypatch.delenv("UAIS_SCOPE_REFERENCE", raising=False)
    from deploy.api import scope_guard
    importlib.reload(scope_guard)
    # strongly conflicting modalities -> high drift AND out of envelope (audit H1:
    # the guard must actually FIRE, not just attach a reason string).
    out = scope_guard.evaluate({"rgb": 0.95, "depth": 0.05}, endpoint="fusion")
    assert out["mode"] == "advisory_no_reference"
    assert 0.0 <= out["drift"] <= 1.0
    assert "high cross-modal disagreement" in out["reasons"]
    assert out["in_envelope"] is False  # regression guard: must not be inert
    # agreeing modalities -> in envelope
    out2 = scope_guard.evaluate({"rgb": 0.5, "depth": 0.52}, endpoint="fusion")
    assert out2["in_envelope"] is True


def test_scope_guard_reference_envelope_detects_out_of_range(tmp_path, monkeypatch):
    # Fix-queue item 30 (reproducibility hygiene), same defect as above: the
    # os.environ[...] = ... / os.environ.pop(...) pair only unsets the variable
    # when the assertions in between all pass. On a failure UAIS_SCOPE_REFERENCE
    # stayed set for the rest of the session and every later scope_guard reload
    # silently ran in reference_envelope mode, so one failure changed the
    # outcome of unrelated tests. monkeypatch unsets it in teardown either way.
    ref = tmp_path / "ref.json"
    ref.write_text('{"rgb": {"q01": 0.0, "q99": 0.6}, "depth": {"q01": 0.0, "q99": 0.6}}')
    monkeypatch.setenv("UAIS_SCOPE_REFERENCE", str(ref))
    from deploy.api import scope_guard
    importlib.reload(scope_guard)
    assert scope_guard.reference_loaded() is True
    out = scope_guard.evaluate({"rgb": 0.99, "depth": 0.1}, endpoint="fusion")  # rgb out of band
    assert out["mode"] == "reference_envelope"
    assert out["drift"] > 0
    # The module is process-global state: reload it once more after teardown has
    # removed the variable so the next test does not inherit this reference.
    monkeypatch.undo()
    importlib.reload(scope_guard)


def test_gate_p_audit_runs_and_is_scoped_ready():
    from scripts.audit_gate_p_production import grade
    rep = grade()
    assert rep["verdict"] in ("PRODUCTION_READY", "SCOPED_PRODUCTION_READY", "CONDITIONAL_SCOPED")
    by = {c["id"]: c["status"] for c in rep["criteria"]}
    assert by["P12"] == "PASS", "live drift monitoring must be wired"
    assert by["P13"] == "PASS", "scope contract must exist"
    assert by["P4"] == "PASS", "safe model loading must hold"
    if rep["verdict"] == "PRODUCTION_READY":
        assert all(c["status"] == "PASS" for c in rep["criteria"])
