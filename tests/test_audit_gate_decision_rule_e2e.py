"""Smoke tests for the gate decision rule end-to-end audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_audit_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "src/scripts/audit_gate_decision_rule_e2e.py"
    spec = importlib.util.spec_from_file_location("audit_gate_decision_rule_e2e", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, root


def test_synthetic_scenarios_pass_audit():
    module, root = _load_audit_module()
    payload = module.run_audit(root)
    synthetic = [s for s in payload["scenarios"] if s["scenario_id"].endswith("_synthetic")]
    assert len(synthetic) == 2
    assert all(s["audit_pass"] for s in synthetic)
