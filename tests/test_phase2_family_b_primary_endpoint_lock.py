"""Phase 2.2B — B-MECH-1 primary endpoint parameters are locked in source."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "src" / "scripts" / "run_phase2_mechanism_replication.py"


def test_b_mech_1_tau_is_locked_at_066():
    t = DRIVER.read_text()
    assert '"tau_mean": 0.66' in t, "B-MECH-1 driver must lock tau_mean=0.66"


def test_b_mech_1_gate_mode_is_locked_at_mean():
    t = DRIVER.read_text()
    assert '"gate_mode": "mean"' in t, "B-MECH-1 driver must lock gate_mode=mean"


def test_b_mech_1_k_values_is_locked_at_4_only():
    t = DRIVER.read_text()
    assert '"k_values": (4,)' in t, "B-MECH-1 driver must lock k_values=(4,) for coherent collapse"


def test_b_mech_1_attacks_are_zero_and_max_only():
    t = DRIVER.read_text()
    assert '"attacks": ("zero_attack", "max_attack")' in t


def test_b_mech_1_writes_selection_used_test_metrics_false():
    t = DRIVER.read_text()
    assert "selection_used_test_metrics=False" in t
