from __future__ import annotations

from pathlib import Path

import yaml


def test_positive_transfer_protocol_is_natural_and_cw_locked():
    root = Path(__file__).resolve().parents[1]
    path = root / "research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert doc["status"] == "SEALED_DEV_THEN_CONFIRM"
    assert doc["target"] == "natural_clean_transfer_beats_sar_and_cw"
    assert doc["primary_endpoints"]["vs_sar"]["minimum_practical_delta"] == 0.010
    assert doc["primary_endpoints"]["vs_cw"]["minimum_practical_delta"] == 0.005
    assert doc["primary_endpoints"]["vs_sar"]["ci_low_must_be_gt"] == 0.0
    assert doc["primary_endpoints"]["vs_cw"]["ci_low_must_be_gt"] == 0.0
    assert doc["confirmation_rules"]["fresh_or_unopened_holdout_required"] is True
    assert doc["confirmation_rules"]["opened_3d_adam_test_is_development_only"] is True
    assert doc["confirmation_rules"]["opened_mulsen_test_is_development_only"] is True
    assert doc["forbidden"]["synthetic_degradation"] is True
    assert doc["forbidden"]["fake_relabeling"] is True
    assert "experiments/fusion/cross_modal_gate_e_result.json" in doc["forbidden_sources"]

