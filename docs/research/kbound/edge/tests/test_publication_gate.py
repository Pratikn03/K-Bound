from __future__ import annotations

from pathlib import Path

from kbound_edge.publication import evaluate_publication_gate
from kbound_edge.real_manifest import canonical_protocol_hash, load_real_protocol
from kbound_edge.recording import build_session_checklist

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "configs" / "edge_real_phone_v1.yaml"


def valid_inputs():
    cfg = load_real_protocol(CONFIG)
    protocol_hash = canonical_protocol_hash(cfg)
    clips = []
    index = 0
    for sid in sorted(cfg["sessions"]):
        for row in build_session_checklist(cfg, sid):
            index += 1
            clips.append(
                {
                    **row,
                    "session_id": sid,
                    "capture_mode": "physical",
                    "sha256": f"{index:064x}",
                    "captured_at": (
                        "2026-07-03T09:00:00+00:00"
                        if sid < "S07"
                        else "2026-07-05T09:00:00+00:00"
                    ),
                }
            )
    audit = {"checks": [{"check": f"check-{i}", "passed": True} for i in range(8)]}
    return cfg, {
        "model_card": {
            "protocol_hash": protocol_hash,
            "training_command": "03_train_source_model.py --epochs 20",
            "metrics": {"val_balanced_acc": 0.85, "val_macro_f1": 0.84},
        },
        "split_audit": {
            "sealed_splits": {"calibration_conformal": True},
            "sealed_at": "2026-07-04T09:00:00+00:00",
        },
        "inventory": {"clips": clips},
        "heldout": {"n_windows": cfg["sessions"]["S07"]["windows"] + cfg["sessions"]["S08"]["windows"]},
        "replication": {"n_windows": cfg["sessions"]["S09"]["windows"] + cfg["sessions"]["S10"]["windows"]},
        "anti_leakage": audit,
    }


def test_publication_gate_passes_complete_physical_study():
    cfg, inputs = valid_inputs()
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"]


def test_publication_gate_rejects_mock_capture():
    cfg, inputs = valid_inputs()
    inputs["inventory"]["clips"][0]["capture_mode"] = "mock"
    report = evaluate_publication_gate(cfg, **inputs)
    assert not report["passed"]
    failed = {row["check"] for row in report["checks"] if not row["passed"]}
    assert "all inventory clips are physical" in failed
