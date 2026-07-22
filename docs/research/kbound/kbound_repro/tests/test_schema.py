"""Schema-validation tests (Phase 6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kbound_repro import schema as S  # noqa: E402

KBOUND = Path(__file__).resolve().parents[2]


def _base_per_condition(**over):
    rec = {
        "schema_version": "kbound-per-condition-v1",
        "dataset": "CIFAR-10-C",
        "protocol": "STRESS_GRID_MULTISEED_PROTOCOL_A_v1",
        "seed": 0,
        "condition_id": "gaussian_noise_5",
        "model_id": "resnet18-cifar",
        "config_hash": "abc123",
        "quantile_rule": "loo_gbr_conformal",
        "source_artifact": "experiments/.../cell.json",
        "resolved_device": "mps",
        "created_at": "2026-07-21T10:00:00Z",
        "decisions": ["adapt", "freeze", "abstain", "adapt"],
        "counts": {"adapt": 2, "freeze": 1, "abstain": 1, "status": "retained"},
    }
    rec.update(over)
    return rec


def test_valid_per_condition_passes():
    S.validate(_base_per_condition(), "per_condition")


def test_missing_provenance_fails():
    rec = _base_per_condition()
    del rec["config_hash"]
    with pytest.raises(S.SchemaError):
        S.validate(rec, "per_condition")


def test_per_condition_requires_decisions_or_counts():
    rec = _base_per_condition()
    del rec["decisions"]
    del rec["counts"]
    with pytest.raises(S.SchemaError):
        S.validate(rec, "per_condition")


def test_counts_not_reconstructed_from_rates():
    # not_retained must carry null counts, never round(rate*n)
    bad = _base_per_condition(counts={"adapt": 12, "freeze": 0, "abstain": 0, "status": "not_retained"})
    with pytest.raises(S.SchemaError, match="not_retained"):
        S.validate(bad, "per_condition")


def test_retained_requires_integer_counts():
    bad = _base_per_condition(counts={"adapt": None, "freeze": 1, "abstain": 1, "status": "retained"})
    with pytest.raises(S.SchemaError):
        S.validate(bad, "per_condition")


def test_seed_uniqueness_and_condition_order():
    S.check_seed_uniqueness([0, 1, 2])
    with pytest.raises(S.SchemaError):
        S.check_seed_uniqueness([0, 1, 1])
    S.check_identical_condition_order([["a", "b"], ["a", "b"]])
    with pytest.raises(S.SchemaError):
        S.check_identical_condition_order([["a", "b"], ["b", "a"]])


def test_empirical_metrics_boundary_is_pinned():
    ok = {
        "schema_version": "kbound-empirical-metrics-v1",
        "false_adapt_boundary": "delta_le_0",
        "fa_u": 0.0,
        "fa_c": None,
        "counts": {"adapt": 0, "freeze": 5, "abstain": 0, "status": "retained"},
        "coverage_kind": "empirical",
    }
    S.validate(ok, "empirical_decision_metrics")
    bad = dict(ok, false_adapt_boundary="delta_lt_0")
    with pytest.raises(S.SchemaError):
        S.validate(bad, "empirical_decision_metrics")


def test_real_claim_ledger_validates():
    ledger = json.loads((KBOUND / "claim_ledger.json").read_text())
    S.validate(ledger, "claim_ledger")


def test_migration_preserves_original_and_marks_not_retained(tmp_path):
    # A historical rate-only artifact (no raw decisions / integer counts).
    src = tmp_path / "hist.json"
    src.write_text(json.dumps({"adapt_rate": 0.75, "freeze_rate": 0.25, "abstain_rate": 0.0}))
    src_bytes = src.read_bytes()
    dst = tmp_path / "norm" / "cell_v1.json"
    rec = S.migrate_historical_per_condition(
        src, dst,
        dataset="Camelyon17", protocol="CAMELYON17_PROTOCOL_G_RECONCILED_v2",
        seed=0, condition_id="hospital4", created_at="2026-07-21T10:00:00Z",
    )
    # original untouched
    assert src.read_bytes() == src_bytes
    # counts NOT reconstructed from rates
    assert rec["counts"] == {"adapt": None, "freeze": None, "abstain": None, "status": "not_retained"}
    assert dst.exists()
    S.validate(json.loads(dst.read_text()), "per_condition")


def test_migration_carries_raw_decisions(tmp_path):
    src = tmp_path / "hist2.json"
    src.write_text(json.dumps({"decisions": ["adapt", "adapt", "freeze"]}))
    dst = tmp_path / "norm2.json"
    rec = S.migrate_historical_per_condition(
        src, dst, dataset="d", protocol="p", seed=1, condition_id="c",
        created_at="2026-07-21T10:00:00Z",
    )
    assert rec["counts"] == {"adapt": 2, "freeze": 1, "abstain": 0, "status": "retained"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
