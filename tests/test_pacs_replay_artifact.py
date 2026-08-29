from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/research/kbound/scripts/validate_pacs_replay.py"
SPEC = importlib.util.spec_from_file_location("validate_pacs_replay", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(*, condition: str, benefit: float, b_hat: float, epsilon: float) -> dict:
    action = MODULE.canonical_action(b_hat, epsilon)
    return {
        "dataset": "PACS",
        "domain": "photo",
        "calibration_domain": "sketch",
        "seed": 0,
        "split": "test",
        "condition": condition,
        "candidate": "tent",
        "metric": "accuracy",
        "Z": [0.1, 0.2],
        "Z_names": ["f0", "f1"],
        "evidence_schema_version": "test_evidence_v1",
        "a0": 0.5,
        "aa": 0.5 + benefit,
        "loss_frozen": 0.5,
        "loss_adapted": 0.5 - benefit,
        "B": benefit,
        "b_hat": b_hat,
        "eps_conformal": epsilon,
        "kga_decision": action,
        "oracle_action": "ADAPT" if benefit > 0 else "FREEZE",
        "source_checkpoint_sha256": "a" * 64,
        "run_config_sha256": "b" * 64,
        "residual_pool_sha256": "c" * 64,
        "record_id": hashlib.sha256(condition.encode()).hexdigest(),
    }


def artifact() -> dict:
    return {
        "schema": "kbound_pacs_percell_v2",
        "dataset": "PACS",
        "domain": "photo",
        "calibration_domain": "sketch",
        "seed": 0,
        "alpha": 0.1,
        "records": [
            record(condition="helpful", benefit=0.1, b_hat=0.2, epsilon=0.05),
            record(condition="harmful", benefit=-0.2, b_hat=-0.3, epsilon=0.05),
            record(condition="uncertain", benefit=0.01, b_hat=0.0, epsilon=0.05),
        ],
    }


def test_replay_matches_canonical_actions_and_metrics() -> None:
    result = MODULE.replay_records(artifact())
    assert result["actions"] == {"ADAPT": 1, "FREEZE": 1, "ABSTAIN": 1}
    assert result["FA_u"] == 0
    assert result["regret"]["K_Bound"] == pytest.approx(0.01 / 3)


def test_replay_rejects_tampered_action() -> None:
    document = artifact()
    document["records"][0]["kga_decision"] = "FREEZE"
    with pytest.raises(MODULE.PACSReplayError, match="action mismatch"):
        MODULE.replay_records(document)


def test_null_radius_fails_closed_to_abstain() -> None:
    document = artifact()
    document["records"][0]["eps_conformal"] = None
    document["records"][0]["kga_decision"] = "ABSTAIN"
    result = MODULE.replay_records(document)
    assert result["actions"] == {"ADAPT": 0, "FREEZE": 1, "ABSTAIN": 2}


def test_seed_summary_is_hash_locked_and_replayed(tmp_path: Path) -> None:
    per_cell = tmp_path / "per_cell" / "pacs_photo_seed0_percell.json"
    per_cell.parent.mkdir()
    per_cell.write_text(json.dumps(artifact(), sort_keys=True) + "\n")
    digest = hashlib.sha256(per_cell.read_bytes()).hexdigest()
    metrics = MODULE.replay_records(artifact())
    summary = {
        "schema": "kbound_pacs_seed_v1",
        "dataset": "PACS",
        "seed": 0,
        "per_domain": {
            "photo": {
                "n_test_cells": metrics["n"],
                "regret": metrics["regret"],
                "FA_u": metrics["FA_u"],
                "adapt_rate": metrics["adapt_rate"],
                "coverage": metrics["coverage"],
                "per_cell_artifact": "per_cell/pacs_photo_seed0_percell.json",
                "per_cell_sha256": digest,
            }
        },
    }
    summary_path = tmp_path / "pacs_seed0.json"
    summary_path.write_text(json.dumps(summary))
    assert MODULE.validate_seed_summary(summary_path)["photo"] == metrics


def test_historical_summary_fails_closed() -> None:
    path = ROOT / "experiments/kbound/results/pacs_seed1.json"
    with pytest.raises(MODULE.PACSReplayError, match="historical aggregate is not replayable"):
        MODULE.validate_seed_summary(path)
