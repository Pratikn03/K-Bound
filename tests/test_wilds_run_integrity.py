from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WILDS = ROOT / "experiments/kbound/wilds"
if str(WILDS) not in sys.path:
    sys.path.insert(0, str(WILDS))

import run_integrity as ri  # noqa: E402


def _cell(name: str) -> str:
    return ri.make_cell_id(dataset="demo", seed=0, condition=name)


def _accept_semantics(_records, _conditions) -> None:
    """Explicit test-only validator for cases exercising structural checks."""


def test_partial_state_fails_closed_without_a_semantic_validator(tmp_path: Path) -> None:
    with pytest.raises(ri.RunIntegrityError, match="semantic_validator"):
        ri.load_partial_state(
            tmp_path / "absent.json",
            run_config_sha256="0" * 64,
            expected_cell_ids=[],
        )


def test_cell_identity_and_seed_are_order_stable() -> None:
    left = ri.make_cell_id(dataset="demo", seed=2, split="test")
    right = ri.make_cell_id(split="test", seed=2, dataset="demo")

    assert left == right
    assert ri.deterministic_seed(left) == ri.deterministic_seed(right)


def test_ledger_fails_closed_for_missing_failed_duplicate_and_unexpected_cells() -> None:
    a, b = _cell("a"), _cell("b")
    incomplete = ri.build_ledger([a, b], [{"cell_id": a}], [])
    assert incomplete["status"] == "INCOMPLETE"
    assert incomplete["execution_complete"] is False
    assert incomplete["missing_cell_ids"] == [b]

    failed = ri.build_ledger([a, b], [{"cell_id": a}], [{"cell_id": b}])
    assert failed["publication_eligible"] is False
    assert failed["failed_cells"] == 1

    with pytest.raises(ri.RunIntegrityError, match="duplicate"):
        ri.build_ledger([a, b], [{"cell_id": a}, {"cell_id": a}], [])
    with pytest.raises(ri.RunIntegrityError, match="unexpected"):
        ri.build_ledger([a], [{"cell_id": b}], [])


def test_complete_ledger_does_not_self_promote_to_publication() -> None:
    cell_id = _cell("complete")
    ledger = ri.build_ledger([cell_id], [{"cell_id": cell_id}], [])

    assert ledger["status"] == "COMPLETE"
    assert ledger["execution_complete"] is True
    assert ledger["publication_eligible"] is False


def test_partial_resume_requires_exact_scientific_identity(tmp_path: Path) -> None:
    cell_id = _cell("a")
    config_hash = ri.stable_sha256({"split": "test", "candidate": "tent"})
    path = tmp_path / "_partial.json"
    payload = ri.partial_document(
        run_config_sha256=config_hash,
        expected_cell_ids=[cell_id],
        records=[{
            "cell_id": cell_id,
            "candidate": "tent",
            "a0": 0.5,
            "aa": 0.6,
            "B": 0.1,
        }],
        conditions=[{
            "cell_id": cell_id,
            "cand_names": ["freeze_f0", "tent"],
            "aa_all": [0.5, 0.6],
            "a0": 0.5,
            "best_adapt": 0.6,
            "oracle": 0.6,
        }],
        failures=[],
        progress="1/1",
        semantic_validator=_accept_semantics,
    )
    ri.atomic_json_dump(payload, path)

    records, conditions, failures = ri.load_partial_state(
        path,
        run_config_sha256=config_hash,
        expected_cell_ids=[cell_id],
        semantic_validator=_accept_semantics,
    )
    assert len(records) == len(conditions) == 1
    assert failures == []
    assert payload["record_inventory"][cell_id]["candidates"] == ["tent"]
    with pytest.raises(ri.RunIntegrityError, match="config mismatch"):
        ri.load_partial_state(
            path,
            run_config_sha256=ri.stable_sha256({"split": "val"}),
            expected_cell_ids=[cell_id],
            semantic_validator=_accept_semantics,
        )


def test_partial_resume_rejects_tampered_record_inventory_commitment(tmp_path: Path) -> None:
    identity = {"dataset": "demo", "seed": 0, "condition": "bound"}
    cell_id = ri.make_cell_id(**identity)
    path = tmp_path / "_partial.json"
    payload = ri.partial_document(
        run_config_sha256="a" * 64,
        expected_cell_ids=[cell_id],
        records=[{
            "cell_id": cell_id,
            "scientific_cell_identity": identity,
            "candidate": "tent",
            "a0": 0.5,
            "aa": 0.6,
            "B": 0.1,
        }],
        conditions=[{
            "cell_id": cell_id,
            "scientific_cell_identity": identity,
            "cand_names": ["freeze_f0", "tent"],
            "aa_all": [0.5, 0.6],
            "a0": 0.5,
        }],
        failures=[],
        progress="1/1",
        require_scientific_cell_identity=True,
        semantic_validator=_accept_semantics,
    )
    payload["record_inventory"][cell_id]["records_sha256"] = "0" * 64
    ri.atomic_json_dump(payload, path)

    with pytest.raises(ri.RunIntegrityError, match="record_inventory commitment mismatch"):
        ri.load_partial_state(
            path,
            run_config_sha256="a" * 64,
            expected_cell_ids=[cell_id],
            require_scientific_cell_identity=True,
            semantic_validator=_accept_semantics,
        )


def test_external_semantic_validator_rejects_a_self_resealed_partial(tmp_path: Path) -> None:
    """Internal hashes cannot authenticate attacker-controlled scientific rows."""

    cell_id = _cell("resealed")
    path = tmp_path / "_partial.json"
    forged = ri.partial_document(
        run_config_sha256="c" * 64,
        expected_cell_ids=[cell_id],
        records=[{
            "cell_id": cell_id,
            "candidate": "tent",
            "a0": 0.5,
            "aa": 0.99,
            "B": 0.49,
        }],
        conditions=[{
            "cell_id": cell_id,
            "cand_names": ["freeze_f0", "tent"],
            "aa_all": [0.5, 0.99],
            "a0": 0.5,
            "best_adapt": 0.99,
            "oracle": 0.99,
        }],
        failures=[],
        progress="1/1",
        # Simulates an attacker who can recompute every in-document hash.
        semantic_validator=lambda _records, _conditions: None,
    )
    ri.atomic_json_dump(forged, path)

    def validate_against_external_predictions(records, _conditions) -> None:
        if records[0]["aa"] != 0.6:
            raise ri.RunIntegrityError("external prediction recomputation mismatch")

    with pytest.raises(ri.RunIntegrityError, match="external prediction recomputation mismatch"):
        ri.load_partial_state(
            path,
            run_config_sha256="c" * 64,
            expected_cell_ids=[cell_id],
            semantic_validator=validate_against_external_predictions,
        )


def test_semantic_validator_cannot_mutate_the_document_being_sealed() -> None:
    cell_id = _cell("isolated-validator")

    def mutate_copy(records, conditions) -> None:
        records[0]["aa"] = 0.0
        conditions[0]["aa_all"][1] = 0.0

    payload = ri.partial_document(
        run_config_sha256="d" * 64,
        expected_cell_ids=[cell_id],
        records=[{
            "cell_id": cell_id,
            "candidate": "tent",
            "a0": 0.5,
            "aa": 0.6,
            "B": 0.1,
        }],
        conditions=[{
            "cell_id": cell_id,
            "cand_names": ["freeze_f0", "tent"],
            "aa_all": [0.5, 0.6],
            "a0": 0.5,
        }],
        failures=[],
        progress="1/1",
        semantic_validator=mutate_copy,
    )

    assert payload["records"][0]["aa"] == 0.6
    assert payload["conditions"][0]["aa_all"][1] == 0.6


def test_required_scientific_cell_identity_is_recomputed_not_trusted() -> None:
    identity = {"dataset": "demo", "seed": 0, "condition": "original"}
    cell_id = ri.make_cell_id(**identity)
    tampered = {**identity, "condition": "changed"}
    with pytest.raises(ri.RunIntegrityError, match="does not match"):
        ri.partial_document(
            run_config_sha256="b" * 64,
            expected_cell_ids=[cell_id],
            records=[{
                "cell_id": cell_id,
                "scientific_cell_identity": tampered,
                "candidate": "tent",
                "a0": 0.5,
                "aa": 0.6,
                "B": 0.1,
            }],
            conditions=[{
                "cell_id": cell_id,
                "scientific_cell_identity": tampered,
                "cand_names": ["freeze_f0", "tent"],
                "aa_all": [0.5, 0.6],
                "a0": 0.5,
            }],
            failures=[],
            progress="1/1",
            require_scientific_cell_identity=True,
            semantic_validator=_accept_semantics,
        )


@pytest.mark.parametrize("defect", ["dimension", "update", "protocol"])
def test_evidence_record_fails_closed_for_adversarial_payloads(defect: str) -> None:
    expected_protocol = {"schema": "demo", "mode": "online"}
    record = {
        "Z": [0.2, 0.4],
        "upd_norm": 0.4,
        "tta_protocol": expected_protocol,
    }
    if defect == "dimension":
        record["Z"].pop()
    elif defect == "update":
        record["upd_norm"] = 0.3
    else:
        record["tta_protocol"] = {"schema": "demo", "mode": "episodic"}

    with pytest.raises(ri.RunIntegrityError):
        ri.validate_evidence_record(
            record,
            ["other", "update_norm"],
            expected_tta_protocol=expected_protocol,
        )


def test_partial_rejects_orphan_records_and_legacy_state(tmp_path: Path) -> None:
    cell_id = _cell("a")
    config_hash = ri.stable_sha256({"split": "test"})
    with pytest.raises(ri.RunIntegrityError, match="outside completed"):
        ri.partial_document(
            run_config_sha256=config_hash,
            expected_cell_ids=[cell_id],
            records=[{"cell_id": cell_id}],
            conditions=[],
            failures=[],
            progress="0/1",
            semantic_validator=_accept_semantics,
        )

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"records": [], "conditions": []}), encoding="utf-8")
    with pytest.raises(ri.RunIntegrityError, match="legacy"):
        ri.load_partial_state(
            legacy,
            run_config_sha256=config_hash,
            expected_cell_ids=[cell_id],
            semantic_validator=_accept_semantics,
        )


@pytest.mark.parametrize("defect", ["empty", "missing", "duplicate", "wrong_score", "wrong_identity"])
def test_completed_cell_requires_exact_semantic_record_inventory(defect: str) -> None:
    cell_id = _cell("semantic")
    other_cell = _cell("other")
    condition = {
        "cell_id": cell_id,
        "seed": 0,
        "domain": "test",
        "cand_names": ["freeze_f0", "tent", "sar"],
        "aa_all": [0.5, 0.6, 0.4],
        "a0": 0.5,
        "best_adapt": 0.6,
        "oracle": 0.6,
    }
    records = [
        {"cell_id": cell_id, "seed": 0, "domain": "test", "candidate": "tent", "a0": 0.5, "aa": 0.6, "B": 0.1},
        {"cell_id": cell_id, "seed": 0, "domain": "test", "candidate": "sar", "a0": 0.5, "aa": 0.4, "B": -0.1},
    ]
    if defect == "empty":
        records = []
    elif defect == "missing":
        records.pop()
    elif defect == "duplicate":
        records[1]["candidate"] = "tent"
    elif defect == "wrong_score":
        records[1]["aa"] = 0.45
        records[1]["B"] = -0.05
    elif defect == "wrong_identity":
        records[1]["domain"] = "val"

    with pytest.raises(ri.RunIntegrityError):
        ri.partial_document(
            run_config_sha256=ri.stable_sha256({"contract": 2}),
            expected_cell_ids=[cell_id, other_cell] if defect == "empty" else [cell_id],
            records=records,
            conditions=[condition],
            failures=[],
            progress="resume",
            semantic_validator=_accept_semantics,
        )


def test_atomic_json_is_strict_and_never_leaves_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    with pytest.raises(ValueError):
        ri.atomic_json_dump({"epsilon": float("inf")}, path)
    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []

    ri.atomic_json_dump({"epsilon": None, "radius_status": "INFEASIBLE"}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["epsilon"] is None
    assert ri.finite_tree(json.loads(path.read_text(encoding="utf-8")))


def test_strict_loader_rejects_valid_json_number_that_overflows_to_infinity(tmp_path: Path) -> None:
    path = tmp_path / "overflow.json"
    path.write_text('{"value": 1e999}\n', encoding="utf-8")

    with pytest.raises(ri.RunIntegrityError, match="non-finite"):
        ri.strict_json_load(path)


def test_scientific_hash_and_writer_reject_unknown_types_instead_of_stringifying(tmp_path: Path) -> None:
    class Unknown:
        pass

    with pytest.raises(TypeError):
        ri.stable_sha256({"value": Unknown()})
    with pytest.raises(TypeError):
        ri.atomic_json_dump({"value": Unknown()}, tmp_path / "unknown.json")
