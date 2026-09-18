"""Safety regressions for the So2Sat v2 pre-target authorization boundary."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.kbound.so2sat import target_runner
from experiments.kbound.so2sat.integrity import (
    ARTIFACT_RECEIPT_SCHEMA_V1,
    IntegrityError,
    stable_sha256,
    strict_json_load,
    write_immutable_json_with_receipt,
)
from experiments.kbound.so2sat.protocol import default_protocol_path, load_protocol
from tests.test_so2sat_target_boundary import _fixture


def _documents(fixture: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        "protocol": load_protocol(),
        "execution_seal": strict_json_load(fixture["seal"]),
        "selection": strict_json_load(fixture["selected"]),
        "calibration_bundle": strict_json_load(fixture["cal_bundle"]),
        "gate": strict_json_load(fixture["gate"]),
    }


def _authorize(fixture: dict[str, object], gate_screen_path: Path, **overrides: Path) -> dict[str, object]:
    from experiments.kbound.so2sat.v2_target import authorize_target_execution

    return authorize_target_execution(
        protocol_path=default_protocol_path(),
        execution_seal_path=overrides.get("execution_seal", fixture["seal"]),
        selection_path=fixture["selected"],
        calibration_bundle_path=fixture["cal_bundle"],
        gate_path=overrides.get("gate", fixture["gate"]),
        gate_screen_path=gate_screen_path,
    )


def test_receipted_mapping_consumes_one_secure_verified_read_without_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.kbound.so2sat import v2_target

    artifact = tmp_path / "authority.json"
    expected = {"schema": "synthetic_authority_v2", "value": 7}
    expected_receipt = write_immutable_json_with_receipt(artifact, expected)

    def forbidden_reopen(*args: object, **kwargs: object) -> object:
        raise AssertionError("verified authority path was reopened")

    monkeypatch.setattr(v2_target, "verify_artifact_receipt", forbidden_reopen, raising=False)
    monkeypatch.setattr(v2_target, "strict_json_load", forbidden_reopen, raising=False)

    document, receipt = v2_target._load_receipted_mapping(artifact)
    assert document == expected
    assert receipt == expected_receipt


def test_v2_target_secure_loader_preserves_legacy_v1_receipt_compatibility(
    tmp_path: Path,
) -> None:
    from experiments.kbound.so2sat import v2_target

    artifact = tmp_path / "legacy-authority.json"
    expected = {"schema": "synthetic_authority_v1", "value": 7}
    expected_receipt = write_immutable_json_with_receipt(
        artifact,
        expected,
        receipt_schema=ARTIFACT_RECEIPT_SCHEMA_V1,
    )

    document, receipt = v2_target._load_receipted_mapping(artifact)
    assert document == expected
    assert receipt == expected_receipt


def test_receipted_mapping_consumes_the_verified_bytes_if_path_changes_later(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.kbound.so2sat import integrity, v2_target

    artifact = tmp_path / "authority.json"
    expected = {"schema": "synthetic_authority_v2", "value": 7}
    write_immutable_json_with_receipt(artifact, expected)
    real_read = integrity._read_regular_file_from_directory
    observed_reads: list[Path] = []

    def mutate_after_verified_artifact_read(
        parent_descriptor: int,
        basename: str,
        display_path: Path,
    ):
        payload, snapshot = real_read(parent_descriptor, basename, display_path)
        observed = Path(display_path)
        observed_reads.append(observed)
        if observed == artifact:
            artifact.chmod(0o644)
            artifact.write_text(
                '{"schema":"synthetic_authority_v2","value":999}\n',
                encoding="utf-8",
            )
        return payload, snapshot

    monkeypatch.setattr(
        integrity,
        "_read_regular_file_from_directory",
        mutate_after_verified_artifact_read,
    )

    document, _ = v2_target._load_receipted_mapping(artifact)
    assert document == expected
    assert observed_reads == [artifact, artifact.with_name(artifact.name + ".receipt.json")]


def test_authorization_consumes_protocol_and_receipt_without_legacy_reopens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.kbound.so2sat import v2_target
    from experiments.kbound.so2sat.v2_target import recompute_gate_screen

    fixture = _fixture(tmp_path)
    screen = recompute_gate_screen(**_documents(fixture))
    screen_path = tmp_path / "canonical-gate-screen.json"
    write_immutable_json_with_receipt(screen_path, screen)

    def forbidden_reopen(*args: object, **kwargs: object) -> object:
        raise AssertionError("protocol or protocol receipt was reopened")

    monkeypatch.setattr(v2_target, "load_protocol", forbidden_reopen, raising=False)
    monkeypatch.setattr(
        v2_target,
        "verify_checked_in_protocol_receipt",
        forbidden_reopen,
        raising=False,
    )

    authorization = _authorize(fixture, screen_path)
    assert authorization["status"] == "AUTHORIZED_AFTER_CANONICAL_95_CELL_SCREEN"


def test_authorization_recomputes_the_receipted_canonical_95_cell_screen(
    tmp_path: Path,
) -> None:
    from experiments.kbound.so2sat.v2_target import recompute_gate_screen

    fixture = _fixture(tmp_path)
    screen = recompute_gate_screen(**_documents(fixture))
    screen_path = tmp_path / "canonical-gate-screen.json"
    receipt = write_immutable_json_with_receipt(screen_path, screen)

    authorization = _authorize(fixture, screen_path)

    assert screen["cell_count"] == 95
    assert screen["passed"] is True
    assert authorization["status"] == "AUTHORIZED_AFTER_CANONICAL_95_CELL_SCREEN"
    assert authorization["gate_screen_artifact_sha256"] == receipt["artifact_sha256"]
    assert authorization["gate_screen_sha256"] == screen["gate_screen_sha256"]


def test_resigned_forged_screen_cannot_self_authorize(tmp_path: Path) -> None:
    from experiments.kbound.so2sat.v2_target import recompute_gate_screen

    fixture = _fixture(tmp_path)
    forged = recompute_gate_screen(**_documents(fixture))
    forged["passed"] = False
    forged["gate_screen_sha256"] = stable_sha256(
        {key: value for key, value in forged.items() if key != "gate_screen_sha256"}
    )
    forged_path = tmp_path / "resigned-forged-screen.json"
    write_immutable_json_with_receipt(forged_path, forged)

    with pytest.raises(IntegrityError, match="canonical recomputation"):
        _authorize(fixture, forged_path)


def test_receipted_failed_all_tie_screen_never_authorizes(tmp_path: Path) -> None:
    from experiments.kbound.so2sat.v2_target import recompute_gate_screen

    fixture = _fixture(tmp_path)
    documents = _documents(fixture)
    failed_gate = copy.deepcopy(documents["gate"])
    calibration = failed_gate["calibration"]
    calibration["city_max_residuals"] = dict.fromkeys(calibration["city_max_residuals"], 1000000.0)
    calibration["residuals_sorted"] = [1_000_000.0] * 19
    calibration["epsilon"] = 1_000_000.0
    failed_gate["gate_sha256"] = stable_sha256(
        {key: value for key, value in failed_gate.items() if key != "gate_sha256"}
    )
    failed_gate_path = tmp_path / "failed-gate.json"
    failed_gate_receipt = write_immutable_json_with_receipt(failed_gate_path, failed_gate)

    failed_seal = copy.deepcopy(documents["execution_seal"])
    failed_seal["gate_sha256"] = failed_gate["gate_sha256"]
    failed_seal["gate_artifact"] = {
        "artifact_sha256": failed_gate_receipt["artifact_sha256"],
        "canonical_document_sha256": failed_gate_receipt["canonical_document_sha256"],
    }
    failed_seal["execution_seal_sha256"] = stable_sha256(
        {key: value for key, value in failed_seal.items() if key != "execution_seal_sha256"}
    )
    failed_seal_path = tmp_path / "failed-execution-seal.json"
    write_immutable_json_with_receipt(failed_seal_path, failed_seal)

    failed_screen = recompute_gate_screen(
        protocol=documents["protocol"],
        execution_seal=failed_seal,
        selection=documents["selection"],
        calibration_bundle=documents["calibration_bundle"],
        gate=failed_gate,
    )
    failed_screen_path = tmp_path / "failed-screen.json"
    write_immutable_json_with_receipt(failed_screen_path, failed_screen)

    assert failed_screen["decision_counts"] == {
        "ADAPT": 0,
        "FREEZE": 0,
        "ABSTAIN": 95,
    }
    assert failed_screen["passed"] is False
    with pytest.raises(IntegrityError, match="failed.*exposure"):
        _authorize(
            fixture,
            failed_screen_path,
            gate=failed_gate_path,
            execution_seal=failed_seal_path,
        )


def test_two_pass_bundle_mismatch_never_publishes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_loader = target_runner.load_complete_target_bundle
    calls = 0

    def disagree_on_second_pass(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        master, cells = real_loader(*args, **kwargs)
        if calls == 2:
            cells = copy.deepcopy(cells)
            cells[0]["cell_sha256"] = "f" * 64
        return master, cells

    monkeypatch.setattr(target_runner, "load_complete_target_bundle", disagree_on_second_pass)

    with pytest.raises(IntegrityError, match="changed across deterministic"):
        _fixture(tmp_path)

    assert calls == 2
    assert not (tmp_path / "target-bundle").exists()
    assert (tmp_path / ".target-bundle.staging").is_dir()
