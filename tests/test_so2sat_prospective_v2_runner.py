"""Adversarial ordering tests for the prospective-v2 live target gateway."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.kbound.so2sat.integrity import (
    IntegrityError,
    stable_sha256,
    write_immutable_json_with_receipt,
)
from tests.test_so2sat_prospective_v2 import (
    _calibration_bundle,
    _feature_document,
    _precalibration_seal,
)


@pytest.fixture(autouse=True)
def synthetic_v2_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real secure loaders without requiring historical authorities.

    These schema fixtures are NOT recovered historical protocol/controller
    evidence. They exist only under pytest's temporary directory, and no
    production default or repository receipt is written or bypassed.
    """
    from experiments.kbound.so2sat import prospective_v2 as protocol

    fixture_root = tmp_path / "synthetic-v2-schema"
    documents = {
        protocol.PROTOCOL_BASENAME: protocol._expected_protocol_v2(),
        protocol.CONTROLLER_BASENAME: protocol._expected_controller_v2(),
        protocol.PRECALIBRATION_TEMPLATE_BASENAME: protocol._expected_template_v2(),
    }
    for name, document in documents.items():
        write_immutable_json_with_receipt(fixture_root / name, document)
    real_module_file = protocol._module_file

    def source_or_fixture(name: str) -> Path:
        return fixture_root / name if name in documents else real_module_file(name)

    monkeypatch.setattr(protocol, "_module_file", source_or_fixture)


def _authority_files(tmp_path: Path) -> dict[str, Path]:
    from experiments.kbound.so2sat.prospective_v2 import (
        authorize_target_execution_v2,
        build_gate_screen_v2,
    )

    seal, controller = _precalibration_seal()
    calibration = _calibration_bundle(seal, controller)
    screen = build_gate_screen_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=calibration,
    )
    authorization = authorize_target_execution_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=calibration,
        submitted_gate_screen=screen,
    )
    documents = {
        "precalibration_seal": seal,
        "calibration_bundle": calibration,
        "gate_screen": screen,
        "target_authorization": authorization,
    }
    paths: dict[str, Path] = {}
    for name, document in documents.items():
        path = tmp_path / f"{name}.json"
        write_immutable_json_with_receipt(path, document)
        paths[name] = path
    return paths


def _city_features(controller: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for city_index in range(10):
        city = f"opaque-target-city-{city_index:02d}"
        result[city] = {
            str(checkpoint): _feature_document(
                controller,
                marker=10_000 + city_index * 5 + checkpoint,
            )
            for checkpoint in range(5)
        }
    return result


def _materialize_complete_synthetic_v2_output_inventory(root: Path) -> None:
    """Create the hand-derived regular-file inventory required at publication."""

    actions = root / "actions"
    cells = root / "cells"
    actions.mkdir(parents=True)
    cells.mkdir()
    root_files = (
        "target_action_plan.json",
        "target_action_plan.json.receipt.json",
        "so2sat_target_bundle_v2.json",
        "so2sat_target_bundle_v2.json.receipt.json",
    )
    for name in root_files:
        (root / name).write_bytes(b"synthetic inventory member\n")
    for city_index in range(10):
        for checkpoint_id in range(5):
            prefix = f"city{city_index:02d}_checkpoint{checkpoint_id}"
            action_name = f"{prefix}.action.json"
            for name in (action_name, f"{action_name}.receipt.json"):
                (actions / name).write_bytes(b"synthetic inventory member\n")
            cell_names = (
                f"{prefix}.json",
                f"{prefix}.json.receipt.json",
                f"{prefix}.logits.npz",
                f"{prefix}.logits.npz.manifest.json",
                f"{prefix}.logits.npz.manifest.json.receipt.json",
            )
            for name in cell_names:
                (cells / name).write_bytes(b"synthetic inventory member\n")


@pytest.mark.parametrize("duplicate_member", ["artifact", "receipt"])
def test_runtime_authority_decoder_rejects_duplicate_json_keys(
    tmp_path: Path,
    duplicate_member: str,
) -> None:
    from experiments.kbound.so2sat import prospective_runner_v2 as runner
    from experiments.kbound.so2sat.integrity import ARTIFACT_RECEIPT_SCHEMA_V2

    artifact = tmp_path / "authority.json"
    document = {"schema": "synthetic_authority_v2", "value": 2}
    canonical_payload = json.dumps(document, sort_keys=True).encode("utf-8") + b"\n"
    artifact_payload = canonical_payload
    if duplicate_member == "artifact":
        artifact_payload = b'{"schema":"synthetic_authority_v2","value":1,"value":2}\n'
    artifact.write_bytes(artifact_payload)
    receipt = {
        "schema": ARTIFACT_RECEIPT_SCHEMA_V2,
        "artifact_basename": artifact.name,
        "artifact_bytes": len(artifact_payload),
        "artifact_sha256": hashlib.sha256(artifact_payload).hexdigest(),
        "canonical_document_sha256": stable_sha256(document),
    }
    receipt_path = artifact.with_name(artifact.name + ".receipt.json")
    if duplicate_member == "receipt":
        receipt_payload = (
            "{"
            f'"schema":{json.dumps(ARTIFACT_RECEIPT_SCHEMA_V2)},'
            f'"schema":{json.dumps(ARTIFACT_RECEIPT_SCHEMA_V2)},'
            f'"artifact_basename":{json.dumps(artifact.name)},'
            f'"artifact_bytes":{len(artifact_payload)},'
            f'"artifact_sha256":{json.dumps(hashlib.sha256(artifact_payload).hexdigest())},'
            f'"canonical_document_sha256":{json.dumps(stable_sha256(document))}'
            "}\n"
        ).encode()
    else:
        receipt_payload = json.dumps(receipt, sort_keys=True).encode("utf-8") + b"\n"
    receipt_path.write_bytes(receipt_payload)

    with pytest.raises(IntegrityError, match="duplicate JSON key"):
        runner._secure_receipted_mapping(artifact)


def test_runtime_gateway_recomputes_every_authority_before_target_factory(
    tmp_path: Path,
) -> None:
    from experiments.kbound.so2sat.prospective_runner_v2 import (
        open_authorized_target_runtime_v2,
    )

    paths = _authority_files(tmp_path)
    events: list[str] = []

    def target_factory(authority: object) -> object:
        events.append("target_factory")
        return authority

    authority = open_authorized_target_runtime_v2(
        **paths,
        target_factory=target_factory,
        _audit_event=events.append,
    )

    assert authority.target_authorization["target_probe_pixel_access_authorized"] is True
    assert events == [
        "precalibration_seal_verified_and_recomputed",
        "complete_95_cell_gate_screen_recomputed",
        "target_authorization_recomputed",
        "target_factory",
    ]


def test_forged_or_partial_authority_never_reaches_target_factory(tmp_path: Path) -> None:
    from experiments.kbound.so2sat.prospective_runner_v2 import (
        open_authorized_target_runtime_v2,
    )

    paths = _authority_files(tmp_path / "forged")
    seal, controller = _precalibration_seal()
    partial = _calibration_bundle(seal, controller)
    partial["cells"] = partial["cells"][:-1]
    partial["bundle_sha256"] = stable_sha256({key: value for key, value in partial.items() if key != "bundle_sha256"})
    partial_path = tmp_path / "partial-calibration.json"
    write_immutable_json_with_receipt(partial_path, partial)
    reached = False

    def target_factory(_: object) -> object:
        nonlocal reached
        reached = True
        return object()

    with pytest.raises(IntegrityError, match="95 cells"):
        open_authorized_target_runtime_v2(
            **{**paths, "calibration_bundle": partial_path},
            target_factory=target_factory,
        )
    assert reached is False

    forged = copy.deepcopy(_precalibration_seal()[0])
    forged["target_access_authorized"] = True
    forged["precalibration_seal_sha256"] = stable_sha256(
        {key: value for key, value in forged.items() if key != "precalibration_seal_sha256"}
    )
    forged_path = tmp_path / "forged-precalibration.json"
    write_immutable_json_with_receipt(forged_path, forged)
    with pytest.raises(IntegrityError, match="pre-calibration|unsealed"):
        open_authorized_target_runtime_v2(
            **{**paths, "precalibration_seal": forged_path},
            target_factory=target_factory,
        )
    assert reached is False


def test_production_entrypoint_checks_authority_before_any_target_path(
    tmp_path: Path,
) -> None:
    from experiments.kbound.so2sat.prospective_runner_v2 import (
        run_production_target_v2,
    )

    paths = _authority_files(tmp_path / "authority")
    forged = copy.deepcopy(_precalibration_seal()[0])
    forged["target_access_authorized"] = True
    forged["precalibration_seal_sha256"] = stable_sha256(
        {key: value for key, value in forged.items() if key != "precalibration_seal_sha256"}
    )
    forged_path = tmp_path / "forged-production-precalibration.json"
    write_immutable_json_with_receipt(forged_path, forged)
    nonexistent = tmp_path / "must-not-be-touched"

    with pytest.raises(IntegrityError, match="pre-calibration|unsealed"):
        run_production_target_v2(
            **{**paths, "precalibration_seal": forged_path},
            population_manifest=nonexistent / "population.json",
            source_postrun_acceptance=nonexistent / "acceptance.json",
            checkpoint_collection=nonexistent / "checkpoints.json",
            checkpoint_dir=nonexistent / "checkpoints",
            normalizer=nonexistent / "normalizer.json",
            environment_identity=nonexistent / "environment.json",
            geo_paths={
                "training": nonexistent / "training_geo.h5",
                "validation": nonexistent / "validation_geo.h5",
                "testing": nonexistent / "testing_geo.h5",
            },
            target_data_paths={
                "validation": nonexistent / "validation.h5",
                "testing": nonexistent / "testing.h5",
            },
            output_dir=nonexistent / "output",
            device_name="cpu",
        )
    assert not nonexistent.exists()


def test_internal_target_core_rejects_forged_authority_before_import_or_path_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    authority = runner.load_v2_runtime_authority(**_authority_files(tmp_path / "authority"))
    forged = copy.copy(authority)
    object.__setattr__(forged, "_runtime_capability", object())
    events: list[str] = []

    real_import = __import__

    def audited_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", 1)[0] in {"numpy", "torch"}:
            events.append(f"import:{name}")
        return real_import(name, *args, **kwargs)

    def forbidden_path_access(path: object) -> Path:
        events.append(f"path:{path}")
        raise AssertionError("an unissued authority reached path normalization")

    monkeypatch.setattr("builtins.__import__", audited_import)
    monkeypatch.setattr(runner, "_absolute_without_resolving", forbidden_path_access)

    with pytest.raises(IntegrityError, match="issued by the verified v2 authority loader"):
        runner._run_authorized_target_v2(
            authority=forged,
            population_manifest_path=tmp_path / "population.json",
            source_postrun_acceptance_path=tmp_path / "acceptance.json",
            checkpoint_collection_path=tmp_path / "checkpoints.json",
            checkpoint_dir=tmp_path / "checkpoints",
            normalizer_path=tmp_path / "normalizer.json",
            environment_identity_path=tmp_path / "environment.json",
            geo_paths={
                "training": tmp_path / "training_geo.h5",
                "validation": tmp_path / "validation_geo.h5",
                "testing": tmp_path / "testing_geo.h5",
            },
            target_data_paths={
                "validation": tmp_path / "validation.h5",
                "testing": tmp_path / "testing.h5",
            },
            output_dir=tmp_path / "output",
            device_name="cpu",
        )
    assert events == []


@pytest.mark.parametrize("link_member", ["artifact", "receipt"])
def test_runtime_gateway_rejects_symlinked_authority_members(
    tmp_path: Path,
    link_member: str,
) -> None:
    from experiments.kbound.so2sat.prospective_runner_v2 import (
        load_v2_runtime_authority,
    )

    paths = _authority_files(tmp_path / "real")
    original = paths["gate_screen"]
    linked_dir = tmp_path / "linked"
    linked_dir.mkdir()
    linked = linked_dir / original.name
    receipt = original.with_name(original.name + ".receipt.json")
    linked_receipt = linked.with_name(linked.name + ".receipt.json")
    if link_member == "artifact":
        linked.symlink_to(original)
        linked_receipt.write_bytes(receipt.read_bytes())
    else:
        linked.write_bytes(original.read_bytes())
        linked_receipt.symlink_to(receipt)

    with pytest.raises(IntegrityError, match="symlink"):
        load_v2_runtime_authority(**{**paths, "gate_screen": linked})


def test_runtime_gateway_detects_toctou_snapshot_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.kbound.so2sat import integrity
    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    paths = _authority_files(tmp_path)
    original = integrity._snapshot_from_stat
    # Target one authority's actual inode, not the third incidental metadata
    # call (which now belongs to a retained private transaction residue).
    identity = paths["precalibration_seal"].stat()
    calls = 0

    def changed(stat_result: object) -> tuple[int, int, int, int, int]:
        nonlocal calls
        snapshot = original(stat_result)
        if (stat_result.st_dev, stat_result.st_ino) == (identity.st_dev, identity.st_ino):
            calls += 1
            if calls >= 3:
                return (*snapshot[:-1], snapshot[-1] + 1)
        return snapshot

    monkeypatch.setattr(integrity, "_snapshot_from_stat", changed)
    with pytest.raises(IntegrityError, match="changed while being read"):
        runner.load_v2_runtime_authority(**paths)


def test_authorized_target_path_validation_rejects_symlinks(tmp_path: Path) -> None:
    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    regular = tmp_path / "validation.h5"
    regular.write_bytes(b"synthetic-not-hdf5")
    linked = tmp_path / "linked-validation.h5"
    linked.symlink_to(regular)
    with pytest.raises(IntegrityError, match="symlink"):
        runner._validate_target_path_map(
            {"validation": linked},
            expected={"validation"},
            field="target_data_paths",
        )


def test_all_50_actions_are_receipted_and_replayed_before_evaluation_access(
    tmp_path: Path,
) -> None:
    from experiments.kbound.so2sat.prospective_runner_v2 import (
        load_v2_runtime_authority,
        stage_v2_target_action_plan,
    )

    authority = load_v2_runtime_authority(**_authority_files(tmp_path / "authority"))
    features = _city_features(authority.controller)
    events: list[str] = []

    plan = stage_v2_target_action_plan(
        authority=authority,
        city_feature_documents=features,
        destination=tmp_path / "actions",
        before_evaluation=lambda: events.append("evaluation_opened"),
        _audit_event=events.append,
    )

    assert plan["cell_count"] == 50
    assert plan["abstain_realized_action"] == "FREEZE"
    assert events[-2:] == ["all_50_actions_independently_replayed", "evaluation_opened"]
    assert len(list((tmp_path / "actions").glob("*.action.json"))) == 50
    assert len(list((tmp_path / "actions").glob("*.action.json.receipt.json"))) == 50


def test_abstain_action_plan_always_retains_frozen_model(tmp_path: Path) -> None:
    from experiments.kbound.so2sat.prospective_runner_v2 import (
        load_v2_runtime_authority,
        stage_v2_target_action_plan,
    )

    authority = load_v2_runtime_authority(**_authority_files(tmp_path / "authority"))
    features = _city_features(authority.controller)
    first_city = sorted(features)[0]
    features[first_city]["0"]["features"]["entropy_change"] = float("nan")
    # Keep the malformed feature document intentionally unresigned: the live
    # planner must fail closed to ABSTAIN/FREEZE rather than trust its hash.

    plan = stage_v2_target_action_plan(
        authority=authority,
        city_feature_documents=features,
        destination=tmp_path / "actions",
    )
    abstentions = [row for row in plan["actions"] if row["decision"] == "ABSTAIN"]
    assert abstentions
    assert {row["realized_action"] for row in abstentions} == {"FREEZE"}


def test_live_probe_feature_failure_routes_to_auditable_unusable_marker() -> None:
    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    class BrokenProbe:
        frozen_probe_logits = [[float("nan")]]
        adapted_probe_logits = [[float("nan")]]
        normalized_adapter_update_norm = float("nan")
        batchnorm_source_statistic_divergence = float("nan")

    marker = runner._feature_document_or_unusable(BrokenProbe(), probe_count=1)
    assert marker == {
        "schema": "kbound_so2sat_unusable_probe_features_v2",
        "status": "INSUFFICIENT_OR_NONFINITE_INPUT_ABSTAIN_RETAIN_FROZEN",
        "probe_count": 1,
        "validation_labels_opened": False,
        "target_outcomes_opened": False,
    }


def test_resigned_action_forgery_is_rejected_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    authority = runner.load_v2_runtime_authority(**_authority_files(tmp_path / "authority"))
    features = _city_features(authority.controller)
    real_write = runner.write_immutable_json_with_receipt
    forged_once = False

    def forge_one(path: Path, document: dict[str, Any]) -> dict[str, Any]:
        nonlocal forged_once
        payload = document
        if not forged_once and path.name.endswith(".action.json"):
            forged_once = True
            payload = copy.deepcopy(document)
            payload["realized_action"] = "FREEZE" if payload["realized_action"] == "ADAPT" else "ADAPT"
            payload["action_sha256"] = stable_sha256(
                {key: value for key, value in payload.items() if key != "action_sha256"}
            )
        return real_write(path, payload)

    monkeypatch.setattr(runner, "write_immutable_json_with_receipt", forge_one)
    called = False

    def evaluation() -> None:
        nonlocal called
        called = True

    with pytest.raises(IntegrityError, match="canonical replay|changed after staging"):
        runner.stage_v2_target_action_plan(
            authority=authority,
            city_feature_documents=features,
            destination=tmp_path / "actions",
            before_evaluation=evaluation,
        )
    assert forged_once is True
    assert called is False


def test_evaluation_callback_is_not_called_for_partial_action_grid(tmp_path: Path) -> None:
    from experiments.kbound.so2sat.prospective_runner_v2 import (
        load_v2_runtime_authority,
        stage_v2_target_action_plan,
    )

    authority = load_v2_runtime_authority(**_authority_files(tmp_path / "authority"))
    features = _city_features(authority.controller)
    features.pop(sorted(features)[-1])
    called = False

    def evaluation() -> None:
        nonlocal called
        called = True

    with pytest.raises(IntegrityError, match="10-city by 5-checkpoint"):
        stage_v2_target_action_plan(
            authority=authority,
            city_feature_documents=features,
            destination=tmp_path / "actions",
            before_evaluation=evaluation,
        )
    assert called is False


def test_target_output_publication_is_atomic_create_only_under_destination_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.kbound.so2sat import integrity
    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    reserve = getattr(runner, "_reserve_output_directory_v2", None)
    publish = getattr(runner, "_publish_output_directory_v2", None)
    assert callable(reserve), "runner lacks a pinned create-only staging reservation"
    assert callable(publish), "runner lacks an atomic create-only publication step"

    publication = tmp_path / "target-publication"
    staging = reserve(publication)
    _materialize_complete_synthetic_v2_output_inventory(staging.staging_path)

    # Simulate an adversary winning the destination name after the publisher's
    # last existence check but immediately before its native no-replace rename.
    real_native_rename = integrity._native_rename_entry_create_only_at

    def inject_destination_at_syscall_boundary(
        source_parent_descriptor: int,
        source_basename: str,
        destination_parent_descriptor: int,
        destination_basename: str,
    ) -> None:
        runner.os.mkdir(
            destination_basename,
            mode=0o700,
            dir_fd=destination_parent_descriptor,
        )
        destination_descriptor = runner.os.open(
            destination_basename,
            runner.os.O_RDONLY | runner.os.O_DIRECTORY | runner.os.O_NOFOLLOW,
            dir_fd=destination_parent_descriptor,
        )
        try:
            intruder_descriptor = runner.os.open(
                "intruder.txt",
                runner.os.O_WRONLY | runner.os.O_CREAT | runner.os.O_EXCL,
                0o600,
                dir_fd=destination_descriptor,
            )
            try:
                runner.os.write(intruder_descriptor, b"preserve me")
            finally:
                runner.os.close(intruder_descriptor)
        finally:
            runner.os.close(destination_descriptor)
        real_native_rename(
            source_parent_descriptor,
            source_basename,
            destination_parent_descriptor,
            destination_basename,
        )

    def forbidden_replace(*args: object, **kwargs: object) -> object:
        raise AssertionError("overwrite-capable os.replace must not publish v2 output")

    monkeypatch.setattr(runner.os, "replace", forbidden_replace)
    monkeypatch.setattr(
        integrity,
        "_native_rename_entry_create_only_at",
        inject_destination_at_syscall_boundary,
    )
    with pytest.raises(IntegrityError, match="already exists|create-only"):
        publish(staging, publication)

    assert (publication / "intruder.txt").read_text(encoding="ascii") == "preserve me"
    assert (staging / "so2sat_target_bundle_v2.json").is_file()


def test_target_output_directory_appears_only_as_a_complete_atomic_rename(
    tmp_path: Path,
) -> None:
    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    reserve = getattr(runner, "_reserve_output_directory_v2", None)
    publish = getattr(runner, "_publish_output_directory_v2", None)
    assert callable(reserve)
    assert callable(publish)

    publication = tmp_path / "target-publication"
    staging = reserve(publication)
    _materialize_complete_synthetic_v2_output_inventory(staging.staging_path)
    assert not publication.exists()

    publish(staging, publication)

    assert not staging.exists()
    assert (publication / "so2sat_target_bundle_v2.json").is_file()


@pytest.mark.parametrize("attack", ["delete_bundle", "external_npz_symlink"])
def test_target_output_revalidates_exact_inventory_after_publication_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    """Mutation immediately after the directory rename must prevent completion."""

    from experiments.kbound.so2sat import integrity
    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    publication = tmp_path / "target-publication"
    reservation = runner._reserve_output_directory_v2(publication)
    _materialize_complete_synthetic_v2_output_inventory(reservation.staging_path)
    required = runner._required_output_regular_files_v2()
    external = tmp_path / "synthetic-external.npz"
    external.write_bytes(b"external synthetic payload\n")
    stolen = tmp_path / "writer-owned.npz"
    real_native_rename = integrity._native_rename_entry_create_only_at
    attack_fired = False

    def mutate_after_payload_rename(
        source_parent_descriptor: int,
        source_basename: str,
        destination_parent_descriptor: int,
        destination_basename: str,
    ) -> None:
        nonlocal attack_fired
        real_native_rename(
            source_parent_descriptor,
            source_basename,
            destination_parent_descriptor,
            destination_basename,
        )
        if source_basename == "payload" and destination_basename == publication.name:
            assert not attack_fired
            if attack == "delete_bundle":
                (publication / "so2sat_target_bundle_v2.json").unlink()
            else:
                member = publication / "cells" / "city00_checkpoint0.logits.npz"
                member.rename(stolen)
                member.symlink_to(external)
            attack_fired = True

    monkeypatch.setattr(
        integrity,
        "_native_rename_entry_create_only_at",
        mutate_after_payload_rename,
    )
    with pytest.raises(IntegrityError, match="post-rename|inventory|incomplete|quarantine"):
        runner._publish_output_directory_v2(reservation, publication)

    assert attack_fired, "the deterministic post-rename mutation did not execute"
    assert not publication.exists()
    failed_payload = reservation.holder_path / "failed-payload"
    assert failed_payload.is_dir()
    assert (failed_payload / ".kbound-publication-incomplete.json").is_file()
    assert not (failed_payload / ".kbound-publication-complete.json").exists()
    with pytest.raises(IntegrityError, match="missing|publication"):
        integrity.verify_complete_directory_publication(
            publication,
            required_relative_regular_files=required,
        )


def test_complete_publication_consumer_requires_descriptor_authenticated_state(
    tmp_path: Path,
) -> None:
    """An exact-looking tree without the publisher's completion state is incomplete."""

    from experiments.kbound.so2sat import integrity
    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    uncommitted = tmp_path / "uncommitted"
    _materialize_complete_synthetic_v2_output_inventory(uncommitted)
    with pytest.raises(IntegrityError, match="completion|complete|publication"):
        integrity.verify_complete_directory_publication(
            uncommitted,
            required_relative_regular_files=runner._required_output_regular_files_v2(),
        )

    publication = tmp_path / "target-publication"
    reservation = runner._reserve_output_directory_v2(publication)
    _materialize_complete_synthetic_v2_output_inventory(reservation.staging_path)
    runner._publish_output_directory_v2(reservation, publication)
    assert (
        integrity.verify_complete_directory_publication(
            publication,
            required_relative_regular_files=runner._required_output_regular_files_v2(),
        )
        == publication
    )


def test_public_v2_loader_rejects_valid_pair_in_uncommitted_directory(
    tmp_path: Path,
) -> None:
    """A valid member pair cannot bypass authentication of its publication root."""

    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    uncommitted = tmp_path / "uncommitted"
    _materialize_complete_synthetic_v2_output_inventory(uncommitted)
    bundle_path = uncommitted / "so2sat_target_bundle_v2.json"
    bundle_path.unlink()
    bundle_path.with_name(bundle_path.name + ".receipt.json").unlink()
    bundle = {"schema": "synthetic_complete_v2_bundle", "value": 7}
    write_immutable_json_with_receipt(bundle_path, bundle)

    assert runner._secure_receipted_mapping(bundle_path)[0] == bundle
    with pytest.raises(IntegrityError, match="completion|complete|publication"):
        runner.load_secure_receipted_mapping_v2(
            bundle_path,
            publication_root=uncommitted,
        )


def test_public_v2_loader_accepts_member_only_after_complete_publication(
    tmp_path: Path,
) -> None:
    """The public member loader consumes a pair only within a committed 354-file tree."""

    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    publication = tmp_path / "target-publication"
    reservation = runner._reserve_output_directory_v2(publication)
    _materialize_complete_synthetic_v2_output_inventory(reservation.staging_path)
    bundle_path = reservation.staging_path / "so2sat_target_bundle_v2.json"
    bundle_path.unlink()
    bundle_path.with_name(bundle_path.name + ".receipt.json").unlink()
    bundle = {"schema": "synthetic_complete_v2_bundle", "value": 11}
    write_immutable_json_with_receipt(bundle_path, bundle)

    runner._publish_output_directory_v2(reservation, publication)

    published_bundle = publication / bundle_path.name
    assert (
        runner.load_secure_receipted_mapping_v2(
            published_bundle,
            publication_root=publication,
        )[0]
        == bundle
    )


def test_v2_producer_replays_committed_bundle_through_public_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-publish inventory loss must prevent the producer returning success."""

    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    publication = tmp_path / "target-publication"
    reservation = runner._reserve_output_directory_v2(publication)
    _materialize_complete_synthetic_v2_output_inventory(reservation.staging_path)
    bundle_path = reservation.staging_path / "so2sat_target_bundle_v2.json"
    bundle_path.unlink()
    bundle_path.with_name(bundle_path.name + ".receipt.json").unlink()
    bundle = {"schema": "synthetic_complete_v2_bundle", "value": 13}
    write_immutable_json_with_receipt(bundle_path, bundle)
    real_publish = runner._publish_output_directory_v2

    def publish_then_remove_required_member(
        staging: object,
        destination: str | Path,
    ) -> Path:
        result = real_publish(staging, destination)  # type: ignore[arg-type]
        (result / "cells" / "city00_checkpoint0.logits.npz").unlink()
        return result

    monkeypatch.setattr(
        runner,
        "_publish_output_directory_v2",
        publish_then_remove_required_member,
    )
    with pytest.raises(IntegrityError, match="inventory|missing|publication"):
        runner._publish_and_replay_output_bundle_v2(
            reservation,
            publication,
            bundle_path.name,
            bundle,
        )


def test_target_output_publish_never_rmdirs_shared_holder_after_identity_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A holder-name intruder swapped at rmdir time must survive untouched."""

    from experiments.kbound.so2sat import integrity

    publication = tmp_path / "target-publication"
    reservation = integrity.reserve_create_only_directory_publication(publication)
    (reservation.staging_path / "complete.json").write_text(
        '{"status":"complete"}\n',
        encoding="ascii",
    )
    intruder = tmp_path / "reviewer-empty-holder-intruder"
    intruder.mkdir(mode=0o700)
    preserved_holder = tmp_path / "reviewer-preserved-authenticated-holder"
    parent_descriptor = reservation._parent_descriptor
    holder_basename = reservation.holder_path.name
    real_rmdir = integrity.os.rmdir
    cleanup_attempts: list[str] = []

    def substitute_at_shared_rmdir(
        basename: str,
        *args: object,
        dir_fd: int | None = None,
        **kwargs: object,
    ) -> object:
        if basename == holder_basename and dir_fd == parent_descriptor:
            cleanup_attempts.append(basename)
            integrity.os.rename(
                basename,
                preserved_holder.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            integrity.os.rename(
                intruder.name,
                basename,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        return real_rmdir(basename, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(integrity.os, "rmdir", substitute_at_shared_rmdir)
    result = integrity.publish_directory_create_only(
        reservation,
        publication,
        required_relative_regular_files={"complete.json"},
    )

    assert result == publication
    assert (publication / "complete.json").is_file()
    assert cleanup_attempts == []
    assert intruder.is_dir()
    assert reservation.holder_path.is_dir()
    assert {entry.name for entry in reservation.holder_path.iterdir()} == {"publication-incomplete.json"}
    assert not preserved_holder.exists()


def test_target_output_reservation_rejects_staging_directory_substitution(
    tmp_path: Path,
) -> None:
    """Publishing must authenticate the exact directory reserved earlier."""

    from experiments.kbound.so2sat import integrity

    publication = tmp_path / "target-publication"
    raw_reservation = integrity.reserve_create_only_directory_publication(publication)
    if hasattr(raw_reservation, "staging_path"):
        staging = raw_reservation.staging_path
        publish_args = (raw_reservation, publication)
    else:
        _, staging = raw_reservation
        publish_args = (staging, publication)
    (staging / "writer-owned.json").write_text("owned\n", encoding="ascii")
    preserved_original = staging.with_name(staging.name + ".reviewer-preserved")
    staging.rename(preserved_original)
    staging.mkdir(mode=0o700)
    (staging / "intruder.json").write_text("preserve me\n", encoding="ascii")

    with pytest.raises(IntegrityError, match="substitut|identity|reserved"):
        integrity.publish_directory_create_only(
            *publish_args,
            required_relative_regular_files={"writer-owned.json"},
        )

    assert not publication.exists()
    assert (staging / "intruder.json").read_text(encoding="ascii") == "preserve me\n"
    assert (preserved_original / "writer-owned.json").read_text(encoding="ascii") == "owned\n"


def test_target_output_publish_rejects_complete_lexical_intruder_when_pinned_payload_is_empty(
    tmp_path: Path,
) -> None:
    """Only the descriptor-pinned complete inventory may become public."""

    from experiments.kbound.so2sat import prospective_runner_v2 as runner

    publication = tmp_path / "target-publication"
    reservation = runner._reserve_output_directory_v2(publication)
    preserved_holder = tmp_path / "reviewer-preserved-authenticated-holder"
    reservation.holder_path.rename(preserved_holder)
    reservation.holder_path.mkdir(mode=0o700)
    reservation.staging_path.mkdir(mode=0o700)
    _materialize_complete_synthetic_v2_output_inventory(reservation.staging_path)

    with pytest.raises(IntegrityError, match="inventory|incomplete"):
        runner._publish_output_directory_v2(reservation, publication)

    assert not publication.exists()
    assert (reservation.staging_path / "so2sat_target_bundle_v2.json").is_file()
    assert {entry.name for entry in (preserved_holder / "payload").iterdir()} == {".kbound-publication-incomplete.json"}
