"""Fail-closed live gateway for the So2Sat prospective-v2 protocol.

This module separates authorization from target I/O.  It first receipt-verifies
and independently recomputes the v2 pre-calibration seal, the complete 95-cell
gate screen, and the target authorization.  Only the returned authority may be
passed to a target factory.  Probe pixels may then form label-free actions, but
all 50 action/receipt pairs must be staged and replayed before evaluation pixels
are opened.  Target outcomes are never an input to this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from .features import validate_feature_document
from .integrity import (
    ARTIFACT_RECEIPT_SCHEMA_V2,
    DirectoryPublicationReservation,
    IntegrityError,
    load_verified_json_mapping_with_receipt,
    publish_directory_create_only,
    require_sha256,
    reserve_create_only_directory_publication,
    stable_sha256,
    verify_complete_directory_publication,
    write_immutable_json_with_receipt,
)
from .prospective_v2 import (
    CHECKPOINT_IDS,
    FEATURE_NAMES,
    PROTOCOL_ID,
    TARGET_CELL_COUNT,
    TARGET_CITY_COUNT,
    authorize_target_execution_v2,
    build_gate_screen_v2,
    build_precalibration_seal_v2,
    load_controller_v2,
    load_protocol_v2,
    route_city_checkpoint_v2,
    validate_precalibration_seal_v2,
    validate_target_authorization_v2,
)

TARGET_ACTION_DOCUMENT_SCHEMA = "kbound_so2sat_label_free_target_action_v2"
TARGET_ACTION_PLAN_SCHEMA = "kbound_so2sat_complete_target_action_plan_v2"
TARGET_CELL_SCHEMA = "kbound_so2sat_label_blind_target_cell_v2"
TARGET_BUNDLE_SCHEMA = "kbound_so2sat_complete_label_blind_target_bundle_v2"

_T = TypeVar("_T")
_RUNTIME_CAPABILITY = object()


def _noop_event(_: str) -> None:
    return None


def _absolute_without_resolving(path: str | os.PathLike[str]) -> Path:
    """Expand a path without following a symlink at any component."""

    expanded = Path(path).expanduser()
    return Path(os.path.abspath(os.fspath(expanded)))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise IntegrityError(f"required authority path is missing: {path}") from exc
        if stat.S_ISLNK(mode):
            raise IntegrityError(f"authority path contains a symlink component: {current}")


def _reserve_output_directory_v2(
    publication: str | os.PathLike[str],
) -> DirectoryPublicationReservation:
    """Reserve one private staging directory without claiming the final name."""

    return reserve_create_only_directory_publication(publication)


def _publish_output_directory_v2(
    staging: DirectoryPublicationReservation,
    publication: str | os.PathLike[str],
) -> Path:
    """Atomically publish a complete output directory with no overwrite path."""

    return publish_directory_create_only(
        staging,
        publication,
        required_relative_regular_files=_required_output_regular_files_v2(),
    )


def _required_output_regular_files_v2() -> frozenset[str]:
    """Return the exact regular-file inventory produced by the live v2 runner."""

    required = {
        "target_action_plan.json",
        "target_action_plan.json.receipt.json",
        "so2sat_target_bundle_v2.json",
        "so2sat_target_bundle_v2.json.receipt.json",
    }
    for city_index in range(TARGET_CITY_COUNT):
        for checkpoint_id in CHECKPOINT_IDS:
            prefix = f"city{city_index:02d}_checkpoint{checkpoint_id}"
            action_name = f"{prefix}.action.json"
            required.update(
                {
                    f"actions/{action_name}",
                    f"actions/{action_name}.receipt.json",
                    f"cells/{prefix}.json",
                    f"cells/{prefix}.json.receipt.json",
                    f"cells/{prefix}.logits.npz",
                    f"cells/{prefix}.logits.npz.manifest.json",
                    f"cells/{prefix}.logits.npz.manifest.json.receipt.json",
                }
            )
    return frozenset(required)


def _secure_receipted_mapping(
    artifact_path: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and verify one portable v2 artifact pair without path reopens."""

    return load_verified_json_mapping_with_receipt(
        artifact_path,
        receipt_schema=ARTIFACT_RECEIPT_SCHEMA_V2,
    )


def verify_published_directory_v2(
    publication_root: str | os.PathLike[str],
) -> Path:
    """Authenticate one completed v2 output and its exact 354-file inventory."""

    return verify_complete_directory_publication(
        publication_root,
        required_relative_regular_files=_required_output_regular_files_v2(),
    )


def load_secure_receipted_mapping_v2(
    artifact_path: str | os.PathLike[str],
    *,
    publication_root: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Consume one member pair only within an authenticated v2 publication."""

    publication = verify_published_directory_v2(publication_root)
    artifact = _absolute_without_resolving(artifact_path)
    try:
        relative_artifact = artifact.relative_to(publication).as_posix()
    except ValueError as exc:
        raise IntegrityError("v2 artifact is outside its authenticated publication root") from exc
    required = _required_output_regular_files_v2()
    if relative_artifact not in required or f"{relative_artifact}.receipt.json" not in required:
        raise IntegrityError("v2 artifact pair is not a declared publication member")

    result = _secure_receipted_mapping(artifact)
    verify_published_directory_v2(publication)
    return result


def _publish_and_replay_output_bundle_v2(
    staging: DirectoryPublicationReservation,
    publication: str | os.PathLike[str],
    bundle_basename: str,
    expected_bundle: Mapping[str, Any],
) -> Path:
    """Publish, then consume the committed bundle through the public v2 gate."""

    published_root = _publish_output_directory_v2(staging, publication)
    published_bundle_path = published_root / bundle_basename
    committed_bundle, _ = load_secure_receipted_mapping_v2(
        published_bundle_path,
        publication_root=published_root,
    )
    if committed_bundle != dict(expected_bundle):
        raise IntegrityError("committed v2 target bundle changed during public consumer replay")
    return published_bundle_path


@dataclass(frozen=True)
class _V2RuntimeAuthority:
    """Fully recomputed authority that is safe to pass to target construction."""

    protocol: dict[str, Any]
    controller: dict[str, Any]
    precalibration_seal: dict[str, Any]
    precalibration_seal_receipt: dict[str, Any]
    calibration_bundle: dict[str, Any]
    calibration_bundle_receipt: dict[str, Any]
    gate_screen: dict[str, Any]
    gate_screen_receipt: dict[str, Any]
    target_authorization: dict[str, Any]
    target_authorization_receipt: dict[str, Any]
    _runtime_capability: object = field(repr=False, compare=False)


def _validate_in_memory_receipt_binding(
    document: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    role: str,
) -> None:
    expected_receipt_keys = {
        "schema",
        "artifact_basename",
        "artifact_bytes",
        "artifact_sha256",
        "canonical_document_sha256",
    }
    if set(receipt) != expected_receipt_keys or receipt.get("schema") != ARTIFACT_RECEIPT_SCHEMA_V2:
        raise IntegrityError(f"{role} runtime receipt has unknown or missing fields")
    basename = receipt.get("artifact_basename")
    if (
        not isinstance(basename, str)
        or not basename
        or basename in {".", ".."}
        or "/" in basename
        or "\\" in basename
        or "\x00" in basename
        or Path(basename).name != basename
    ):
        raise IntegrityError(f"{role} runtime receipt basename is not portable")
    canonical_hash = require_sha256(
        receipt.get("canonical_document_sha256"),
        field=f"{role}.canonical_document_sha256",
    )
    if stable_sha256(document) != canonical_hash:
        raise IntegrityError(f"{role} runtime document changed after verified loading")
    payload = (
        json.dumps(
            dict(document),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    artifact_bytes = receipt.get("artifact_bytes")
    if isinstance(artifact_bytes, bool) or not isinstance(artifact_bytes, int):
        raise IntegrityError(f"{role} runtime receipt byte count is invalid")
    if artifact_bytes != len(payload):
        raise IntegrityError(f"{role} runtime artifact was not canonical JSON")
    artifact_hash = require_sha256(
        receipt.get("artifact_sha256"),
        field=f"{role}.artifact_sha256",
    )
    if hashlib.sha256(payload).hexdigest() != artifact_hash:
        raise IntegrityError(f"{role} runtime artifact bytes changed after verified loading")


def _validate_runtime_authority(authority: _V2RuntimeAuthority) -> None:
    if type(authority) is not _V2RuntimeAuthority or authority._runtime_capability is not _RUNTIME_CAPABILITY:
        raise IntegrityError("runtime authority was not issued by the verified v2 authority loader")
    for role, document, receipt in (
        (
            "precalibration_seal",
            authority.precalibration_seal,
            authority.precalibration_seal_receipt,
        ),
        (
            "calibration_bundle",
            authority.calibration_bundle,
            authority.calibration_bundle_receipt,
        ),
        ("gate_screen", authority.gate_screen, authority.gate_screen_receipt),
        (
            "target_authorization",
            authority.target_authorization,
            authority.target_authorization_receipt,
        ),
    ):
        _validate_in_memory_receipt_binding(document, receipt, role=role)
    validate_precalibration_seal_v2(authority.precalibration_seal)
    expected_screen = build_gate_screen_v2(
        precalibration_seal=authority.precalibration_seal,
        controller=authority.controller,
        calibration_bundle=authority.calibration_bundle,
    )
    if authority.gate_screen != expected_screen:
        raise IntegrityError("v2 runtime authority gate screen changed after loading")
    expected_authorization = authorize_target_execution_v2(
        precalibration_seal=authority.precalibration_seal,
        controller=authority.controller,
        calibration_bundle=authority.calibration_bundle,
        submitted_gate_screen=expected_screen,
    )
    if authority.target_authorization != expected_authorization:
        raise IntegrityError("v2 runtime authority target authorization changed after loading")
    validate_target_authorization_v2(
        authority.target_authorization,
        precalibration_seal=authority.precalibration_seal,
        controller=authority.controller,
        calibration_bundle=authority.calibration_bundle,
        gate_screen=authority.gate_screen,
    )


def load_v2_runtime_authority(
    *,
    precalibration_seal: str | os.PathLike[str],
    calibration_bundle: str | os.PathLike[str],
    gate_screen: str | os.PathLike[str],
    target_authorization: str | os.PathLike[str],
    _audit_event: Callable[[str], None] = _noop_event,
) -> _V2RuntimeAuthority:
    """Securely load and recompute all authorities before any target I/O."""

    protocol = load_protocol_v2()
    controller = load_controller_v2()
    seal, seal_receipt = _secure_receipted_mapping(precalibration_seal)
    validate_precalibration_seal_v2(seal)
    rebuilt_seal = build_precalibration_seal_v2(
        protocol=protocol,
        controller=controller,
        source_bindings=seal["source_bindings"],
        runtime_bindings=seal["runtime_bindings"],
        opaque_target_identities=seal["opaque_target_identities"],
        chronology=seal["chronology"],
    )
    if seal != rebuilt_seal:
        raise IntegrityError("v2 pre-calibration seal differs from canonical recomputation")
    _audit_event("precalibration_seal_verified_and_recomputed")

    calibration, calibration_receipt = _secure_receipted_mapping(calibration_bundle)
    submitted_screen, screen_receipt = _secure_receipted_mapping(gate_screen)
    expected_screen = build_gate_screen_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=calibration,
    )
    if submitted_screen != expected_screen:
        raise IntegrityError("receipted v2 gate screen differs from complete canonical recomputation")
    _audit_event("complete_95_cell_gate_screen_recomputed")

    submitted_authorization, authorization_receipt = _secure_receipted_mapping(target_authorization)
    expected_authorization = authorize_target_execution_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=calibration,
        submitted_gate_screen=expected_screen,
    )
    if submitted_authorization != expected_authorization:
        raise IntegrityError("receipted v2 target authorization differs from canonical recomputation")
    validate_target_authorization_v2(
        submitted_authorization,
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=calibration,
        gate_screen=expected_screen,
    )
    _audit_event("target_authorization_recomputed")
    authority = _V2RuntimeAuthority(
        protocol=copy.deepcopy(protocol),
        controller=copy.deepcopy(controller),
        precalibration_seal=copy.deepcopy(seal),
        precalibration_seal_receipt=copy.deepcopy(seal_receipt),
        calibration_bundle=copy.deepcopy(calibration),
        calibration_bundle_receipt=copy.deepcopy(calibration_receipt),
        gate_screen=copy.deepcopy(expected_screen),
        gate_screen_receipt=copy.deepcopy(screen_receipt),
        target_authorization=copy.deepcopy(expected_authorization),
        target_authorization_receipt=copy.deepcopy(authorization_receipt),
        _runtime_capability=_RUNTIME_CAPABILITY,
    )
    _validate_runtime_authority(authority)
    return authority


def open_authorized_target_runtime_v2(
    *,
    precalibration_seal: str | os.PathLike[str],
    calibration_bundle: str | os.PathLike[str],
    gate_screen: str | os.PathLike[str],
    target_authorization: str | os.PathLike[str],
    target_factory: Callable[[_V2RuntimeAuthority], _T],
    _audit_event: Callable[[str], None] = _noop_event,
) -> _T:
    """Invoke a target factory only after the whole authority chain replays."""

    authority = load_v2_runtime_authority(
        precalibration_seal=precalibration_seal,
        calibration_bundle=calibration_bundle,
        gate_screen=gate_screen,
        target_authorization=target_authorization,
        _audit_event=_audit_event,
    )
    return target_factory(authority)


def _safe_feature_fingerprint(value: Any) -> str:
    """Hash malformed inputs deterministically without emitting non-standard JSON."""

    def sanitize(member: Any) -> Any:
        if isinstance(member, Mapping):
            return {str(key): sanitize(item) for key, item in sorted(member.items(), key=lambda item: str(item[0]))}
        if isinstance(member, (list, tuple)):
            return [sanitize(item) for item in member]
        if isinstance(member, float) and not math.isfinite(member):
            return {"nonfinite_float": "nan" if math.isnan(member) else "+inf" if member > 0 else "-inf"}
        if member is None or isinstance(member, (str, int, float, bool)):
            return member
        return {"unsupported_type": type(member).__qualname__}

    return stable_sha256(sanitize(value))


def _usable_city_features(
    feature_documents: Any,
) -> tuple[dict[str, dict[str, float]], dict[str, str], bool]:
    fingerprints: dict[str, str] = {}
    selected: dict[str, dict[str, float]] = {}
    if not isinstance(feature_documents, Mapping) or set(feature_documents) != set(CHECKPOINT_IDS):
        return selected, fingerprints, False
    for checkpoint_id in CHECKPOINT_IDS:
        document = feature_documents[checkpoint_id]
        fingerprints[checkpoint_id] = _safe_feature_fingerprint(document)
        try:
            if not isinstance(document, Mapping):
                raise IntegrityError("feature document must be a mapping")
            validate_feature_document(document)
            values = document.get("features")
            if not isinstance(values, Mapping):
                raise IntegrityError("feature payload must be a mapping")
            selected[checkpoint_id] = {name: float(values[name]) for name in FEATURE_NAMES}
            if not all(math.isfinite(value) for value in selected[checkpoint_id].values()):
                raise IntegrityError("v2 selected feature is non-finite")
        except (IntegrityError, KeyError, TypeError, ValueError, OverflowError):
            return {}, fingerprints, False
    return selected, fingerprints, True


def _target_action_document(
    *,
    authority: _V2RuntimeAuthority,
    city_id: str,
    checkpoint_id: str,
    feature_documents: Any,
) -> dict[str, Any]:
    selected, fingerprints, usable = _usable_city_features(feature_documents)
    route = route_city_checkpoint_v2(
        authority.controller,
        checkpoint_id=checkpoint_id,
        city_probe_features=selected if usable else {},
        interval_radius=authority.gate_screen["calibration"]["interval_radius"],
    )
    document: dict[str, Any] = {
        "schema": TARGET_ACTION_DOCUMENT_SCHEMA,
        "status": "SEALED_AFTER_LABEL_FREE_PROBE_BEFORE_EVALUATION_PIXELS",
        "protocol_id": PROTOCOL_ID,
        "controller_sha256": authority.controller["controller_sha256"],
        "authorization_sha256": authority.target_authorization["authorization_sha256"],
        "action_unit": "city_checkpoint",
        "city_id": city_id,
        "checkpoint_id": checkpoint_id,
        "city_probe_feature_fingerprints": fingerprints,
        "support_status": route["support_status"],
        "delta_hat": route["delta_hat"],
        "interval_radius": route["interval_radius"],
        "lower": route["lower"],
        "upper": route["upper"],
        "decision": route["decision"],
        "realized_action": route["realized_action"],
        "validation_labels_opened": False,
        "target_outcomes_opened": False,
    }
    document["action_sha256"] = stable_sha256(document)
    return document


def _validate_target_action_document(
    document: Mapping[str, Any],
    *,
    authority: _V2RuntimeAuthority,
    city_id: str,
    checkpoint_id: str,
    feature_documents: Any,
) -> None:
    expected = _target_action_document(
        authority=authority,
        city_id=city_id,
        checkpoint_id=checkpoint_id,
        feature_documents=feature_documents,
    )
    if dict(document) != expected:
        raise IntegrityError("receipted v2 target action differs from canonical replay")
    if document.get("decision") == "ABSTAIN" and document.get("realized_action") != "FREEZE":
        raise IntegrityError("v2 ABSTAIN must retain the frozen model")


def _validate_city_feature_grid(value: Any) -> list[str]:
    if not isinstance(value, Mapping) or len(value) != TARGET_CITY_COUNT:
        raise IntegrityError("v2 action staging requires the exact 10-city by 5-checkpoint grid")
    cities = sorted(value)
    if any(
        not isinstance(city, str) or not city or city in {".", ".."} or "/" in city or "\\" in city or "\x00" in city
        for city in cities
    ):
        raise IntegrityError("v2 target city identifiers must be non-path opaque strings")
    if any(not isinstance(value[city], Mapping) or set(value[city]) != set(CHECKPOINT_IDS) for city in cities):
        raise IntegrityError("v2 action staging requires the exact 10-city by 5-checkpoint grid")
    return cities


def stage_v2_target_action_plan(
    *,
    authority: _V2RuntimeAuthority,
    city_feature_documents: Mapping[str, Mapping[str, Mapping[str, Any]]],
    destination: str | os.PathLike[str],
    before_evaluation: Callable[[], None] | None = None,
    _audit_event: Callable[[str], None] = _noop_event,
) -> dict[str, Any]:
    """Create and replay all 50 actions before allowing evaluation access.

    ``city_feature_documents`` must come only from validation/probe pixels.  The
    optional callback is invoked after two independent receipt-backed replays;
    a production caller uses that boundary to make evaluation pixels reachable.
    """

    _validate_runtime_authority(authority)
    cities = _validate_city_feature_grid(city_feature_documents)
    output = _absolute_without_resolving(destination)
    if output.exists() or output.is_symlink():
        raise IntegrityError("v2 action destination must be new and empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output.parent)
    output.mkdir(mode=0o700)

    paths: dict[tuple[str, str], Path] = {}
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for city_index, city_id in enumerate(cities):
        documents = city_feature_documents[city_id]
        for checkpoint_id in CHECKPOINT_IDS:
            action = _target_action_document(
                authority=authority,
                city_id=city_id,
                checkpoint_id=checkpoint_id,
                feature_documents=documents,
            )
            path = output / f"city{city_index:02d}_checkpoint{checkpoint_id}.action.json"
            write_immutable_json_with_receipt(path, action)
            paths[(city_id, checkpoint_id)] = path
            expected[(city_id, checkpoint_id)] = action
    if len(paths) != TARGET_CELL_COUNT:
        raise IntegrityError("v2 action staging did not create all 50 actions")
    _audit_event("all_50_actions_staged")

    replay_digests: list[str] = []
    receipt_bindings: dict[tuple[str, str], dict[str, Any]] = {}
    for pass_index in range(2):
        observed: list[dict[str, Any]] = []
        for key in sorted(paths):
            action, receipt = _secure_receipted_mapping(paths[key])
            _validate_target_action_document(
                action,
                authority=authority,
                city_id=key[0],
                checkpoint_id=key[1],
                feature_documents=city_feature_documents[key[0]],
            )
            if action != expected[key]:
                raise IntegrityError("v2 target action changed after staging")
            observed.append(action)
            if pass_index == 0:
                receipt_bindings[key] = receipt
        replay_digests.append(stable_sha256(observed))
    if replay_digests[0] != replay_digests[1]:
        raise IntegrityError("v2 target action plan changed across independent replays")
    _audit_event("all_50_actions_independently_replayed")

    rows = []
    for key in sorted(paths):
        action = expected[key]
        receipt = receipt_bindings[key]
        rows.append(
            {
                "city_id": key[0],
                "checkpoint_id": key[1],
                "action_basename": paths[key].name,
                "action_sha256": action["action_sha256"],
                "artifact_sha256": receipt["artifact_sha256"],
                "canonical_document_sha256": receipt["canonical_document_sha256"],
                "decision": action["decision"],
                "realized_action": action["realized_action"],
            }
        )
    if before_evaluation is not None:
        before_evaluation()
    plan: dict[str, Any] = {
        "schema": TARGET_ACTION_PLAN_SCHEMA,
        "status": "COMPLETE_50_ACTIONS_REPLAYED_BEFORE_EVALUATION_PIXELS",
        "protocol_id": PROTOCOL_ID,
        "controller_sha256": authority.controller["controller_sha256"],
        "authorization_sha256": authority.target_authorization["authorization_sha256"],
        "action_unit": "city_checkpoint",
        "target_city_count": len(cities),
        "checkpoint_count": len(CHECKPOINT_IDS),
        "cell_count": len(rows),
        "actions": rows,
        "replay_sha256": replay_digests,
        "all_actions_sealed_before_evaluation_pixels": True,
        "validation_labels_opened": False,
        "target_outcomes_opened": False,
        "abstain_realized_action": "FREEZE",
    }
    plan["action_plan_sha256"] = stable_sha256(plan)
    return plan


def _receipt_binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": receipt["schema"],
        "artifact_basename": receipt["artifact_basename"],
        "artifact_bytes": receipt["artifact_bytes"],
        "artifact_sha256": receipt["artifact_sha256"],
        "canonical_document_sha256": receipt["canonical_document_sha256"],
    }


def _require_bound_source_artifacts(
    *,
    authority: _V2RuntimeAuthority,
    population_manifest_path: str | os.PathLike[str],
    source_postrun_acceptance_path: str | os.PathLike[str],
    checkpoint_collection_path: str | os.PathLike[str],
    checkpoint_dir: str | os.PathLike[str],
    normalizer_path: str | os.PathLike[str],
    environment_identity_path: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Verify every source/runtime artifact after target authorization exists."""

    from .metadata_manifest import validate_population_manifest
    from .target_contract import validate_checkpoint_collection

    bindings = authority.precalibration_seal["source_bindings"]
    manifest, manifest_receipt = _secure_receipted_mapping(population_manifest_path)
    validate_population_manifest(manifest)
    if (
        _receipt_binding(manifest_receipt) != bindings["population_manifest_artifact"]
        or manifest.get("population_identity_sha256") != bindings["population_identity_sha256"]
    ):
        raise IntegrityError("population manifest differs from the v2 pre-calibration seal")

    acceptance, acceptance_receipt = _secure_receipted_mapping(source_postrun_acceptance_path)
    if (
        _receipt_binding(acceptance_receipt) != bindings["source_postrun_acceptance_artifact"]
        or stable_sha256(acceptance) != bindings["source_postrun_acceptance_sha256"]
    ):
        raise IntegrityError("source post-run acceptance differs from the v2 seal")

    collection, collection_receipt = _secure_receipted_mapping(checkpoint_collection_path)
    if (
        _receipt_binding(collection_receipt) != bindings["source_checkpoint_collection_artifact"]
        or stable_sha256(collection) != bindings["source_checkpoint_collection_sha256"]
    ):
        raise IntegrityError("checkpoint collection differs from the v2 seal")
    checkpoint_root = _absolute_without_resolving(checkpoint_dir)
    _reject_symlink_components(checkpoint_root)
    checkpoints = validate_checkpoint_collection(
        collection,
        collection_receipt=collection_receipt,
        collection_path=Path(checkpoint_collection_path),
        checkpoint_dir=checkpoint_root,
    )
    for checkpoint_id in CHECKPOINT_IDS:
        expected = bindings["source_checkpoints"][checkpoint_id]
        if any(
            checkpoints[checkpoint_id][field] != expected[field]
            for field in ("checkpoint_file_sha256", "checkpoint_tensor_sha256")
        ):
            raise IntegrityError("source checkpoint differs from the frozen v2 controller")
        _reject_symlink_components(_absolute_without_resolving(checkpoints[checkpoint_id]["checkpoint_path"]))

    normalizer, normalizer_receipt = _secure_receipted_mapping(normalizer_path)
    if (
        _receipt_binding(normalizer_receipt) != bindings["source_normalizer_artifact"]
        or normalizer.get("normalizer_sha256") != bindings["source_normalizer_sha256"]
    ):
        raise IntegrityError("source normalizer differs from the frozen v2 controller")

    environment, _ = _secure_receipted_mapping(environment_identity_path)
    claimed_environment = require_sha256(
        environment.get("environment_identity_sha256"),
        field="environment_identity_sha256",
    )
    unsigned_environment = dict(environment)
    unsigned_environment.pop("environment_identity_sha256")
    if (
        stable_sha256(unsigned_environment) != claimed_environment
        or claimed_environment != authority.precalibration_seal["runtime_bindings"]["environment_identity_sha256"]
    ):
        raise IntegrityError("runtime environment differs from the v2 pre-calibration seal")
    return manifest, checkpoints


def _feature_document_from_probe(computation: Any, *, probe_count: int) -> dict[str, Any]:
    import numpy as np

    from .features import N_CLASSES, extract_label_free_features

    def logits(value: Any, *, field: str) -> Any:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise IntegrityError(f"{field} must be numeric logits") from exc
        if array.shape != (probe_count, N_CLASSES) or not np.isfinite(array).all():
            raise IntegrityError(f"{field} must have finite shape {(probe_count, N_CLASSES)}")
        return np.ascontiguousarray(array)

    document = extract_label_free_features(
        logits(computation.frozen_probe_logits, field="frozen_probe_logits"),
        logits(computation.adapted_probe_logits, field="adapted_probe_logits"),
        normalized_adapter_update_norm=computation.normalized_adapter_update_norm,
        batchnorm_source_statistic_divergence=(computation.batchnorm_source_statistic_divergence),
    )
    if not isinstance(document, Mapping):  # defensive narrowing across the lazy boundary
        raise IntegrityError("label-free feature extraction did not return a mapping")
    validate_feature_document(document)
    return dict(document)


def _feature_document_or_unusable(computation: Any, *, probe_count: int) -> dict[str, Any]:
    """Convert unusable probe features into the frozen ABSTAIN pathway."""

    try:
        return _feature_document_from_probe(computation, probe_count=probe_count)
    except (IntegrityError, KeyError, TypeError, ValueError, OverflowError):
        return {
            "schema": "kbound_so2sat_unusable_probe_features_v2",
            "status": "INSUFFICIENT_OR_NONFINITE_INPUT_ABSTAIN_RETAIN_FROZEN",
            "probe_count": probe_count,
            "validation_labels_opened": False,
            "target_outcomes_opened": False,
        }


def _validate_target_path_map(
    value: Mapping[str, str | os.PathLike[str]],
    *,
    expected: set[str],
    field: str,
) -> dict[str, Path]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise IntegrityError(f"{field} must contain exactly {sorted(expected)}")
    result: dict[str, Path] = {}
    for name in sorted(expected):
        path = _absolute_without_resolving(value[name])
        _reject_symlink_components(path)
        result[name] = path
    return result


def _run_authorized_target_v2(
    *,
    authority: _V2RuntimeAuthority,
    population_manifest_path: str | os.PathLike[str],
    source_postrun_acceptance_path: str | os.PathLike[str],
    checkpoint_collection_path: str | os.PathLike[str],
    checkpoint_dir: str | os.PathLike[str],
    normalizer_path: str | os.PathLike[str],
    environment_identity_path: str | os.PathLike[str],
    geo_paths: Mapping[str, str | os.PathLike[str]],
    target_data_paths: Mapping[str, str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    device_name: str,
) -> Path:
    """Execute label-free v2 after the authorization gateway has completed."""

    _validate_runtime_authority(authority)

    import numpy as np
    import torch

    from .adapters import candidate_spec
    from .label_firewall import LabelFreeTargetLoader, VerifiedGeoIndex
    from .target_inference import TorchTargetCellExecutor
    from .target_runner import (
        EvaluationComputation,
        ProbeComputation,
        _freeze_sample_pixels,
        _partition,
        _verify_loader_audit,
        _write_logit_archive,
    )

    if device_name not in {"cpu", "mps"}:
        raise IntegrityError("prospective-v2 target device must be exactly cpu or mps")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise IntegrityError("MPS prospective-v2 execution requested but unavailable")
    manifest, checkpoints = _require_bound_source_artifacts(
        authority=authority,
        population_manifest_path=population_manifest_path,
        source_postrun_acceptance_path=source_postrun_acceptance_path,
        checkpoint_collection_path=checkpoint_collection_path,
        checkpoint_dir=checkpoint_dir,
        normalizer_path=Path(normalizer_path),
        environment_identity_path=environment_identity_path,
    )

    # This is the first point where any target path is normalized, inspected,
    # or passed to a loader.  The complete authority chain already replayed.
    safe_geo_paths = _validate_target_path_map(
        geo_paths,
        expected={"training", "validation", "testing"},
        field="geo_paths",
    )
    safe_target_paths = _validate_target_path_map(
        target_data_paths,
        expected={"validation", "testing"},
        field="target_data_paths",
    )
    target_identities = {
        split: {
            "bytes": authority.precalibration_seal["opaque_target_identities"][split]["artifact_bytes"],
            "sha256": authority.precalibration_seal["opaque_target_identities"][split]["raw_file_sha256"],
        }
        for split in ("validation", "testing")
    }
    geo_index = VerifiedGeoIndex(manifest, safe_geo_paths)
    target_loader = LabelFreeTargetLoader(
        geo_index,
        safe_target_paths,
        target_identities,
        modality="sen2_10_band",
    )
    executor = TorchTargetCellExecutor(
        candidate_id=authority.controller["candidate_id"],
        normalizer_path=Path(normalizer_path),
        device=torch.device(device_name),
    )
    if (
        executor.candidate_id != authority.controller["candidate_id"]
        or executor.normalizer_sha256 != authority.precalibration_seal["source_bindings"]["source_normalizer_sha256"]
        or executor.environment_identity_sha256
        != authority.precalibration_seal["runtime_bindings"]["environment_identity_sha256"]
    ):
        raise IntegrityError("live executor differs from the v2 pre-calibration seal")
    if target_loader.access_log:
        raise IntegrityError("prospective-v2 requires a fresh target loader")
    observed_containers = target_loader.verify_containers()
    observed_identities = {
        row["split"]: {
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in observed_containers["containers"]
    }
    if observed_identities != target_identities:
        raise IntegrityError("target containers differ from the opaque v2 identities")

    cities = list(manifest["cities"]["target"])
    if len(cities) != TARGET_CITY_COUNT or cities != sorted(set(cities)):
        raise IntegrityError("v2 target manifest must contain exactly ten target cities")
    validation_by_city = _partition(geo_index, manifest, split="validation", cities=cities)
    testing_by_city = _partition(geo_index, manifest, split="testing", cities=cities)
    selected_spec = candidate_spec(authority.controller["candidate_id"])
    if selected_spec["candidate_config_sha256"] != authority.controller["candidate_config_sha256"]:
        raise IntegrityError("live Tent configuration differs from the v2 controller")

    publication = _absolute_without_resolving(output_dir)
    staging = _reserve_output_directory_v2(publication)

    probe_samples_by_city: dict[str, Any] = {}
    city_feature_documents: dict[str, dict[str, dict[str, Any]]] = {}
    for city_id in cities:
        probe_samples = target_loader.read_verified_many("validation", validation_by_city[city_id])
        _freeze_sample_pixels(probe_samples)
        probe_samples_by_city[city_id] = probe_samples
        documents: dict[str, dict[str, Any]] = {}
        for checkpoint_id in CHECKPOINT_IDS:
            computation = executor.prepare_probe(checkpoints[checkpoint_id], selected_spec, probe_samples)
            if not isinstance(computation, ProbeComputation):
                raise IntegrityError("v2 executor must return ProbeComputation")
            documents[checkpoint_id] = _feature_document_or_unusable(
                computation,
                probe_count=len(probe_samples),
            )
        city_feature_documents[city_id] = documents

    evaluation_unlocked = False

    def unlock_evaluation() -> None:
        nonlocal evaluation_unlocked
        evaluation_unlocked = True

    action_plan = stage_v2_target_action_plan(
        authority=authority,
        city_feature_documents=city_feature_documents,
        destination=staging / "actions",
        before_evaluation=unlock_evaluation,
    )
    if not evaluation_unlocked or action_plan["cell_count"] != TARGET_CELL_COUNT:
        raise IntegrityError("v2 evaluation remained locked by an incomplete action plan")
    write_immutable_json_with_receipt(staging / "target_action_plan.json", action_plan)

    action_by_key = {(row["city_id"], row["checkpoint_id"]): row for row in action_plan["actions"]}
    cell_rows: list[dict[str, Any]] = []
    cells_dir = staging / "cells"
    cells_dir.mkdir()
    for city_index, city_id in enumerate(cities):
        probe_samples = probe_samples_by_city[city_id]
        prepared: dict[str, Any] = {}
        for checkpoint_id in CHECKPOINT_IDS:
            computation = executor.prepare_probe(checkpoints[checkpoint_id], selected_spec, probe_samples)
            if not isinstance(computation, ProbeComputation):
                raise IntegrityError("v2 executor must return ProbeComputation")
            replayed_feature = _feature_document_or_unusable(
                computation,
                probe_count=len(probe_samples),
            )
            if replayed_feature != city_feature_documents[city_id][checkpoint_id]:
                raise IntegrityError("v2 probe features changed before evaluation")
            prepared[checkpoint_id] = computation

        if not evaluation_unlocked:
            raise IntegrityError("testing pixels cannot open before all 50 actions")
        evaluation_samples = target_loader.read_verified_many("testing", testing_by_city[city_id])
        _freeze_sample_pixels(evaluation_samples)
        for checkpoint_id in CHECKPOINT_IDS:
            action_row = action_by_key[(city_id, checkpoint_id)]
            action_path = staging / "actions" / action_row["action_basename"]
            action, action_receipt = _secure_receipted_mapping(action_path)
            _validate_target_action_document(
                action,
                authority=authority,
                city_id=city_id,
                checkpoint_id=checkpoint_id,
                feature_documents=city_feature_documents[city_id],
            )
            evaluation = executor.evaluate_after_action(prepared[checkpoint_id], evaluation_samples)
            if not isinstance(evaluation, EvaluationComputation):
                raise IntegrityError("v2 executor must return EvaluationComputation")
            frozen = np.asarray(evaluation.frozen_evaluation_logits, dtype=np.float64)
            adapted = np.asarray(evaluation.adapted_evaluation_logits, dtype=np.float64)
            expected_shape = (len(evaluation_samples), 17)
            if (
                frozen.shape != expected_shape
                or adapted.shape != expected_shape
                or not np.isfinite(frozen).all()
                or not np.isfinite(adapted).all()
            ):
                raise IntegrityError("v2 executor returned malformed evaluation logits")
            selected_logits = adapted if action["realized_action"] == "ADAPT" else frozen
            if action["decision"] == "ABSTAIN" and selected_logits is not frozen:
                raise IntegrityError("v2 ABSTAIN did not retain the frozen model")
            archive_path = cells_dir / (f"city{city_index:02d}_checkpoint{checkpoint_id}.logits.npz")
            logit_archive = _write_logit_archive(
                archive_path,
                arrays={
                    "frozen_probe_logits": np.asarray(
                        prepared[checkpoint_id].frozen_probe_logits,
                        dtype=np.float64,
                    ),
                    "adapted_probe_logits": np.asarray(
                        prepared[checkpoint_id].adapted_probe_logits,
                        dtype=np.float64,
                    ),
                    "frozen_evaluation_logits": frozen,
                    "adapted_evaluation_logits": adapted,
                },
                execution_mode="PRODUCTION",
            )
            cell: dict[str, Any] = {
                "schema": TARGET_CELL_SCHEMA,
                "status": "SEALED_BEFORE_TARGET_OUTCOME_ACCESS",
                "protocol_id": PROTOCOL_ID,
                "controller_sha256": authority.controller["controller_sha256"],
                "authorization_sha256": authority.target_authorization["authorization_sha256"],
                "action_unit": "city_checkpoint",
                "city_id": city_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_file_sha256": checkpoints[checkpoint_id]["checkpoint_file_sha256"],
                "checkpoint_tensor_sha256": checkpoints[checkpoint_id]["checkpoint_tensor_sha256"],
                "probe_feature_document": city_feature_documents[city_id][checkpoint_id],
                "action": action,
                "action_artifact": _receipt_binding(action_receipt),
                "evaluation": {
                    "sample_count": len(evaluation_samples),
                    "frozen_prediction_class_ids": frozen.argmax(axis=1).tolist(),
                    "adapted_prediction_class_ids": adapted.argmax(axis=1).tolist(),
                    "selected_prediction_class_ids": selected_logits.argmax(axis=1).tolist(),
                    "selected_policy": action["realized_action"],
                    "target_outcomes_opened": False,
                },
                "logit_archive": logit_archive,
                "validation_labels_opened": False,
                "target_outcomes_opened": False,
            }
            cell["cell_sha256"] = stable_sha256(cell)
            cell_path = cells_dir / f"city{city_index:02d}_checkpoint{checkpoint_id}.json"
            cell_receipt = write_immutable_json_with_receipt(cell_path, cell)
            cell_rows.append(
                {
                    "city_id": city_id,
                    "checkpoint_id": checkpoint_id,
                    "cell_basename": f"cells/{cell_path.name}",
                    "cell_sha256": cell["cell_sha256"],
                    "artifact_sha256": cell_receipt["artifact_sha256"],
                    "canonical_document_sha256": cell_receipt["canonical_document_sha256"],
                    "action_sha256": action["action_sha256"],
                }
            )

    postrun_containers = target_loader.verify_containers()
    if postrun_containers["containers"] != observed_containers["containers"]:
        raise IntegrityError("v2 target containers changed during execution")
    access_audit = _verify_loader_audit(target_loader, manifest)
    cell_rows.sort(key=lambda row: (row["city_id"], row["checkpoint_id"]))
    bundle: dict[str, Any] = {
        "schema": TARGET_BUNDLE_SCHEMA,
        "status": "COMPLETE_50_CELLS_SEALED_BEFORE_TARGET_OUTCOME_ACCESS",
        "protocol_id": PROTOCOL_ID,
        "precalibration_seal_sha256": authority.precalibration_seal["precalibration_seal_sha256"],
        "gate_screen_sha256": authority.gate_screen["gate_screen_sha256"],
        "authorization_sha256": authority.target_authorization["authorization_sha256"],
        "controller_sha256": authority.controller["controller_sha256"],
        "action_plan_sha256": action_plan["action_plan_sha256"],
        "action_unit": "city_checkpoint",
        "target_cities": cities,
        "checkpoint_ids": list(CHECKPOINT_IDS),
        "cell_count": len(cell_rows),
        "cells": cell_rows,
        "access_audit": access_audit,
        "validation_labels_opened": False,
        "target_outcomes_opened": False,
        "complete_before_scoring": True,
    }
    if len(cell_rows) != TARGET_CELL_COUNT:
        raise IntegrityError("v2 target bundle is not the complete 50-cell grid")
    bundle["bundle_sha256"] = stable_sha256(bundle)
    bundle_path = staging / "so2sat_target_bundle_v2.json"
    write_immutable_json_with_receipt(bundle_path, bundle)
    for _ in range(2):
        replayed, _ = _secure_receipted_mapping(bundle_path)
        if replayed != bundle:
            raise IntegrityError("v2 target bundle changed during final replay")
        for row in cell_rows:
            cell_path = staging / row["cell_basename"]
            cell, receipt = _secure_receipted_mapping(cell_path)
            if (
                cell.get("cell_sha256") != row["cell_sha256"]
                or receipt["artifact_sha256"] != row["artifact_sha256"]
                or stable_sha256({key: value for key, value in cell.items() if key != "cell_sha256"})
                != cell["cell_sha256"]
            ):
                raise IntegrityError("v2 target cell changed during final replay")
    return _publish_and_replay_output_bundle_v2(
        staging,
        publication,
        bundle_path.name,
        bundle,
    )


def run_production_target_v2(
    *,
    precalibration_seal: str | os.PathLike[str],
    calibration_bundle: str | os.PathLike[str],
    gate_screen: str | os.PathLike[str],
    target_authorization: str | os.PathLike[str],
    population_manifest: str | os.PathLike[str],
    source_postrun_acceptance: str | os.PathLike[str],
    checkpoint_collection: str | os.PathLike[str],
    checkpoint_dir: str | os.PathLike[str],
    normalizer: str | os.PathLike[str],
    environment_identity: str | os.PathLike[str],
    geo_paths: Mapping[str, str | os.PathLike[str]],
    target_data_paths: Mapping[str, str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    device_name: str,
) -> Path:
    """Canonical v2 production entry point with authorization-before-path order."""

    authority = load_v2_runtime_authority(
        precalibration_seal=precalibration_seal,
        calibration_bundle=calibration_bundle,
        gate_screen=gate_screen,
        target_authorization=target_authorization,
    )
    return _run_authorized_target_v2(
        authority=authority,
        population_manifest_path=population_manifest,
        source_postrun_acceptance_path=source_postrun_acceptance,
        checkpoint_collection_path=checkpoint_collection,
        checkpoint_dir=checkpoint_dir,
        normalizer_path=normalizer,
        environment_identity_path=environment_identity,
        geo_paths=geo_paths,
        target_data_paths=target_data_paths,
        output_dir=output_dir,
        device_name=device_name,
    )


__all__ = [
    "TARGET_ACTION_DOCUMENT_SCHEMA",
    "TARGET_ACTION_PLAN_SCHEMA",
    "load_v2_runtime_authority",
    "load_secure_receipted_mapping_v2",
    "open_authorized_target_runtime_v2",
    "run_production_target_v2",
    "stage_v2_target_action_plan",
    "verify_published_directory_v2",
]
