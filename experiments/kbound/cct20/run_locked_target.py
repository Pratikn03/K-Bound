#!/usr/bin/env python3
"""Execute the sealed label-free CCT-20 target streams and prediction grid.

This entry point accepts only the verified image manifest, pixels, sealed gate,
and execution seal.  It cannot accept or import the held-out scorer.  For each
fresh checkpoint-camera session it runs every probe sequence, seals the action,
then runs every evaluation sequence.  Completed shards are verified and
skipped; an action-only interrupted shard is resumed only after exact probe
replay against the immutable action artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .integrity import IntegrityError
from .label_free_traces import assert_label_free
from .prediction_artifacts import (
    build_prediction_collection,
    validate_prediction_cell,
)
from .prospective_data import validate_locked_target_population
from .protocol_seal import verify_execution_environment
from .ridge_gate import validate_gate_document
from .runner_runtime import (
    VerifiedImageStore,
    clear_device_cache,
    configure_deterministic_inference,
    load_checkpoint_model_pair,
    load_json_object,
    load_sealed_json_object,
    paired_forward,
    select_inference_device,
    shared_runtime_dependency_paths,
    validate_shared_runtime_identity,
    write_or_verify_immutable_json,
)
from .target_executor import LabelFreeTargetCell
from .tent_official import (
    FrozenBatchNormMomentAccumulator,
    install_locked_root_bn_cpu_fallback,
    new_checkpoint_location_session,
    verify_official_tent,
)

TARGET_LOCATIONS = ("0", "7", "28", "40", "46", "78", "100", "105", "130")


def _dependency_map(rows: Any, *, field: str) -> dict[str, Path]:
    if not isinstance(rows, list):
        raise IntegrityError(f"execution seal {field} is not a dependency list")
    result = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise IntegrityError(f"execution seal {field} contains a non-object")
        name = str(row.get("name", ""))
        path = Path(str(row.get("path", ""))).expanduser().resolve()
        if not name or name in result:
            raise IntegrityError(f"execution seal {field} has an empty/duplicate name")
        result[name] = path
    return result


def normalize_target_manifest(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return only the six label-free fields needed by the target runtime."""

    images = manifest.get("images")
    samples = manifest.get("samples")
    if not isinstance(images, list) or not isinstance(samples, list) or len(images) != len(samples):
        raise IntegrityError("target manifest image/sample arrays are missing or misaligned")
    metadata_rows = []
    expected_samples = []
    seen: set[str] = set()
    for index, (image, sample) in enumerate(zip(images, samples, strict=True)):
        if not isinstance(image, Mapping) or not isinstance(sample, Mapping):
            raise IntegrityError(f"target manifest row {index} is not an object")
        if "frame_num" not in image or "date_captured" not in image:
            raise IntegrityError(f"target manifest row {index} lacks frame_num/date_captured stream metadata")
        image_id = str(image.get("id", ""))
        if not image_id or image_id in seen:
            raise IntegrityError("string-normalized target image ids are empty or duplicate")
        seen.add(image_id)
        if image.get("id") != sample.get("id") or image.get("file_name") != sample.get("file_name"):
            raise IntegrityError(f"target manifest row {index} image/sample mismatch")
        row = {
            "image_id": image_id,
            "sequence_id": str(image.get("seq_id", "")),
            "location_id": str(image.get("location", "")),
            "file_name": image.get("file_name"),
            "frame_num": image["frame_num"],
            "date_captured": image["date_captured"],
        }
        assert_label_free(row, path=f"target_metadata[{index}]")
        metadata_rows.append(row)
        expected_samples.append(
            {
                "image_id": image_id,
                "file_name": sample.get("file_name"),
                "image_bytes": sample.get("image_bytes"),
                "image_sha256": sample.get("image_sha256"),
            }
        )
    if {row["location_id"] for row in metadata_rows} != set(TARGET_LOCATIONS):
        raise IntegrityError("target manifest does not contain exactly the nine sealed cameras")
    return metadata_rows, expected_samples


def _artifact_pair_state(path: Path) -> str:
    receipt = path.with_name(path.name + ".receipt.json")
    if path.exists() and receipt.exists():
        return "complete"
    if not path.exists() and not receipt.exists():
        return "absent"
    raise IntegrityError(f"incomplete immutable artifact/receipt pair: {path}")


def _validate_completed_cell(
    path: Path,
    *,
    action_path: Path,
    execution_artifact_sha256: str,
    gate_sha256: str,
    manifest_sha256: str,
    checkpoint_row: Mapping[str, Any],
    location_id: str,
) -> dict[str, Any]:
    document, _ = load_sealed_json_object(path)
    validate_prediction_cell(document)
    expected = {
        "protocol_seal_sha256": execution_artifact_sha256,
        "gate_sha256": gate_sha256,
        "target_manifest_sha256": manifest_sha256,
        "checkpoint_seed": checkpoint_row["model_seed"],
        "checkpoint_tensor_sha256": checkpoint_row["tensor_sha256"],
        "location_id": location_id,
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise IntegrityError(f"completed prediction cell {field} identity mismatch")
    embedded_path = Path(document["gate"]["action_receipt"]["artifact_path"]).resolve()
    if embedded_path != action_path.resolve():
        raise IntegrityError("completed cell points to a different immutable action path")
    return document


def _forward_role(
    *,
    cell: LabelFreeTargetCell,
    role: str,
    frozen_model: Any,
    adapted_model: Any,
    image_store: VerifiedImageStore,
    device: Any,
) -> None:
    batches = cell.batch_plan(role)
    for batch_index, batch in enumerate(batches):
        images = image_store.tensor_batch(batch)
        frozen_logits, adapted_logits = paired_forward(frozen_model, adapted_model, images, device=device)
        cell.record_batch(
            role=role,
            batch_index=batch_index,
            image_ids=[row["image_id"] for row in batch],
            frozen_logits=frozen_logits,
            tent_logits=adapted_logits,
        )
        if batch_index == 0 or (batch_index + 1) % 10 == 0 or batch_index + 1 == len(batches):
            print(
                f"  {role}: batch {batch_index + 1}/{len(batches)} "
                f"images={sum(len(value) for value in batches[: batch_index + 1])}",
                flush=True,
            )


def _execute_cell(
    *,
    metadata_rows: Sequence[Mapping[str, Any]],
    image_store: VerifiedImageStore,
    checkpoint_row: Mapping[str, Any],
    location_id: str,
    execution_artifact_sha256: str,
    manifest_sha256: str,
    gate: Mapping[str, Any],
    tent_repo: Path,
    device: Any,
    action_path: Path,
    cell_path: Path,
) -> dict[str, Any]:
    frozen, adapted_source = load_checkpoint_model_pair(checkpoint_row, device=device)
    backend_installation = install_locked_root_bn_cpu_fallback(adapted_source)
    binding = new_checkpoint_location_session(
        adapted_source,
        repo_root=tent_repo,
        checkpoint_tensor_sha256=checkpoint_row["tensor_sha256"],
        location_id=location_id,
        backend_installation_receipt=backend_installation,
    )
    cell = LabelFreeTargetCell(
        metadata_rows,
        checkpoint_seed=int(checkpoint_row["model_seed"]),
        checkpoint_tensor_sha256=checkpoint_row["tensor_sha256"],
        location_id=location_id,
        protocol_seal_sha256=execution_artifact_sha256,
        target_manifest_sha256=manifest_sha256,
        gate_document=gate,
        tent_binding_receipt=binding.receipt(),
    )
    accumulator = FrozenBatchNormMomentAccumulator(frozen)
    _forward_role(
        cell=cell,
        role="probe",
        frozen_model=frozen,
        adapted_model=binding.adapter,
        image_store=image_store,
        device=device,
    )
    moments = accumulator.finalize()
    update = binding.probe_update_receipt()
    action_state = _artifact_pair_state(action_path)
    if action_state == "complete":
        gate_result = cell.restore_sealed_probe_action(
            tent_update_receipt=update,
            frozen_bn_probe_moment_receipt=moments,
            action_output_path=action_path,
        )
        action_verb = "restored after exact probe replay"
    else:
        gate_result = cell.seal_probe_action(
            tent_update_receipt=update,
            frozen_bn_probe_moment_receipt=moments,
            action_output_path=action_path,
        )
        action_verb = "sealed"
    print(
        f"  action {action_verb}: {gate_result['decision']} "
        f"interval=[{gate_result.get('lower')}, {gate_result.get('upper')}]",
        flush=True,
    )
    _forward_role(
        cell=cell,
        role="evaluation",
        frozen_model=frozen,
        adapted_model=binding.adapter,
        image_store=image_store,
        device=device,
    )
    document = cell.finalize()
    write_or_verify_immutable_json(cell_path, document)
    del cell, binding, adapted_source, frozen
    clear_device_cache(device)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-seal", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--tent-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    configure_deterministic_inference()
    device = select_inference_device(args.device)
    seal, seal_receipt = load_sealed_json_object(args.execution_seal)
    gate, _ = load_sealed_json_object(args.gate)
    validate_gate_document(gate)
    manifest = load_json_object(args.target_manifest)
    validate_locked_target_population(manifest)
    verify_official_tent(args.tent_repo)
    dataset_dependencies = _dependency_map(seal.get("dataset_dependencies"), field="dataset_dependencies")
    code_dependencies = _dependency_map(seal.get("code_dependencies"), field="code_dependencies")
    if dataset_dependencies.get("label_free_target_manifest") != args.target_manifest.resolve():
        raise IntegrityError("target manifest path differs from the execution seal")
    if code_dependencies.get("development_gate") != args.gate.resolve():
        raise IntegrityError("gate path differs from the execution seal")
    if code_dependencies.get("official_tent_py") != (args.tent_repo / "tent.py").resolve():
        raise IntegrityError("official Tent path differs from the execution seal")
    runtime_path = code_dependencies.get("shared_runtime_identity")
    if runtime_path is None:
        raise IntegrityError("execution seal lacks the shared runtime identity")
    runtime, runtime_receipt = load_sealed_json_object(runtime_path)
    if code_dependencies.get("shared_runtime_identity_receipt") != runtime_path.with_name(
        runtime_path.name + ".receipt.json"
    ):
        raise IntegrityError("execution seal shared-runtime receipt path mismatch")
    if runtime_receipt.get("artifact_sha256") is None:
        raise IntegrityError("shared runtime identity lacks an immutable receipt")
    runtime_addendum = code_dependencies.get("downstream_execution_runtime_addendum")
    if runtime_addendum is None:
        raise IntegrityError("execution seal lacks the downstream runtime addendum")
    source_seal_path = code_dependencies.get("source_training_seal")
    if source_seal_path is None:
        raise IntegrityError("execution seal lacks the immutable source-training seal")
    source_seal, source_seal_receipt = load_sealed_json_object(source_seal_path)
    if code_dependencies.get("source_training_seal_receipt") != source_seal_path.with_name(
        source_seal_path.name + ".receipt.json"
    ):
        raise IntegrityError("execution seal source-training receipt path mismatch")
    if (
        source_seal.get("schema") != "kbound_cct20_source_training_seal_v1"
        or source_seal.get("status") != "SEALED_BEFORE_SOURCE_TRAINING_AND_TARGET_OUTCOMES"
        or runtime.get("source_training_seal_artifact_sha256") != source_seal_receipt.get("artifact_sha256")
        or runtime.get("source_training_seal_document_sha256") != source_seal_receipt.get("canonical_document_sha256")
    ):
        raise IntegrityError("shared runtime is not bound to the immutable source-training seal")
    runtime_dependencies = shared_runtime_dependency_paths(
        Path(__file__).resolve().parents[3],
        tent_repo=args.tent_repo,
        runtime_addendum=runtime_addendum,
    )
    validate_shared_runtime_identity(
        runtime,
        device=device,
        dependency_paths=runtime_dependencies,
    )
    if seal.get("gate_sha256") != gate.get("gate_sha256"):
        raise IntegrityError("gate hash differs from the execution seal")
    population = seal.get("population", {})
    if population.get("target_manifest_sha256") != manifest.get("manifest_sha256"):
        raise IntegrityError("target-manifest hash differs from the execution seal")
    if tuple(str(value) for value in population.get("target_location_ids", ())) != TARGET_LOCATIONS:
        raise IntegrityError("target camera order differs from the execution seal")
    execution_artifact_sha256 = seal_receipt["artifact_sha256"]
    metadata, expected_samples = normalize_target_manifest(manifest)
    image_store = VerifiedImageStore(args.image_root, expected_samples)
    by_location = {
        location: [row for row in metadata if row["location_id"] == location] for location in TARGET_LOCATIONS
    }

    output = args.output_dir.expanduser().resolve()
    actions_dir = output / "actions"
    cells_dir = output / "cells"
    actions_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [dict(row) for row in seal.get("checkpoints", ())]
    if [row.get("model_seed") for row in checkpoints] != list(range(5)):
        raise IntegrityError("execution seal checkpoint grid is not seeds 0..4")
    cells = []
    environment_verified = False
    for checkpoint_row in checkpoints:
        seed = int(checkpoint_row["model_seed"])
        for location_id in TARGET_LOCATIONS:
            action_path = actions_dir / f"seed{seed}_location{location_id}_action.json"
            cell_path = cells_dir / f"seed{seed}_location{location_id}_predictions.json"
            cell_state = _artifact_pair_state(cell_path)
            action_state = _artifact_pair_state(action_path)
            if cell_state == "complete":
                if action_state != "complete":
                    raise IntegrityError("completed cell lacks its immutable action pair")
                cell = _validate_completed_cell(
                    cell_path,
                    action_path=action_path,
                    execution_artifact_sha256=execution_artifact_sha256,
                    gate_sha256=gate["gate_sha256"],
                    manifest_sha256=manifest["manifest_sha256"],
                    checkpoint_row=checkpoint_row,
                    location_id=location_id,
                )
                print(f"target cell verified/resumed: seed={seed} location={location_id}", flush=True)
            else:
                if not environment_verified:
                    # This is intentionally immediately before the first target forward.
                    verify_execution_environment(seal)
                    environment_verified = True
                    print(
                        "execution-seal receipt and complete environment verified before target forward",
                        flush=True,
                    )
                print(f"target cell start: seed={seed} location={location_id}", flush=True)
                cell = _execute_cell(
                    metadata_rows=by_location[location_id],
                    image_store=image_store,
                    checkpoint_row=checkpoint_row,
                    location_id=location_id,
                    execution_artifact_sha256=execution_artifact_sha256,
                    manifest_sha256=manifest["manifest_sha256"],
                    gate=gate,
                    tent_repo=args.tent_repo,
                    device=device,
                    action_path=action_path,
                    cell_path=cell_path,
                )
                print(
                    f"target cell sealed: seed={seed} location={location_id} decision={cell['gate']['decision']}",
                    flush=True,
                )
            cells.append(cell)

    # Revalidate all sealed dependencies after the last forward as well.  This
    # covers an all-complete resume and detects any mid-run dependency drift
    # before the collection becomes immutable.
    verify_execution_environment(seal)
    validate_shared_runtime_identity(
        runtime,
        device=device,
        dependency_paths=runtime_dependencies,
    )
    print("execution environment reverified before collection seal", flush=True)
    collection = build_prediction_collection(
        cells,
        target_index=metadata,
        target_location_ids=TARGET_LOCATIONS,
    )
    collection_path = output / "prediction_collection.json"
    receipt = write_or_verify_immutable_json(collection_path, collection)
    print(
        "label-free CCT-20 prediction grid sealed: "
        f"cells={collection['cell_count']} actions={collection['action_counts_at_cell_unit']} "
        f"artifact_sha256={receipt['artifact_sha256']} -> {collection_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
