#!/usr/bin/env python3
"""Create the immutable pre-inference CCT-20 execution seal.

The held-out annotation file and the two official archives are byte-hashed as
opaque dependencies; this command never deserializes the held-out annotation
document.  The separate target runner does not accept that document at all.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from .integrity import IntegrityError, file_sha256
from .prospective_data import validate_locked_target_population
from .protocol_seal import build_execution_seal, validate_execution_seal
from .ridge_gate import validate_gate_document
from .run_development_gate import validate_development_trace_collection
from .runner_runtime import (
    configure_deterministic_inference,
    load_json_object,
    load_sealed_json_object,
    select_inference_device,
    shared_runtime_dependency_paths,
    validate_shared_runtime_identity,
    validate_source_training_seal_identity,
    verify_checkpoint_audit_document,
    write_or_verify_immutable_json,
)
from .tent_official import verify_official_tent

TARGET_LOCATIONS = (0, 7, 28, 40, 46, 78, 100, 105, 130)


def _code_dependencies(
    repository_root: Path,
    args: argparse.Namespace,
    *,
    development_trace_artifacts: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, Path]:
    directory = repository_root / "experiments" / "kbound" / "cct20"
    gate_receipt = args.gate.with_name(args.gate.name + ".receipt.json")
    source_training_seal = repository_root / "research_lock" / "KBOUND_CCT20_SOURCE_TRAINING_SEAL_v1.json"
    development_collection = args.gate.parent / "development_trace_collection.json"
    dependencies = {
        "checkpoint_audit": args.checkpoint_audit,
        "development_gate": args.gate,
        "development_gate_receipt": gate_receipt,
        "development_trace_collection": development_collection,
        "development_trace_collection_receipt": development_collection.with_name(
            development_collection.name + ".receipt.json"
        ),
        "prospective_protocol": directory / "prospective_protocol_v1.yaml",
        "seal_cct20_execution": directory / "seal_cct20_execution.py",
        "run_development_gate": directory / "run_development_gate.py",
        "run_locked_target": directory / "run_locked_target.py",
        "runner_runtime": directory / "runner_runtime.py",
        "target_executor": directory / "target_executor.py",
        "prediction_artifacts": directory / "prediction_artifacts.py",
        "label_free_traces": directory / "label_free_traces.py",
        "ridge_gate": directory / "ridge_gate.py",
        "tent_official_binding": directory / "tent_official.py",
        "protocol_seal": directory / "protocol_seal.py",
        "integrity": directory / "integrity.py",
        "audit_checkpoints": directory / "audit_checkpoints.py",
        "prospective_data": directory / "prospective_data.py",
        "score_once": directory / "score_once.py",
        "two_way_inference": directory / "two_way_inference.py",
        "train_source": directory / "train_source.py",
        "official_tent_py": args.tent_repo / "tent.py",
        "source_training_seal": source_training_seal,
        "source_training_seal_receipt": source_training_seal.with_name(source_training_seal.name + ".receipt.json"),
        "shared_runtime_identity": args.shared_runtime_identity,
        "shared_runtime_identity_receipt": args.shared_runtime_identity.with_name(
            args.shared_runtime_identity.name + ".receipt.json"
        ),
        "downstream_execution_runtime_addendum": args.runtime_addendum,
    }
    if development_trace_artifacts is None:
        trace_paths = [args.gate.parent / "traces" / f"trace_{index:02d}.json" for index in range(55)]
    else:
        if len(development_trace_artifacts) != 55:
            raise IntegrityError("development collection must bind exactly 55 traces")
        ordered = sorted(
            development_trace_artifacts,
            key=lambda record: str(record.get("trace_id", "")),
        )
        trace_paths = [Path(str(record.get("artifact_path", ""))).expanduser().resolve() for record in ordered]
    for index, trace_path in enumerate(trace_paths):
        dependencies[f"development_trace_{index:02d}"] = trace_path
        dependencies[f"development_trace_receipt_{index:02d}"] = trace_path.with_name(trace_path.name + ".receipt.json")
    return dependencies


def _assert_source_seal_file(record: object, path: Path, *, name: str) -> None:
    if not isinstance(record, dict):
        raise IntegrityError(f"source-training seal lacks {name}")
    resolved = path.expanduser().resolve()
    if record.get("path") != str(resolved):
        raise IntegrityError(f"{name} path differs from the immutable source-training seal")
    if record.get("bytes") != resolved.stat().st_size:
        raise IntegrityError(f"{name} byte count differs from the immutable source-training seal")
    if record.get("sha256") != file_sha256(resolved):
        raise IntegrityError(f"{name} hash differs from the immutable source-training seal")


def _validate_source_training_lineage(
    source_seal: dict,
    *,
    source_receipt: dict,
    args: argparse.Namespace,
    runtime: dict,
) -> None:
    validate_source_training_seal_identity(source_seal, source_receipt)
    if runtime.get("source_training_seal_artifact_sha256") != source_receipt.get("artifact_sha256") or runtime.get(
        "source_training_seal_document_sha256"
    ) != source_receipt.get("canonical_document_sha256"):
        raise IntegrityError("shared runtime is not bound to the source-training seal")
    dataset = source_seal.get("dataset", {})
    _assert_source_seal_file(dataset.get("image_archive"), args.official_image_archive, name="official image archive")
    _assert_source_seal_file(
        dataset.get("annotation_archive"),
        args.official_annotation_archive,
        name="official annotation archive",
    )
    _assert_source_seal_file(
        dataset.get("annotation_splits", {}).get("trans_test"),
        args.target_annotations_json,
        name="held-out annotation JSON",
    )
    _assert_source_seal_file(
        source_seal.get("preflights", {}).get("target_label_free"),
        args.target_manifest,
        name="label-free target manifest",
    )
    source_runtime = source_seal.get("runtime", {})
    for field in ("python", "torch", "torchvision", "numpy", "pillow", "platform"):
        if runtime.get(field) != source_runtime.get(field):
            raise IntegrityError(f"shared runtime {field} differs from source-training seal")
    if (
        runtime.get("mps_available") != source_runtime.get("mps_available")
        or runtime.get("mps_built") != source_runtime.get("mps_built")
        or runtime.get("deterministic_algorithms_enabled") != source_runtime.get("deterministic_algorithms_required")
    ):
        raise IntegrityError("shared accelerator/determinism differs from source-training seal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-annotations-json", type=Path, required=True)
    parser.add_argument("--official-image-archive", type=Path, required=True)
    parser.add_argument("--official-annotation-archive", type=Path, required=True)
    parser.add_argument("--tent-repo", type=Path, required=True)
    parser.add_argument("--shared-runtime-identity", type=Path)
    parser.add_argument(
        "--runtime-addendum",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "research_lock"
        / "KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    args.checkpoint_audit = args.checkpoint_audit.expanduser().resolve()
    args.gate = args.gate.expanduser().resolve()
    args.target_manifest = args.target_manifest.expanduser().resolve()
    args.target_annotations_json = args.target_annotations_json.expanduser().resolve()
    args.official_image_archive = args.official_image_archive.expanduser().resolve()
    args.official_annotation_archive = args.official_annotation_archive.expanduser().resolve()
    args.tent_repo = args.tent_repo.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.runtime_addendum = args.runtime_addendum.expanduser().resolve()
    args.shared_runtime_identity = (
        args.shared_runtime_identity.expanduser().resolve()
        if args.shared_runtime_identity is not None
        else args.gate.parent / "shared_runtime_identity.json"
    )
    if args.target_annotations_json.name != "trans_test_annotations.json":
        raise IntegrityError("sealed held-out annotation dependency must be named trans_test_annotations.json")
    configure_deterministic_inference()
    device = select_inference_device(args.device)
    verify_official_tent(args.tent_repo)
    audit = load_json_object(args.checkpoint_audit)
    verify_checkpoint_audit_document(audit)
    gate, _ = load_sealed_json_object(args.gate)
    validate_gate_document(gate)
    development_collection_path = args.gate.parent / "development_trace_collection.json"
    development_collection, _ = load_sealed_json_object(development_collection_path)
    manifest = load_json_object(args.target_manifest)
    validate_locked_target_population(manifest)

    repository_root = Path(__file__).resolve().parents[3]
    source_training_seal_path = repository_root / "research_lock" / "KBOUND_CCT20_SOURCE_TRAINING_SEAL_v1.json"
    source_training_seal, source_training_receipt = load_sealed_json_object(source_training_seal_path)
    runtime, runtime_receipt = load_sealed_json_object(args.shared_runtime_identity)
    runtime_dependencies = shared_runtime_dependency_paths(
        repository_root,
        tent_repo=args.tent_repo,
        runtime_addendum=args.runtime_addendum,
    )
    validate_shared_runtime_identity(
        runtime,
        device=device,
        dependency_paths=runtime_dependencies,
    )
    _validate_source_training_lineage(
        source_training_seal,
        source_receipt=source_training_receipt,
        args=args,
        runtime=runtime,
    )
    runtime_binding = development_collection.get("shared_runtime_identity", {})
    if (
        not isinstance(runtime_binding, Mapping)
        or runtime_binding.get("artifact_path") != str(args.shared_runtime_identity)
        or runtime_binding.get("artifact_receipt") != runtime_receipt
        or runtime_binding.get("shared_runtime_sha256") != runtime.get("runtime_sha256")
    ):
        raise IntegrityError("development collection does not bind the selected shared runtime")
    validate_development_trace_collection(
        development_collection,
        gate_document=gate,
        checkpoint_audit=audit,
        verify_trace_files=True,
        verify_runtime_file=True,
    )
    protocol_path = Path(__file__).resolve().with_name("prospective_protocol_v1.yaml")
    try:
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise IntegrityError(f"cannot load authoritative prospective protocol: {exc}") from exc
    if not isinstance(protocol, dict):
        raise IntegrityError("authoritative prospective protocol is not a mapping")
    dataset_dependencies = {
        "target_annotations_json": args.target_annotations_json,
        "label_free_target_manifest": args.target_manifest,
        "official_image_archive": args.official_image_archive,
        "official_annotation_archive": args.official_annotation_archive,
    }
    seal = build_execution_seal(
        target_location_ids=TARGET_LOCATIONS,
        target_manifest_sha256=manifest["manifest_sha256"],
        dataset_dependencies=dataset_dependencies,
        code_dependencies=_code_dependencies(
            repository_root,
            args,
            development_trace_artifacts=development_collection["trace_artifacts"],
        ),
        checkpoint_audit=audit,
        gate_sha256=gate["gate_sha256"],
        protocol_config=protocol,
    )
    validate_execution_seal(seal)
    receipt = write_or_verify_immutable_json(args.output, seal)
    print(
        "CCT-20 execution sealed before target inference: "
        f"artifact_sha256={receipt['artifact_sha256']} -> {args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
