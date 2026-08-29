"""Create the final receipt-verified So2Sat target execution seal.

This entry point hashes ``validation.h5`` and ``testing.h5`` as opaque byte
containers.  It never opens either file as HDF5 and has no dataset-name API.
The resulting create-only seal must exist before the live target runner starts.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from .adapters import CANDIDATE_IDS
from .development import load_gate_authorization_with_receipt
from .gate import load_gate_with_receipt, load_study_binding
from .integrity import (
    IntegrityError,
    stable_sha256,
    strict_json_load,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from .metadata_manifest import validate_population_manifest
from .precalibration_seal import (
    load_precalibration_seal_with_receipt,
    precalibration_code_identity,
    validate_reveal_registry_directory,
)
from .target_amendment import load_target_boundary_amendment
from .target_contract import (
    PRODUCTION_MODE,
    TEST_ONLY_MODE,
    load_source_postrun_acceptance_pair,
    opaque_target_identities_from_paths,
    target_scorer_code_identity,
    target_scorer_environment_identity,
    validate_checkpoint_collection,
)
from .target_contract import artifact_binding as _artifact_binding
from .target_inference import TorchTargetCellExecutor
from .target_runner import (
    _PRODUCTION_SEAL_BUILD_AUTHORITY,
    build_execution_seal,
)

_PRODUCTION_SEAL_AUTHORITY = object()


def _verified_json(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = verify_artifact_receipt(path)
    document = strict_json_load(path)
    if not isinstance(document, dict):
        raise IntegrityError(f"receipt-verified seal input is not a JSON mapping: {path}")
    return document, receipt


def _create_execution_seal_core(
    *,
    population_manifest_path: str | Path,
    source_postrun_acceptance_path: str | Path,
    selected_candidate_path: str | Path,
    selected_gate_fit_bundle_path: str | Path,
    selected_gate_cal_bundle_path: str | Path,
    precalibration_seal_path: str | Path,
    gate_path: str | Path,
    gate_authorization_path: str | Path,
    target_boundary_amendment_path: str | Path,
    checkpoint_collection_path: str | Path,
    checkpoint_dir: str | Path,
    reveal_registry_dir: str | Path,
    target_data_paths: Mapping[str, str | Path],
    output_path: str | Path,
    live_code_identity_sha256: str,
    live_environment_identity_sha256: str,
    live_normalizer_sha256: str,
    execution_mode: str,
    population_manifest_validator: Callable[[Mapping[str, Any]], None],
    production_authority: object | None = None,
) -> Path:
    if execution_mode == PRODUCTION_MODE and (
        production_authority is not _PRODUCTION_SEAL_AUTHORITY
        or population_manifest_validator is not validate_population_manifest
    ):
        raise IntegrityError("production seal core rejects injected construction authority")
    if execution_mode not in {PRODUCTION_MODE, TEST_ONLY_MODE}:
        raise IntegrityError("target seal execution mode is invalid")
    manifest, _ = _verified_json(population_manifest_path)
    population_manifest_validator(manifest)
    source_acceptance, _, source_acceptance_binding = (
        load_source_postrun_acceptance_pair(
            source_postrun_acceptance_path,
            strict_document=execution_mode == PRODUCTION_MODE,
        )
    )
    study_binding = load_study_binding(population_manifest_path)
    gate = load_gate_with_receipt(gate_path)
    gate_receipt = verify_artifact_receipt(gate_path)
    if gate.get("study_binding") != study_binding:
        raise IntegrityError("target seal gate and manifest study binding differ")
    selection, selection_receipt = _verified_json(selected_candidate_path)
    fit_bundle, fit_bundle_receipt = _verified_json(selected_gate_fit_bundle_path)
    authorization, authorized_selection, authorized_gate = (
        load_gate_authorization_with_receipt(
            gate_authorization_path,
            selection_path=selected_candidate_path,
            gate_path=gate_path,
            population_manifest_path=population_manifest_path,
            fit_bundle_path=selected_gate_fit_bundle_path,
            calibration_bundle_path=selected_gate_cal_bundle_path,
        )
    )
    authorization_receipt = verify_artifact_receipt(gate_authorization_path)
    if authorized_selection != selection or authorized_gate != gate:
        raise IntegrityError("target seal gate authorization replay changed its inputs")
    amendment, amendment_receipt = load_target_boundary_amendment(
        target_boundary_amendment_path
    )
    collection, collection_receipt = _verified_json(checkpoint_collection_path)
    validate_checkpoint_collection(
        collection,
        collection_receipt=collection_receipt,
        collection_path=checkpoint_collection_path,
        checkpoint_dir=checkpoint_dir,
    )
    if (
        authorization["checkpoint_collection_canonical_sha256"]
        != stable_sha256(collection)
        or authorization["normalizer_sha256"] != collection["normalizer_sha256"]
        or live_normalizer_sha256 != collection["normalizer_sha256"]
    ):
        raise IntegrityError("target seal checkpoint collection differs from gate authorization")
    target_identities = opaque_target_identities_from_paths(target_data_paths)
    scorer_code = target_scorer_code_identity()
    scorer_environment = target_scorer_environment_identity()
    precalibration_seal, precalibration_seal_receipt = (
        load_precalibration_seal_with_receipt(
            precalibration_seal_path,
            study_binding=study_binding,
            selection=selection,
            fit_bundle=fit_bundle,
            target_boundary_amendment=amendment,
            checkpoint_collection=collection,
        )
    )
    if (
        precalibration_seal["execution_mode"] != execution_mode
        or precalibration_seal["population_manifest_artifact"]
        != _artifact_binding(verify_artifact_receipt(population_manifest_path))
        or precalibration_seal["selection_artifact"] != _artifact_binding(selection_receipt)
        or precalibration_seal["selected_gate_fit_bundle_artifact"]
        != _artifact_binding(fit_bundle_receipt)
        or precalibration_seal["target_boundary_amendment_artifact"]
        != _artifact_binding(amendment_receipt)
        or precalibration_seal["checkpoint_collection_artifact"]
        != _artifact_binding(collection_receipt)
        or precalibration_seal["target_data_identities"] != target_identities
        or precalibration_seal["source_postrun_acceptance"]
        != source_acceptance_binding
        or precalibration_seal["source_postrun_training_container"]
        != source_acceptance["postrun_source_container"]
        or precalibration_seal["source_hdf5_runtime_disclosure"]
        != source_acceptance["source_hdf5_runtime_disclosure"]
        or precalibration_seal["source_checkpoint_selection_disclosure"]
        != source_acceptance["source_checkpoint_selection_disclosure"]
        or precalibration_seal["source_initialization_clarification"]
        != source_acceptance["source_initialization_clarification"]
        or precalibration_seal["frozen_gate_fit_model"] != gate["ridge"]
        or precalibration_seal["package_code_identity"]
        != precalibration_code_identity()
        or precalibration_seal["target_live_environment_identity"]
        .get("environment_identity_sha256")
        != live_environment_identity_sha256
        or precalibration_seal["offline_scorer_environment_identity"]
        != scorer_environment
    ):
        raise IntegrityError("final target seal does not extend the prior precalibration seal")
    validate_reveal_registry_directory(
        reveal_registry_dir, precalibration_seal["outcome_reveal_registry"]
    )
    document = build_execution_seal(
        study_binding=study_binding,
        selected_candidate=selection,
        selected_candidate_receipt=selection_receipt,
        selected_gate_fit_bundle=fit_bundle,
        gate=gate,
        gate_receipt=gate_receipt,
        gate_authorization=authorization,
        gate_authorization_receipt=authorization_receipt,
        target_boundary_amendment=amendment,
        target_boundary_amendment_receipt=amendment_receipt,
        checkpoint_collection=collection,
        checkpoint_collection_receipt=collection_receipt,
        precalibration_seal=precalibration_seal,
        precalibration_seal_receipt=precalibration_seal_receipt,
        target_data_identities=target_identities,
        code_identity_sha256=live_code_identity_sha256,
        environment_identity_sha256=live_environment_identity_sha256,
        scorer_code_identity_sha256=scorer_code["code_identity_sha256"],
        scorer_environment_identity_sha256=scorer_environment[
            "environment_identity_sha256"
        ],
        execution_mode=execution_mode,
        _production_authority=(
            _PRODUCTION_SEAL_BUILD_AUTHORITY
            if execution_mode == PRODUCTION_MODE
            else None
        ),
    )
    destination = Path(output_path).expanduser().resolve()
    write_immutable_json_with_receipt(destination, document)
    return destination


def create_production_execution_seal(
    *,
    population_manifest_path: str | Path,
    source_postrun_acceptance_path: str | Path,
    selected_candidate_path: str | Path,
    selected_gate_fit_bundle_path: str | Path,
    selected_gate_cal_bundle_path: str | Path,
    precalibration_seal_path: str | Path,
    gate_path: str | Path,
    gate_authorization_path: str | Path,
    target_boundary_amendment_path: str | Path,
    checkpoint_collection_path: str | Path,
    checkpoint_dir: str | Path,
    reveal_registry_dir: str | Path,
    normalizer_path: str | Path,
    target_data_paths: Mapping[str, str | Path],
    output_path: str | Path,
    device_name: str,
) -> Path:
    """Create a PRODUCTION seal using only canonical concrete implementations."""

    if device_name not in {"cpu", "mps"}:
        raise IntegrityError("target seal device must be exactly 'cpu' or 'mps'")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise IntegrityError("MPS target sealing was requested but MPS is unavailable")
    selection, _ = _verified_json(selected_candidate_path)
    candidate_id = selection.get("selected_candidate_id")
    if candidate_id not in CANDIDATE_IDS:
        raise IntegrityError("target seal requires one selected adapter")
    executor = TorchTargetCellExecutor(
        candidate_id=str(candidate_id),
        normalizer_path=normalizer_path,
        device=torch.device(device_name),
    )
    return _create_execution_seal_core(
        population_manifest_path=population_manifest_path,
        source_postrun_acceptance_path=source_postrun_acceptance_path,
        selected_candidate_path=selected_candidate_path,
        selected_gate_fit_bundle_path=selected_gate_fit_bundle_path,
        selected_gate_cal_bundle_path=selected_gate_cal_bundle_path,
        precalibration_seal_path=precalibration_seal_path,
        gate_path=gate_path,
        gate_authorization_path=gate_authorization_path,
        target_boundary_amendment_path=target_boundary_amendment_path,
        checkpoint_collection_path=checkpoint_collection_path,
        checkpoint_dir=checkpoint_dir,
        reveal_registry_dir=reveal_registry_dir,
        target_data_paths=target_data_paths,
        output_path=output_path,
        live_code_identity_sha256=executor.code_identity_sha256,
        live_environment_identity_sha256=executor.environment_identity_sha256,
        live_normalizer_sha256=executor.normalizer_sha256,
        execution_mode=PRODUCTION_MODE,
        population_manifest_validator=validate_population_manifest,
        production_authority=_PRODUCTION_SEAL_AUTHORITY,
    )


def _create_execution_seal_for_test(
    *,
    live_code_identity_sha256: str,
    live_environment_identity_sha256: str,
    live_normalizer_sha256: str,
    population_manifest_validator: Callable[[Mapping[str, Any]], None],
    **kwargs: Any,
) -> Path:
    """Create only a visibly TEST_ONLY seal for deterministic synthetic tests."""

    return _create_execution_seal_core(
        **kwargs,
        live_code_identity_sha256=live_code_identity_sha256,
        live_environment_identity_sha256=live_environment_identity_sha256,
        live_normalizer_sha256=live_normalizer_sha256,
        execution_mode=TEST_ONLY_MODE,
        population_manifest_validator=population_manifest_validator,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-manifest", required=True)
    parser.add_argument("--source-postrun-acceptance", required=True)
    parser.add_argument("--selected-candidate", required=True)
    parser.add_argument("--selected-gate-fit-bundle", required=True)
    parser.add_argument("--selected-gate-cal-bundle", required=True)
    parser.add_argument("--precalibration-seal", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--gate-authorization", required=True)
    parser.add_argument("--target-boundary-amendment", required=True)
    parser.add_argument("--checkpoint-collection", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--reveal-registry-dir", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--testing-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = create_production_execution_seal(
        population_manifest_path=arguments.population_manifest,
        source_postrun_acceptance_path=arguments.source_postrun_acceptance,
        selected_candidate_path=arguments.selected_candidate,
        selected_gate_fit_bundle_path=arguments.selected_gate_fit_bundle,
        selected_gate_cal_bundle_path=arguments.selected_gate_cal_bundle,
        precalibration_seal_path=arguments.precalibration_seal,
        gate_path=arguments.gate,
        gate_authorization_path=arguments.gate_authorization,
        target_boundary_amendment_path=arguments.target_boundary_amendment,
        checkpoint_collection_path=arguments.checkpoint_collection,
        checkpoint_dir=arguments.checkpoint_dir,
        reveal_registry_dir=arguments.reveal_registry_dir,
        normalizer_path=arguments.normalizer,
        target_data_paths={
            "validation": arguments.validation_data,
            "testing": arguments.testing_data,
        },
        output_path=arguments.output,
        device_name=arguments.device,
    )
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI integration
    raise SystemExit(main())
