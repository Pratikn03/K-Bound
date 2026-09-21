"""Current-run preparation and locked calibration for the unchanged So2Sat v2.

No calibration image/label reader is constructed before both current-run and
v2 pre-calibration locks validate. The historical v1 feasibility stop is kept.
Target execution remains behind the existing independently recomputed v2 gate.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import resource
import shutil
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from experiments.kbound.so2sat.integrity import (
    file_sha256,
    load_verified_json_mapping_with_receipt,
    stable_sha256,
    strict_json_load,
    write_immutable_json_with_receipt,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/research/kbound/next_phase/natural_protocol.json"
REFERENCE = Path("/Users/pratik_n/Desktop/AutoML_Flagship_V8 2")
DEFAULT_DATA = Path("/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/data/so2sat_lcz42_v4.2/v4")
DEFAULT_SOURCE = Path(
    "/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/so2sat_lcz42_prospective_v1/source"
)
DEFAULT_OUTPUT = ROOT / "output/next_phase/natural_v2"
DEFAULT_OPAQUE = Path("/Volumes/T9/kbound_next_phase_20260921/natural_v2_opaque")
OPENED_DEV = ROOT / "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def zero_error_upper_bound(n: int, delta: float) -> float:
    if n < 1 or not 0 < delta < 1:
        raise ValueError("invalid binomial sample size or confidence level")
    return 1 - delta ** (1 / n)


def validate_access(audit: Mapping[str, Any]) -> None:
    fields = (
        "current_run_calibration_pixels_read",
        "current_run_calibration_labels_read",
        "current_run_target_pixels_read",
        "current_run_target_labels_read",
    )
    if any(type(audit.get(key)) is not int or audit[key] != 0 for key in fields):
        raise ValueError("current-run access already occurred or is unknown")
    if audit.get("unrecognized_prior_access_artifacts") != []:
        raise ValueError("unrecognized prior access artifacts require an audit")
    if audit.get("historical_access_status") != "DOCUMENTED_ZERO_RESERVED_ACCESS_NOT_EXTERNALLY_PROVEN":
        raise ValueError("historical access must retain its documented uncertainty")


def make_run_lock(protocol: Mapping[str, Any], files: Mapping[str, Path], access: Mapping[str, Any]) -> dict:
    validate_access(access)
    doc = {
        "schema": "kbound_natural_current_run_lock_v1",
        "created_utc": utc(),
        "protocol": copy.deepcopy(dict(protocol)),
        "access": copy.deepcopy(dict(access)),
        "files": {
            name: {"basename": Path(path).name, "bytes": Path(path).stat().st_size, "sha256": file_sha256(path)}
            for name, path in sorted(files.items())
        },
    }
    doc["lock_sha256"] = stable_sha256(doc)
    return doc


def verify_run_lock(lock: Mapping[str, Any], files: Mapping[str, Path]) -> None:
    unsigned = dict(lock)
    claimed = unsigned.pop("lock_sha256", None)
    if stable_sha256(unsigned) != claimed:
        raise ValueError("run lock hash mismatch")
    validate_access(lock["access"])
    if set(files) != set(lock["files"]):
        raise ValueError("run lock input inventory changed")
    for name, path in files.items():
        row = lock["files"][name]
        if (
            Path(path).name != row["basename"]
            or Path(path).stat().st_size != row["bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise ValueError(f"bound input changed: {name}")


def dispatch_eligible_target(screen: Mapping[str, Any], target: Callable[[], Any]) -> Any:
    if (
        screen.get("passed") is not True
        or screen.get("cell_count") != 95
        or screen.get("status") != "PASSED_GATE_AUTHORIZATION_SCREEN"
        or set(screen.get("checks", {}))
        != {
            "complete_19_city_95_cell_grid",
            "at_least_7_direct_adapt_cities",
            "at_least_7_direct_freeze_cities",
            "zero_target_access",
        }
        or not all(value is True for value in screen["checks"].values())
    ):
        raise ValueError("NO_TARGET_ACCESS: incomplete or failed calibration screen")
    return target()


def score_cell(cell: Mapping[str, Any], truth: Any) -> dict:
    """Score fixed predictions without modifying or reselecting the action."""
    import numpy as np

    action, evaluation = cell["action"], cell["evaluation"]
    truth = np.asarray(truth)
    frozen = np.asarray(evaluation["frozen_prediction_class_ids"])
    adapted = np.asarray(evaluation["adapted_prediction_class_ids"])
    selected = np.asarray(evaluation["selected_prediction_class_ids"])
    expected_policy = "ADAPT" if action["decision"] == "ADAPT" else "FREEZE"
    expected = adapted if expected_policy == "ADAPT" else frozen
    if action["realized_action"] != expected_policy or not np.array_equal(selected, expected):
        raise ValueError("predictions contradict sealed action")
    if (
        truth.ndim != 1
        or len(truth) != evaluation["sample_count"]
        or not len(truth)
        or any(array.shape != truth.shape for array in (frozen, adapted, selected))
    ):
        raise ValueError("scoring vectors are misaligned")
    for array in (truth, frozen, adapted, selected):
        if not np.issubdtype(array.dtype, np.integer) or np.any((array < 0) | (array > 16)):
            raise ValueError("invalid class IDs")
    return {
        "city_id": cell["city_id"],
        "checkpoint_id": cell["checkpoint_id"],
        "decision": action["decision"],
        "realized_action": action["realized_action"],
        "observed_benefit": float(np.mean(adapted == truth) - np.mean(frozen == truth)),
        "action_sha256": action["action_sha256"],
    }


def load_pair(path: Path) -> tuple[dict, dict]:
    return load_verified_json_mapping_with_receipt(path)


def _write(path: Path, document: Mapping[str, Any]) -> dict:
    return write_immutable_json_with_receipt(path, document)


def _materialize_prerequisites() -> None:
    """Copy exact pinned public adapter files, retaining original licenses."""
    from experiments.kbound.so2sat.adapters import CANDIDATE_IDS, verify_official_adapter_sources

    for package, names in {
        "tent_official": ("tent.py", "LICENSE"),
        "sar_official": ("sar.py", "sam.py", "LICENSE"),
    }.items():
        for name in names:
            src, dst = REFERENCE / "external" / package / name, ROOT / "external" / package / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if file_sha256(dst) != file_sha256(src):
                    raise ValueError("existing adapter prerequisite differs")
            else:
                with src.open("rb") as source, dst.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
    for candidate in CANDIDATE_IDS:
        verify_official_adapter_sources(candidate)


def _materialize_v2() -> tuple[dict, dict]:
    from experiments.kbound.so2sat import prospective_v2 as v2

    tent, receipt = load_pair(OPENED_DEV / "so2sat_tent_adam_bn_affine_probe_transfer_v1.gate_fit.json")
    controller = v2.derive_controller_v2(tent, receipt)
    protocol = v2._expected_protocol_v2()
    for path, doc in (
        (v2.default_protocol_v2_path(), protocol),
        (v2.default_controller_v2_path(), controller),
        (v2.default_precalibration_template_v2_path(), v2._expected_template_v2()),
    ):
        if path.exists():
            previous, _ = load_pair(path)
            if previous != doc:
                raise ValueError("existing v2 configuration differs")
        else:
            _write(path, doc)
    return protocol, controller


def _chronology() -> dict:
    return {
        "schema": "kbound_so2sat_zero_access_chronology_v2",
        "status": "DECLARED_ZERO_ACCESS_NOT_AN_EXTERNAL_TIMESTAMP",
        "ordered_events": [
            "v1_negative_result_closed",
            "v2_protocol_and_controller_fixed",
            "v2_precalibration_seal_created",
            "gate_calibration_may_begin_only_after_seal",
            "target_may_begin_only_after_gate_authorization",
        ],
        "gate_calibration_rows_read_before_v2_seal": 0,
        "gate_calibration_pixels_read_before_v2_seal": 0,
        "gate_calibration_labels_read_before_v2_seal": 0,
        "target_pixels_read_before_v2_seal": 0,
        "target_labels_read_before_v2_seal": 0,
        "external_timestamp_claimed": False,
    }


def prior_run_access_artifacts(root: Path) -> list[str]:
    """Discover access markers by filename without opening outcome artifacts."""
    return (
        sorted(
            str(path)
            for marker in ("calibration_started.json", "outcome_reveal_started.json")
            for path in root.rglob(marker)
        )
        if root.exists()
        else []
    )


def _access_audit(source: Path, output: Path) -> dict:
    negative, negative_receipt = load_pair(OPENED_DEV / "so2sat_candidate_selection.json")
    if (
        negative.get("status") != "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL"
        or negative.get("selected_candidate_id") is not None
    ):
        raise ValueError("historical negative selection identity changed")
    if any(
        negative.get(k) != 0
        for k in ("gate_cal_rows_read_before_selection", "target_pixels_read", "target_labels_read")
    ):
        raise ValueError("historical artifact records reserved access")
    # Directory-name inspection only; no unknown calibration/target artifacts opened.
    inspected = [source.parent, OPENED_DEV.parent, REFERENCE / "experiments/kbound/results/so2sat_lcz42_prospective_v1"]
    suspicious = prior_run_access_artifacts(ROOT / "output/next_phase")
    inventories = {}
    for root in inspected:
        names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.json") if not p.name.startswith("._"))
        inventories[str(root)] = names
        suspicious.extend(
            str(root / name)
            for name in names
            if any(
                token in name.lower()
                for token in (
                    "gate_cal.",
                    "calibration_bundle",
                    "target_bundle",
                    "target_score",
                    "target_authorization",
                )
            )
        )
    return {
        "historical_access_status": "DOCUMENTED_ZERO_RESERVED_ACCESS_NOT_EXTERNALLY_PROVEN",
        "current_run_calibration_pixels_read": 0,
        "current_run_calibration_labels_read": 0,
        "current_run_target_pixels_read": 0,
        "current_run_target_labels_read": 0,
        "unrecognized_prior_access_artifacts": suspicious,
        "known_opened": "nine gate_fit cities for Tent and SAR; all v1 candidate selection outcomes",
        "negative_receipt": negative_receipt,
        "metadata_inventory": inventories,
        "limitation": "Targeted documented-history audit, not universal filesystem or external custody proof; declarations concern this current run.",
    }


def prepare(output: Path, data: Path, source: Path, opaque_root: Path = DEFAULT_OPAQUE) -> dict:
    import torch

    from experiments.kbound.so2sat import prospective_v2 as v2
    from experiments.kbound.so2sat.development import load_verified_checkpoints
    from experiments.kbound.so2sat.metadata_manifest import build_population_manifest
    from experiments.kbound.so2sat.target_inference import target_runtime_environment_identity

    started = time.monotonic()
    if output.exists():
        raise ValueError("occupied output: preparation is create-only")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    _materialize_prerequisites()
    protocol, controller = _materialize_v2()
    collection, checkpoint_rows = load_verified_checkpoints(source)
    checkpoints = {row.checkpoint_id: row for row in checkpoint_rows}
    acceptance, acceptance_receipt = load_pair(source / "so2sat_source_postrun_acceptance.json")
    manifest = build_population_manifest(
        {split: data / f"{split}_geo.h5" for split in ("training", "validation", "testing")}
    )
    historical = acceptance["population_manifest"]
    if stable_sha256(manifest) != historical["canonical_document_sha256"]:
        raise ValueError("reconstructed metadata differs from historical source identity")
    manifest_path = output / "KBOUND_SO2SAT_POPULATION_MANIFEST_v1.json"
    manifest_receipt = _write(manifest_path, manifest)
    if any(
        manifest_receipt[k] != historical[k] for k in ("artifact_bytes", "artifact_sha256", "canonical_document_sha256")
    ):
        raise ValueError("metadata reconstruction is not byte-identical to historical source binding")
    environment = target_runtime_environment_identity(torch.device("mps"))
    _write(output / "environment.json", environment)
    audit = _access_audit(source, output)
    _write(output / "access_audit.json", audit)
    files = {
        "runner": Path(__file__),
        "scorer_reader": ROOT / "experiments/kbound/so2sat/target_scorer.py",
        "natural_protocol": PROTOCOL,
        "v2_protocol": v2.default_protocol_v2_path(),
        "v2_controller": v2.default_controller_v2_path(),
        "manifest": manifest_path,
        "acceptance": source / "so2sat_source_postrun_acceptance.json",
        "normalizer": source / "so2sat_sen2_source_normalizer.json",
        "collection": source / "so2sat_source_checkpoint_collection.json",
        "environment": output / "environment.json",
    }
    files.update({f"checkpoint_{key}": checkpoint.checkpoint_path for key, checkpoint in checkpoints.items()})
    lock = make_run_lock(strict_json_load(PROTOCOL), files, audit)
    _write(output / "current_run_lock.json", lock)
    # Opaque decompression, no h5py call or dataset inspection.
    raw_dir = opaque_root
    raw_dir.mkdir(parents=True, exist_ok=False)
    opaque = {}
    for split, role in (
        ("validation", "label_free_probe_pixels"),
        ("testing", "sealed_evaluation_pixels_and_outcomes"),
    ):
        archive, target = data / f"{split}.h5.gz", raw_dir / f"{split}.h5"
        archive_hash = file_sha256(archive)
        digest = hashlib.sha256()
        with gzip.open(archive, "rb") as src, target.open("xb") as dst:
            while block := src.read(8 * 1024 * 1024):
                digest.update(block)
                dst.write(block)
        opaque[split] = {
            "container_role": role,
            "artifact_basename": target.name,
            "artifact_bytes": target.stat().st_size,
            "raw_file_sha256": digest.hexdigest(),
            "hashing_method": "sha256_raw_bytes_without_hdf5_deserialization",
            "hdf5_datasets_opened": 0,
        }
        _write(
            output / f"{split}_opaque_decompression.json",
            {"archive_sha256": archive_hash, "raw_identity": opaque[split], "hdf5_datasets_opened": 0},
        )
    _, collection_receipt = load_pair(files["collection"])
    normalizer, normalizer_receipt = load_pair(files["normalizer"])
    bindings = {
        "population_manifest_artifact": manifest_receipt,
        "population_identity_sha256": manifest["population_identity_sha256"],
        "source_postrun_acceptance_artifact": acceptance_receipt,
        "source_postrun_acceptance_sha256": stable_sha256(acceptance),
        "source_checkpoint_collection_artifact": collection_receipt,
        "source_checkpoint_collection_sha256": stable_sha256(collection),
        "source_checkpoints": {
            key: {
                "checkpoint_file_sha256": val.checkpoint_file_sha256,
                "checkpoint_tensor_sha256": val.checkpoint_tensor_sha256,
            }
            for key, val in checkpoints.items()
        },
        "source_normalizer_artifact": normalizer_receipt,
        "source_normalizer_sha256": normalizer["normalizer_sha256"],
    }
    runtime = {
        "code_identity_sha256": v2.prospective_v2_code_identity()["code_identity_sha256"],
        "configuration_identity_sha256": v2.configuration_identity_v2(),
        "environment_identity_sha256": environment["environment_identity_sha256"],
    }
    seal = v2.build_precalibration_seal_v2(
        protocol=protocol,
        controller=controller,
        source_bindings=bindings,
        runtime_bindings=runtime,
        opaque_target_identities=opaque,
        chronology=_chronology(),
    )
    _write(output / "precalibration_seal_v2.json", seal)
    paths = {
        "bound_files": {k: str(p) for k, p in files.items()},
        "data": str(data),
        "source": str(source),
        "output": str(output),
        "opaque_root": str(raw_dir),
    }
    _write(output / "execution_paths.json", paths)
    receipt = {
        "status": "CURRENT_RUN_LOCKED_CALIBRATION_AUTHORIZED_TARGET_FORBIDDEN",
        "created_utc": utc(),
        "elapsed_seconds": time.monotonic() - started,
        "lock_sha256": lock["lock_sha256"],
        "precalibration_seal_sha256": seal["precalibration_seal_sha256"],
        "metadata_reconstruction": "byte-identical to historical source acceptance; new current-run receipt",
        "target_hdf5_datasets_opened": 0,
        "calibration_pixels_or_labels_read": 0,
    }
    _write(output / "preparation_summary.json", receipt)
    return receipt


def calibrate(output: Path) -> dict:
    import torch

    from experiments.kbound.so2sat import prospective_v2 as v2
    from experiments.kbound.so2sat.development import (
        GATE_CAL_ROLE,
        DevelopmentData,
        _load_manifest_and_inventory,
        _runner_code_identity,
        load_verified_checkpoints,
        run_development_cell,
    )
    from experiments.kbound.so2sat.source_data import load_sealed_band_normalizer
    from experiments.kbound.so2sat.target_inference import target_runtime_environment_identity

    torch.set_num_threads(4)
    started = time.monotonic()
    paths, _ = load_pair(output / "execution_paths.json")
    files = {key: Path(value) for key, value in paths["bound_files"].items()}
    lock, _ = load_pair(output / "current_run_lock.json")
    verify_run_lock(lock, files)
    seal, _ = load_pair(output / "precalibration_seal_v2.json")
    v2.validate_precalibration_seal_v2(seal)
    if (
        target_runtime_environment_identity(torch.device("mps"))["environment_identity_sha256"]
        != seal["runtime_bindings"]["environment_identity_sha256"]
    ):
        raise ValueError("execution environment changed before calibration")
    marker = output / "calibration_started.json"
    _write(
        marker, {"utc": utc(), "precalibration_seal_sha256": seal["precalibration_seal_sha256"], "target_access": False}
    )
    data_root, source = Path(paths["data"]), Path(paths["source"])
    binding, manifest, inventory = _load_manifest_and_inventory(files["manifest"], data_root / "training_geo.h5")
    collection, checkpoint_rows = load_verified_checkpoints(source)
    checkpoints = {row.checkpoint_id: row for row in checkpoint_rows}
    normalizer = load_sealed_band_normalizer(files["normalizer"])
    controller = v2.load_controller_v2()
    # First reader capable of consuming calibration payloads: both locks validated above.
    data = DevelopmentData(data_root / "training.h5", inventory, normalizer, authorized_role=GATE_CAL_ROLE)
    if data.container.identity_sha256 != controller["source_bindings"]["source_container_identity_sha256"]:
        raise ValueError("calibration training container differs from frozen source identity")
    code = _runner_code_identity()
    cells, timings = [], []
    for city, partition in sorted(inventory.partitions[GATE_CAL_ROLE].items()):
        for checkpoint_id in v2.CHECKPOINT_IDS:
            torch.mps.synchronize()
            cell_start = time.monotonic()
            cell = run_development_cell(
                candidate_id=controller["candidate_id"],
                role=GATE_CAL_ROLE,
                partition=partition,
                checkpoint=checkpoints[checkpoint_id],
                data=data,
                study_binding=binding,
                device=torch.device("mps"),
                code_identity=code,
            )
            torch.mps.synchronize()
            row = cell["gate_row"]
            cells.append(
                {
                    key: row[key]
                    for key in ("city_id", "checkpoint_id", "feature_document", "observed_benefit", "trace_sha256")
                }
            )
            timing = {
                "city_id": city,
                "checkpoint_id": checkpoint_id,
                "total_cell_seconds": time.monotonic() - cell_start,
                "probe_images": cell["probe_n"],
                "evaluation_images": cell["evaluation_n"],
                "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                "scope": "full cell including source reset, paired probe/evaluation inference, Tent candidate and feature extraction; no stage decomposition",
            }
            timings.append(timing)
            _write(output / "calibration_cells" / f"{city}_checkpoint{checkpoint_id}.json", cell)
            _write(output / "timings" / f"{city}_checkpoint{checkpoint_id}.json", timing)
            print(
                json.dumps(
                    {
                        "completed_cells": len(cells),
                        "total_cells": 95,
                        "city": city,
                        "checkpoint": checkpoint_id,
                        "seconds": timing["total_cell_seconds"],
                    }
                ),
                flush=True,
            )
    data.container.close()
    verify_run_lock(lock, files)
    bundle = {
        "schema": v2.CALIBRATION_BUNDLE_SCHEMA,
        "status": "COMPLETE_19_CITY_95_CELL_GATE_CALIBRATION",
        "precalibration_seal_sha256": seal["precalibration_seal_sha256"],
        "controller_sha256": controller["controller_sha256"],
        "action_unit": "city_checkpoint",
        "cells": cells,
        "gate_calibration_city_count": 19,
        "checkpoint_count": 5,
        "gate_calibration_rows_read": 95,
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "target_inputs": [],
    }
    bundle["bundle_sha256"] = stable_sha256(bundle)
    _write(output / "calibration_bundle_v2.json", bundle)
    screen = v2.build_gate_screen_v2(precalibration_seal=seal, controller=controller, calibration_bundle=bundle)
    _write(output / "gate_screen_v2.json", screen)
    if screen["passed"]:
        authorization = v2.authorize_target_execution_v2(
            precalibration_seal=seal, controller=controller, calibration_bundle=bundle, submitted_gate_screen=screen
        )
        _write(output / "target_authorization_v2.json", authorization)
    summary = {
        "status": screen["status"],
        "created_utc": utc(),
        "elapsed_seconds": time.monotonic() - started,
        "independent_calibration_cities": 19,
        "independently_trained_checkpoints": 5,
        "cells_not_independent_environments": 95,
        "interval_radius": screen["calibration"]["interval_radius"],
        "rank": 18,
        "decision_counts": screen["decision_counts"],
        "direct_action_cities": screen["direct_action_cities"],
        "utility_diagnostics": screen["utility_diagnostics"],
        "checks": screen["checks"],
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "historical_freshness": lock["access"]["historical_access_status"],
        "zero_error_19_city_one_sided_95_percent_upper_risk": zero_error_upper_bound(19, 0.05),
        "conditional_risk_at_most_point_one_certified": False,
        "exchangeable_new_city_coverage": "nominal .90 for all five checkpoints jointly, conditional assumptions required; not evidence culture-10 target cities are exchangeable or coverage conditional on passing selection",
        "timings": timings,
        "gate_screen_sha256": screen["gate_screen_sha256"],
        "v1_negative_preserved": True,
    }
    _write(output / "calibration_summary.json", summary)
    public = ROOT / "output/next_phase/natural_v2"
    public.mkdir(parents=True, exist_ok=True)
    for name in ("calibration_summary.json", "gate_screen_v2.json", "preparation_summary.json"):
        doc, _ = load_pair(output / name)
        if public.resolve() != output.resolve():
            _write(public / name, doc)
    return summary


def target(output: Path) -> Path:
    """Execute only the receipt-recomputed eligible v2 target branch."""
    import torch

    from experiments.kbound.so2sat.prospective_runner_v2 import run_production_target_v2

    torch.set_num_threads(4)
    paths, _ = load_pair(output / "execution_paths.json")
    files = {key: Path(value) for key, value in paths["bound_files"].items()}
    lock, _ = load_pair(output / "current_run_lock.json")
    verify_run_lock(lock, files)
    screen, _ = load_pair(output / "gate_screen_v2.json")

    def execute():
        return run_production_target_v2(
            precalibration_seal=output / "precalibration_seal_v2.json",
            calibration_bundle=output / "calibration_bundle_v2.json",
            gate_screen=output / "gate_screen_v2.json",
            target_authorization=output / "target_authorization_v2.json",
            population_manifest=files["manifest"],
            source_postrun_acceptance=files["acceptance"],
            checkpoint_collection=files["collection"],
            checkpoint_dir=paths["source"],
            normalizer=files["normalizer"],
            environment_identity=files["environment"],
            geo_paths={
                split: Path(paths["data"]) / f"{split}_geo.h5" for split in ("training", "validation", "testing")
            },
            target_data_paths={
                split: Path(paths["opaque_root"]) / f"{split}.h5" for split in ("validation", "testing")
            },
            output_dir=output / "target_predictions",
            device_name="mps",
        )

    return dispatch_eligible_target(screen, execute)


def score_target(output: Path) -> dict:
    """Authenticate all actions and predictions before the one outcome reveal."""
    import numpy as np

    from experiments.kbound.so2sat import prospective_v2 as v2
    from experiments.kbound.so2sat.label_firewall import VerifiedGeoIndex
    from experiments.kbound.so2sat.prospective_runner_v2 import load_v2_runtime_authority, verify_published_directory_v2
    from experiments.kbound.so2sat.target_scorer import _read_testing_outcomes_once

    paths, _ = load_pair(output / "execution_paths.json")
    files = {key: Path(value) for key, value in paths["bound_files"].items()}
    lock, _ = load_pair(output / "current_run_lock.json")
    verify_run_lock(lock, files)
    authority = load_v2_runtime_authority(
        precalibration_seal=output / "precalibration_seal_v2.json",
        calibration_bundle=output / "calibration_bundle_v2.json",
        gate_screen=output / "gate_screen_v2.json",
        target_authorization=output / "target_authorization_v2.json",
    )
    publication = output / "target_predictions"
    verify_published_directory_v2(publication)
    bundle, _ = load_pair(publication / "so2sat_target_bundle_v2.json")
    plan, _ = load_pair(publication / "target_action_plan.json")
    auth_hash = authority.target_authorization["authorization_sha256"]
    if (
        bundle.get("authorization_sha256") != auth_hash
        or bundle.get("cell_count") != 50
        or bundle.get("complete_before_scoring") is not True
        or plan.get("cell_count") != 50
        or plan.get("all_actions_sealed_before_evaluation_pixels") is not True
    ):
        raise ValueError("target prediction publication is incomplete or stale")
    manifest, _ = load_pair(files["manifest"])
    geo = VerifiedGeoIndex(
        manifest, {split: Path(paths["data"]) / f"{split}_geo.h5" for split in ("training", "validation", "testing")}
    )
    indices = {city: [] for city in manifest["cities"]["target"]}
    for record in geo.iter_records("testing"):
        indices[record.city_id].append(record.row_index)
    cells = []
    grid = set()
    for row in bundle["cells"]:
        cell, receipt = load_pair(publication / row["cell_basename"])
        if (
            receipt["artifact_sha256"] != row["artifact_sha256"]
            or cell["authorization_sha256"] != auth_hash
            or cell["cell_sha256"] != row["cell_sha256"]
            or cell["action"]["action_sha256"] != row["action_sha256"]
        ):
            raise ValueError("target cell identity or action changed")
        identity = (cell["city_id"], cell["checkpoint_id"])
        if identity in grid:
            raise ValueError("duplicate target cell")
        grid.add(identity)
        # Validate prediction/action consistency before reading any outcome.
        score_cell(cell, np.zeros(len(indices[cell["city_id"]]), dtype=np.int64))
        cells.append(cell)
    if grid != {(city, checkpoint) for city in indices for checkpoint in v2.CHECKPOINT_IDS}:
        raise ValueError("target grid is not complete")
    for split in ("validation", "testing"):
        raw = Path(paths["opaque_root"]) / f"{split}.h5"
        expected = authority.precalibration_seal["opaque_target_identities"][split]
        if raw.stat().st_size != expected["artifact_bytes"] or file_sha256(raw) != expected["raw_file_sha256"]:
            raise ValueError("opaque target container changed before scoring")
    verify_published_directory_v2(publication)
    _write(
        output / "outcome_reveal_started.json",
        {
            "utc": utc(),
            "authorization_sha256": auth_hash,
            "bundle_sha256": bundle["bundle_sha256"],
            "testing_label_read_passes_authorized": 1,
            "validation_label_read_passes_authorized": 0,
        },
    )
    truth = _read_testing_outcomes_once(
        Path(paths["opaque_root"]) / "testing.h5",
        expected_rows=manifest["splits"]["testing"]["observed_samples"],
        h5_factory=None,
    )
    scored = {
        "schema": v2.SCORED_TARGET_BUNDLE_SCHEMA,
        "status": "COMPLETE_10_CITY_50_CELL_SINGLE_REVEAL",
        "protocol_id": v2.PROTOCOL_ID,
        "controller_sha256": authority.controller["controller_sha256"],
        "authorization_sha256": auth_hash,
        "action_unit": "city_checkpoint",
        "cells": [
            score_cell(cell, truth[indices[cell["city_id"]]])
            for cell in sorted(cells, key=lambda row: (row["city_id"], int(row["checkpoint_id"])))
        ],
        "target_city_count": 10,
        "checkpoint_count": 5,
        "outcome_reveal_count": 1,
    }
    scored["bundle_sha256"] = stable_sha256(scored)
    _write(output / "scored_target_bundle_v2.json", scored)
    report = v2.summarize_target_inference_v2(scored, authorization=authority.target_authorization)
    _write(output / "target_inference_v2.json", report)
    public_target = ROOT / "output/next_phase/natural_v2/target_inference_v2.json"
    if public_target.resolve() != (output / "target_inference_v2.json").resolve():
        _write(public_target, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "calibrate", "target", "score"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--opaque-root", type=Path, default=DEFAULT_OPAQUE)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.data, args.source, args.opaque_root)
    elif args.command == "calibrate":
        result = calibrate(args.output)
    elif args.command == "target":
        result = {"prediction_bundle": str(target(args.output))}
    else:
        result = score_target(args.output)
    print(json.dumps({key: val for key, val in result.items() if key != "timings"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
