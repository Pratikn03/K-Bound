"""Stage-separated painting development pilot; logical APIs, not independent custody.

The private archive broker sees class-bearing paths and ingestion labels. Model
APIs receive ordered tensors only. Complete check action/prediction seals precede
all check scoring joins. This is neither production custody nor confirmation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import plistlib
import re
import struct
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn

from experiments.kbound.domainnet import pilot_analysis, pilot_candidate, pilot_data, source_data, train_source
from kga.benefit import FrozenLinearBenefitEstimator, fit_frozen_linear_benefit_estimator
from kga.certificate import Certificate, InsufficientCalibrationError, split_conformal_rank_radius
from kga.policy import decide

AMENDMENT_SHA256 = "e13985b0fd46bca222dd63041b55b81059b3efd2e37de7d5562a82b24e196f66"
TRAINER_SHA256 = "5dfb95a2e869b154b1ea32f99cd58dbcadb1e5fa2aa207f08e7be4c1963b91e6"
EVIDENCE_SCHEMA = "kbound-domainnet-pilot-evidence/2"
Observer = Callable[[str, str, Path], None]


@dataclass(frozen=True)
class SyntheticHooks:
    """Library-only reduced-class synthetic mechanisms; no CLI selection."""

    archive_sha256: str
    class_count: int
    model_factory: Callable[[], nn.Module]
    transform: Callable[[Image.Image], torch.Tensor]
    observer: Observer | None = None

    def __post_init__(self) -> None:
        if (
            type(self.class_count) is not int
            or not 2 <= self.class_count < 126
            or not isinstance(self.archive_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.archive_sha256) is None
            or self.archive_sha256 in {pilot_data.ARCHIVE_SHA256, source_data.SourceIdentity().archive_sha256}
        ):
            raise ValueError("synthetic hooks require reduced classes and nonproduction identity")


def _plain(path: Path, *, directory: bool = False) -> None:
    if not path.is_absolute():
        raise ValueError("absolute input/output paths required")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink input/output paths are forbidden")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"missing regular {'directory' if directory else 'file'}: {path}")


def _hash_file(path: Path, expected: str) -> None:
    _plain(path)
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("externally expected SHA256 required")
    if source_data.file_hash(path) != expected:
        raise ValueError(f"input SHA256 mismatch: {path.name}")


def _digest(value: Any) -> str:
    return hashlib.sha256(source_data.json_bytes(value)).hexdigest()


def _scope(value: dict[str, Any], synthetic: bool) -> None:
    expected = {
        "development_pilot": True,
        "purpose": "SOURCE_ONLY_DEVELOPMENT_PILOT",
        "eligible_for_confirmatory": False,
        "execution_scope": "SYNTHETIC_TEST" if synthetic else "SOURCE_ONLY_DEVELOPMENT_PILOT",
    }
    if any(value.get(k) != v or type(value.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("source pilot scope/flags mismatch")


def _finite_checkpoint(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite full checkpoint tensor")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite full checkpoint scalar")
    elif isinstance(value, dict):
        for member in value.values():
            _finite_checkpoint(member)
    elif isinstance(value, (list, tuple)):
        for member in value:
            _finite_checkpoint(member)


def validate_source_run(
    source_run_dir: Path,
    source_inventory: Path,
    config_path: Path,
    policy_path: Path,
    expected_hashes: dict[str, str],
    *,
    synthetic_hooks: SyntheticHooks | None = None,
) -> dict[str, Any]:
    """Verify one complete seed-zero final checkpoint before constructing a model."""
    if synthetic_hooks is not None:
        if type(synthetic_hooks) is not SyntheticHooks:
            raise ValueError("explicit synthetic hooks required")
        synthetic_hooks.__post_init__()
    _plain(source_run_dir, directory=True)
    paths = {
        "config": config_path,
        "policy": policy_path,
        "source_inventory": source_inventory,
        "source_receipt": source_run_dir / "receipt.json",
        "source_seed_receipt": source_run_dir / "seed-0/receipt.json",
        "source_checkpoint": source_run_dir / "seed-0/final.pt",
        "source_epochs": source_run_dir / "seed-0/epochs.jsonl",
    }
    for key, path in paths.items():
        _hash_file(path, expected_hashes[key])
    if sorted(p.name for p in source_run_dir.glob("seed-*")) != ["seed-0"]:
        raise ValueError("exactly one source seed-zero run required")
    policy = source_data.load_source_policy(policy_path, expected_hashes["policy"])
    config = source_data.strict_json(config_path)
    if source_data.json_bytes(config) != source_data.json_bytes(
        train_source.approved_config(source_policy=policy, development_pilot=True)
    ):
        raise ValueError("source configuration differs from approved pilot recipe")
    inventory = source_data.strict_json(source_inventory)
    identity = inventory["identity"]
    if synthetic_hooks is None:
        if (
            source_data.json_bytes(identity) != source_data.json_bytes(asdict(source_data.SourceIdentity()))
            or expected_hashes["source_inventory"] != pilot_data.SOURCE_INVENTORY_SHA256
        ):
            raise ValueError("production source inventory/identity mismatch")
    elif (
        identity.get("class_count") != synthetic_hooks.class_count
        or identity.get("archive_sha256") == source_data.SourceIdentity().archive_sha256
    ):
        raise ValueError("synthetic source identity mismatch")
    receipt = source_data.strict_json(paths["source_receipt"])
    seed = source_data.strict_json(paths["source_seed_receipt"])
    _scope(receipt, synthetic_hooks is not None)
    _scope(seed, synthetic_hooks is not None)
    if receipt.get("status") != "COMPLETED" or receipt.get("runs") != [seed]:
        raise ValueError("source aggregate and seed receipt mismatch")
    context = {
        "config": config,
        "config_sha256": expected_hashes["config"],
        "source_identity": identity,
        "inventory_sha256": expected_hashes["source_inventory"],
        "source_policy": policy,
        "effective_epochs": 20,
        "effective_batch_size": 32,
    }
    if any(source_data.json_bytes(receipt.get(k)) != source_data.json_bytes(v) for k, v in context.items()):
        raise ValueError("source context/configuration mismatch")
    for name in ("__init__.py", "source_data.py", "train_source.py"):
        key = f"experiments/kbound/domainnet/{name}"
        digest = source_data.file_hash(Path(__file__).with_name(name))
        if receipt.get("code_identity", {}).get("files", {}).get(key) != digest:
            raise ValueError("source code identity mismatch")
        if name == "train_source.py" and digest != TRAINER_SHA256:
            raise ValueError("approved original source trainer changed")
    expected_seed = {
        "status": "COMPLETED",
        "seed": 0,
        "seed_streams": train_source.seed_streams(0),
        "epochs": 20,
        "processed_count": 16811 * 20,
        "batches": math.ceil(16811 / 32) * 20,
        "checkpoint": "seed-0/final.pt",
        "candidate": "final_epoch_only",
        "source_policy": policy,
        "checkpoint_sha256": expected_hashes["source_checkpoint"],
        "final_tensor_sha256": expected_hashes["source_checkpoint_tensor"],
    }
    if any(source_data.json_bytes(seed.get(k)) != source_data.json_bytes(v) for k, v in expected_seed.items()):
        raise ValueError("source final epoch/seed/count/checkpoint receipt mismatch")
    # The epoch log is independently externally hash-bound: source readiness never selects a best epoch.
    epochs = []
    for line in paths["source_epochs"].read_text().splitlines():
        epochs.append(json.loads(line))
    readiness = pilot_analysis.source_readiness(epochs)
    if any(type(row.get("batches")) is not int or row["batches"] != 526 for row in epochs):
        raise ValueError("source epoch batch count mismatch")
    checkpoint = torch.load(paths["source_checkpoint"], map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("invalid source checkpoint root")
    _finite_checkpoint(checkpoint)
    _scope(checkpoint, synthetic_hooks is not None)
    if (
        any(source_data.json_bytes(checkpoint.get(k)) != source_data.json_bytes(v) for k, v in context.items())
        or checkpoint.get("code_identity") != receipt["code_identity"]
    ):
        raise ValueError("checkpoint source context mismatch")
    if checkpoint.get("record") != {k: v for k, v in seed.items() if k != "checkpoint_sha256"}:
        raise ValueError("checkpoint internal and external receipts mismatch")
    state = checkpoint.get("model")
    if (
        not isinstance(state, dict)
        or not state
        or any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all()) for v in state.values())
    ):
        raise ValueError("nonfinite or malformed full source state")
    if train_source.tensor_hash(state) != expected_hashes["source_checkpoint_tensor"]:
        raise ValueError("source checkpoint tensor SHA256 mismatch")
    return {"state": state, "readiness": readiness, "receipt": receipt}


class ArchiveBroker:
    """Private class-bearing paths/labels; every image read checks CRC and RGB identity."""

    def __init__(
        self,
        archive: Path,
        rows: list[dict[str, Any]],
        public: dict[str, Any],
        *,
        class_count: int,
        transform: Callable[[Image.Image], torch.Tensor] = pilot_candidate.image_transform,
        observer: Observer | None = None,
    ):
        self._path = archive
        self._rows = {r["sample_id"]: copy.deepcopy(r) for r in rows}
        if len(self._rows) != len(rows):
            raise ValueError("duplicate private sample IDs")
        self._public = copy.deepcopy(public)
        self.class_count = class_count
        self._transform = transform
        self._observer = observer
        self._zip: zipfile.ZipFile | None = None
        self._output: Path | None = None

    def observe(self, event: str, cell_id: str = "") -> None:
        if self._observer is not None and self._output is not None:
            self._observer(event, cell_id, self._output)

    def validate(self, public: dict[str, Any]) -> None:
        if public != self._public:
            raise ValueError("public row/window identities mutated")
        if list(self._rows) != [row["sample_id"] for row in public["rows"]]:
            raise ValueError("private/public sample IDs differ")
        for row in public["rows"]:
            private = self._rows[row["sample_id"]]
            if row != {k: v for k, v in private.items() if k not in {"path", "label"}}:
                raise ValueError("private/public row metadata mismatch")
            if type(private["label"]) is not int or not 0 <= private["label"] < self.class_count:
                raise ValueError("private label outside class range")
            pilot_data.canonical_painting_path(private["path"])

    def _ids(self, cell_id: str, window: str, ids: tuple[str, ...]) -> None:
        partition, index = cell_id.split(":")
        expected = self._public["packing"][partition]["cells"][int(index)][window]
        if type(ids) is not tuple or list(ids) != expected:
            raise ValueError("broker refuses missing/extra/reordered/substituted window IDs")

    def images(self, cell_id: str, window: str, ids: tuple[str, ...], device: torch.device) -> Iterator[torch.Tensor]:
        self._ids(cell_id, window, ids)
        self.observe(f"images:{window}", cell_id)
        if self._zip is None:
            self._zip = zipfile.ZipFile(self._path)
            pilot_data.validate_painting_members(self._zip)
        for sample_id in ids:
            row = self._rows[sample_id]
            raw = self._zip.read(row["path"])
            if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
                raise ValueError("encoded painting image SHA256 mismatch")
            rgb = source_data.decode_rgb(raw)
            try:
                if (
                    rgb.size != (row["width"], row["height"])
                    or hashlib.sha256(struct.pack(">QQ", *rgb.size) + rgb.tobytes()).hexdigest() != row["group_id"]
                ):
                    raise ValueError("painting decoded RGB group/size mismatch")
                yield self._transform(rgb).to(device)
            finally:
                rgb.close()

    def labels(self, cell_id: str, ids: tuple[str, ...], *, verified_check: bool = False) -> tuple[int, ...]:
        self._ids(cell_id, "E", ids)
        if cell_id.startswith("DEV_check:") and not verified_check:
            raise ValueError("all check seals must verify before check label joins")
        self.observe("labels", cell_id)
        return tuple(self._rows[i]["label"] for i in ids)

    def close(self) -> None:
        if self._zip is not None:
            owned, self._zip = self._zip, None
            owned.close()


def _cells(public: dict[str, Any], synthetic: bool) -> dict[str, list[tuple[str, dict[str, tuple[str, ...]]]]]:
    if public.get("schema") != "kbound-painting-pilot-public/2" or set(public["packing"]) != set(pilot_data.PARTITIONS):
        raise ValueError("V2 public schema/partitions required")
    rows = public["rows"]
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate active row ID")
    result: dict[str, list[tuple[str, dict[str, tuple[str, ...]]]]] = {}
    used: set[str] = set()
    groups: dict[str, tuple[str, int | None, str]] = {}
    location: tuple[str, int | None, str]
    for partition in pilot_data.PARTITIONS:
        packed = public["packing"][partition]
        if set(packed) != {"cells", "tail"} or (
            not synthetic and len(packed["cells"]) < pilot_data.MINIMUM_CELLS[partition]
        ):
            raise ValueError("incomplete full partition geometry")
        if not packed["cells"]:
            raise ValueError("empty required partition")
        result[partition] = []
        for index, cell in enumerate(packed["cells"]):
            if set(cell) != set(pilot_data.WINDOWS):
                raise ValueError("exact U/V/E windows required")
            windows = {w: tuple(cell[w]) for w in pilot_data.WINDOWS}
            result[partition].append((f"{partition}:{index}", windows))
            for window, ids in windows.items():
                if len(ids) < pilot_data.WINDOW_MINIMA[window]:
                    raise ValueError("incomplete window")
                for sample_id in ids:
                    row = by_id.get(sample_id)
                    location = (partition, index, window)
                    if sample_id in used or row is None or (row["partition"], row["cell"], row["window"]) != location:
                        raise ValueError("row window identity/overlap mismatch")
                    used.add(sample_id)
                    if row["group_id"] in groups and groups[row["group_id"]] != location:
                        raise ValueError("decoded group straddles windows")
                    groups[row["group_id"]] = location
        if set(packed["tail"]) != set(pilot_data.WINDOWS):
            raise ValueError("explicit U/V/E tail inventory required")
        for ids in packed["tail"].values():
            for sample_id in ids:
                row = by_id.get(sample_id)
                location = (partition, None, "TAIL")
                if sample_id in used or row is None or (row["partition"], row["cell"], row["window"]) != location:
                    raise ValueError("invalid predeclared tail")
                used.add(sample_id)
                if row["group_id"] in groups and groups[row["group_id"]] != location:
                    raise ValueError("tail group straddles windows")
                groups[row["group_id"]] = location
    if used != set(by_id):
        raise ValueError("unaccounted active rows")
    return result


def _verify_inputs(lock: dict[str, Any], *, include_large_inputs: bool = True) -> None:
    for records in (lock.get("input_files", {}), lock.get("executable_files", {})):
        for name, value in records.items():
            # Already pinned before execution. Archive image bytes are verified on
            # every read; loaded source state/config is checked by kernel snapshots.
            if not include_large_inputs and name in {"archive", "source_checkpoint"}:
                continue
            _hash_file(Path(value["path"]), value["sha256"])


def _write(path: Path, value: dict[str, Any]) -> dict[str, str]:
    source_data.write_new_json(path, value)
    return {"path": str(path), "sha256": source_data.file_hash(path)}


def _verify_records(records: list[dict[str, str]]) -> None:
    for record in records:
        _hash_file(Path(record["path"]), record["sha256"])


def _action(delta_hat: float, epsilon: float | None, n: int) -> str:
    return decide(
        Certificate(
            delta_hat=delta_hat, epsilon=math.inf if epsilon is None else epsilon, method="conformal", alpha=0.1, n=n
        )
    ).value


def _error(output: Path, exc: BaseException) -> None:
    if output.is_dir() and not output.is_symlink():
        try:
            source_data.write_new_json(
                output / "error_stop.json",
                {
                    "status": "ERROR_STOP",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "eligible_for_confirmatory": False,
                },
            )
        except Exception as publication_error:
            exc.add_note(f"ERROR_STOP could not be written: {publication_error}")


def execute_pilot(
    source: nn.Module,
    public: dict[str, Any],
    broker: ArchiveBroker,
    lock: dict[str, Any],
    output_dir: Path,
    *,
    synthetic_hooks: SyntheticHooks | None = None,
) -> dict[str, Any]:
    """Execute reviewed kernels from validated public geometry and private broker."""
    started = time.perf_counter()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    _plain(output_dir.parent, directory=True)
    synthetic = synthetic_hooks is not None
    if synthetic:
        if type(synthetic_hooks) is not SyntheticHooks:
            raise ValueError("explicit synthetic hooks required")
        synthetic_hooks.__post_init__()
        if lock.get("execution_scope") != "SYNTHETIC_TEST" or lock.get("class_count") != synthetic_hooks.class_count:
            raise ValueError("synthetic execution scope/class mismatch")
        _hash_file(broker._path, synthetic_hooks.archive_sha256)
    elif lock.get("execution_scope") != "PAINTING_DEVELOPMENT_ONLY" or lock.get("class_count") != 126:
        raise ValueError("production execution scope/class mismatch")
    broker.validate(public)
    cells = _cells(public, synthetic)
    pilot_candidate._validate_model(source)
    source_snapshot = pilot_candidate._snapshot(source)
    if not synthetic and next(source.parameters()).device.type != "mps":
        raise ValueError("production painting images/model must use MPS")
    if broker.class_count != lock["class_count"]:
        raise ValueError("broker class count mismatch")
    _verify_inputs(lock)
    output_dir.mkdir(mode=0o700)
    broker._output = output_dir
    records = []
    try:
        lock = {
            **lock,
            "expected_cells": {p: [cid for cid, _ in entries] for p, entries in cells.items()},
            "public_manifest_payload_sha256": _digest(public),
            "source_kernel_state_sha256": source_snapshot.state_sha256,
        }
        lock_record = _write(output_dir / "execution_lock.json", lock)
        records.append(lock_record)
        protocol_hash = lock_record["sha256"]
        (output_dir / "cells").mkdir()
        device = next(source.parameters()).device
        features: dict[str, list[list[float]]] = {p: [] for p in pilot_data.PARTITIONS}
        benefits: dict[str, list[float]] = {p: [] for p in pilot_data.PARTITIONS}
        pairs: dict[str, dict[str, Any]] = {}
        actions: dict[str, dict[str, Any]] = {}
        check_records: list[dict[str, Any]] = []
        kernel_seconds = {"adaptation": 0.0, "inference": 0.0}
        estimator: FrozenLinearBenefitEstimator | None = None
        epsilon: float | None = None
        calibration_records: list[dict[str, str]] = []
        for partition in pilot_data.PARTITIONS:
            if partition == "DEV_check":
                estimator = fit_frozen_linear_benefit_estimator(
                    features["DEV_fit"],
                    benefits["DEV_fit"],
                    features["DEV_radius"],
                    benefits["DEV_radius"],
                    feature_names=pilot_candidate.FEATURE_NAMES,
                    evidence_schema_version=EVIDENCE_SCHEMA,
                    protocol_sha256=protocol_hash,
                    fit_unit="DEV_fit",
                    calibration_unit="DEV_radius",
                    ridge=10.0,
                )
                try:
                    radius = split_conformal_rank_radius(estimator.residuals, 0.1, on_infeasible="raise")
                    epsilon = radius if math.isfinite(radius) else None
                except InsufficientCalibrationError:
                    epsilon = None
                estimator_record = _write(output_dir / "estimator.json", estimator.to_dict())
                calibration: dict[str, Any] = {
                    "schema": "kbound-domainnet-pilot-calibration/2",
                    "ridge": 10.0,
                    "alpha": 0.1,
                    "epsilon": epsilon,
                    "radius_status": "FINITE" if epsilon is not None else "UNAVAILABLE_ABSTAIN",
                    "fit_cell_ids": [cid for cid, _ in cells["DEV_fit"]],
                    "radius_cell_ids": [cid for cid, _ in cells["DEV_radius"]],
                    "estimator_payload_sha256": estimator.payload_sha256,
                    "execution_lock_sha256": protocol_hash,
                    "fit_radius_records": copy.deepcopy(records[1:]),
                }
                calibration_record = _write(output_dir / "calibration.json", calibration)
                seal_record = _write(
                    output_dir / "calibration_seal.json",
                    {
                        "estimator_sha256": estimator_record["sha256"],
                        "calibration_sha256": calibration_record["sha256"],
                        "estimator_payload_sha256": estimator.payload_sha256,
                        "execution_lock_sha256": protocol_hash,
                    },
                )
                calibration_records = [estimator_record, calibration_record, seal_record]
                records.extend(calibration_records)
            for cell_id, windows in cells[partition]:
                pilot_candidate._unchanged(source, source_snapshot)
                _verify_records(calibration_records)
                cell_dir = output_dir / "cells" / cell_id.replace(":", "-")
                cell_dir.mkdir()
                candidate = pilot_candidate.create_candidate(
                    source, broker.images(cell_id, "U", windows["U"], device), expected_count=len(windows["U"])
                )
                if candidate.class_count != lock["class_count"]:
                    raise ValueError("candidate class count mismatch")
                kernel_seconds["adaptation"] += candidate.adaptation_seconds
                v = pilot_candidate.predict_pair(
                    source,
                    candidate,
                    broker.images(cell_id, "V", windows["V"], device),
                    expected_count=len(windows["V"]),
                )
                kernel_seconds["inference"] += v.inference_seconds
                vector = list(pilot_candidate.extract_features(v, candidate.normalized_bn_affine_update))
                features[partition].append(vector)
                common = {
                    "cell_id": cell_id,
                    "execution_lock_sha256": protocol_hash,
                    "source_state_sha256": candidate.source_state_sha256,
                    "candidate_state_sha256": candidate.candidate_state_sha256,
                    "window_sample_ids": {w: list(ids) for w, ids in windows.items()},
                    "window_sample_sha256": {w: _digest(ids) for w, ids in windows.items()},
                }
                evidence_record = _write(
                    cell_dir / "evidence.json",
                    dict(
                        schema=EVIDENCE_SCHEMA,
                        features=vector,
                        feature_names=list(pilot_candidate.FEATURE_NAMES),
                        adaptation_seconds=candidate.adaptation_seconds,
                        v_inference_seconds=v.inference_seconds,
                        **common,
                    ),
                )
                records.append(evidence_record)
                action_record = None
                if estimator is not None:
                    _verify_records(records)
                    estimator.require_payload_identity(calibration["estimator_payload_sha256"])
                    delta_hat = estimator.predict(
                        dict(zip(pilot_candidate.FEATURE_NAMES, vector, strict=True)),
                        evidence_schema_version=EVIDENCE_SCHEMA,
                        protocol_sha256=protocol_hash,
                    )
                    action = _action(delta_hat, epsilon, len(estimator.residuals))
                    actions[cell_id] = dict(
                        delta_hat=delta_hat,
                        epsilon=epsilon,
                        action=action,
                        no_radius_action=_action(delta_hat, 0.0, len(estimator.residuals)),
                        lower=None if epsilon is None else delta_hat - epsilon,
                        upper=None if epsilon is None else delta_hat + epsilon,
                        served_model="candidate" if action == "ADAPT" else "frozen",
                        estimator_payload_sha256=estimator.payload_sha256,
                        calibration_sha256=calibration_record["sha256"],
                        evidence_sha256=evidence_record["sha256"],
                        **common,
                    )
                    action_record = _write(cell_dir / "action.json", actions[cell_id])
                    records.append(action_record)
                e = pilot_candidate.predict_pair(
                    source,
                    candidate,
                    broker.images(cell_id, "E", windows["E"], device),
                    expected_count=len(windows["E"]),
                )
                kernel_seconds["inference"] += e.inference_seconds
                pairs[cell_id] = dict(
                    frozen_predictions=list(e.frozen_predictions),
                    candidate_predictions=list(e.candidate_predictions),
                    e_sample_ids=list(windows["E"]),
                    e_inference_seconds=e.inference_seconds,
                    **common,
                )
                prediction_record = _write(cell_dir / "predictions.json", pairs[cell_id])
                records.append(prediction_record)
                if partition == "DEV_check":
                    assert action_record is not None
                    check_records.append(
                        {
                            "cell_id": cell_id,
                            "action": {
                                **action_record,
                                "path": str(Path(action_record["path"]).relative_to(output_dir)),
                            },
                            "predictions": {
                                **prediction_record,
                                "path": str(Path(prediction_record["path"]).relative_to(output_dir)),
                            },
                        }
                    )
                else:
                    _verify_records([evidence_record, prediction_record])
                    labels = broker.labels(cell_id, windows["E"])
                    delta = (
                        sum(p == y for p, y in zip(e.candidate_predictions, labels, strict=True))
                        - sum(p == y for p, y in zip(e.frozen_predictions, labels, strict=True))
                    ) / len(labels)
                    benefits[partition].append(delta)
                    records.append(
                        _write(
                            cell_dir / "outcome.json",
                            {
                                "cell_id": cell_id,
                                "delta": delta,
                                "labels": list(labels),
                                "predictions_sha256": prediction_record["sha256"],
                                "sample_ids": list(windows["E"]),
                            },
                        )
                    )
                del candidate
        expected_check = [cid for cid, _ in cells["DEV_check"]]
        panel = {
            "schema": "kbound-domainnet-pilot-check-panel-seal/2",
            "expected_cell_ids": expected_check,
            "cells": check_records,
            "execution_lock_sha256": protocol_hash,
            "calibration_seal_sha256": calibration_records[-1]["sha256"],
        }
        panel_record = _write(output_dir / "check_panel_seal.json", panel)
        records.append(panel_record)
        broker.observe("check_panel_sealed")
        expected_dirs = sorted(cid.replace(":", "-") for entries in cells.values() for cid, _ in entries)

        def verify_before_join() -> None:
            _verify_records(records)
            _verify_inputs(lock, include_large_inputs=False)
            broker.validate(public)
            pilot_candidate._unchanged(source, source_snapshot)
            if sorted(p.name for p in (output_dir / "cells").iterdir()) != expected_dirs:
                raise ValueError("unexpected/missing cell panel directory")
            if (
                source_data.strict_json(output_dir / "check_panel_seal.json") != panel
                or [r["cell_id"] for r in check_records] != expected_check
            ):
                raise ValueError("incomplete/reordered expected check panel")
            for row, (cid, windows) in zip(check_records, cells["DEV_check"], strict=True):
                actual_action = source_data.strict_json(output_dir / row["action"]["path"])
                actual_pair = source_data.strict_json(output_dir / row["predictions"]["path"])
                if (
                    actual_action != actions[cid]
                    or actual_pair != pairs[cid]
                    or actual_pair["e_sample_ids"] != list(windows["E"])
                ):
                    raise ValueError("check action/prediction/sample identities differ")

        scored = []
        for cell_id, windows in cells["DEV_check"]:
            verify_before_join()
            labels = broker.labels(cell_id, windows["E"], verified_check=True)
            sealed_action, pair = actions[cell_id], pairs[cell_id]
            scored.append(
                pilot_analysis.ScoredCell(
                    cell_id,
                    labels,
                    tuple(pair["frozen_predictions"]),
                    tuple(pair["candidate_predictions"]),
                    sealed_action["delta_hat"],
                    epsilon,
                    sealed_action["action"],
                    sealed_action["no_radius_action"],
                )
            )
        verify_before_join()
        outcome = {
            "schema": "kbound-domainnet-pilot-check-outcomes/2",
            "check_panel_seal_sha256": panel_record["sha256"],
            "cells": [asdict(row) for row in scored],
        }
        _write(output_dir / "check_outcomes.json", outcome)
        summary = pilot_analysis.summarize_check(scored)
        broker.close()
        summary.update(
            execution_scope=lock["execution_scope"],
            execution_lock_sha256=protocol_hash,
            check_panel_seal_sha256=panel_record["sha256"],
            kernel_seconds=kernel_seconds,
            observed_total_wallclock_seconds=time.perf_counter()
            - started
            + lock.get("preexecution_wallclock_seconds", 0.0),
            wallclock_scope="Painting runner invocation through summary preparation; excludes source training, imports, and final summary file publication.",
            memory_scope="Peak memory not measured; no peak claim.",
            custody_scope="Logical process/API separation; broker sees class-bearing paths and ingestion labels. No independent or OS-enforced custody.",
        )
        source_data.write_new_json(output_dir / "summary.json", summary)
        return summary
    except BaseException as exc:
        _error(output_dir, exc)
        raise
    finally:
        if broker._zip is not None:
            broker.close()


def run_pilot(
    prepared_dir: Path,
    v1_dir: Path,
    archive: Path,
    source_run_dir: Path,
    source_inventory: Path,
    config_path: Path,
    policy_path: Path,
    design_path: Path,
    amendment_path: Path,
    v1_stop: Path,
    expected_hashes: dict[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    """Production CLI path: exact approved inputs, one final ResNet18, MPS, fresh APFS."""
    from experiments.kbound.domainnet.pilot_data_v2 import verify_prepared_painting_v2

    started = time.perf_counter()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    _plain(output_dir.parent, directory=True)
    paths = {
        "archive": archive,
        "source_inventory": source_inventory,
        "config": config_path,
        "policy": policy_path,
        "design": design_path,
        "amendment": amendment_path,
        "v1_stop": v1_stop,
        "prepared_summary": prepared_dir / "summary.json",
        "source_receipt": source_run_dir / "receipt.json",
        "source_seed_receipt": source_run_dir / "seed-0/receipt.json",
        "source_checkpoint": source_run_dir / "seed-0/final.pt",
        "source_epochs": source_run_dir / "seed-0/epochs.jsonl",
    }
    try:
        if set(expected_hashes) != {*paths, "source_checkpoint_tensor"}:
            raise ValueError("exact externally expected input hash set required")
        mount = output_dir.parent
        while mount.parent != mount and not mount.is_mount():
            mount = mount.parent
        volume = subprocess.run(["diskutil", "info", "-plist", str(mount)], check=False, capture_output=True)
        if volume.returncode or plistlib.loads(volume.stdout).get("FilesystemType") != "apfs":
            raise ValueError("fresh APFS output directory required")
        for name, path in paths.items():
            _hash_file(path, expected_hashes[name])
        if (
            expected_hashes["design"] != pilot_data.DESIGN_SHA256
            or expected_hashes["amendment"] != AMENDMENT_SHA256
            or expected_hashes["archive"] != pilot_data.ARCHIVE_SHA256
        ):
            raise ValueError("approved production painting/design/amendment identities required")
        for directory in (prepared_dir, v1_dir, source_run_dir):
            _plain(directory, directory=True)
        for name in ("inventory.json", "public_manifest.json", "labels.json", "quarantine.json", "summary.json"):
            _plain(prepared_dir / name)
        inventory, public = verify_prepared_painting_v2(
            prepared_dir,
            v1_dir,
            source_inventory,
            design_path,
            amendment_path,
            v1_stop,
            expected_summary_sha256=expected_hashes["prepared_summary"],
        )
        if (
            inventory.get("eligible_for_pilot") is not True
            or inventory.get("eligible_for_confirmatory") is not False
            or inventory.get("execution_scope") != "PAINTING_DEVELOPMENT_ONLY"
        ):
            raise ValueError("prepared V2 geometry not eligible for development pilot")
        source = validate_source_run(source_run_dir, source_inventory, config_path, policy_path, expected_hashes)
        if not source["readiness"]["ready"]:
            output_dir.mkdir(mode=0o700)
            _write(output_dir / "source_readiness.json", source["readiness"])
            return dict(source["readiness"])
        if not torch.backends.mps.is_available():
            raise ValueError("required MPS backend unavailable")
        train_source.seed_all(0)
        model = torchvision.models.resnet18(weights=None, num_classes=126)
        model.load_state_dict(source["state"], strict=True)
        model.requires_grad_(False).eval().to("mps")
        pilot_candidate._validate_model(model)
        root = Path(__file__).resolve().parents[3]
        code_paths = [
            Path(__file__).with_name(name)
            for name in (
                "__init__.py",
                "pilot_runner.py",
                "pilot_data.py",
                "pilot_data_v2.py",
                "pilot_candidate.py",
                "pilot_analysis.py",
                "source_data.py",
                "train_source.py",
            )
        ]
        code_paths.extend(
            root / "kga" / name
            for name in ("__init__.py", "benefit.py", "certificate.py", "policy.py", "evaluation.py", "_validation.py")
        )
        lock = {
            "schema": "kbound-domainnet-pilot-execution-lock/2",
            "status": "DEVELOPMENT_ONLY",
            "execution_scope": "PAINTING_DEVELOPMENT_ONLY",
            "class_count": 126,
            "eligible_for_confirmatory": False,
            "source_receipt_code_identity": source["receipt"]["code_identity"],
            "candidate": source_data.strict_json(design_path)["candidate"],
            "gate": source_data.strict_json(design_path)["gate"],
            "feature_names": list(pilot_candidate.FEATURE_NAMES),
            "feature_schema": EVIDENCE_SCHEMA,
            "input_files": {name: {"path": str(path), "sha256": expected_hashes[name]} for name, path in paths.items()},
            "executable_files": {
                str(path.relative_to(root)): {"path": str(path), "sha256": source_data.file_hash(path)}
                for path in code_paths
            },
            "source_checkpoint_tensor_sha256": expected_hashes["source_checkpoint_tensor"],
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
                "backend": "mps",
                "platform": platform.platform(),
                "mps_bitwise_determinism_claim": False,
            },
            "source_readiness": source["readiness"],
        }
        # Parent derivation was replayed above. Keep every derived and parent
        # artifact byte identity in the lock, and recheck these before label joins.
        prepared_summary = source_data.strict_json(prepared_dir / "summary.json")
        for name, digest in prepared_summary["output_sha256"].items():
            lock["input_files"]["prepared/" + name] = {"path": str(prepared_dir / name), "sha256": digest}
        for name, parent_key in (
            ("inventory.json", "v1_inventory_sha256"),
            ("public_manifest.json", "v1_public_manifest_sha256"),
            ("labels.json", "v1_labels_sha256"),
            ("summary.json", "v1_summary_sha256"),
        ):
            lock["input_files"]["v1/" + name] = {"path": str(v1_dir / name), "sha256": inventory["parents"][parent_key]}
        broker = ArchiveBroker(archive, inventory["rows"], public, class_count=126)
        lock["preexecution_wallclock_seconds"] = time.perf_counter() - started
        return execute_pilot(model, public, broker, lock, output_dir)
    except BaseException as exc:
        if not output_dir.exists():
            output_dir.mkdir(mode=0o700)
        if not (output_dir / "error_stop.json").exists():
            _error(output_dir, exc)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-dir",
        "v1-dir",
        "archive",
        "source-run-dir",
        "source-inventory",
        "config",
        "source-policy",
        "design",
        "amendment",
        "v1-stop",
        "expected-hashes",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    _plain(args.expected_hashes)
    result = run_pilot(
        args.prepared_dir,
        args.v1_dir,
        args.archive,
        args.source_run_dir,
        args.source_inventory,
        args.config,
        args.source_policy,
        args.design,
        args.amendment,
        args.v1_stop,
        source_data.strict_json(args.expected_hashes),
        args.output_dir,
    )
    return 0 if result["status"] == "GO_FURTHER_DEVELOPMENT" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        active_error = sys.exc_info()[1]
        try:
            train_source._cleanup_standalone_torch_temp()
        except Exception as cleanup_error:
            if active_error is None or (isinstance(active_error, SystemExit) and active_error.code in (None, 0)):
                raise
            active_error.add_note(f"Torch temporary cleanup failed: {cleanup_error}")
