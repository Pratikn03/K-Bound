#!/usr/bin/env python3
"""Train five independent So2Sat source models without target access.

The production command accepts only ``training_geo.h5``, ``training.h5``, and
the receipt-verified population manifest.  ``source_train`` is the sole
optimization role; ``source_monitor`` is the sole checkpoint-selection role.
There is deliberately no validation/testing path, adapter, or target-scoring
import in this module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from .integrity import (
    IntegrityError,
    file_sha256,
    require_sha256,
    stable_sha256,
    strict_json_load,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from .model import (
    ARCHITECTURE_ID,
    CANONICAL_MODEL_SEEDS,
    NUM_CLASSES,
    architecture_spec,
    assert_model_contract,
    build_so2sat_resnet18,
    clone_cpu_state,
    tensor_state_sha256,
)
from .source_data import (
    SOURCE_MONITOR_ROLE,
    SOURCE_TRAIN_ROLE,
    ArraySourceContainer,
    BandNormalizer,
    H5SourceContainer,
    SourceDataBundle,
    SourceRoleInventory,
    build_source_data_bundle,
    fit_band_normalizer,
    load_sealed_band_normalizer,
    load_verified_source_inventory,
    seal_band_normalizer,
    synthetic_source_inventory,
)

TRAINING_CONFIG_SCHEMA = "kbound_so2sat_source_training_config_v1"
CHECKPOINT_SCHEMA = "kbound_so2sat_source_checkpoint_v1"
TRAINING_RECEIPT_SCHEMA = "kbound_so2sat_source_training_receipt_v1"
RESUME_SCHEMA = "kbound_so2sat_source_resume_v1"
RESUME_RECEIPT_SCHEMA = "kbound_so2sat_source_resume_receipt_v1"
COLLECTION_SCHEMA = "kbound_so2sat_source_checkpoint_collection_v1"

TRAINING_EPOCHS = 30
PHYSICAL_BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
SCHEDULER_ETA_MIN = 1e-6


@dataclass(frozen=True)
class TrainingConfig:
    """Complete scientific source-training configuration."""

    epochs: int = TRAINING_EPOCHS
    batch_size: int = PHYSICAL_BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    label_smoothing: float = LABEL_SMOOTHING
    scheduler_eta_min: float = SCHEDULER_ETA_MIN
    workers: int = 0
    model_seeds: tuple[int, ...] = CANONICAL_MODEL_SEEDS
    run_mode: str = "production"

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise IntegrityError("source training epochs must be a positive integer")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise IntegrityError("source training batch size must be a positive integer")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 0:
            raise IntegrityError("source training workers must be a non-negative integer")
        if tuple(self.model_seeds) != CANONICAL_MODEL_SEEDS:
            raise IntegrityError(
                f"source training requires the exact five model seeds {CANONICAL_MODEL_SEEDS}"
            )
        if not 0.0 < self.learning_rate < 1.0:
            raise IntegrityError("source training learning rate is outside (0,1)")
        if not 0.0 <= self.weight_decay < 1.0:
            raise IntegrityError("source training weight decay is outside [0,1)")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise IntegrityError("source training label smoothing is outside [0,1)")
        if not 0.0 <= self.scheduler_eta_min < self.learning_rate:
            raise IntegrityError("source scheduler eta_min must lie in [0, learning_rate)")
        if self.run_mode not in {"production", "synthetic_smoke"}:
            raise IntegrityError("unknown So2Sat source-training run mode")
        if self.run_mode == "production":
            expected = {
                "epochs": TRAINING_EPOCHS,
                "batch_size": PHYSICAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "label_smoothing": LABEL_SMOOTHING,
                "scheduler_eta_min": SCHEDULER_ETA_MIN,
            }
            observed = {field: getattr(self, field) for field in expected}
            if observed != expected:
                raise IntegrityError("production source hyperparameters differ from the locked recipe")

    def document(self) -> dict[str, Any]:
        return {
            "schema": TRAINING_CONFIG_SCHEMA,
            "run_mode": self.run_mode,
            "model": architecture_spec(),
            "model_seeds": list(self.model_seeds),
            "epochs": self.epochs,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "amsgrad": False,
            },
            "scheduler": {
                "name": "CosineAnnealingLR",
                "t_max_epochs": self.epochs,
                "eta_min": self.scheduler_eta_min,
            },
            "loss": {
                "name": "cross_entropy",
                "label_smoothing": self.label_smoothing,
            },
            "batching": {
                "physical_batch_size": self.batch_size,
                "workers": self.workers,
                "drop_last": False,
                "epoch_order": "seeded_random_permutation_bound_to_model_seed_and_epoch",
            },
            "augmentation": {
                "source_train": "stateless_uniform_dihedral_d4_by_model_seed_epoch_row",
                "source_monitor": "none",
            },
            "normalization": "sealed_per_band_population_mean_std_fit_on_source_train_only",
            "optimization_role": "source_train",
            "checkpoint_selection": {
                "role": "source_monitor",
                "primary": "macro_recall_over_supported_classes",
                "secondary": "top1_accuracy",
                "tertiary": "lower_cross_entropy",
                "tie_break": "earliest_epoch",
            },
            "optimization_scope": "full_network",
            "deterministic_algorithms": True,
            "target_data_inputs": [],
            "target_scoring_imports": [],
        }


@dataclass(frozen=True)
class ArtifactPaths:
    checkpoint: Path
    training_receipt: Path
    training_receipt_byte_receipt: Path
    resume_state: Path
    resume_receipt: Path


@dataclass(frozen=True)
class TrainingResult:
    model_seed: int
    checkpoint_path: Path
    training_receipt_path: Path
    checkpoint_file_sha256: str
    checkpoint_tensor_sha256: str
    initial_tensor_sha256: str
    best_epoch: int
    best_source_monitor_macro_recall: float
    best_source_monitor_accuracy: float

    def collection_row(self) -> dict[str, Any]:
        return {
            "model_seed": self.model_seed,
            "checkpoint_basename": self.checkpoint_path.name,
            "training_receipt_basename": self.training_receipt_path.name,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "checkpoint_tensor_sha256": self.checkpoint_tensor_sha256,
            "initial_tensor_sha256": self.initial_tensor_sha256,
            "best_epoch": self.best_epoch,
            "best_source_monitor_macro_recall": self.best_source_monitor_macro_recall,
            "best_source_monitor_accuracy": self.best_source_monitor_accuracy,
        }


def seed_everything(seed: int) -> None:
    """Seed all supported generators and require deterministic PyTorch kernels."""

    if seed not in CANONICAL_MODEL_SEEDS:
        raise IntegrityError(f"model seed must be one of {CANONICAL_MODEL_SEEDS}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> torch.device:
    """Select only the two execution backends covered by this local protocol."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "mps" if torch.backends.mps.is_available() else "cpu"
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise IntegrityError("MPS was requested but is unavailable")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    raise IntegrityError("So2Sat source training supports only device=auto, mps, or cpu")


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _epoch_order_seed(model_seed: int, epoch: int) -> int:
    digest = stable_sha256(
        {
            "domain": "KBOUND_SO2SAT_SOURCE_EPOCH_ORDER_v1",
            "model_seed": model_seed,
            "epoch": epoch,
        }
    )
    return int(digest[:16], 16) % (2**63 - 1)


def _code_identity() -> tuple[dict[str, str], str]:
    directory = Path(__file__).resolve().parent
    names = (
        "integrity.py",
        "protocol.py",
        "metadata_manifest.py",
        "label_firewall.py",
        "model.py",
        "source_data.py",
        "train_source.py",
        "prospective_protocol_v1.json",
        "prospective_protocol_v1.json.receipt.json",
    )
    files = {name: file_sha256(directory / name) for name in names}
    return files, stable_sha256(files)


def _artifact_paths(output_dir: Path, model_seed: int) -> ArtifactPaths:
    stem = f"so2sat_resnet18_seed{model_seed}"
    training_receipt = output_dir / f"{stem}.training.json"
    return ArtifactPaths(
        checkpoint=output_dir / f"{stem}.pt",
        training_receipt=training_receipt,
        training_receipt_byte_receipt=training_receipt.with_name(
            training_receipt.name + ".receipt.json"
        ),
        resume_state=output_dir / f".{stem}.resume.pt",
        resume_receipt=output_dir / f".{stem}.resume.json",
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(document),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    _atomic_bytes(path, payload)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise IntegrityError(f"refusing to overwrite immutable checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_torch_mapping(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise IntegrityError(f"cannot load trusted local training state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"training state {path} must contain a mapping")
    return value


def _scientific_identity(
    config: TrainingConfig,
    bundle: SourceDataBundle,
    code_sha256: str,
    model_seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], str]:
    runtime = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_basename": Path(sys.executable).name,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device_type": device.type,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    document = {
        "schema": "kbound_so2sat_source_seed_identity_v1",
        "model_seed": model_seed,
        "config": config.document(),
        "config_sha256": stable_sha256(config.document()),
        "data_identity_sha256": bundle.data_identity_sha256,
        "population_manifest_sha256": bundle.inventory.population_manifest_sha256,
        "source_rows_sha256": bundle.inventory.source_rows_sha256,
        "source_container_identity_sha256": bundle.container.identity_sha256,
        "normalizer_sha256": bundle.normalizer.normalizer_sha256,
        "code_sha256": code_sha256,
        "runtime": runtime,
        "runtime_sha256": stable_sha256(runtime),
        "target_data_inputs": [],
    }
    return document, stable_sha256(document)


def _make_loader(
    dataset: Any,
    config: TrainingConfig,
    *,
    shuffle: bool,
    generator_seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(generator_seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.workers,
        drop_last=False,
        generator=generator,
        worker_init_fn=_seed_worker,
        persistent_workers=False,
        pin_memory=False,
    )


@torch.inference_mode()
def evaluate_source_monitor(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    expected_rows: Sequence[int],
) -> dict[str, Any]:
    """Compute checkpoint metrics on source_monitor and verify exact coverage."""

    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    observed_rows: list[int] = []
    for images, targets, row_indices in loader:
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.long)
        logits = model(images)
        if logits.shape != (targets.shape[0], NUM_CLASSES):
            raise IntegrityError("So2Sat source model emitted an invalid logit shape")
        loss = F.cross_entropy(logits, targets, reduction="sum")
        predictions = logits.argmax(dim=1)
        batch_n = int(targets.shape[0])
        total_loss += float(loss.detach().cpu())
        total += batch_n
        matches = predictions.eq(targets)
        correct += int(matches.sum().detach().cpu())
        target_cpu = targets.detach().cpu().numpy()
        match_cpu = matches.detach().cpu().numpy()
        class_total += np.bincount(target_cpu, minlength=NUM_CLASSES)
        class_correct += np.bincount(target_cpu[match_cpu], minlength=NUM_CLASSES)
        observed_rows.extend(int(value) for value in row_indices.tolist())
    if total != len(expected_rows) or observed_rows != list(expected_rows):
        raise IntegrityError("source_monitor evaluation did not consume its exact sealed rows in order")
    supported = class_total > 0
    if not supported.any():
        raise IntegrityError("source_monitor contains no supported class")
    per_class = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    per_class[supported] = class_correct[supported] / class_total[supported]
    metrics = {
        "cross_entropy": total_loss / total,
        "top1_accuracy": correct / total,
        "macro_recall_supported_classes": float(per_class[supported].mean()),
        "supported_class_count": int(supported.sum()),
        "class_support": class_total.tolist(),
        "class_correct": class_correct.tolist(),
        "n": total,
        "role": "source_monitor",
    }
    if not all(
        np.isfinite(metrics[field])
        for field in ("cross_entropy", "top1_accuracy", "macro_recall_supported_classes")
    ):
        raise IntegrityError("source_monitor evaluation produced a non-finite metric")
    return metrics


def _selection_key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["macro_recall_supported_classes"]),
        float(metrics["top1_accuracy"]),
        -float(metrics["cross_entropy"]),
    )


def _save_resume(
    paths: ArtifactPaths,
    *,
    scientific_identity_sha256: str,
    model_seed: int,
    completed_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    initial_tensor_sha256: str,
    best_state: Mapping[str, torch.Tensor],
    best_epoch: int,
    best_key: Sequence[float],
    history: Sequence[Mapping[str, Any]],
) -> None:
    latest_state = clone_cpu_state(model)
    best_cpu = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in best_state.items()}
    payload = {
        "schema": RESUME_SCHEMA,
        "scientific_identity_sha256": scientific_identity_sha256,
        "model_seed": model_seed,
        "completed_epochs": completed_epochs,
        "model_state": latest_state,
        "model_tensor_sha256": tensor_state_sha256(latest_state),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "initial_tensor_sha256": initial_tensor_sha256,
        "best_state": best_cpu,
        "best_tensor_sha256": tensor_state_sha256(best_cpu),
        "best_epoch": best_epoch,
        "best_key": list(best_key),
        "history": [dict(row) for row in history],
    }
    _atomic_torch_save(paths.resume_state, payload, overwrite=True)
    receipt = {
        "schema": RESUME_RECEIPT_SCHEMA,
        "resume_state_basename": paths.resume_state.name,
        "resume_state_bytes": paths.resume_state.stat().st_size,
        "resume_state_sha256": file_sha256(paths.resume_state),
        "scientific_identity_sha256": scientific_identity_sha256,
        "model_seed": model_seed,
        "completed_epochs": completed_epochs,
        "model_tensor_sha256": payload["model_tensor_sha256"],
        "best_tensor_sha256": payload["best_tensor_sha256"],
    }
    _atomic_json(paths.resume_receipt, receipt)


def _load_resume(
    paths: ArtifactPaths,
    *,
    scientific_identity_sha256: str,
    model_seed: int,
    target_epochs: int,
) -> dict[str, Any]:
    if not paths.resume_state.is_file() or not paths.resume_receipt.is_file():
        raise IntegrityError("resume requires a complete state/receipt pair")
    receipt = strict_json_load(paths.resume_receipt)
    if not isinstance(receipt, Mapping) or receipt.get("schema") != RESUME_RECEIPT_SCHEMA:
        raise IntegrityError("unknown So2Sat source resume receipt")
    if receipt.get("resume_state_basename") != paths.resume_state.name:
        raise IntegrityError("resume-state basename mismatch")
    if receipt.get("resume_state_bytes") != paths.resume_state.stat().st_size:
        raise IntegrityError("resume-state byte count mismatch")
    if receipt.get("resume_state_sha256") != file_sha256(paths.resume_state):
        raise IntegrityError("resume-state file SHA-256 mismatch")
    if receipt.get("scientific_identity_sha256") != scientific_identity_sha256:
        raise IntegrityError("resume scientific identity differs from the current run")
    if receipt.get("model_seed") != model_seed:
        raise IntegrityError("resume model seed mismatch")
    payload = _load_torch_mapping(paths.resume_state)
    required = {
        "schema": RESUME_SCHEMA,
        "scientific_identity_sha256": scientific_identity_sha256,
        "model_seed": model_seed,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise IntegrityError(f"resume payload {field} mismatch")
    completed = payload.get("completed_epochs")
    history = payload.get("history")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 1 <= completed <= target_epochs
        or not isinstance(history, list)
        or len(history) != completed
    ):
        raise IntegrityError("resume completed-epoch/history contract is invalid")
    model_state = payload.get("model_state")
    best_state = payload.get("best_state")
    if not isinstance(model_state, Mapping) or not isinstance(best_state, Mapping):
        raise IntegrityError("resume payload lacks model/best tensor states")
    if tensor_state_sha256(model_state) != payload.get("model_tensor_sha256"):
        raise IntegrityError("resume current tensor SHA-256 mismatch")
    if tensor_state_sha256(best_state) != payload.get("best_tensor_sha256"):
        raise IntegrityError("resume best tensor SHA-256 mismatch")
    if receipt.get("model_tensor_sha256") != payload["model_tensor_sha256"]:
        raise IntegrityError("resume receipt/current tensor identity mismatch")
    if receipt.get("best_tensor_sha256") != payload["best_tensor_sha256"]:
        raise IntegrityError("resume receipt/best tensor identity mismatch")
    if receipt.get("completed_epochs") != completed:
        raise IntegrityError("resume receipt epoch mismatch")
    best_epoch = payload.get("best_epoch")
    best_key = payload.get("best_key")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 0 <= best_epoch < completed
        or not isinstance(best_key, list)
        or len(best_key) != 3
        or not all(np.isfinite(float(value)) for value in best_key)
    ):
        raise IntegrityError("resume best-checkpoint selection state is invalid")
    return payload


def _require_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntegrityError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise IntegrityError(f"{field} must be a finite number")
    return result


def _validated_monitor_metrics(
    value: Any,
    *,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """Validate one source-monitor metric row and replay its aggregates."""

    expected_fields = {
        "cross_entropy",
        "top1_accuracy",
        "macro_recall_supported_classes",
        "supported_class_count",
        "class_support",
        "class_correct",
        "n",
        "role",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise IntegrityError("source-monitor metrics have unknown or missing fields")
    if value.get("role") != "source_monitor":
        raise IntegrityError("source-monitor metrics carry another data role")
    n = value.get("n")
    supported_count = value.get("supported_class_count")
    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n < 1
        or isinstance(supported_count, bool)
        or not isinstance(supported_count, int)
        or not 1 <= supported_count <= NUM_CLASSES
    ):
        raise IntegrityError("source-monitor metric counts are invalid")
    if expected_n is not None and n != expected_n:
        raise IntegrityError("source-monitor metric population differs from the sealed role")
    support = value.get("class_support")
    correct = value.get("class_correct")
    if (
        not isinstance(support, list)
        or not isinstance(correct, list)
        or len(support) != NUM_CLASSES
        or len(correct) != NUM_CLASSES
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in support)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in correct)
        or any(hit > total for hit, total in zip(correct, support, strict=True))
        or sum(support) != n
    ):
        raise IntegrityError("source-monitor class support/correct counts are invalid")
    observed_supported = sum(total > 0 for total in support)
    if observed_supported != supported_count:
        raise IntegrityError("source-monitor supported-class count does not replay")

    cross_entropy = _require_finite_number(value.get("cross_entropy"), field="cross_entropy")
    top1 = _require_finite_number(value.get("top1_accuracy"), field="top1_accuracy")
    macro = _require_finite_number(
        value.get("macro_recall_supported_classes"),
        field="macro_recall_supported_classes",
    )
    if cross_entropy < 0.0 or not 0.0 <= top1 <= 1.0 or not 0.0 <= macro <= 1.0:
        raise IntegrityError("source-monitor metric value is outside its valid range")
    replay_top1 = sum(correct) / n
    replay_macro = float(
        np.mean([hit / total for hit, total in zip(correct, support, strict=True) if total > 0])
    )
    if not math.isclose(top1, replay_top1, rel_tol=0.0, abs_tol=1e-15):
        raise IntegrityError("source-monitor top-1 accuracy does not replay from class counts")
    if not math.isclose(macro, replay_macro, rel_tol=0.0, abs_tol=1e-15):
        raise IntegrityError("source-monitor macro recall does not replay from class counts")
    return dict(value)


def _verify_training_history(
    receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Replay checkpoint selection from the immutable epoch history."""

    config = receipt.get("config")
    data = receipt.get("data")
    if not isinstance(config, Mapping) or not isinstance(data, Mapping):
        raise IntegrityError("source training receipt lacks config/data mappings")
    epochs = config.get("epochs")
    completed = receipt.get("epochs_completed")
    history = receipt.get("history")
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed != epochs
        or not isinstance(history, list)
        or len(history) != completed
    ):
        raise IntegrityError("source training epochs/history are incomplete or inconsistent")
    expected_train_n = data.get("source_train_unique_label_rows_authorized")
    expected_monitor_n = data.get("source_monitor_unique_label_rows_authorized")
    if (
        isinstance(expected_train_n, bool)
        or not isinstance(expected_train_n, int)
        or expected_train_n < 1
        or isinstance(expected_monitor_n, bool)
        or not isinstance(expected_monitor_n, int)
        or expected_monitor_n < 1
    ):
        raise IntegrityError("source training receipt has invalid role populations")
    if (
        data.get("source_train_label_read_passes") != completed
        or data.get("source_monitor_label_read_passes") != completed
    ):
        raise IntegrityError("source training receipt label-read passes differ from completed epochs")

    running_key = (-float("inf"), -float("inf"), -float("inf"))
    selected_epoch = -1
    selected_metrics: dict[str, Any] | None = None
    invariant_support: list[int] | None = None
    expected_history_fields = {
        "epoch",
        "source_train",
        "source_monitor",
        "learning_rate_after_scheduler_step",
        "selected",
    }
    for epoch, row in enumerate(history):
        if not isinstance(row, Mapping) or set(row) != expected_history_fields:
            raise IntegrityError("source training history row has unknown or missing fields")
        if row.get("epoch") != epoch or not isinstance(row.get("selected"), bool):
            raise IntegrityError("source training history epoch/selection marker drift")
        source_train = row.get("source_train")
        if not isinstance(source_train, Mapping) or set(source_train) != {
            "role",
            "cross_entropy_with_label_smoothing",
            "n",
        }:
            raise IntegrityError("source_train history metrics have unknown or missing fields")
        if source_train.get("role") != "source_train" or source_train.get("n") != expected_train_n:
            raise IntegrityError("source_train history metrics differ from the sealed role")
        if _require_finite_number(
            source_train.get("cross_entropy_with_label_smoothing"),
            field="source_train.cross_entropy_with_label_smoothing",
        ) < 0.0:
            raise IntegrityError("source_train loss is negative")
        learning_rate = _require_finite_number(
            row.get("learning_rate_after_scheduler_step"),
            field="learning_rate_after_scheduler_step",
        )
        if learning_rate < 0.0:
            raise IntegrityError("source scheduler emitted a negative learning rate")
        monitor = _validated_monitor_metrics(row.get("source_monitor"), expected_n=expected_monitor_n)
        support = list(monitor["class_support"])
        if invariant_support is None:
            invariant_support = support
        elif support != invariant_support:
            raise IntegrityError("source_monitor class support changed across epochs")
        key = _selection_key(monitor)
        expected_selected = key > running_key
        if row.get("selected") is not expected_selected:
            raise IntegrityError("source training selected flag does not replay")
        if expected_selected:
            running_key = key
            selected_epoch = epoch
            selected_metrics = monitor

    best_epoch = receipt.get("best_epoch")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch != selected_epoch
        or checkpoint.get("best_epoch") != best_epoch
    ):
        raise IntegrityError("source best_epoch does not replay across receipt/checkpoint/history")
    receipt_best = receipt.get("best_source_monitor")
    checkpoint_best = checkpoint.get("best_source_monitor")
    if selected_metrics is None or receipt_best != selected_metrics or checkpoint_best != selected_metrics:
        raise IntegrityError(
            "source best_source_monitor does not replay across receipt/checkpoint/history"
        )
    return best_epoch, selected_metrics


def _verify_final_identities(
    receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    """Recompute every JSON-level identity stored by the source trainer."""

    config = receipt.get("config")
    data = receipt.get("data")
    scientific = receipt.get("scientific_identity")
    code_files = receipt.get("code_files_sha256")
    if not all(isinstance(value, Mapping) for value in (config, data, scientific, code_files)):
        raise IntegrityError("source training receipt identity mappings are incomplete")
    assert isinstance(config, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(scientific, Mapping)
    assert isinstance(code_files, Mapping)
    expected_config_fields = {
        "schema",
        "run_mode",
        "model",
        "model_seeds",
        "epochs",
        "optimizer",
        "scheduler",
        "loss",
        "batching",
        "augmentation",
        "normalization",
        "optimization_role",
        "checkpoint_selection",
        "optimization_scope",
        "deterministic_algorithms",
        "target_data_inputs",
        "target_scoring_imports",
    }
    if (
        set(config) != expected_config_fields
        or config.get("schema") != TRAINING_CONFIG_SCHEMA
        or config.get("target_data_inputs") != []
        or config.get("target_scoring_imports") != []
    ):
        raise IntegrityError("source training config schema/access contract drift")
    optimizer = config.get("optimizer")
    scheduler = config.get("scheduler")
    loss = config.get("loss")
    batching = config.get("batching")
    if not all(isinstance(value, Mapping) for value in (optimizer, scheduler, loss, batching)):
        raise IntegrityError("source training optimizer/scheduler config is incomplete")
    assert isinstance(optimizer, Mapping)
    assert isinstance(scheduler, Mapping)
    assert isinstance(loss, Mapping)
    assert isinstance(batching, Mapping)
    try:
        replay_config = TrainingConfig(
            epochs=config.get("epochs"),
            batch_size=batching.get("physical_batch_size"),
            learning_rate=optimizer.get("learning_rate"),
            weight_decay=optimizer.get("weight_decay"),
            label_smoothing=loss.get("label_smoothing"),
            scheduler_eta_min=scheduler.get("eta_min"),
            workers=batching.get("workers"),
            model_seeds=tuple(config.get("model_seeds", ())),
            run_mode=config.get("run_mode"),
        )
    except (TypeError, ValueError) as exc:
        raise IntegrityError("source training config cannot be replayed") from exc
    if replay_config.document() != dict(config):
        raise IntegrityError("source training config differs from the declared recipe")
    model_spec = config.get("model")
    if (
        not isinstance(model_spec, Mapping)
        or model_spec.get("architecture_id") != ARCHITECTURE_ID
        or checkpoint.get("architecture_id") != ARCHITECTURE_ID
    ):
        raise IntegrityError("source training architecture identity drift")
    config_sha = stable_sha256(dict(config))
    if receipt.get("config_sha256") != config_sha or checkpoint.get("config_sha256") != config_sha:
        raise IntegrityError("source training config SHA-256 does not replay")
    for name, digest in code_files.items():
        if not isinstance(name, str) or not name:
            raise IntegrityError("source code identity contains an invalid filename")
        require_sha256(digest, field=f"code_files_sha256.{name}")
    code_sha = stable_sha256(dict(code_files))
    if receipt.get("code_sha256") != code_sha or checkpoint.get("code_sha256") != code_sha:
        raise IntegrityError("source code aggregate SHA-256 does not replay")

    container_identity = data.get("source_container_identity")
    normalizer_document = data.get("normalizer")
    expected_data_fields = {
        "population_identity_sha256",
        "population_manifest_sha256",
        "training_geo_sha256",
        "training_population_n",
        "source_train_n",
        "source_monitor_n",
        "source_rows_sha256",
        "source_container_identity",
        "source_container_identity_sha256",
        "normalizer",
        "source_train_unique_label_rows_authorized",
        "source_train_label_read_passes",
        "source_monitor_unique_label_rows_authorized",
        "source_monitor_label_read_passes",
        "other_role_label_rows_read",
        "target_split_pixels_read",
        "target_split_labels_read",
    }
    if (
        set(data) != expected_data_fields
        or not isinstance(container_identity, Mapping)
        or not isinstance(normalizer_document, Mapping)
    ):
        raise IntegrityError("source receipt lacks container/normalizer identity mappings")
    container_sha = stable_sha256(dict(container_identity))
    if data.get("source_container_identity_sha256") != container_sha:
        raise IntegrityError("source container identity SHA-256 does not replay")
    normalizer = BandNormalizer.from_document(normalizer_document)
    if normalizer.source_container_identity_sha256 != container_sha:
        raise IntegrityError("source normalizer belongs to another source container")
    inventory_fields = (
        "population_identity_sha256",
        "population_manifest_sha256",
        "training_geo_sha256",
        "training_population_n",
        "source_train_n",
        "source_monitor_n",
        "source_rows_sha256",
    )
    inventory = {field: data.get(field) for field in inventory_fields}
    for field in (
        "population_identity_sha256",
        "population_manifest_sha256",
        "training_geo_sha256",
        "source_rows_sha256",
    ):
        require_sha256(inventory[field], field=field)
    population_n = inventory["training_population_n"]
    source_train_n = inventory["source_train_n"]
    source_monitor_n = inventory["source_monitor_n"]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (population_n, source_train_n, source_monitor_n)
        )
        or source_train_n + source_monitor_n > population_n
        or data.get("source_train_unique_label_rows_authorized") != source_train_n
        or data.get("source_monitor_unique_label_rows_authorized") != source_monitor_n
        or normalizer.source_train_n != source_train_n
        or normalizer.source_rows_sha256 != inventory["source_rows_sha256"]
    ):
        raise IntegrityError("source normalizer belongs to another row inventory")
    data_identity_document = {
        "inventory": inventory,
        "source_container_identity": dict(container_identity),
        "source_container_identity_sha256": container_sha,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "label_access_roles": [SOURCE_TRAIN_ROLE, SOURCE_MONITOR_ROLE],
        "optimization_role": SOURCE_TRAIN_ROLE,
        "checkpoint_selection_role": SOURCE_MONITOR_ROLE,
        "target_split_paths": [],
    }
    data_sha = stable_sha256(data_identity_document)
    if receipt.get("data_identity_sha256") != data_sha or checkpoint.get("data_identity_sha256") != data_sha:
        raise IntegrityError("source data identity SHA-256 does not replay")
    if (
        checkpoint.get("normalizer_sha256") != normalizer.normalizer_sha256
        or checkpoint.get("source_rows_sha256") != inventory["source_rows_sha256"]
    ):
        raise IntegrityError("source checkpoint normalizer/row identity drift")

    expected_scientific_keys = {
        "schema",
        "model_seed",
        "config",
        "config_sha256",
        "data_identity_sha256",
        "population_manifest_sha256",
        "source_rows_sha256",
        "source_container_identity_sha256",
        "normalizer_sha256",
        "code_sha256",
        "runtime",
        "runtime_sha256",
        "target_data_inputs",
    }
    if (
        set(scientific) != expected_scientific_keys
        or scientific.get("schema") != "kbound_so2sat_source_seed_identity_v1"
    ):
        raise IntegrityError("source scientific identity schema drift")
    scientific_sha = stable_sha256(dict(scientific))
    if (
        receipt.get("scientific_identity_sha256") != scientific_sha
        or checkpoint.get("scientific_identity_sha256") != scientific_sha
    ):
        raise IntegrityError("source scientific identity SHA-256 does not replay")
    runtime = scientific.get("runtime")
    if not isinstance(runtime, Mapping) or scientific.get("runtime_sha256") != stable_sha256(dict(runtime)):
        raise IntegrityError("source runtime identity SHA-256 does not replay")
    expected_scientific_fields = {
        "model_seed": receipt.get("model_seed"),
        "config": dict(config),
        "config_sha256": config_sha,
        "data_identity_sha256": data_sha,
        "population_manifest_sha256": inventory["population_manifest_sha256"],
        "source_rows_sha256": inventory["source_rows_sha256"],
        "source_container_identity_sha256": container_sha,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "code_sha256": code_sha,
        "target_data_inputs": [],
    }
    for field, expected in expected_scientific_fields.items():
        if scientific.get(field) != expected:
            raise IntegrityError(f"source scientific identity field {field} drift")
    if (
        receipt.get("optimization_data_role") != SOURCE_TRAIN_ROLE
        or receipt.get("selection_data_role") != SOURCE_MONITOR_ROLE
        or data.get("other_role_label_rows_read") != 0
        or data.get("target_split_pixels_read") != 0
        or data.get("target_split_labels_read") != 0
        or checkpoint.get("target_data_inputs") != []
    ):
        raise IntegrityError("source final artifact access/role contract drift")


def _verify_complete_result(
    paths: ArtifactPaths,
    *,
    model_seed: int,
    expected_scientific_identity_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_data_identity_sha256: str | None = None,
    expected_code_sha256: str | None = None,
) -> TrainingResult:
    verify_artifact_receipt(paths.training_receipt)
    receipt = strict_json_load(paths.training_receipt)
    expected_receipt_fields = {
        "schema",
        "status",
        "model_seed",
        "checkpoint_basename",
        "checkpoint_file_sha256",
        "checkpoint_tensor_sha256",
        "initial_tensor_sha256",
        "best_epoch",
        "best_source_monitor",
        "selection_data_role",
        "optimization_data_role",
        "config",
        "config_sha256",
        "scientific_identity",
        "scientific_identity_sha256",
        "data",
        "data_identity_sha256",
        "code_files_sha256",
        "code_sha256",
        "epochs_completed",
        "history",
        "device",
        "wall_seconds",
    }
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != expected_receipt_fields
        or receipt.get("schema") != TRAINING_RECEIPT_SCHEMA
        or receipt.get("status") != "SOURCE_TRAINING_COMPLETE"
    ):
        raise IntegrityError("unknown So2Sat source training receipt")
    wall_seconds = _require_finite_number(receipt.get("wall_seconds"), field="wall_seconds")
    if wall_seconds < 0.0 or not isinstance(receipt.get("device"), str) or not receipt["device"]:
        raise IntegrityError("source training runtime summary is invalid")
    if receipt.get("model_seed") != model_seed:
        raise IntegrityError("source training receipt model seed mismatch")
    if receipt.get("checkpoint_basename") != paths.checkpoint.name:
        raise IntegrityError("source training checkpoint basename mismatch")
    if not paths.checkpoint.is_file():
        raise IntegrityError("source training receipt has no checkpoint")
    checkpoint_file_hash = file_sha256(paths.checkpoint)
    if receipt.get("checkpoint_file_sha256") != checkpoint_file_hash:
        raise IntegrityError("source checkpoint file SHA-256 mismatch")
    checkpoint = _load_torch_mapping(paths.checkpoint)
    expected_checkpoint_fields = {
        "schema",
        "architecture_id",
        "model_state",
        "model_seed",
        "checkpoint_tensor_sha256",
        "initial_tensor_sha256",
        "best_epoch",
        "best_source_monitor",
        "scientific_identity_sha256",
        "config_sha256",
        "data_identity_sha256",
        "code_sha256",
        "normalizer_sha256",
        "source_rows_sha256",
        "target_data_inputs",
    }
    if (
        set(checkpoint) != expected_checkpoint_fields
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("model_seed") != model_seed
        or checkpoint.get("architecture_id") != ARCHITECTURE_ID
    ):
        raise IntegrityError("unknown So2Sat source checkpoint identity")
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise IntegrityError("So2Sat source checkpoint lacks a model state")
    tensor_hash = tensor_state_sha256(state)
    if tensor_hash != checkpoint.get("checkpoint_tensor_sha256"):
        raise IntegrityError("So2Sat source checkpoint tensor SHA-256 mismatch")
    if receipt.get("checkpoint_tensor_sha256") != tensor_hash:
        raise IntegrityError("source training receipt/checkpoint tensor identity mismatch")
    for field in (
        "initial_tensor_sha256",
        "scientific_identity_sha256",
        "config_sha256",
        "data_identity_sha256",
        "code_sha256",
    ):
        if receipt.get(field) != checkpoint.get(field):
            raise IntegrityError(f"source training receipt/checkpoint {field} mismatch")
    expected_fields = {
        "scientific_identity_sha256": expected_scientific_identity_sha256,
        "config_sha256": expected_config_sha256,
        "data_identity_sha256": expected_data_identity_sha256,
        "code_sha256": expected_code_sha256,
    }
    for field, expected in expected_fields.items():
        if expected is not None and receipt.get(field) != expected:
            raise IntegrityError(f"complete source result differs from current {field}")
    _verify_final_identities(receipt, checkpoint)
    best_epoch, metrics = _verify_training_history(receipt, checkpoint)
    return TrainingResult(
        model_seed=model_seed,
        checkpoint_path=paths.checkpoint,
        training_receipt_path=paths.training_receipt,
        checkpoint_file_sha256=checkpoint_file_hash,
        checkpoint_tensor_sha256=tensor_hash,
        initial_tensor_sha256=require_sha256(
            receipt.get("initial_tensor_sha256"), field="initial_tensor_sha256"
        ),
        best_epoch=best_epoch,
        best_source_monitor_macro_recall=float(metrics["macro_recall_supported_classes"]),
        best_source_monitor_accuracy=float(metrics["top1_accuracy"]),
    )


def verify_complete_source_result(
    checkpoint_dir: str | os.PathLike[str],
    model_seed: int,
    *,
    expected_scientific_identity_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_data_identity_sha256: str | None = None,
    expected_code_sha256: str | None = None,
) -> TrainingResult:
    """Strictly replay one immutable source checkpoint/receipt pair."""

    if model_seed not in CANONICAL_MODEL_SEEDS:
        raise IntegrityError(f"model seed must be one of {CANONICAL_MODEL_SEEDS}")
    directory = Path(checkpoint_dir).expanduser().resolve()
    return _verify_complete_result(
        _artifact_paths(directory, model_seed),
        model_seed=model_seed,
        expected_scientific_identity_sha256=expected_scientific_identity_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_data_identity_sha256=expected_data_identity_sha256,
        expected_code_sha256=expected_code_sha256,
    )


def train_one_seed(
    bundle: SourceDataBundle,
    output_dir: str | os.PathLike[str],
    *,
    model_seed: int,
    config: TrainingConfig,
    device: torch.device,
    resume: bool,
    stop_after_epoch: int | None = None,
) -> TrainingResult | None:
    """Train or exactly resume one seed; return ``None`` only for a test stop."""

    if model_seed not in config.model_seeds:
        raise IntegrityError(f"model seed must be one of {config.model_seeds}")
    if device.type not in {"cpu", "mps"}:
        raise IntegrityError("So2Sat source training supports only CPU or MPS")
    if stop_after_epoch is not None and config.run_mode != "synthetic_smoke":
        raise IntegrityError("stop_after_epoch is reserved for synthetic resume tests")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config.epochs:
        raise IntegrityError("stop_after_epoch is outside the configured epoch range")
    bundle.source_train.set_augmentation_seed(model_seed)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(destination, model_seed)
    code_files_start, code_sha256_start = _code_identity()
    scientific_document, scientific_identity_sha256 = _scientific_identity(
        config, bundle, code_sha256_start, model_seed, device
    )
    config_sha256 = stable_sha256(config.document())
    final_flags = (
        paths.checkpoint.exists(),
        paths.training_receipt.exists(),
        paths.training_receipt_byte_receipt.exists(),
    )
    if all(final_flags):
        if not resume:
            raise IntegrityError(f"refusing to overwrite complete model-seed {model_seed} artifacts")
        return _verify_complete_result(
            paths,
            model_seed=model_seed,
            expected_scientific_identity_sha256=scientific_identity_sha256,
            expected_config_sha256=config_sha256,
            expected_data_identity_sha256=bundle.data_identity_sha256,
            expected_code_sha256=code_sha256_start,
        )
    if any(final_flags):
        raise IntegrityError(f"partial immutable final artifacts for model seed {model_seed}")
    resume_flags = (paths.resume_state.exists(), paths.resume_receipt.exists())
    if any(resume_flags) and not all(resume_flags):
        raise IntegrityError(f"partial resume pair for model seed {model_seed}")
    if all(resume_flags) and not resume:
        raise IntegrityError(f"model seed {model_seed} has resume state; pass --resume")

    seed_everything(model_seed)
    model = build_so2sat_resnet18()
    assert_model_contract(model)
    initial_state = clone_cpu_state(model)
    initial_tensor_sha256 = tensor_state_sha256(initial_state)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.scheduler_eta_min,
    )
    start_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_key = (-float("inf"), -float("inf"), -float("inf"))
    history: list[dict[str, Any]] = []
    if all(resume_flags):
        payload = _load_resume(
            paths,
            scientific_identity_sha256=scientific_identity_sha256,
            model_seed=model_seed,
            target_epochs=config.epochs,
        )
        if payload.get("initial_tensor_sha256") != initial_tensor_sha256:
            raise IntegrityError("resume independent initialization identity mismatch")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        best_state = dict(payload["best_state"])
        best_epoch = int(payload["best_epoch"])
        best_key = tuple(float(value) for value in payload["best_key"])
        history = [dict(row) for row in payload["history"]]
        start_epoch = int(payload["completed_epochs"])

    started = time.time()
    expected_train_rows = bundle.inventory.source_train_indices
    expected_monitor_rows = bundle.inventory.source_monitor_indices
    for epoch in range(start_epoch, config.epochs):
        bundle.source_train.set_epoch(epoch)
        train_loader = _make_loader(
            bundle.source_train,
            config,
            shuffle=True,
            generator_seed=_epoch_order_seed(model_seed, epoch),
        )
        model.train()
        loss_sum = 0.0
        seen_rows: set[int] = set()
        for images, targets, row_indices in train_loader:
            batch_rows = [int(value) for value in row_indices.tolist()]
            if len(batch_rows) != len(set(batch_rows)) or seen_rows.intersection(batch_rows):
                raise IntegrityError("source_train epoch repeated a sealed row")
            seen_rows.update(batch_rows)
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape != (targets.shape[0], NUM_CLASSES):
                raise IntegrityError("So2Sat source model emitted an invalid training logit shape")
            loss = F.cross_entropy(
                logits,
                targets,
                label_smoothing=config.label_smoothing,
            )
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise IntegrityError("source_train loss became non-finite")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * int(targets.shape[0])
        if seen_rows != set(expected_train_rows):
            raise IntegrityError("source_train epoch did not consume exactly its sealed rows")
        scheduler.step()
        monitor_loader = _make_loader(
            bundle.source_monitor,
            config,
            shuffle=False,
            generator_seed=_epoch_order_seed(model_seed, epoch),
        )
        monitor = evaluate_source_monitor(
            model,
            monitor_loader,
            device,
            expected_rows=expected_monitor_rows,
        )
        key = _selection_key(monitor)
        selected = key > best_key
        if selected:
            best_key = key
            best_epoch = epoch
            best_state = clone_cpu_state(model)
        epoch_record = {
            "epoch": epoch,
            "source_train": {
                "role": "source_train",
                "cross_entropy_with_label_smoothing": loss_sum / len(expected_train_rows),
                "n": len(expected_train_rows),
            },
            "source_monitor": monitor,
            "learning_rate_after_scheduler_step": float(optimizer.param_groups[0]["lr"]),
            "selected": selected,
        }
        history.append(epoch_record)
        if best_state is None:
            raise IntegrityError("source_monitor failed to select a checkpoint")
        _save_resume(
            paths,
            scientific_identity_sha256=scientific_identity_sha256,
            model_seed=model_seed,
            completed_epochs=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            initial_tensor_sha256=initial_tensor_sha256,
            best_state=best_state,
            best_epoch=best_epoch,
            best_key=best_key,
            history=history,
        )
        print(
            f"[So2Sat seed={model_seed} epoch={epoch + 1}/{config.epochs}] "
            f"train_loss={epoch_record['source_train']['cross_entropy_with_label_smoothing']:.5f} "
            f"monitor_macro={monitor['macro_recall_supported_classes']:.5f} "
            f"monitor_acc={monitor['top1_accuracy']:.5f} selected={selected}",
            flush=True,
        )
        if stop_after_epoch is not None and epoch + 1 == stop_after_epoch:
            return None

    if best_state is None or best_epoch < 0:
        raise IntegrityError("source training produced no source_monitor-selected checkpoint")
    code_files_end, code_sha256_end = _code_identity()
    if code_files_end != code_files_start or code_sha256_end != code_sha256_start:
        raise IntegrityError("So2Sat source code/protocol changed during model training")
    best_metrics = history[best_epoch]["source_monitor"]
    checkpoint_tensor_sha256 = tensor_state_sha256(best_state)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture_id": ARCHITECTURE_ID,
        "model_state": best_state,
        "model_seed": model_seed,
        "checkpoint_tensor_sha256": checkpoint_tensor_sha256,
        "initial_tensor_sha256": initial_tensor_sha256,
        "best_epoch": best_epoch,
        "best_source_monitor": best_metrics,
        "scientific_identity_sha256": scientific_identity_sha256,
        "config_sha256": config_sha256,
        "data_identity_sha256": bundle.data_identity_sha256,
        "code_sha256": code_sha256_start,
        "normalizer_sha256": bundle.normalizer.normalizer_sha256,
        "source_rows_sha256": bundle.inventory.source_rows_sha256,
        "target_data_inputs": [],
    }
    _atomic_torch_save(paths.checkpoint, checkpoint, overwrite=False)
    checkpoint_file_sha256 = file_sha256(paths.checkpoint)
    training_receipt = {
        "schema": TRAINING_RECEIPT_SCHEMA,
        "status": "SOURCE_TRAINING_COMPLETE",
        "model_seed": model_seed,
        "checkpoint_basename": paths.checkpoint.name,
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "checkpoint_tensor_sha256": checkpoint_tensor_sha256,
        "initial_tensor_sha256": initial_tensor_sha256,
        "best_epoch": best_epoch,
        "best_source_monitor": best_metrics,
        "selection_data_role": "source_monitor",
        "optimization_data_role": "source_train",
        "config": config.document(),
        "config_sha256": config_sha256,
        "scientific_identity": scientific_document,
        "scientific_identity_sha256": scientific_identity_sha256,
        "data": {
            **bundle.inventory.identity(),
            "source_container_identity": dict(bundle.container.identity),
            "source_container_identity_sha256": bundle.container.identity_sha256,
            "normalizer": bundle.normalizer.document(),
            "source_train_unique_label_rows_authorized": len(expected_train_rows),
            "source_train_label_read_passes": config.epochs,
            "source_monitor_unique_label_rows_authorized": len(expected_monitor_rows),
            "source_monitor_label_read_passes": config.epochs,
            "other_role_label_rows_read": 0,
            "target_split_pixels_read": 0,
            "target_split_labels_read": 0,
        },
        "data_identity_sha256": bundle.data_identity_sha256,
        "code_files_sha256": code_files_start,
        "code_sha256": code_sha256_start,
        "epochs_completed": len(history),
        "history": history,
        "device": str(device),
        "wall_seconds": round(time.time() - started, 3),
    }
    write_immutable_json_with_receipt(paths.training_receipt, training_receipt)
    result = _verify_complete_result(paths, model_seed=model_seed)
    # These two files are mutable progress state owned by this exact seed.  They
    # have no purpose after the immutable checkpoint/receipt pair verifies.
    paths.resume_receipt.unlink()
    paths.resume_state.unlink()
    return result


def write_checkpoint_collection(
    output_dir: str | os.PathLike[str],
    results: Sequence[TrainingResult],
    *,
    config: TrainingConfig,
    bundle: SourceDataBundle,
    resume: bool,
) -> Path:
    """Seal the five-seed collection and enforce independent tensors."""

    ordered = sorted(results, key=lambda row: row.model_seed)
    if [row.model_seed for row in ordered] != list(CANONICAL_MODEL_SEEDS):
        raise IntegrityError("source checkpoint collection requires all five canonical seeds")
    tensor_hashes = [row.checkpoint_tensor_sha256 for row in ordered]
    initial_hashes = [row.initial_tensor_sha256 for row in ordered]
    if len(set(tensor_hashes)) != len(CANONICAL_MODEL_SEEDS):
        raise IntegrityError("source checkpoint tensors are not independent across all five seeds")
    if len(set(initial_hashes)) != len(CANONICAL_MODEL_SEEDS):
        raise IntegrityError("source model initializations are not independent across all five seeds")
    document = {
        "schema": COLLECTION_SCHEMA,
        "status": "FIVE_INDEPENDENT_SOURCE_CHECKPOINTS_VERIFIED",
        "model_seeds": list(CANONICAL_MODEL_SEEDS),
        "all_checkpoint_tensor_hashes_distinct": True,
        "all_initial_tensor_hashes_distinct": True,
        "config_sha256": stable_sha256(config.document()),
        "data_identity_sha256": bundle.data_identity_sha256,
        "normalizer_sha256": bundle.normalizer.normalizer_sha256,
        "source_rows_sha256": bundle.inventory.source_rows_sha256,
        "checkpoints": [row.collection_row() for row in ordered],
        "target_data_inputs": [],
    }
    path = Path(output_dir).expanduser().resolve() / "so2sat_source_checkpoint_collection.json"
    if path.exists() or path.with_name(path.name + ".receipt.json").exists():
        if not resume:
            raise IntegrityError("refusing to overwrite the source checkpoint collection")
        verify_artifact_receipt(path)
        existing = strict_json_load(path)
        if existing != document:
            raise IntegrityError("existing source checkpoint collection differs from verified results")
        return path
    write_immutable_json_with_receipt(path, document)
    return path


def train_all_seeds(
    bundle: SourceDataBundle,
    output_dir: str | os.PathLike[str],
    *,
    config: TrainingConfig,
    device: torch.device,
    resume: bool,
) -> tuple[TrainingResult, ...]:
    """Train the exact five-seed ensemble and seal its collection receipt."""

    results: list[TrainingResult] = []
    for model_seed in config.model_seeds:
        bundle.source_train.set_augmentation_seed(model_seed)
        result = train_one_seed(
            bundle,
            output_dir,
            model_seed=model_seed,
            config=config,
            device=device,
            resume=resume,
        )
        if result is None:  # pragma: no cover - only explicit synthetic interruption uses None
            raise IntegrityError("full five-seed training stopped before a seed completed")
        results.append(result)
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
    write_checkpoint_collection(
        output_dir,
        results,
        config=config,
        bundle=bundle,
        resume=resume,
    )
    return tuple(results)


def _load_or_fit_normalizer(
    output_dir: Path,
    inventory: SourceRoleInventory,
    container: Any,
) -> BandNormalizer:
    path = output_dir / "so2sat_sen2_source_normalizer.json"
    receipt = path.with_name(path.name + ".receipt.json")
    flags = (path.exists(), receipt.exists())
    if any(flags) and not all(flags):
        raise IntegrityError("partial source normalizer artifact/receipt pair")
    if all(flags):
        normalizer = load_sealed_band_normalizer(path)
        if normalizer.source_rows_sha256 != inventory.source_rows_sha256:
            raise IntegrityError("sealed normalizer belongs to a different source inventory")
        if normalizer.source_container_identity_sha256 != container.identity_sha256:
            raise IntegrityError("sealed normalizer belongs to a different source container")
        return normalizer
    normalizer = fit_band_normalizer(container, inventory)
    seal_band_normalizer(path, normalizer)
    return normalizer


def run_production_training(
    *,
    population_manifest: str | os.PathLike[str],
    training_geo: str | os.PathLike[str],
    training_data: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    device_name: str,
    workers: int,
    resume: bool,
) -> tuple[TrainingResult, ...]:
    """Execute the source-only production path with fixed scientific settings."""

    _, inventory = load_verified_source_inventory(population_manifest, training_geo)
    container = H5SourceContainer(
        training_data,
        expected_rows=inventory.training_population_n,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    normalizer = _load_or_fit_normalizer(destination, inventory, container)
    config = TrainingConfig(workers=workers)
    bundle = build_source_data_bundle(
        inventory,
        container,
        normalizer,
        augmentation_seed=0,
    )
    return train_all_seeds(
        bundle,
        destination,
        config=config,
        device=select_device(device_name),
        resume=resume,
    )


def build_synthetic_smoke_bundle(
    output_dir: str | os.PathLike[str],
    *,
    sample_seed: int = 104729,
) -> SourceDataBundle:
    """Create a tiny, deterministic 17-class source-only dataset in memory."""

    rng = np.random.default_rng(sample_seed)
    train_n = 34
    monitor_n = 17
    population_n = train_n + monitor_n
    labels = np.zeros((population_n, NUM_CLASSES), dtype=np.float32)
    class_ids = np.arange(population_n) % NUM_CLASSES
    labels[np.arange(population_n), class_ids] = 1.0
    pixels = rng.normal(0.0, 0.6, size=(population_n, 32, 32, 10)).astype(np.float32)
    for row_index, class_id in enumerate(class_ids):
        pixels[row_index, :, :, class_id % 10] += 0.15 + 0.02 * class_id
        pixels[row_index, class_id % 32, :, :] += 0.05
    inventory = synthetic_source_inventory(
        range(train_n),
        range(train_n, population_n),
        population_n=population_n,
    )
    container = ArraySourceContainer(
        pixels,
        labels,
        identity_tag="kbound_so2sat_source_synthetic_smoke_v1",
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    normalizer = _load_or_fit_normalizer(destination, inventory, container)
    return build_source_data_bundle(
        inventory,
        container,
        normalizer,
        augmentation_seed=0,
    )


def run_synthetic_smoke(
    output_dir: str | os.PathLike[str],
    *,
    device_name: str = "cpu",
    resume: bool = False,
    epochs: int = 1,
) -> tuple[TrainingResult, ...]:
    """Exercise the real model/trainer/receipts on tiny in-memory data."""

    bundle = build_synthetic_smoke_bundle(output_dir)
    config = TrainingConfig(
        epochs=epochs,
        batch_size=34,
        learning_rate=1e-3,
        weight_decay=1e-2,
        label_smoothing=0.0,
        scheduler_eta_min=1e-6,
        workers=0,
        run_mode="synthetic_smoke",
    )
    return train_all_seeds(
        bundle,
        output_dir,
        config=config,
        device=select_device(device_name),
        resume=resume,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-manifest", type=Path)
    parser.add_argument("--training-geo", type=Path)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        if any(value is not None for value in (args.population_manifest, args.training_geo, args.training_data)):
            raise IntegrityError("synthetic smoke mode does not accept real-data paths")
        results = run_synthetic_smoke(
            args.output_dir,
            device_name=args.device,
            resume=args.resume,
        )
    else:
        missing = [
            name
            for name, value in (
                ("--population-manifest", args.population_manifest),
                ("--training-geo", args.training_geo),
                ("--training-data", args.training_data),
            )
            if value is None
        ]
        if missing:
            parser.error(f"production source training requires {', '.join(missing)}")
        results = run_production_training(
            population_manifest=args.population_manifest,
            training_geo=args.training_geo,
            training_data=args.training_data,
            output_dir=args.output_dir,
            device_name=args.device,
            workers=args.workers,
            resume=args.resume,
        )
    print(
        "So2Sat source training: PASS "
        + " ".join(
            f"seed{row.model_seed}={row.checkpoint_tensor_sha256[:12]}" for row in results
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
