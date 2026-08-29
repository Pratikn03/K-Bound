#!/usr/bin/env python3
"""Train five independent CCT-20 source models without target access.

The command line accepts only ``train_annotations.json``.  Complete camera
sequences are assigned to source-fit or a 10% source-monitor role by the sealed
metadata hash; the official ``cis_val`` file is intentionally unused because
it overlaps train sequence identifiers.  There is no target annotation,
manifest, pixel, or metric input.

Each model seed starts from torchvision's ResNet-50 ImageNet-1K V2 weights,
fine-tunes the full network for ten epochs, and writes a hash-bound checkpoint.
Every declared train image is decoded and hashed before training; a missing or
unreadable sample aborts the run.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as transforms
import yaml
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

try:
    from .audit_checkpoints import CANONICAL_MODEL_SEEDS, tensor_state_sha256
    from .integrity import (
        IntegrityError,
        atomic_json_dump,
        file_sha256,
        stable_sha256,
        strict_json_load,
    )
except ImportError:  # pragma: no cover - direct script execution
    from audit_checkpoints import CANONICAL_MODEL_SEEDS, tensor_state_sha256
    from integrity import IntegrityError, atomic_json_dump, file_sha256, stable_sha256, strict_json_load


TRAIN_ANNOTATIONS_BASENAME = "train_annotations.json"
CHECKPOINT_SCHEMA = "kbound_cct20_source_checkpoint_v1"
METADATA_SCHEMA = "kbound_cct20_source_training_metadata_v1"
EXPECTED_NUM_CLASSES = 16
FROZEN_CATEGORY_ID_TO_NAME = {
    1: "opossum",
    3: "raccoon",
    5: "squirrel",
    6: "bobcat",
    7: "skunk",
    8: "dog",
    9: "coyote",
    10: "rabbit",
    11: "bird",
    16: "cat",
    21: "badger",
    30: "empty",
    33: "car",
    34: "deer",
    51: "fox",
    99: "rodent",
}
LABEL_CONTRACT_SHA256 = "295403f261932df1e3118225b30fe813a51247cc56ae4759bba9c19f92aa79c1"
LABEL_CONTRACT_RELATIVE_PATH = Path(
    "research_lock/KBOUND_CCT20_TARGET_SELECTION_v1_LABEL_CONTRACT_ADDENDUM.yaml"
)
PROTOCOL_RELATIVE_PATH = Path("experiments/kbound/cct20/prospective_protocol_v1.yaml")
SOURCE_MONITOR_SALT = "KBOUND_CCT20_SOURCE_MONITOR_v1"
SOURCE_MONITOR_FRACTION = 0.10
SOURCE_PARTITION_UNIT = ("location", "seq_id")
TRAINING_EPOCHS = 10
PHYSICAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SCHEDULER_ETA_MIN = 1e-6
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LabeledSample:
    image_id: int | str
    file_name: str
    path: Path
    location: int | str
    seq_id: str
    category_keys: tuple[str, ...]
    labels: tuple[int, ...]
    image_bytes: int
    image_sha256: str


@dataclass(frozen=True)
class LabeledSplit:
    role: str
    annotation_path: Path
    annotation_sha256: str
    samples: tuple[LabeledSample, ...]
    categories: tuple[dict[str, Any], ...]
    population_sha256: str


@dataclass(frozen=True)
class TrainingBundle:
    source: LabeledSplit
    source_fit_samples: tuple[LabeledSample, ...]
    source_monitor_samples: tuple[LabeledSample, ...]
    partition_manifest_sha256: str
    categories: tuple[dict[str, Any], ...]
    data_sha256: str


def _category_key(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrityError(f"category id must be an integer, found {value!r}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _image_id_key(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise IntegrityError(f"image id must be an integer or string, found {value!r}")
    if isinstance(value, str) and not value.strip():
        raise IntegrityError("image id must not be an empty string")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _relative_path(value: Any, *, context: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise IntegrityError(f"{context} must be a non-empty relative path")
    posix = PurePosixPath(value.replace("\\", "/"))
    if posix.is_absolute() or not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise IntegrityError(f"unsafe image path in {context}: {value!r}")
    normalized = posix.as_posix()
    return normalized, Path(*posix.parts)


def _resolve_image(image_root: Path, relative: Path, *, context: str) -> Path:
    root = image_root.expanduser().resolve(strict=True)
    try:
        path = (root / relative).resolve(strict=True)
    except FileNotFoundError as exc:
        raise IntegrityError(f"declared {context} image is missing: {relative.as_posix()}") from exc
    if root != path and root not in path.parents:
        raise IntegrityError(f"declared {context} image escapes image root: {relative.as_posix()}")
    if not path.is_file():
        raise IntegrityError(f"declared {context} image is not a regular file: {path}")
    return path


def _decode_and_hash(path: Path, *, context: str) -> tuple[int, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"cannot read declared {context} image {path}: {exc}") from exc
    if not payload:
        raise IntegrityError(f"declared {context} image is empty: {path}")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise IntegrityError(f"declared {context} image does not decode completely: {path}: {exc}") from exc
    import hashlib

    return len(payload), hashlib.sha256(payload).hexdigest()


def _parse_categories(raw: Any, *, role: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list) or not raw:
        raise IntegrityError(f"{role} annotations require a non-empty categories array")
    by_key: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or "id" not in row:
            raise IntegrityError(f"{role} categories[{index}] requires id")
        key = _category_key(row["id"])
        if key in by_key:
            raise IntegrityError(f"duplicate {role} category id: {row['id']!r}")
        name = row.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise IntegrityError(f"{role} categories[{index}].name must be a non-empty string")
        by_key[key] = {"id": row["id"], "name": name}
    actual = {
        int(row["id"]): str(row.get("name", "")).strip().casefold()
        for row in by_key.values()
    }
    if actual != FROZEN_CATEGORY_ID_TO_NAME:
        raise IntegrityError(
            f"{role} categories do not match the sealed 16-output sparse-id mapping: {actual}"
        )
    return tuple(
        {"id": category_id, "name": FROZEN_CATEGORY_ID_TO_NAME[category_id]}
        for category_id in sorted(FROZEN_CATEGORY_ID_TO_NAME)
    )


def _category_index(categories: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {_category_key(row["id"]): index for index, row in enumerate(categories)}


def load_labeled_split(
    annotation_path: str | os.PathLike[str],
    image_root: str | os.PathLike[str],
    *,
    role: str,
    expected_basename: str,
    reference_categories: Sequence[Mapping[str, Any]] | None = None,
) -> LabeledSplit:
    """Load one allowed source-side role and verify its complete population."""

    annotation = Path(annotation_path).expanduser().resolve()
    if annotation.name != expected_basename:
        raise IntegrityError(
            f"{role} must be loaded from a file named {expected_basename!r}; found {annotation.name!r}"
        )
    if not annotation.is_file():
        raise FileNotFoundError(f"missing {role} annotation file: {annotation}")
    root = Path(image_root)
    if not root.is_dir():
        raise FileNotFoundError(f"CCT-20 image root is missing: {root}")
    document = strict_json_load(annotation)
    if not isinstance(document, dict):
        raise IntegrityError(f"{role} annotation document must be an object")
    images = document.get("images")
    annotations = document.get("annotations")
    if not isinstance(images, list) or not images:
        raise IntegrityError(f"{role} annotations require a non-empty images array")
    if not isinstance(annotations, list) or not annotations:
        raise IntegrityError(f"{role} annotations require a non-empty annotations array")
    categories = _parse_categories(document.get("categories"), role=role)
    if reference_categories is not None and list(categories) != [dict(row) for row in reference_categories]:
        raise IntegrityError(f"{role} category vocabulary differs from train annotations")
    label_index = _category_index(categories)

    image_by_key: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for index, row in enumerate(images):
        if (
            not isinstance(row, dict)
            or "id" not in row
            or "file_name" not in row
            or "location" not in row
            or "seq_id" not in row
        ):
            raise IntegrityError(
                f"{role} images[{index}] requires id, file_name, location, and seq_id"
            )
        key = _image_id_key(row["id"])
        if key in image_by_key:
            raise IntegrityError(f"duplicate {role} image id: {row['id']!r}")
        normalized, relative = _relative_path(row["file_name"], context=f"{role} images[{index}].file_name")
        if normalized in paths:
            raise IntegrityError(f"duplicate {role} image path: {normalized!r}")
        paths.add(normalized)
        location = row["location"]
        if isinstance(location, bool) or not isinstance(location, (int, str)):
            raise IntegrityError(f"{role} images[{index}].location must be an integer or string")
        if isinstance(location, str) and not location.strip():
            raise IntegrityError(f"{role} images[{index}].location must not be empty")
        seq_id = row["seq_id"]
        if not isinstance(seq_id, str) or not seq_id.strip():
            raise IntegrityError(f"{role} images[{index}].seq_id must be a non-empty string")
        image_by_key[key] = {
            "id": row["id"],
            "file_name": normalized,
            "relative": relative,
            "location": location,
            "seq_id": seq_id,
        }

    categories_by_image: dict[str, set[str]] = {key: set() for key in image_by_key}
    annotation_ids: set[str] = set()
    for index, row in enumerate(annotations):
        if (
            not isinstance(row, dict)
            or "id" not in row
            or "image_id" not in row
            or "category_id" not in row
        ):
            raise IntegrityError(
                f"{role} annotations[{index}] requires id, image_id, and category_id"
            )
        image_key = _image_id_key(row["image_id"])
        if image_key not in image_by_key:
            raise IntegrityError(f"{role} annotation references unknown image id: {row['image_id']!r}")
        annotation_id = _image_id_key(row["id"])
        if annotation_id in annotation_ids:
            raise IntegrityError(f"duplicate {role} annotation id: {row['id']!r}")
        annotation_ids.add(annotation_id)
        category_key = _category_key(row["category_id"])
        if category_key not in label_index:
            raise IntegrityError(f"{role} annotation uses unknown category id: {row['category_id']!r}")
        # Repeated boxes of the same class collapse to one image-level member;
        # distinct classes are all retained by the sealed soft-target contract.
        categories_by_image[image_key].add(category_key)
    missing_labels = sorted(key for key, values in categories_by_image.items() if not values)
    if missing_labels:
        raise IntegrityError(f"{role} has {len(missing_labels)} images without annotations")

    samples: list[LabeledSample] = []
    population_rows: list[dict[str, Any]] = []
    for image_key, image in image_by_key.items():
        path = _resolve_image(root, image["relative"], context=role)
        byte_count, content_hash = _decode_and_hash(path, context=role)
        category_keys = tuple(
            sorted(categories_by_image[image_key], key=lambda key: label_index[key])
        )
        labels = tuple(label_index[key] for key in category_keys)
        sample = LabeledSample(
            image_id=image["id"],
            file_name=image["file_name"],
            path=path,
            location=image["location"],
            seq_id=image["seq_id"],
            category_keys=category_keys,
            labels=labels,
            image_bytes=byte_count,
            image_sha256=content_hash,
        )
        samples.append(sample)
        population_rows.append(
            {
                "image_id": sample.image_id,
                "file_name": sample.file_name,
                "location": sample.location,
                "seq_id": sample.seq_id,
                "category_keys": list(sample.category_keys),
                "label_indices": list(sample.labels),
                "image_bytes": sample.image_bytes,
                "image_sha256": sample.image_sha256,
            }
        )
    return LabeledSplit(
        role=role,
        annotation_path=annotation,
        annotation_sha256=file_sha256(annotation),
        samples=tuple(samples),
        categories=categories,
        population_sha256=stable_sha256(population_rows),
    )


def source_monitor_role(location: int | str, seq_id: str) -> str:
    """Assign a complete source sequence using metadata only."""

    payload = {
        "salt": SOURCE_MONITOR_SALT,
        "location": location,
        "seq_id": seq_id,
    }
    fraction = int(stable_sha256(payload), 16) / (1 << 256)
    return "source_monitor" if fraction < SOURCE_MONITOR_FRACTION else "source_fit"


def load_training_bundle(
    train_annotations: str | os.PathLike[str],
    image_root: str | os.PathLike[str],
) -> TrainingBundle:
    source = load_labeled_split(
        train_annotations,
        image_root,
        role="train",
        expected_basename=TRAIN_ANNOTATIONS_BASENAME,
    )
    if len(source.categories) != EXPECTED_NUM_CLASSES:
        raise IntegrityError(
            "the sealed primary protocol is 16-way full-frame classification "
            f"(14 animals, car, and empty); train annotations expose {len(source.categories)} categories"
        )
    fit_samples: list[LabeledSample] = []
    monitor_samples: list[LabeledSample] = []
    sequence_roles: dict[tuple[str, str], str] = {}
    assignments: list[dict[str, Any]] = []
    for sample in source.samples:
        unit = (_image_id_key(sample.location), sample.seq_id)
        role = source_monitor_role(sample.location, sample.seq_id)
        prior = sequence_roles.setdefault(unit, role)
        if prior != role:  # defensive: the pure hash function makes this impossible
            raise IntegrityError(f"source sequence was assigned to two roles: {unit}")
        (monitor_samples if role == "source_monitor" else fit_samples).append(sample)
        assignments.append(
            {
                "image_id": sample.image_id,
                "location": sample.location,
                "seq_id": sample.seq_id,
                "role": role,
            }
        )
    if not fit_samples or not monitor_samples:
        raise IntegrityError(
            "source sequence hash produced an empty fit or monitor role; refusing training"
        )
    if len(fit_samples) + len(monitor_samples) != len(source.samples):
        raise IntegrityError("source fit/monitor partition does not cover every train image exactly once")
    partition_manifest_sha256 = stable_sha256(assignments)
    identity = {
        "source_train": {
            "annotation_sha256": source.annotation_sha256,
            "population_sha256": source.population_sha256,
            "n": len(source.samples),
        },
        "partition": {
            "unit": list(SOURCE_PARTITION_UNIT),
            "salt": SOURCE_MONITOR_SALT,
            "monitor_fraction": SOURCE_MONITOR_FRACTION,
            "manifest_sha256": partition_manifest_sha256,
            "fit_n": len(fit_samples),
            "monitor_n": len(monitor_samples),
        },
        "categories_sha256": stable_sha256(source.categories),
    }
    return TrainingBundle(
        source=source,
        source_fit_samples=tuple(fit_samples),
        source_monitor_samples=tuple(monitor_samples),
        partition_manifest_sha256=partition_manifest_sha256,
        categories=source.categories,
        data_sha256=stable_sha256(identity),
    )


class CCTClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Exact-index dataset: image failures propagate; no fallback sample exists."""

    def __init__(self, samples: Sequence[LabeledSample], transform: Callable[[Image.Image], torch.Tensor]):
        self.samples = tuple(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as image:
                rgb = image.convert("RGB")
                tensor = self.transform(rgb)
        except Exception as exc:
            raise IntegrityError(
                f"failed to load exact dataset index {index} ({sample.file_name}); no substitution allowed"
            ) from exc
        target = torch.zeros(EXPECTED_NUM_CLASSES, dtype=torch.float32)
        target[list(sample.labels)] = 1.0 / len(sample.labels)
        return tensor, target


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(232, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_resnet50_imagenet(num_classes: int) -> nn.Module:
    if num_classes <= 1:
        raise IntegrityError("CCT-20 source training requires at least two classes")
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise IntegrityError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise IntegrityError("MPS requested but unavailable")
    return torch.device(requested)


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for images, soft_targets in loader:
        logits = model(images.to(device))
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(soft_targets.cpu())
    if not targets:
        raise IntegrityError("source-monitor loader produced no samples")
    predicted = torch.cat(predictions).numpy()
    truth = torch.cat(targets).numpy()
    truth_membership = truth > 0.0
    row_indices = np.arange(len(predicted))
    set_correct = truth_membership[row_indices, predicted]
    prediction_sets = np.zeros_like(truth_membership, dtype=bool)
    prediction_sets[row_indices, predicted] = True
    f1_values = []
    for category in range(truth_membership.shape[1]):
        true_category = truth_membership[:, category]
        predicted_category = prediction_sets[:, category]
        true_positive = int(np.sum(true_category & predicted_category))
        false_positive = int(np.sum(~true_category & predicted_category))
        false_negative = int(np.sum(true_category & ~predicted_category))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "set_membership_top1_accuracy": float(np.mean(set_correct)),
        "multilabel_macro_f1": float(np.mean(f1_values)),
        "n": int(len(truth)),
        "n_ground_truth_categories_observed": int(np.sum(np.any(truth_membership, axis=0))),
    }


def normalized_soft_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy for the sealed uniform-over-distinct-categories target."""

    if logits.ndim != 2 or targets.shape != logits.shape:
        raise IntegrityError(
            f"soft-target shape must equal logits shape, found {targets.shape} and {logits.shape}"
        )
    if not torch.all(torch.isfinite(targets)) or torch.any(targets < 0):
        raise IntegrityError("soft targets must be finite and non-negative")
    row_sums = targets.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=0.0, atol=1e-7):
        raise IntegrityError("every image soft target must sum exactly to one within tolerance")
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


@dataclass
class BestEpochSelector:
    """Select the highest source-monitor score; retain the earliest tie."""

    best_metric: float = float("-inf")
    best_epoch: int = -1

    def observe(self, metric: float, epoch: int) -> bool:
        if not np.isfinite(metric):
            raise IntegrityError(
                f"non-finite source-monitor selection metric at epoch {epoch}: {metric}"
            )
        improved = metric > self.best_metric
        if improved:
            self.best_metric = float(metric)
            self.best_epoch = int(epoch)
        return improved


def _clone_cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _code_identity() -> tuple[dict[str, str], str]:
    directory = Path(__file__).resolve().parent
    repository = Path(__file__).resolve().parents[3]
    label_contract = Path(__file__).resolve().parents[3] / LABEL_CONTRACT_RELATIVE_PATH
    actual_label_contract_sha256 = file_sha256(label_contract)
    if actual_label_contract_sha256 != LABEL_CONTRACT_SHA256:
        raise IntegrityError(
            "sealed CCT-20 label contract is missing or changed: "
            f"expected {LABEL_CONTRACT_SHA256}, found {actual_label_contract_sha256}"
        )
    protocol_path = repository / PROTOCOL_RELATIVE_PATH
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    model_recipe = protocol.get("source_model", {}) if isinstance(protocol, dict) else {}
    expected_recipe_fields = {
        "implementation": "torchvision.models.resnet50",
        "pretrained_weights": "IMAGENET1K_V2",
        "replace_classifier_with_outputs": EXPECTED_NUM_CLASSES,
        "trainable_parameters": "full_network",
        "seeds": list(CANONICAL_MODEL_SEEDS),
        "epochs": TRAINING_EPOCHS,
        "selection": "highest source-monitor set-membership top1; earliest epoch breaks ties",
        "loss": "normalized_soft_target_cross_entropy_over_distinct_ground_truth_set",
    }
    for field, expected in expected_recipe_fields.items():
        if model_recipe.get(field) != expected:
            raise IntegrityError(
                f"prospective protocol source_model.{field} differs from trainer: "
                f"{model_recipe.get(field)!r} != {expected!r}"
            )
    expected_nested = {
        "optimizer": {
            "name": "SGD",
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "nesterov": False,
        },
        "scheduler": {
            "name": "cosine_annealing",
            "t_max_epochs": TRAINING_EPOCHS,
            "eta_min": SCHEDULER_ETA_MIN,
        },
        "batching": {
            "physical_batch_size": PHYSICAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "incomplete_final_accumulation": "mean_over_actual_samples_in_accumulation_group",
        },
        "train_transform": {
            "random_resized_crop": {"size": 224, "scale": [0.65, 1.0]},
            "random_horizontal_flip_probability": 0.5,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
            },
            "normalization": "IMAGENET1K_V2",
        },
        "evaluation_transform": {
            "resize_shorter_side": 232,
            "center_crop": 224,
            "normalization": "IMAGENET1K_V2",
        },
    }
    for field, expected in expected_nested.items():
        if model_recipe.get(field) != expected:
            raise IntegrityError(f"prospective protocol source_model.{field} differs from trainer")
    source_role = protocol.get("roles", {}).get("source_fit_and_monitor", {})
    expected_source_role = {
        "split": "train",
        "partition_unit": list(SOURCE_PARTITION_UNIT),
        "monitor_fraction": SOURCE_MONITOR_FRACTION,
        "hash": "sha256",
        "salt": SOURCE_MONITOR_SALT,
    }
    if source_role != expected_source_role:
        raise IntegrityError("prospective protocol source-fit/monitor partition differs from trainer")
    files = {
        "train_source.py": file_sha256(directory / "train_source.py"),
        "prospective_data.py": file_sha256(directory / "prospective_data.py"),
        "integrity.py": file_sha256(directory / "integrity.py"),
        "audit_checkpoints.py": file_sha256(directory / "audit_checkpoints.py"),
        str(LABEL_CONTRACT_RELATIVE_PATH): actual_label_contract_sha256,
        str(PROTOCOL_RELATIVE_PATH): file_sha256(protocol_path),
    }
    return files, stable_sha256(files)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def accumulation_divisor(microbatch_index: int, total_microbatches: int) -> int:
    """Return the sealed group size, including a short final accumulation."""

    if total_microbatches <= 0 or not 0 <= microbatch_index < total_microbatches:
        raise IntegrityError("invalid microbatch index/count for gradient accumulation")
    group_start = (
        microbatch_index // GRADIENT_ACCUMULATION_STEPS
    ) * GRADIENT_ACCUMULATION_STEPS
    return min(GRADIENT_ACCUMULATION_STEPS, total_microbatches - group_start)


def accumulation_loss_weight(
    microbatch_index: int,
    total_microbatches: int,
    current_batch_size: int,
    total_samples: int,
) -> float:
    """Weight a mean microbatch loss by its samples in the accumulation group.

    Dividing every microbatch mean by the number of microbatches is correct only
    when all physical batches have the same size.  The final CCT-20 batch is
    short, so sample weighting is required to make each optimizer step the mean
    loss over the actual accumulation group.
    """

    group_microbatches = accumulation_divisor(microbatch_index, total_microbatches)
    if current_batch_size <= 0 or total_samples <= 0:
        raise IntegrityError("batch and sample counts must be positive")
    expected_microbatches = (total_samples + PHYSICAL_BATCH_SIZE - 1) // PHYSICAL_BATCH_SIZE
    if total_microbatches != expected_microbatches:
        raise IntegrityError(
            "microbatch count does not match samples and sealed physical batch size"
        )
    expected_current_batch_size = min(
        PHYSICAL_BATCH_SIZE,
        total_samples - microbatch_index * PHYSICAL_BATCH_SIZE,
    )
    if current_batch_size != expected_current_batch_size:
        raise IntegrityError("current physical batch size does not match its dataset position")
    group_start = (
        microbatch_index // GRADIENT_ACCUMULATION_STEPS
    ) * GRADIENT_ACCUMULATION_STEPS
    group_sample_start = group_start * PHYSICAL_BATCH_SIZE
    group_sample_count = min(
        group_microbatches * PHYSICAL_BATCH_SIZE,
        total_samples - group_sample_start,
    )
    if group_sample_count <= 0 or current_batch_size > group_sample_count:
        raise IntegrityError("invalid sample counts for gradient accumulation group")
    return current_batch_size / group_sample_count


def train_one_seed(
    bundle: TrainingBundle,
    output_dir: str | os.PathLike[str],
    *,
    model_seed: int,
    device: torch.device,
    workers: int,
) -> tuple[Path, Path]:
    if model_seed not in CANONICAL_MODEL_SEEDS:
        raise IntegrityError(f"model_seed must be one of {CANONICAL_MODEL_SEEDS}")
    if workers < 0:
        raise IntegrityError("workers cannot be negative")
    destination = Path(output_dir)
    checkpoint_path = destination / f"cct20_resnet50_seed{model_seed}.pt"
    metadata_path = destination / f"cct20_resnet50_seed{model_seed}.json"
    if checkpoint_path.exists() or metadata_path.exists():
        raise IntegrityError(
            f"refusing to overwrite model-seed {model_seed} artifacts: {checkpoint_path}, {metadata_path}"
        )
    code_files_start, code_sha256_start = _code_identity()

    seed_everything(model_seed)
    model = build_resnet50_imagenet(len(bundle.categories))
    initial_state = model.state_dict()
    initial_tensor_sha256 = tensor_state_sha256(initial_state)
    imagenet_backbone_sha256 = tensor_state_sha256(
        {name: value for name, value in initial_state.items() if not name.startswith("fc.")}
    )
    model = model.to(device)

    generator = torch.Generator().manual_seed(model_seed)
    train_loader = DataLoader(
        CCTClassificationDataset(bundle.source_fit_samples, train_transform()),
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
        generator=generator,
        worker_init_fn=_seed_worker,
        persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
    )
    source_monitor_loader = DataLoader(
        CCTClassificationDataset(bundle.source_monitor_samples, evaluation_transform()),
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
        worker_init_fn=_seed_worker,
        persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_EPOCHS,
        eta_min=SCHEDULER_ETA_MIN,
    )
    selector = BestEpochSelector()
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(TRAINING_EPOCHS):
        model.train()
        loss_sum = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        total_microbatches = len(train_loader)
        for microbatch_index, (images, soft_targets) in enumerate(train_loader):
            images = images.to(device)
            soft_targets = soft_targets.to(device, dtype=torch.float32)
            loss = normalized_soft_target_cross_entropy(model(images), soft_targets)
            count = int(soft_targets.shape[0])
            loss_weight = accumulation_loss_weight(
                microbatch_index,
                total_microbatches,
                count,
                len(bundle.source_fit_samples),
            )
            (loss * loss_weight).backward()
            loss_sum += float(loss.detach().cpu()) * count
            seen += count
            group_complete = (
                (microbatch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
                or microbatch_index + 1 == total_microbatches
            )
            if group_complete:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if seen != len(bundle.source_fit_samples):
            raise IntegrityError(
                "training epoch consumed "
                f"{seen} samples, expected exactly {len(bundle.source_fit_samples)}"
            )
        scheduler.step()
        source_monitor_metrics = evaluate(model, source_monitor_loader, device)
        improved = selector.observe(
            source_monitor_metrics["set_membership_top1_accuracy"], epoch
        )
        if improved:
            best_state = _clone_cpu_state(model)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "source_monitor": source_monitor_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "selected": improved,
        }
        history.append(record)
        print(
            f"[seed {model_seed} epoch {epoch}] train_loss={record['train_loss']:.5f} "
            "source_monitor_set_top1="
            f"{source_monitor_metrics['set_membership_top1_accuracy']:.5f} selected={improved}",
            flush=True,
        )
    if best_state is None or selector.best_epoch < 0:
        raise IntegrityError("source training produced no source-monitor-selected checkpoint")

    code_files, code_sha256 = _code_identity()
    if code_files != code_files_start or code_sha256 != code_sha256_start:
        raise IntegrityError("training code or protocol changed while this model seed was running")
    config = {
        "schema": "kbound_cct20_source_training_config_v1",
        "architecture": "torchvision_resnet50",
        "initialization": "ResNet50_Weights.IMAGENET1K_V2",
        "model_seed": model_seed,
        "canonical_model_seeds": list(CANONICAL_MODEL_SEEDS),
        "protocol_sha256": code_files[str(PROTOCOL_RELATIVE_PATH)],
        "epochs": TRAINING_EPOCHS,
        "selection": {
            "role": "source_monitor",
            "metric": "set_membership_top1_accuracy",
            "rule": "highest_metric_earliest_epoch_breaks_ties",
        },
        "source_partition": {
            "unit": list(SOURCE_PARTITION_UNIT),
            "hash": "sha256_canonical_json",
            "salt": SOURCE_MONITOR_SALT,
            "monitor_fraction": SOURCE_MONITOR_FRACTION,
            "partition_manifest_sha256": bundle.partition_manifest_sha256,
        },
        "optimizer": "SGD",
        "optimization_scope": "full_network",
        "learning_rate": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "nesterov": False,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max_epochs": TRAINING_EPOCHS,
        "scheduler_eta_min": SCHEDULER_ETA_MIN,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "incomplete_final_accumulation": "mean_over_actual_samples_in_accumulation_group",
        "gradient_mean_weighting": "actual_samples_in_accumulation_group",
        "workers": workers,
        "train_transform": {
            "random_resized_crop": [224, [0.65, 1.0]],
            "random_horizontal_flip_probability": 0.5,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
            },
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
        },
        "source_monitor_transform": {
            "resize": 232,
            "center_crop": 224,
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
        },
        "deterministic_algorithms": True,
        "label_contract": {
            "sha256": LABEL_CONTRACT_SHA256,
            "category_order": "ascending_sparse_archive_category_id",
            "category_ids": sorted(FROZEN_CATEGORY_ID_TO_NAME),
            "repeated_same_category": "collapse",
            "distinct_categories": "complete_set",
            "training_target": "uniform_probability_over_distinct_category_set",
            "loss": "normalized_soft_target_cross_entropy",
            "top1_correctness": "prediction_in_complete_ground_truth_set",
        },
        "target_data_inputs": [],
    }
    config_sha256 = stable_sha256(config)
    best_tensor_sha256 = tensor_state_sha256(best_state)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model_state": best_state,
        "architecture": "resnet50",
        "num_classes": len(bundle.categories),
        "model_seed": model_seed,
        "best_epoch": selector.best_epoch,
        "best_source_monitor_set_membership_top1_accuracy": selector.best_metric,
        "checkpoint_tensor_sha256": best_tensor_sha256,
        "initial_tensor_sha256": initial_tensor_sha256,
        "imagenet_backbone_tensor_sha256": imagenet_backbone_sha256,
        "config": config,
        "config_sha256": config_sha256,
        "data_sha256": bundle.data_sha256,
        "code_sha256": code_sha256,
    }
    _atomic_torch_save(checkpoint_path, payload)
    checkpoint_file_sha256 = file_sha256(checkpoint_path)
    metadata = {
        "schema": METADATA_SCHEMA,
        "status": "SOURCE_TRAINING_COMPLETE",
        "model_seed": model_seed,
        "checkpoint_basename": checkpoint_path.name,
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "checkpoint_tensor_sha256": best_tensor_sha256,
        "initial_tensor_sha256": initial_tensor_sha256,
        "imagenet_backbone_tensor_sha256": imagenet_backbone_sha256,
        "config": config,
        "config_sha256": config_sha256,
        "data": {
            "train_annotations_basename": TRAIN_ANNOTATIONS_BASENAME,
            "train_annotations_sha256": bundle.source.annotation_sha256,
            "train_population_sha256": bundle.source.population_sha256,
            "train_total_n": len(bundle.source.samples),
            "source_fit_n": len(bundle.source_fit_samples),
            "source_monitor_n": len(bundle.source_monitor_samples),
            "partition_manifest_sha256": bundle.partition_manifest_sha256,
            "selection_role": "source_monitor_sequence_hash_partition_of_train",
            "unused_roles": ["cis_val"],
            "target_roles_read": [],
        },
        "data_sha256": bundle.data_sha256,
        "code_files_sha256": code_files,
        "code_sha256": code_sha256,
        "best_epoch": selector.best_epoch,
        "best_source_monitor_set_membership_top1_accuracy": selector.best_metric,
        "epochs_completed": len(history),
        "history": history,
        "device": str(device),
        "wall_seconds": round(time.time() - started, 3),
    }
    atomic_json_dump(metadata_path, metadata)
    return checkpoint_path, metadata_path


def _parse_model_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if not seeds or len(set(seeds)) != len(seeds):
        raise IntegrityError("model seeds must be a non-empty unique subset of 0,1,2,3,4")
    if any(seed not in CANONICAL_MODEL_SEEDS for seed in seeds):
        raise IntegrityError(f"model seeds must be a subset of {CANONICAL_MODEL_SEEDS}")
    return seeds


def build_source_preflight_manifest(bundle: TrainingBundle) -> dict[str, Any]:
    code_files, code_sha256 = _code_identity()
    document: dict[str, Any] = {
        "schema": "kbound_cct20_source_preflight_v1",
        "status": "PASS",
        "source_role": "train",
        "unused_annotation_roles": ["cis_val", "cis_test", "trans_val", "trans_test"],
        "full_image_decode_and_hash": True,
        "sample_substitution": False,
        "train_annotations_sha256": bundle.source.annotation_sha256,
        "train_population_sha256": bundle.source.population_sha256,
        "train_total_n": len(bundle.source.samples),
        "source_fit_n": len(bundle.source_fit_samples),
        "source_monitor_n": len(bundle.source_monitor_samples),
        "partition_unit": list(SOURCE_PARTITION_UNIT),
        "partition_salt": SOURCE_MONITOR_SALT,
        "partition_monitor_fraction": SOURCE_MONITOR_FRACTION,
        "partition_manifest_sha256": bundle.partition_manifest_sha256,
        "category_id_to_output_index": {
            str(category_id): index
            for index, category_id in enumerate(sorted(FROZEN_CATEGORY_ID_TO_NAME))
        },
        "label_contract_sha256": LABEL_CONTRACT_SHA256,
        "data_sha256": bundle.data_sha256,
        "code_files_sha256": code_files,
        "code_sha256": code_sha256,
    }
    document["manifest_sha256"] = stable_sha256(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=list(CANONICAL_MODEL_SEEDS))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()

    seeds = _parse_model_seeds(args.model_seeds)
    bundle = load_training_bundle(args.train_annotations, args.image_root)
    if args.preflight_only:
        if args.preflight_output is None:
            raise IntegrityError("--preflight-only requires --preflight-output")
        if args.preflight_output.exists():
            raise IntegrityError(
                f"refusing to overwrite source preflight: {args.preflight_output}"
            )
        preflight = build_source_preflight_manifest(bundle)
        atomic_json_dump(args.preflight_output, preflight)
        print(
            f"source preflight: PASS n={len(bundle.source.samples)} "
            f"data_sha256={bundle.data_sha256} -> {args.preflight_output}",
            flush=True,
        )
        return
    if args.preflight_output is not None:
        raise IntegrityError("--preflight-output is valid only with --preflight-only")
    if args.output_dir is None:
        raise IntegrityError("source training requires --output-dir")
    device = select_device(args.device)
    print(
        f"[CCT-20 source] train_total={len(bundle.source.samples)} "
        f"source_fit={len(bundle.source_fit_samples)} "
        f"source_monitor={len(bundle.source_monitor_samples)} "
        f"classes={len(bundle.categories)} data_sha256={bundle.data_sha256} "
        f"device={device} seeds={seeds}",
        flush=True,
    )
    for seed in seeds:
        checkpoint, metadata = train_one_seed(
            bundle,
            args.output_dir,
            model_seed=seed,
            device=device,
            workers=args.workers,
        )
        print(f"[seed {seed}] checkpoint -> {checkpoint}\n[seed {seed}] metadata -> {metadata}", flush=True)


if __name__ == "__main__":
    main()
