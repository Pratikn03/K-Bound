"""Deterministic sequence handling and fixed label-free trace features."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from .integrity import IntegrityError, stable_sha256

FEATURE_NAMES = (
    "frozen_mean_entropy",
    "tent_mean_entropy",
    "entropy_change",
    "frozen_mean_confidence",
    "tent_mean_confidence",
    "confidence_change",
    "prediction_disagreement",
    "marginal_jensen_shannon_divergence",
    "normalized_predicted_class_effective_count",
    "normalized_tent_update_norm",
    "batchnorm_batch_source_statistic_divergence",
)
PROBE_FRACTION = 0.30
PARTITION_SALT = "KBOUND_CCT20_PROBE_EVAL_v1"
TARGET_BATCH_SIZE = 32

_FORBIDDEN_KEY_TOKENS = {
    "annotation",
    "annotations",
    "accuracy",
    "benefit",
    "category",
    "categoryid",
    "classid",
    "classlabel",
    "correct",
    "groundtruth",
    "label",
    "labels",
    "loss",
    "oracle",
    "outcome",
    "regret",
    "truth",
    "y",
}


def _key_token(key: Any) -> str:
    return "".join(character for character in str(key).lower() if character.isalnum())


def assert_label_free(value: Any, *, path: str = "root") -> None:
    """Reject label/outcome-bearing fields before they enter target code."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            token = _key_token(key)
            embedded = {
                "accuracy",
                "annotation",
                "benefit",
                "categoryid",
                "classid",
                "classlabel",
                "correct",
                "groundtruth",
                "label",
                "loss",
                "oracle",
                "outcome",
                "regret",
                "truth",
            }
            if token in _FORBIDDEN_KEY_TOKENS or any(word in token for word in embedded):
                raise IntegrityError(f"label/outcome field {key!r} is forbidden at {path}")
            assert_label_free(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_label_free(item, path=f"{path}[{index}]")


def _first(row: Mapping[str, Any], names: Sequence[str], *, field: str) -> Any:
    present = [row[name] for name in names if name in row]
    if len(present) != 1:
        raise IntegrityError(f"metadata row must contain exactly one {field} field from {tuple(names)}")
    return present[0]


@dataclass(frozen=True)
class SequenceItem:
    image_id: str
    sequence_id: str
    location_id: str
    file_name: str
    frame_num: int
    date_captured: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "sequence_id": self.sequence_id,
            "location_id": self.location_id,
            "file_name": self.file_name,
            "frame_num": self.frame_num,
            "date_captured": self.date_captured,
        }


def normalize_metadata_row(row: Mapping[str, Any]) -> SequenceItem:
    assert_label_free(row)
    image_id = str(_first(row, ("image_id", "id"), field="image id"))
    sequence_id = str(_first(row, ("sequence_id", "seq_id"), field="sequence id"))
    location_id = str(_first(row, ("location_id", "location"), field="location id"))
    file_name = str(_first(row, ("file_name",), field="file name"))
    if "frame_num" not in row:
        raise IntegrityError(f"metadata row for image {image_id!r} lacks frame_num")
    frame = row["frame_num"]
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise IntegrityError(f"invalid non-negative frame_num for image {image_id!r}: {frame!r}")
    if not image_id or not sequence_id or not location_id or not file_name:
        raise IntegrityError("image, sequence, location, and file identifiers must be non-empty")
    raw_date = _first(row, ("date_captured", "datetime"), field="capture date")
    date_captured = raw_date if isinstance(raw_date, str) else ""
    if not date_captured:
        raise IntegrityError(f"metadata row for image {image_id!r} lacks date_captured")
    return SequenceItem(image_id, sequence_id, location_id, file_name, frame, date_captured)


def _sequence_order_key(sequence_id: str, salt: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{salt}\x00{sequence_id}".encode()).hexdigest()
    return digest, sequence_id


def sequence_atomic_batches(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_images: int,
    salt: str | None = None,
    order: str = "native",
    merge_singleton_final: bool = True,
) -> list[list[dict[str, Any]]]:
    """Pack deterministic whole sequences into batches.

    A sequence larger than ``max_images`` becomes one oversize batch; it is
    never truncated or split.  Rows from different locations are rejected.
    """

    if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 1:
        raise IntegrityError("max_images must be a positive integer")
    if order not in {"native", "hash"}:
        raise IntegrityError("batch order must be 'native' or 'hash'")
    if order == "hash" and not salt:
        raise IntegrityError("hash batch ordering requires a non-empty salt")
    items = [normalize_metadata_row(row) for row in rows]
    if not items:
        raise IntegrityError("cannot batch an empty metadata collection")
    image_ids = [item.image_id for item in items]
    if len(set(image_ids)) != len(image_ids):
        raise IntegrityError("duplicate image_id in sequence batch input")
    locations = {item.location_id for item in items}
    if len(locations) != 1:
        raise IntegrityError(f"one Tent stream cannot cross camera locations: {sorted(locations)}")

    grouped: dict[str, list[SequenceItem]] = {}
    for item in items:
        grouped.setdefault(item.sequence_id, []).append(item)
    sequences = []
    for sequence_id, members in grouped.items():
        if order == "native" and any(not item.date_captured for item in members):
            raise IntegrityError("native CCT-20 stream order requires date_captured metadata")
        ordered = sorted(
            members,
            key=lambda item: (item.date_captured, item.frame_num, item.image_id),
        )
        sequences.append((sequence_id, ordered))
    if order == "native":
        sequences.sort(key=lambda pair: (pair[1][0].date_captured, pair[0]))
    else:
        sequences.sort(key=lambda pair: _sequence_order_key(pair[0], str(salt)))

    packed: list[list[dict[str, Any]]] = []
    current: list[SequenceItem] = []
    for _, sequence in sequences:
        if current and len(current) + len(sequence) > max_images:
            packed.append([item.as_dict() for item in current])
            current = []
        if len(sequence) > max_images:
            if current:  # pragma: no cover - handled by branch above; defensive
                packed.append([item.as_dict() for item in current])
                current = []
            packed.append([item.as_dict() for item in sequence])
        else:
            current.extend(sequence)
    if current:
        packed.append([item.as_dict() for item in current])
    if merge_singleton_final and len(packed) > 1 and len(packed[-1]) == 1:
        packed[-2].extend(packed.pop())

    flattened = [row["image_id"] for batch in packed for row in batch]
    if sorted(flattened) != sorted(image_ids):  # pragma: no cover - construction guard
        raise IntegrityError("sequence batching lost or duplicated images")
    membership: dict[str, int] = {}
    for batch_index, batch in enumerate(packed):
        for row in batch:
            prior = membership.setdefault(row["sequence_id"], batch_index)
            if prior != batch_index:  # pragma: no cover - construction guard
                raise IntegrityError("a sequence was split across batches")
    return packed


def sequence_atomic_partition(
    rows: Iterable[Mapping[str, Any]],
    *,
    probe_fraction: float,
    salt: str,
) -> dict[str, Any]:
    """Assign complete sequences to probe/evaluation using a frozen hash rule."""

    if not (0.0 < float(probe_fraction) < 1.0):
        raise IntegrityError("probe_fraction must lie strictly between zero and one")
    if not salt:
        raise IntegrityError("partition salt cannot be empty")
    items = [normalize_metadata_row(row) for row in rows]
    if not items:
        raise IntegrityError("cannot partition an empty metadata collection")
    if len({item.image_id for item in items}) != len(items):
        raise IntegrityError("duplicate image_id in partition input")
    grouped: dict[tuple[str, str], list[SequenceItem]] = {}
    for item in items:
        grouped.setdefault((item.location_id, item.sequence_id), []).append(item)
    roles: dict[tuple[str, str], str] = {}
    fraction = Fraction(str(float(probe_fraction)))
    # Compare the integer digest to the rational threshold without rounding.
    # Using ``floor(fraction * 2**256)`` with ``digest < threshold`` drops the
    # single boundary integer whenever the rational product is non-integral.
    # The cross-multiplied form is exactly the sealed rule
    # ``digest / 2**256 < probe_fraction`` for every possible SHA-256 digest.
    fraction_numerator = fraction.numerator
    fraction_denominator = fraction.denominator
    digest_denominator = 1 << 256
    for location_id, sequence_id in grouped:
        digest = hashlib.sha256(f"{salt}\x00{location_id}\x00{sequence_id}".encode()).digest()
        digest_integer = int.from_bytes(digest, "big")
        roles[(location_id, sequence_id)] = (
            "probe" if digest_integer * fraction_denominator < fraction_numerator * digest_denominator else "evaluation"
        )
    locations = {item.location_id for item in items}
    for location in locations:
        observed_roles = {role for (unit_location, _), role in roles.items() if unit_location == location}
        if observed_roles != {"probe", "evaluation"}:
            raise IntegrityError(f"hash partition produced an empty probe or evaluation role for location {location!r}")
    output = {"probe": [], "evaluation": []}
    for unit, members in grouped.items():
        role = roles[unit]
        output[role].extend(item.as_dict() for item in members)
    for role in output:
        output[role].sort(
            key=lambda row: (
                row["location_id"],
                row["date_captured"],
                row["sequence_id"],
                row["frame_num"],
                row["image_id"],
            )
        )
    return {
        "schema": "kbound_cct20_sequence_partition_v1",
        "probe_fraction": float(probe_fraction),
        "salt_sha256": stable_sha256(salt),
        "n_sequences": len(grouped),
        "n_images": len(items),
        "roles": output,
    }


def _as_logits(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{name} must be a finite numeric matrix") from exc
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 16:
        raise IntegrityError(f"{name} must have shape (n, 16), found {array.shape}")
    if not np.isfinite(array).all():
        raise IntegrityError(f"{name} contains NaN or Infinity")
    return array


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def extract_label_free_features(
    frozen_logits: Any,
    adapted_logits: Any,
    *,
    normalized_tent_update_norm: Any,
    batchnorm_batch_source_statistic_divergence: Any,
) -> dict[str, Any]:
    """Compute the protocol's fixed eleven-dimensional label-free gate signal."""

    frozen = _as_logits(frozen_logits, name="frozen_logits")
    adapted = _as_logits(adapted_logits, name="adapted_logits")
    if frozen.shape != adapted.shape:
        raise IntegrityError(f"frozen/adapted logits shape mismatch: {frozen.shape} != {adapted.shape}")
    p0 = _probabilities(frozen)
    pa = _probabilities(adapted)
    tiny = np.finfo(np.float64).tiny
    conf0 = p0.max(axis=1)
    confa = pa.max(axis=1)
    entropy0 = -(p0 * np.log(np.maximum(p0, tiny))).sum(axis=1) / math.log(16.0)
    entropya = -(pa * np.log(np.maximum(pa, tiny))).sum(axis=1) / math.log(16.0)
    marginal0 = p0.mean(axis=0)
    marginala = pa.mean(axis=0)
    midpoint = 0.5 * (marginal0 + marginala)
    js = 0.5 * np.sum(
        marginal0 * (np.log(np.maximum(marginal0, tiny)) - np.log(np.maximum(midpoint, tiny)))
    ) + 0.5 * np.sum(marginala * (np.log(np.maximum(marginala, tiny)) - np.log(np.maximum(midpoint, tiny))))
    js /= math.log(2.0)
    predicted_counts = np.bincount(p0.argmax(axis=1), minlength=16).astype(np.float64)
    predicted_distribution = predicted_counts / predicted_counts.sum()
    nonzero = predicted_distribution[predicted_distribution > 0.0]
    effective_count = math.exp(float(-np.sum(nonzero * np.log(nonzero)))) / 16.0
    update_norm = float(normalized_tent_update_norm)
    bn_divergence = float(batchnorm_batch_source_statistic_divergence)
    if not math.isfinite(update_norm) or update_norm < 0.0:
        raise IntegrityError("normalized_tent_update_norm must be finite and non-negative")
    if not math.isfinite(bn_divergence) or bn_divergence < 0.0:
        raise IntegrityError("batchnorm_batch_source_statistic_divergence must be finite and non-negative")
    values = (
        float(entropy0.mean()),
        float(entropya.mean()),
        float((entropy0 - entropya).mean()),
        float(conf0.mean()),
        float(confa.mean()),
        float((confa - conf0).mean()),
        float(np.mean(p0.argmax(axis=1) != pa.argmax(axis=1))),
        float(js),
        float(effective_count),
        update_norm,
        bn_divergence,
    )
    features = dict(zip(FEATURE_NAMES, values, strict=True))
    if not all(math.isfinite(value) for value in features.values()):  # pragma: no cover
        raise IntegrityError("derived label-free feature is non-finite")
    return {
        "schema": "kbound_cct20_label_free_trace_features_v1",
        "n_probe_images": int(frozen.shape[0]),
        "n_classes": 16,
        "feature_names": list(FEATURE_NAMES),
        "features": features,
    }
