"""Pure, label-free probe features for the prospective So2Sat gate.

The extractor deliberately accepts only two logit tensors and two scalar
adaptation diagnostics.  It has no data-loader, outcome, or scoring interface.
Every returned document is content hashed and records the identities of both
input logit tensors so downstream trace receipts can bind the exact evidence
used by the gate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .integrity import IntegrityError, canonical_json_bytes, require_sha256, stable_sha256

N_CLASSES = 17
FEATURE_SCHEMA = "kbound_so2sat_label_free_probe_features_v1"
FEATURE_STATUS = "LABEL_FREE_PROBE_FEATURES"
FEATURE_NAMES = (
    "frozen_mean_entropy",
    "adapted_mean_entropy",
    "entropy_change",
    "frozen_mean_confidence",
    "adapted_mean_confidence",
    "confidence_change",
    "prediction_disagreement",
    "marginal_jensen_shannon_divergence",
    "normalized_predicted_class_effective_count",
    "normalized_adapter_update_norm",
    "batchnorm_source_statistic_divergence",
)

_DOCUMENT_KEYS = {
    "schema",
    "status",
    "n_probe_images",
    "n_classes",
    "feature_names",
    "frozen_logits_tensor_sha256",
    "adapted_logits_tensor_sha256",
    "features",
    "feature_sha256",
}
_UNIT_INTERVAL_FEATURES = frozenset(
    {
        "frozen_mean_entropy",
        "adapted_mean_entropy",
        "frozen_mean_confidence",
        "adapted_mean_confidence",
        "prediction_disagreement",
        "marginal_jensen_shannon_divergence",
        "normalized_predicted_class_effective_count",
    }
)
_SIGNED_UNIT_INTERVAL_FEATURES = frozenset({"entropy_change", "confidence_change"})
_NONNEGATIVE_FEATURES = frozenset({"normalized_adapter_update_norm", "batchnorm_source_statistic_divergence"})


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise IntegrityError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{field} must be a finite number")
    return result


def _as_logits(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError(f"{name} must be a finite numeric matrix") from exc
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != N_CLASSES:
        raise IntegrityError(f"{name} must have shape (n, {N_CLASSES}), found {array.shape}")
    if not np.isfinite(array).all():
        raise IntegrityError(f"{name} contains NaN or Infinity")
    return np.ascontiguousarray(array, dtype=np.float64)


def _tensor_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    header = canonical_json_bytes(
        {
            "dtype": "float64",
            "shape": list(contiguous.shape),
            "order": "C",
        }
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    raw = contiguous.tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    denominator = exponentiated.sum(axis=1, keepdims=True)
    probabilities = exponentiated / denominator
    if not np.isfinite(probabilities).all():  # pragma: no cover - defensive
        raise IntegrityError("softmax produced a non-finite probability")
    return probabilities


def _unit_interval(value: float) -> float:
    """Remove harmless floating-point excursions at theoretical [0, 1] bounds."""

    return min(1.0, max(0.0, float(value)))


def _unsigned_feature_document(
    *,
    n_probe_images: int,
    frozen_logits_tensor_sha256: str,
    adapted_logits_tensor_sha256: str,
    features: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "schema": FEATURE_SCHEMA,
        "status": FEATURE_STATUS,
        "n_probe_images": n_probe_images,
        "n_classes": N_CLASSES,
        "feature_names": list(FEATURE_NAMES),
        "frozen_logits_tensor_sha256": frozen_logits_tensor_sha256,
        "adapted_logits_tensor_sha256": adapted_logits_tensor_sha256,
        "features": {name: float(features[name]) for name in FEATURE_NAMES},
    }


def extract_label_free_features(
    frozen_probe_logits: Any,
    adapted_probe_logits: Any,
    *,
    normalized_adapter_update_norm: Any,
    batchnorm_source_statistic_divergence: Any,
) -> dict[str, Any]:
    """Return the fixed eleven-dimensional So2Sat probe feature document.

    Both matrices must contain the same probe rows and exactly 17 class logits.
    No outcome or class-label argument exists by design.
    """

    frozen = _as_logits(frozen_probe_logits, name="frozen_probe_logits")
    adapted = _as_logits(adapted_probe_logits, name="adapted_probe_logits")
    if frozen.shape != adapted.shape:
        raise IntegrityError(f"frozen/adapted probe logits shape mismatch: {frozen.shape} != {adapted.shape}")

    update_norm = _finite_number(
        normalized_adapter_update_norm,
        field="normalized_adapter_update_norm",
    )
    bn_divergence = _finite_number(
        batchnorm_source_statistic_divergence,
        field="batchnorm_source_statistic_divergence",
    )
    if update_norm < 0.0:
        raise IntegrityError("normalized_adapter_update_norm must be non-negative")
    if bn_divergence < 0.0:
        raise IntegrityError("batchnorm_source_statistic_divergence must be non-negative")

    frozen_probabilities = _probabilities(frozen)
    adapted_probabilities = _probabilities(adapted)
    tiny = np.finfo(np.float64).tiny

    frozen_confidence = frozen_probabilities.max(axis=1)
    adapted_confidence = adapted_probabilities.max(axis=1)
    entropy_denominator = math.log(float(N_CLASSES))
    frozen_entropy = (
        -(frozen_probabilities * np.log(np.maximum(frozen_probabilities, tiny))).sum(axis=1) / entropy_denominator
    )
    adapted_entropy = (
        -(adapted_probabilities * np.log(np.maximum(adapted_probabilities, tiny))).sum(axis=1) / entropy_denominator
    )

    frozen_marginal = frozen_probabilities.mean(axis=0)
    adapted_marginal = adapted_probabilities.mean(axis=0)
    midpoint = 0.5 * (frozen_marginal + adapted_marginal)
    jensen_shannon = 0.5 * np.sum(
        frozen_marginal * (np.log(np.maximum(frozen_marginal, tiny)) - np.log(np.maximum(midpoint, tiny)))
    ) + 0.5 * np.sum(
        adapted_marginal * (np.log(np.maximum(adapted_marginal, tiny)) - np.log(np.maximum(midpoint, tiny)))
    )
    jensen_shannon /= math.log(2.0)

    frozen_predictions = frozen_probabilities.argmax(axis=1)
    adapted_predictions = adapted_probabilities.argmax(axis=1)
    predicted_counts = np.bincount(
        frozen_predictions,
        minlength=N_CLASSES,
    ).astype(np.float64)
    predicted_distribution = predicted_counts / predicted_counts.sum()
    nonzero = predicted_distribution[predicted_distribution > 0.0]
    effective_count = math.exp(float(-np.sum(nonzero * np.log(nonzero)))) / N_CLASSES

    frozen_entropy_mean = _unit_interval(float(frozen_entropy.mean()))
    adapted_entropy_mean = _unit_interval(float(adapted_entropy.mean()))
    frozen_confidence_mean = _unit_interval(float(frozen_confidence.mean()))
    adapted_confidence_mean = _unit_interval(float(adapted_confidence.mean()))
    values = {
        "frozen_mean_entropy": frozen_entropy_mean,
        "adapted_mean_entropy": adapted_entropy_mean,
        "entropy_change": frozen_entropy_mean - adapted_entropy_mean,
        "frozen_mean_confidence": frozen_confidence_mean,
        "adapted_mean_confidence": adapted_confidence_mean,
        "confidence_change": adapted_confidence_mean - frozen_confidence_mean,
        "prediction_disagreement": _unit_interval(float(np.mean(frozen_predictions != adapted_predictions))),
        "marginal_jensen_shannon_divergence": _unit_interval(float(jensen_shannon)),
        "normalized_predicted_class_effective_count": _unit_interval(effective_count),
        "normalized_adapter_update_norm": update_norm,
        "batchnorm_source_statistic_divergence": bn_divergence,
    }
    if not all(math.isfinite(value) for value in values.values()):  # pragma: no cover
        raise IntegrityError("derived label-free feature is non-finite")

    document = _unsigned_feature_document(
        n_probe_images=int(frozen.shape[0]),
        frozen_logits_tensor_sha256=_tensor_sha256(frozen),
        adapted_logits_tensor_sha256=_tensor_sha256(adapted),
        features=values,
    )
    document["feature_sha256"] = stable_sha256(document)
    validate_feature_document(document)
    return document


def validate_feature_document(document: Mapping[str, Any]) -> None:
    """Validate a serialized feature document without accepting extra fields."""

    if not isinstance(document, Mapping) or set(document) != _DOCUMENT_KEYS:
        raise IntegrityError("So2Sat feature document has unknown or missing fields")
    if document.get("schema") != FEATURE_SCHEMA or document.get("status") != FEATURE_STATUS:
        raise IntegrityError("unknown or unsealed So2Sat feature schema")
    n_probe_images = document.get("n_probe_images")
    if isinstance(n_probe_images, bool) or not isinstance(n_probe_images, int) or n_probe_images < 1:
        raise IntegrityError("feature n_probe_images must be a positive integer")
    if document.get("n_classes") != N_CLASSES:
        raise IntegrityError(f"So2Sat probe features require exactly {N_CLASSES} classes")
    if tuple(document.get("feature_names", ())) != FEATURE_NAMES:
        raise IntegrityError("So2Sat feature-name order drift")
    require_sha256(
        document.get("frozen_logits_tensor_sha256"),
        field="frozen_logits_tensor_sha256",
    )
    require_sha256(
        document.get("adapted_logits_tensor_sha256"),
        field="adapted_logits_tensor_sha256",
    )
    features = document.get("features")
    if not isinstance(features, Mapping) or set(features) != set(FEATURE_NAMES):
        raise IntegrityError("feature values must contain exactly the frozen 11-feature schema")
    normalized = {name: _finite_number(features[name], field=f"features.{name}") for name in FEATURE_NAMES}
    for name in _UNIT_INTERVAL_FEATURES:
        if not 0.0 <= normalized[name] <= 1.0:
            raise IntegrityError(f"features.{name} must lie in [0, 1]")
    for name in _SIGNED_UNIT_INTERVAL_FEATURES:
        if not -1.0 <= normalized[name] <= 1.0:
            raise IntegrityError(f"features.{name} must lie in [-1, 1]")
    for name in _NONNEGATIVE_FEATURES:
        if normalized[name] < 0.0:
            raise IntegrityError(f"features.{name} must be non-negative")
    if not math.isclose(
        normalized["entropy_change"],
        normalized["frozen_mean_entropy"] - normalized["adapted_mean_entropy"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise IntegrityError("entropy_change is inconsistent with the two entropy features")
    if not math.isclose(
        normalized["confidence_change"],
        normalized["adapted_mean_confidence"] - normalized["frozen_mean_confidence"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise IntegrityError("confidence_change is inconsistent with the two confidence features")
    claimed = require_sha256(document.get("feature_sha256"), field="feature_sha256")
    unsigned = dict(document)
    unsigned.pop("feature_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("feature_sha256 does not match the feature document")


def feature_vector(document: Mapping[str, Any]) -> np.ndarray:
    """Return one validated feature vector in the sealed order."""

    validate_feature_document(document)
    values = document["features"]
    return np.asarray([float(values[name]) for name in FEATURE_NAMES], dtype=np.float64)
