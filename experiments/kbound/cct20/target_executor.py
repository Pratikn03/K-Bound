"""State machine for one label-free CCT-20 target checkpoint/location cell.

The state machine makes the protocol order executable: hash-split complete
sequences, process probe batches first, seal the gate action from probe-only
signals, and only then process evaluation images.  It never accepts a label or
outcome field and does not import the post-seal scorer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .integrity import IntegrityError, require_sha256, stable_sha256, strict_json_load
from .label_free_traces import (
    PARTITION_SALT,
    PROBE_FRACTION,
    TARGET_BATCH_SIZE,
    assert_label_free,
    extract_label_free_features,
    sequence_atomic_batches,
    sequence_atomic_partition,
)
from .prediction_artifacts import build_prediction_cell
from .protocol_seal import (
    EXPECTED_MODEL_SEEDS,
    EXPECTED_TARGET_LOCATIONS,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from .ridge_gate import apply_gate, validate_gate_document
from .tent_official import (
    BN_GAUSSIAN_KL_NUMERIC_CLIPPING,
    BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION,
    BN_GAUSSIAN_KL_SCHEMA,
    BN_GAUSSIAN_KL_TAYLOR_TERMS,
    BN_GAUSSIAN_KL_TAYLOR_THRESHOLD,
    LOCKED_BACKEND_STRATEGY,
    OFFICIAL_TENT_COMMIT,
    OFFICIAL_TENT_FILE_SHA256,
    OFFICIAL_TENT_TREE,
)


def _logits(value: Any, *, expected_rows: int, field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{field} must be a finite numeric matrix") from exc
    if result.shape != (expected_rows, 16) or not np.isfinite(result).all():
        raise IntegrityError(f"{field} must be finite with shape ({expected_rows}, 16), found {result.shape}")
    return result


class LabelFreeTargetCell:
    """Fail-closed builder for one of the 45 target execution cells."""

    def __init__(
        self,
        metadata_rows: Iterable[Mapping[str, Any]],
        *,
        checkpoint_seed: int,
        checkpoint_tensor_sha256: str,
        location_id: str | int,
        protocol_seal_sha256: str,
        target_manifest_sha256: str,
        gate_document: Mapping[str, Any],
        tent_binding_receipt: Mapping[str, Any],
    ) -> None:
        rows = [dict(row) for row in metadata_rows]
        assert_label_free(rows, path="metadata_rows")
        validate_gate_document(gate_document)
        protocol_hash = require_sha256(protocol_seal_sha256, field="protocol_seal_sha256")
        manifest_hash = require_sha256(target_manifest_sha256, field="target_manifest_sha256")
        checkpoint_hash = require_sha256(checkpoint_tensor_sha256, field="checkpoint_tensor_sha256")
        if checkpoint_seed not in EXPECTED_MODEL_SEEDS:
            raise IntegrityError(f"checkpoint_seed must be one of {EXPECTED_MODEL_SEEDS}")
        location = str(location_id)
        if location not in EXPECTED_TARGET_LOCATIONS:
            raise IntegrityError(f"location_id {location!r} is not a sealed target location")
        if {str(row.get("location_id", row.get("location", ""))) for row in rows} != {location}:
            raise IntegrityError("target cell metadata must belong to exactly its declared location")
        if tent_binding_receipt.get("schema") != "kbound_cct20_official_tent_binding_v1":
            raise IntegrityError("Tent binding receipt has an unknown schema")
        if tent_binding_receipt.get("checkpoint_tensor_sha256") != checkpoint_hash:
            raise IntegrityError("Tent binding/checkpoint tensor identity mismatch")
        if str(tent_binding_receipt.get("location_id")) != location:
            raise IntegrityError("Tent binding/location identity mismatch")
        parameter_names = tent_binding_receipt.get("parameter_names")
        if (
            not isinstance(parameter_names, list)
            or not parameter_names
            or any(not isinstance(name, str) or not name for name in parameter_names)
            or len(set(parameter_names)) != len(parameter_names)
            or tent_binding_receipt.get("n_parameters") != len(parameter_names)
            or tent_binding_receipt.get("reset_scope") != f"{checkpoint_hash}:{location}"
            or tent_binding_receipt.get("update_norm_formula")
            != "l2(after_probe-before_probe)/max(l2(before_probe),1e-12)"
        ):
            raise IntegrityError("Tent binding parameter/reset receipt is invalid")
        try:
            initial_norm = float(tent_binding_receipt.get("initial_bn_affine_l2"))
        except (TypeError, ValueError) as exc:
            raise IntegrityError("Tent binding initial BN-affine norm is invalid") from exc
        if not np.isfinite(initial_norm) or initial_norm < 0.0:
            raise IntegrityError("Tent binding initial BN-affine norm is invalid")
        backend = tent_binding_receipt.get("backend_installation")
        expected_backend_fields = {
            "schema",
            "strategy",
            "fallback_layer",
            "source_module_class",
            "fallback_module_class",
            "fallback_input_device",
            "fallback_compute_device",
            "fallback_parameter_device",
            "fallback_output_device",
            "num_features",
            "eps",
            "momentum",
            "affine",
            "preconfigure_track_running_stats",
            "source_bn_state_sha256",
            "installed_bn_state_sha256",
            "state_hash_equal",
            "official_tent_parameter_devices",
            "configured_track_running_stats",
            "configured_running_moments_absent",
        }
        if (
            not isinstance(backend, Mapping)
            or set(backend) != expected_backend_fields
            or backend.get("schema") != "kbound_cct20_backend_installation_v1"
            or backend.get("strategy") != LOCKED_BACKEND_STRATEGY
            or backend.get("fallback_layer") != "bn1"
            or backend.get("source_module_class") != "torch.nn.BatchNorm2d"
            or backend.get("fallback_module_class") != "KBoundCPUFallbackBatchNorm2d"
            or backend.get("fallback_input_device") != "mps"
            or backend.get("fallback_compute_device") != "cpu"
            or backend.get("fallback_parameter_device") != "cpu"
            or backend.get("fallback_output_device") != "mps"
            or backend.get("num_features") != 64
            or backend.get("eps") != 1.0e-5
            or backend.get("momentum") != 0.1
            or backend.get("affine") is not True
            or backend.get("preconfigure_track_running_stats") is not True
            or backend.get("state_hash_equal") is not True
            or backend.get("official_tent_parameter_devices") != ["cpu", "mps"]
            or backend.get("configured_track_running_stats") is not False
            or backend.get("configured_running_moments_absent") is not True
        ):
            raise IntegrityError("Tent binding hybrid-backend receipt is invalid")
        source_bn_hash = require_sha256(backend.get("source_bn_state_sha256"), field="source_bn_state_sha256")
        installed_bn_hash = require_sha256(backend.get("installed_bn_state_sha256"), field="installed_bn_state_sha256")
        if source_bn_hash != installed_bn_hash:
            raise IntegrityError("Tent binding hybrid-backend state hashes differ")
        provenance = tent_binding_receipt.get("provenance", {})
        if (
            provenance.get("git_commit") != OFFICIAL_TENT_COMMIT
            or provenance.get("git_tree") != OFFICIAL_TENT_TREE
            or provenance.get("tent_py_sha256") != OFFICIAL_TENT_FILE_SHA256
            or provenance.get("tracked_worktree_clean") is not True
            or provenance.get("configure_function") != "tent.configure_model"
            or provenance.get("parameter_function") != "tent.collect_params"
            or provenance.get("adapter_class") != "tent.Tent"
        ):
            raise IntegrityError("target cell is not bound to the pinned official Tent commit")
        if provenance.get("reset_scope") != "source_checkpoint_x_camera_location":
            raise IntegrityError("Tent binding reset boundary is not checkpoint x location")
        if not (
            provenance.get("optimizer", {}).get("class") == "torch.optim.Adam"
            and provenance.get("optimizer", {}).get("lr") == 0.001
            and provenance.get("optimizer", {}).get("betas") == [0.9, 0.999]
            and provenance.get("optimizer", {}).get("weight_decay") == 0.0
            and provenance.get("steps") == 1
            and provenance.get("episodic") is False
        ):
            raise IntegrityError("Tent binding optimizer/update settings drift from the seal")

        partition = sequence_atomic_partition(
            rows,
            probe_fraction=PROBE_FRACTION,
            salt=PARTITION_SALT,
        )
        self._partition_sha256 = stable_sha256(partition)
        self._batches = {
            role: sequence_atomic_batches(
                partition["roles"][role],
                max_images=TARGET_BATCH_SIZE,
                order="native",
                merge_singleton_final=True,
            )
            for role in ("probe", "evaluation")
        }
        if any(len(batch) == 1 for batches in self._batches.values() for batch in batches):
            raise IntegrityError("a singleton target batch could not be merged; experiment fails closed")
        self.checkpoint_seed = checkpoint_seed
        self.checkpoint_tensor_sha256 = checkpoint_hash
        self.location_id = location
        self.protocol_seal_sha256 = protocol_hash
        self.target_manifest_sha256 = manifest_hash
        self.gate_document = dict(gate_document)
        self._tent_parameter_names = tuple(parameter_names)
        self._recorded: dict[str, list[dict[str, Any]]] = {"probe": [], "evaluation": []}
        self._gate_result: dict[str, Any] | None = None
        self._action_receipt: dict[str, Any] | None = None

    def batch_plan(self, role: str) -> list[list[dict[str, Any]]]:
        if role not in self._batches:
            raise IntegrityError("target role must be probe or evaluation")
        return [[dict(row) for row in batch] for batch in self._batches[role]]

    def record_batch(
        self,
        *,
        role: str,
        batch_index: int,
        image_ids: Iterable[str | int],
        frozen_logits: Any,
        tent_logits: Any,
    ) -> None:
        if role not in {"probe", "evaluation"}:
            raise IntegrityError("target role must be probe or evaluation")
        if role == "evaluation" and self._action_receipt is None:
            raise IntegrityError("evaluation cannot run until the probe-only gate action is sealed")
        expected_index = len(self._recorded[role])
        if batch_index != expected_index or batch_index >= len(self._batches[role]):
            raise IntegrityError(
                f"{role} batch must be recorded in order; expected {expected_index}, found {batch_index}"
            )
        batch = self._batches[role][batch_index]
        observed_ids = [str(value) for value in image_ids]
        expected_ids = [row["image_id"] for row in batch]
        if observed_ids != expected_ids:
            raise IntegrityError(f"{role} batch image order does not match the sealed batch plan")
        frozen = _logits(frozen_logits, expected_rows=len(batch), field="frozen_logits")
        adapted = _logits(tent_logits, expected_rows=len(batch), field="tent_logits")
        self._recorded[role].append(
            {
                "metadata": batch,
                "frozen_logits": frozen,
                "tent_logits": adapted,
            }
        )

    def seal_probe_action(
        self,
        *,
        tent_update_receipt: Mapping[str, Any],
        frozen_bn_probe_moment_receipt: Mapping[str, Any],
        action_output_path: str | Path,
        restore_existing: bool = False,
    ) -> dict[str, Any]:
        if self._gate_result is not None:
            raise IntegrityError("probe gate action is already sealed")
        if len(self._recorded["probe"]) != len(self._batches["probe"]):
            raise IntegrityError("all probe batches must be recorded before sealing the action")
        frozen = np.vstack([batch["frozen_logits"] for batch in self._recorded["probe"]])
        adapted = np.vstack([batch["tent_logits"] for batch in self._recorded["probe"]])
        if not (
            tent_update_receipt.get("schema") == "kbound_cct20_tent_probe_update_v1"
            and tent_update_receipt.get("checkpoint_tensor_sha256") == self.checkpoint_tensor_sha256
            and str(tent_update_receipt.get("location_id")) == self.location_id
            and tent_update_receipt.get("reset_scope") == f"{self.checkpoint_tensor_sha256}:{self.location_id}"
            and tent_update_receipt.get("parameter_names") == list(self._tent_parameter_names)
            and tent_update_receipt.get("formula") == "l2(after_probe-before_probe)/max(l2(before_probe),1e-12)"
        ):
            raise IntegrityError("Tent probe-update receipt does not match this execution cell")
        if not (
            frozen_bn_probe_moment_receipt.get("schema") == BN_GAUSSIAN_KL_SCHEMA
            and frozen_bn_probe_moment_receipt.get("formula") == "channel_weighted_mean_gaussian_kl_probe_to_source"
            and frozen_bn_probe_moment_receipt.get("numeric_implementation") == BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION
            and frozen_bn_probe_moment_receipt.get("taylor_threshold") == BN_GAUSSIAN_KL_TAYLOR_THRESHOLD
            and frozen_bn_probe_moment_receipt.get("taylor_terms") == BN_GAUSSIAN_KL_TAYLOR_TERMS
            and frozen_bn_probe_moment_receipt.get("numeric_clipping") == BN_GAUSSIAN_KL_NUMERIC_CLIPPING
            and int(frozen_bn_probe_moment_receipt.get("channel_count", 0)) > 0
        ):
            raise IntegrityError("frozen BN probe-moment receipt is invalid")
        layers = frozen_bn_probe_moment_receipt.get("layers")
        if not isinstance(layers, list) or not layers:
            raise IntegrityError("frozen BN probe-moment receipt has no layer audit")
        channel_count = 0
        weighted_kl = 0.0
        taylor_branch_channels = 0
        minimum_channel_kl = np.inf
        layer_names: set[str] = set()
        for layer in layers:
            if not isinstance(layer, Mapping):
                raise IntegrityError("frozen BN layer receipt is not a mapping")
            channels = layer.get("channels")
            layer_name = layer.get("layer")
            values_per_channel = layer.get("values_per_channel")
            layer_taylor_channels = layer.get("taylor_branch_channels")
            mean_kl = layer.get("mean_kl")
            min_kl = layer.get("min_kl")
            bn_eps = layer.get("bn_eps")
            if (
                isinstance(channels, bool)
                or not isinstance(layer_name, str)
                or not layer_name
                or layer_name in layer_names
                or not isinstance(channels, int)
                or channels < 1
                or isinstance(values_per_channel, bool)
                or not isinstance(values_per_channel, int)
                or values_per_channel < 1
                or isinstance(layer_taylor_channels, bool)
                or not isinstance(layer_taylor_channels, int)
                or layer_taylor_channels < 0
                or layer_taylor_channels > channels
            ):
                raise IntegrityError("frozen BN layer receipt has invalid counts")
            layer_names.add(layer_name)
            try:
                mean_kl_value = float(mean_kl)
                min_kl_value = float(min_kl)
                eps_value = float(bn_eps)
            except (TypeError, ValueError) as exc:
                raise IntegrityError("frozen BN layer receipt has invalid numeric values") from exc
            if (
                not np.isfinite(mean_kl_value)
                or mean_kl_value < 0.0
                or not np.isfinite(min_kl_value)
                or min_kl_value < 0.0
                or min_kl_value > mean_kl_value
                or not np.isfinite(eps_value)
                or eps_value <= 0.0
            ):
                raise IntegrityError("frozen BN layer receipt has invalid KL/epsilon")
            channel_count += channels
            weighted_kl += channels * mean_kl_value
            taylor_branch_channels += layer_taylor_channels
            minimum_channel_kl = min(minimum_channel_kl, min_kl_value)
        expected_layer_names: dict[str, set[str]] = {}
        for parameter_name in self._tent_parameter_names:
            layer_name, separator, suffix = parameter_name.rpartition(".")
            if not separator or suffix not in {"weight", "bias"}:
                raise IntegrityError("Tent binding contains an invalid BN-affine parameter name")
            expected_layer_names.setdefault(layer_name, set()).add(suffix)
        if any(suffixes != {"weight", "bias"} for suffixes in expected_layer_names.values()) or layer_names != set(
            expected_layer_names
        ):
            raise IntegrityError("frozen BN layer receipt does not match Tent BN-affine layers")
        declared_channels = int(frozen_bn_probe_moment_receipt["channel_count"])
        declared_divergence = float(frozen_bn_probe_moment_receipt.get("batchnorm_batch_source_statistic_divergence"))
        declared_taylor_channels = int(frozen_bn_probe_moment_receipt.get("taylor_branch_channels"))
        declared_minimum_kl = float(frozen_bn_probe_moment_receipt.get("minimum_channel_kl"))
        if (
            channel_count != declared_channels
            or taylor_branch_channels != declared_taylor_channels
            or minimum_channel_kl != declared_minimum_kl
            or not np.isclose(
                weighted_kl / channel_count,
                declared_divergence,
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise IntegrityError("frozen BN aggregate divergence does not reconcile to layers")
        feature_record = extract_label_free_features(
            frozen,
            adapted,
            normalized_tent_update_norm=tent_update_receipt.get("normalized_tent_update_norm"),
            batchnorm_batch_source_statistic_divergence=(
                frozen_bn_probe_moment_receipt.get("batchnorm_batch_source_statistic_divergence")
            ),
        )
        feature_record["diagnostic_receipts"] = {
            "tent_update": dict(tent_update_receipt),
            "frozen_bn_probe_moments": dict(frozen_bn_probe_moment_receipt),
        }
        gate_result = apply_gate(self.gate_document, feature_record["features"])
        gate_result["probe_feature_record"] = feature_record
        probe_trace = []
        for batch in self._recorded["probe"]:
            for metadata, frozen_logits, tent_logits in zip(
                batch["metadata"],
                batch["frozen_logits"],
                batch["tent_logits"],
                strict=True,
            ):
                probe_trace.append(
                    {
                        "image_id": metadata["image_id"],
                        "sequence_id": metadata["sequence_id"],
                        "location_id": metadata["location_id"],
                        "frozen_logits": [float(value) for value in frozen_logits],
                        "adapted_logits": [float(value) for value in tent_logits],
                    }
                )
        gate_result["partition_sha256"] = self._partition_sha256
        gate_result["probe_trace_sha256"] = stable_sha256(probe_trace)
        action_document = {
            "schema": "kbound_cct20_label_free_action_v1",
            "status": "SEALED_BEFORE_EVALUATION_STREAM",
            "protocol_seal_sha256": self.protocol_seal_sha256,
            "gate_sha256": self.gate_document["gate_sha256"],
            "target_manifest_sha256": self.target_manifest_sha256,
            "partition_sha256": self._partition_sha256,
            "probe_trace_sha256": gate_result["probe_trace_sha256"],
            "checkpoint_seed": self.checkpoint_seed,
            "checkpoint_tensor_sha256": self.checkpoint_tensor_sha256,
            "location_id": self.location_id,
            "gate_result": gate_result,
        }
        assert_label_free(action_document, path="action_document")
        action_document["action_sha256"] = stable_sha256(action_document)
        if restore_existing:
            receipt = verify_artifact_receipt(action_output_path)
            existing = strict_json_load(action_output_path)
            if existing != action_document:
                raise IntegrityError("replayed probe evidence/action differs from the existing immutable action")
        else:
            receipt = write_immutable_json_with_receipt(action_output_path, action_document)
        gate_result["action_sha256"] = action_document["action_sha256"]
        gate_result["action_artifact_sha256"] = receipt["artifact_sha256"]
        gate_result["action_receipt"] = receipt
        self._gate_result = gate_result
        self._action_receipt = receipt
        assert_label_free(self._gate_result, path="gate_result")
        return dict(self._gate_result)

    def restore_sealed_probe_action(
        self,
        *,
        tent_update_receipt: Mapping[str, Any],
        frozen_bn_probe_moment_receipt: Mapping[str, Any],
        action_output_path: str | Path,
    ) -> dict[str, Any]:
        """Replay probe state and resume only if it exactly matches an existing action."""

        return self.seal_probe_action(
            tent_update_receipt=tent_update_receipt,
            frozen_bn_probe_moment_receipt=frozen_bn_probe_moment_receipt,
            action_output_path=action_output_path,
            restore_existing=True,
        )

    def finalize(self) -> dict[str, Any]:
        if self._gate_result is None or self._action_receipt is None:
            raise IntegrityError("cannot finalize before sealing the probe action")
        if len(self._recorded["evaluation"]) != len(self._batches["evaluation"]):
            raise IntegrityError("all evaluation batches must be recorded before finalization")
        rows = []
        for role in ("probe", "evaluation"):
            for batch in self._recorded[role]:
                for metadata, frozen, adapted in zip(
                    batch["metadata"],
                    batch["frozen_logits"],
                    batch["tent_logits"],
                    strict=True,
                ):
                    rows.append(
                        {
                            "stream_index": len(rows),
                            "image_id": metadata["image_id"],
                            "sequence_id": metadata["sequence_id"],
                            "location_id": self.location_id,
                            "role": role,
                            "frozen_prediction": int(np.argmax(frozen)),
                            "adapted_prediction": int(np.argmax(adapted)),
                            "frozen_logits": [float(value) for value in frozen],
                            "adapted_logits": [float(value) for value in adapted],
                        }
                    )
        return build_prediction_cell(
            protocol_seal_sha256=self.protocol_seal_sha256,
            gate_sha256=self.gate_document["gate_sha256"],
            target_manifest_sha256=self.target_manifest_sha256,
            checkpoint_seed=self.checkpoint_seed,
            checkpoint_tensor_sha256=self.checkpoint_tensor_sha256,
            location_id=self.location_id,
            gate_result=self._gate_result,
            rows=rows,
        )
