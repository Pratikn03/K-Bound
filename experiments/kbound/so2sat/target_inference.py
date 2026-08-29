"""Production PyTorch cell executor for label-blind So2Sat target inference.

The executor receives only verified checkpoint metadata and ``PixelSample``
objects produced by the target label firewall.  It has no HDF5 path and no
outcome interface.  Every city/checkpoint call starts from a fresh source
model, adapts on validation/probe pixels only, and returns frozen/adapted logits
for the live runner to seal before offline scoring.
"""

from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adapters import (
    ADAPTER_BATCH_SIZE,
    CANDIDATE_IDS,
    adapt_on_probe,
    candidate_spec,
    fixed_model_logits,
    frozen_logits_and_bn_divergence,
)
from .integrity import IntegrityError, file_sha256, stable_sha256
from .label_firewall import PixelSample
from .model import build_so2sat_resnet18, tensor_state_sha256
from .source_data import BandNormalizer, load_sealed_band_normalizer
from .target_runner import CellComputation, EvaluationComputation, ProbeComputation


@dataclass
class _PreparedTargetCell:
    """Opaque model state retained only until the sealed action permits evaluation."""

    checkpoint_id: str
    city_id: str
    frozen_model: torch.nn.Module
    adapted_model: torch.nn.Module
    consumed: bool = False


def target_inference_code_identity() -> dict[str, Any]:
    """Hash the complete live inference implementation used by the seal."""

    directory = Path(__file__).resolve().parent
    names = (
        "integrity.py",
        "protocol.py",
        "metadata_manifest.py",
        "label_firewall.py",
        "model.py",
        "source_data.py",
        "train_source.py",
        "source_acceptance.py",
        "source_preflight.py",
        "adapters.py",
        "features.py",
        "gate.py",
        "development.py",
        "target_amendment.py",
        "precalibration_seal.py",
        "target_contract.py",
        "target_runner.py",
        "target_inference.py",
        "target_seal.py",
        "target_boundary_amendment_v1_1.json",
        "target_boundary_amendment_v1_1.json.receipt.json",
        "prospective_protocol_v1.json",
        "prospective_protocol_v1.json.receipt.json",
    )
    files = {name: file_sha256(directory / name) for name in names}
    return {"files_sha256": files, "code_identity_sha256": stable_sha256(files)}


def target_runtime_environment_identity(device: torch.device) -> dict[str, Any]:
    """Return the live software/hardware identity that the execution seal binds."""

    if device.type not in {"cpu", "mps"}:
        raise IntegrityError("target runtime identity supports only CPU or MPS")
    try:
        h5py_version = importlib.metadata.version("h5py")
    except importlib.metadata.PackageNotFoundError:
        h5py_version = "NOT_INSTALLED"
    try:
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision_version = "NOT_INSTALLED"
    document = {
        "schema": "kbound_so2sat_live_runtime_environment_v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision_version,
        "h5py_version": h5py_version,
        "device_type": device.type,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    document["environment_identity_sha256"] = stable_sha256(document)
    return document


class TorchTargetCellExecutor:
    """Run the sealed adapter on one city/checkpoint cell without outcomes."""

    def __init__(
        self,
        *,
        candidate_id: str,
        normalizer_path: str | Path,
        device: torch.device,
        batch_size: int = ADAPTER_BATCH_SIZE,
    ) -> None:
        if candidate_id not in CANDIDATE_IDS:
            raise IntegrityError("target executor received an unknown adapter candidate")
        if device.type not in {"cpu", "mps"}:
            raise IntegrityError("target executor supports only CPU or MPS")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise IntegrityError("target executor batch size must be a positive integer")
        if batch_size != ADAPTER_BATCH_SIZE:
            raise IntegrityError(
                f"target executor batch size is sealed at {ADAPTER_BATCH_SIZE}"
            )
        self.candidate_id = candidate_id
        self.candidate_spec = candidate_spec(candidate_id)
        self.normalizer: BandNormalizer = load_sealed_band_normalizer(normalizer_path)
        self.normalizer_sha256 = self.normalizer.normalizer_sha256
        self.device = device
        self.batch_size = batch_size
        self.code_identity = target_inference_code_identity()
        self.code_identity_sha256 = self.code_identity["code_identity_sha256"]
        self.environment_identity = target_runtime_environment_identity(device)
        self.environment_identity_sha256 = self.environment_identity[
            "environment_identity_sha256"
        ]
        self._checkpoint_payloads: dict[str, Mapping[str, Any]] = {}

    def _load_source_model(self, checkpoint: Mapping[str, Any]) -> torch.nn.Module:
        checkpoint_id = checkpoint.get("checkpoint_id")
        path_value = checkpoint.get("checkpoint_path")
        if checkpoint_id not in {"0", "1", "2", "3", "4"} or not isinstance(
            path_value, str
        ):
            raise IntegrityError("target executor checkpoint identity is invalid")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file() or path.name != checkpoint.get("checkpoint_basename"):
            raise IntegrityError("target executor checkpoint path is missing or misnamed")
        if file_sha256(path) != checkpoint.get("checkpoint_file_sha256"):
            raise IntegrityError("target executor checkpoint byte hash changed")
        payload = self._checkpoint_payloads.get(checkpoint_id)
        if payload is None:
            try:
                loaded = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise IntegrityError(f"cannot safely load source checkpoint {path}: {exc}") from exc
            if not isinstance(loaded, Mapping):
                raise IntegrityError("target executor checkpoint payload must be a mapping")
            payload = dict(loaded)
            self._checkpoint_payloads[checkpoint_id] = payload
        state = payload.get("model_state")
        if not isinstance(state, Mapping):
            raise IntegrityError("target executor checkpoint lacks model state")
        if (
            payload.get("schema") != "kbound_so2sat_source_checkpoint_v1"
            or payload.get("model_seed") != int(checkpoint_id)
            or payload.get("checkpoint_tensor_sha256")
            != checkpoint.get("checkpoint_tensor_sha256")
            or tensor_state_sha256(state) != checkpoint.get("checkpoint_tensor_sha256")
            or payload.get("normalizer_sha256") != self.normalizer_sha256
            or payload.get("target_data_inputs") != []
        ):
            raise IntegrityError("target executor checkpoint tensor/provenance mismatch")
        model = build_so2sat_resnet18()
        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _batches(self, samples: Sequence[PixelSample]) -> Iterable[torch.Tensor]:
        if not samples:
            raise IntegrityError("target executor received an empty city partition")
        for start in range(0, len(samples), self.batch_size):
            batch_samples = samples[start : start + self.batch_size]
            try:
                values = np.stack(
                    [np.asarray(sample.pixels, dtype=np.float32) for sample in batch_samples]
                )
            except (TypeError, ValueError, MemoryError) as exc:
                raise IntegrityError("target executor could not stack firewall pixels") from exc
            if values.shape != (len(batch_samples), 32, 32, 10):
                raise IntegrityError(
                    f"target executor expected N x 32 x 32 x 10 pixels, found {values.shape}"
                )
            if not np.isfinite(values).all():
                raise IntegrityError("target executor pixels contain NaN or Infinity")
            tensor = torch.from_numpy(values).permute(0, 3, 1, 2).contiguous()
            mean = torch.tensor(self.normalizer.mean, dtype=torch.float32)[None, :, None, None]
            std = torch.tensor(self.normalizer.std, dtype=torch.float32)[None, :, None, None]
            yield (tensor - mean) / std

    def prepare_probe(
        self,
        checkpoint: Mapping[str, Any],
        selected_spec: Mapping[str, Any],
        probe_samples: Sequence[PixelSample],
    ) -> ProbeComputation:
        if dict(selected_spec) != self.candidate_spec:
            raise IntegrityError("target executor candidate specification differs from the selection")
        probe_city = {sample.metadata.city_id for sample in probe_samples}
        if (
            len(probe_city) != 1
            or any(sample.metadata.sample_role != "target_probe" for sample in probe_samples)
        ):
            raise IntegrityError("target executor probe samples violate the city/role boundary")

        frozen_model = self._load_source_model(checkpoint)
        frozen_probe, batchnorm_divergence = frozen_logits_and_bn_divergence(
            frozen_model,
            self._batches(probe_samples),
            device=self.device,
        )
        adaptation_source = self._load_source_model(checkpoint)
        adapted_model, diagnostics = adapt_on_probe(
            adaptation_source,
            self._batches(probe_samples),
            candidate_id=self.candidate_id,
            device=self.device,
            batchnorm_source_statistic_divergence=batchnorm_divergence,
        )
        adapted_probe = fixed_model_logits(
            adapted_model,
            self._batches(probe_samples),
            device=self.device,
        )
        expected_probe = len(probe_samples)
        if (
            frozen_probe.shape != (expected_probe, 17)
            or adapted_probe.shape != (expected_probe, 17)
        ):
            raise IntegrityError("target executor probe-logit coverage is incomplete")
        state = _PreparedTargetCell(
            checkpoint_id=str(checkpoint["checkpoint_id"]),
            city_id=next(iter(probe_city)),
            frozen_model=frozen_model,
            adapted_model=adapted_model,
        )
        return ProbeComputation(
            frozen_probe_logits=frozen_probe.numpy(),
            adapted_probe_logits=adapted_probe.numpy(),
            normalized_adapter_update_norm=diagnostics.normalized_adapter_update_norm,
            batchnorm_source_statistic_divergence=(
                diagnostics.batchnorm_source_statistic_divergence
            ),
            opaque_evaluation_state=state,
        )

    def evaluate_after_action(
        self,
        probe_computation: ProbeComputation,
        evaluation_samples: Sequence[PixelSample],
    ) -> EvaluationComputation:
        """Evaluate both fixed policies only after the runner seals the action."""

        state = probe_computation.opaque_evaluation_state
        evaluation_city = {sample.metadata.city_id for sample in evaluation_samples}
        if (
            type(state) is not _PreparedTargetCell
            or state.consumed
            or evaluation_city != {state.city_id}
            or any(
                sample.metadata.sample_role != "target_evaluation"
                for sample in evaluation_samples
            )
        ):
            raise IntegrityError("target executor evaluation state/city/role is invalid")
        state.consumed = True
        frozen_evaluation = fixed_model_logits(
            state.frozen_model,
            self._batches(evaluation_samples),
            device=self.device,
        )
        adapted_evaluation = fixed_model_logits(
            state.adapted_model,
            self._batches(evaluation_samples),
            device=self.device,
        )
        expected_evaluation = len(evaluation_samples)
        if (
            frozen_evaluation.shape != (expected_evaluation, 17)
            or adapted_evaluation.shape != (expected_evaluation, 17)
        ):
            raise IntegrityError("target executor evaluation-logit coverage is incomplete")
        result = EvaluationComputation(
            frozen_evaluation_logits=frozen_evaluation.numpy(),
            adapted_evaluation_logits=adapted_evaluation.numpy(),
        )
        del state.frozen_model, state.adapted_model
        if self.device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        return result

    def __call__(
        self,
        checkpoint: Mapping[str, Any],
        selected_spec: Mapping[str, Any],
        probe_samples: Sequence[PixelSample],
        evaluation_samples: Sequence[PixelSample],
    ) -> CellComputation:
        """Replay convenience; production runner uses the two-phase methods above."""

        probe = self.prepare_probe(checkpoint, selected_spec, probe_samples)
        evaluation = self.evaluate_after_action(probe, evaluation_samples)
        return CellComputation(
            frozen_probe_logits=probe.frozen_probe_logits,
            adapted_probe_logits=probe.adapted_probe_logits,
            frozen_evaluation_logits=evaluation.frozen_evaluation_logits,
            adapted_evaluation_logits=evaluation.adapted_evaluation_logits,
            normalized_adapter_update_norm=probe.normalized_adapter_update_norm,
            batchnorm_source_statistic_divergence=(
                probe.batchnorm_source_statistic_divergence
            ),
        )
