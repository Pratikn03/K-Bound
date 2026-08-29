"""Frozen, probe-only test-time adapters for the So2Sat confirmation.

The two candidates in this module are fixed before any target pixel access.
Both start from a fresh source checkpoint for every city/checkpoint cell, use
only the west/probe images for gradient updates, and are converted to a fixed
evaluation model before any east/evaluation image is passed through them.

The implementation follows the official Tent and SAR repositories pinned in
``CANDIDATE_SPECS``.  SAR adds one explicit numerical guard: an empty reliable
sample set produces a no-op update rather than a NaN backward pass.  This guard
is declared in the candidate specification and counted in every trace.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .integrity import IntegrityError, file_sha256, stable_sha256
from .model import clone_cpu_state

ADAPTER_BATCH_SIZE = 128
ADAPTER_STEPS_PER_BATCH = 1
TENT_LEARNING_RATE = 1.0e-3
SAR_LEARNING_RATE = 2.5e-4
SAR_MOMENTUM = 0.9
SAR_RHO = 0.05
SAR_ENTROPY_MARGIN = 0.4 * math.log(17.0)
SAR_EMA_DECAY = 0.9
SAR_RESET_CONSTANT = 0.2

TENT_OFFICIAL_COMMIT = "e9e926a668d85244c66a6d5c006efbd2b82e83e8"
SAR_OFFICIAL_COMMIT = "20f6e24b17525f34503510afccedc0629b67b7c4"
TENT_OFFICIAL_FILE_SHA256 = "d854e1f65f741aa1632dfb55420ca1bcf5405c978bef890e45caad04980514c8"
SAR_OFFICIAL_FILE_SHA256 = "0553a395ac2bc087049720f1f162789079fe5ee25aa6a1cf82b2ea6286d66eff"
SAM_OFFICIAL_FILE_SHA256 = "b0569de29015016996feae257d30be6cc36f80d42d0f51d4a201eb39d2491712"

TENT_CANDIDATE_ID = "tent_adam_bn_affine_probe_transfer_v1"
SAR_CANDIDATE_ID = "sar_sam_bn_affine_probe_transfer_v1"
CANDIDATE_IDS = (TENT_CANDIDATE_ID, SAR_CANDIDATE_ID)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _official_paths(candidate_id: str) -> tuple[tuple[Path, str], ...]:
    root = _repository_root()
    if candidate_id == TENT_CANDIDATE_ID:
        return ((root / "external" / "tent_official" / "tent.py", TENT_OFFICIAL_FILE_SHA256),)
    if candidate_id == SAR_CANDIDATE_ID:
        return (
            (root / "external" / "sar_official" / "sar.py", SAR_OFFICIAL_FILE_SHA256),
            (root / "external" / "sar_official" / "sam.py", SAM_OFFICIAL_FILE_SHA256),
        )
    raise IntegrityError(f"unknown So2Sat adapter candidate {candidate_id!r}")


def verify_official_adapter_sources(candidate_id: str) -> dict[str, str]:
    """Verify and return the exact upstream source-file commitments."""

    verified: dict[str, str] = {}
    for path, expected in _official_paths(candidate_id):
        if not path.is_file():
            raise IntegrityError(f"missing pinned official adapter source: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise IntegrityError(f"official adapter source hash drift for {path}: {observed} != {expected}")
        verified[str(path.relative_to(_repository_root()))] = observed
    return verified


def candidate_spec(candidate_id: str, *, verify_official_sources: bool = True) -> dict[str, Any]:
    """Return the complete, deterministic scientific specification."""

    if candidate_id not in CANDIDATE_IDS:
        raise IntegrityError(f"candidate_id must be one of {CANDIDATE_IDS}")
    upstream_files = (
        verify_official_adapter_sources(candidate_id)
        if verify_official_sources
        else {str(path.relative_to(_repository_root())): expected for path, expected in _official_paths(candidate_id)}
    )
    common: dict[str, Any] = {
        "schema": "kbound_so2sat_adapter_candidate_spec_v1",
        "candidate_id": candidate_id,
        "modality": "sen2_10_band",
        "class_count": 17,
        "reset_scope": "fresh_source_checkpoint_per_city_checkpoint_cell",
        "adaptation_data_role": "development_probe_only",
        "adaptation_labels_read": 0,
        "evaluation_data_role": "development_evaluation_only",
        "probe_order": "ascending_training_h5_row_index",
        "probe_batch_size": ADAPTER_BATCH_SIZE,
        "steps_per_probe_batch": ADAPTER_STEPS_PER_BATCH,
        "parameter_scope": "batchnorm2d_affine_weight_and_bias",
        "adaptation_objective": "mean_softmax_entropy",
        "deployment": {
            "gradient_updates_after_probe": 0,
            "batchnorm_affine": "probe_adapted_and_frozen",
            "batchnorm_running_statistics": "restore_source_checkpoint_buffers",
            "mode": "eval",
            "evaluation_batch_statistics_used": False,
        },
        "upstream_source_files_sha256": upstream_files,
        "target_inputs": [],
        "target_tuning": False,
    }
    if candidate_id == TENT_CANDIDATE_ID:
        common.update(
            {
                "family": "Tent",
                "official_repository": "https://github.com/DequanWang/tent",
                "official_commit": TENT_OFFICIAL_COMMIT,
                "optimizer": {
                    "name": "Adam",
                    "learning_rate": TENT_LEARNING_RATE,
                    "betas": [0.9, 0.999],
                    "eps": 1.0e-8,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                },
                "layer_exclusions": [],
                "numerical_guard": "finite_entropy_loss_required",
            }
        )
    else:
        common.update(
            {
                "family": "SAR",
                "official_repository": "https://github.com/mr-eggplant/SAR",
                "official_commit": SAR_OFFICIAL_COMMIT,
                "optimizer": {
                    "name": "SAM",
                    "base_optimizer": "SGD",
                    "learning_rate": SAR_LEARNING_RATE,
                    "momentum": SAR_MOMENTUM,
                    "weight_decay": 0.0,
                    "rho": SAR_RHO,
                    "adaptive": False,
                },
                "entropy_margin": SAR_ENTROPY_MARGIN,
                "layer_exclusions": ["layer4"],
                "model_recovery": {
                    "entropy_ema_decay": SAR_EMA_DECAY,
                    "reset_when_ema_below": SAR_RESET_CONSTANT,
                    "reset_state": "source_checkpoint_affine_and_optimizer",
                },
                "numerical_guard": "empty_reliable_subset_is_declared_no_op",
            }
        )
    common["candidate_config_sha256"] = stable_sha256(common)
    return common


def validate_candidate_spec(document: Mapping[str, Any]) -> None:
    candidate_id = document.get("candidate_id")
    if candidate_id not in CANDIDATE_IDS:
        raise IntegrityError("unknown So2Sat adapter candidate specification")
    expected = candidate_spec(str(candidate_id), verify_official_sources=False)
    if dict(document) != expected:
        raise IntegrityError("So2Sat adapter candidate specification drift")
    verify_official_adapter_sources(str(candidate_id))


def all_candidate_specs() -> tuple[dict[str, Any], ...]:
    return tuple(candidate_spec(candidate_id) for candidate_id in CANDIDATE_IDS)


@dataclass(frozen=True)
class AdaptationDiagnostics:
    candidate_id: str
    selected_parameter_names: tuple[str, ...]
    probe_batches: int
    optimizer_updates: int
    reliable_examples: int
    skipped_empty_reliable_batches: int
    model_recovery_resets: int
    normalized_adapter_update_norm: float
    batchnorm_source_statistic_divergence: float

    def __post_init__(self) -> None:
        if self.candidate_id not in CANDIDATE_IDS:
            raise IntegrityError("adaptation diagnostics have an unknown candidate")
        for field in (
            "probe_batches",
            "optimizer_updates",
            "reliable_examples",
            "skipped_empty_reliable_batches",
            "model_recovery_resets",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise IntegrityError(f"adaptation diagnostics {field} must be non-negative")
        if self.probe_batches < 1:
            raise IntegrityError("adaptation requires at least one probe batch")
        if not self.selected_parameter_names:
            raise IntegrityError("adapter selected no BatchNorm affine parameters")
        if len(set(self.selected_parameter_names)) != len(self.selected_parameter_names):
            raise IntegrityError("adapter parameter names are not unique")
        for field in (
            "normalized_adapter_update_norm",
            "batchnorm_source_statistic_divergence",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0.0:
                raise IntegrityError(f"adaptation diagnostic {field} must be finite and non-negative")

    def document(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "selected_parameter_names": list(self.selected_parameter_names),
            "probe_batches": self.probe_batches,
            "optimizer_updates": self.optimizer_updates,
            "reliable_examples": self.reliable_examples,
            "skipped_empty_reliable_batches": self.skipped_empty_reliable_batches,
            "model_recovery_resets": self.model_recovery_resets,
            "normalized_adapter_update_norm": self.normalized_adapter_update_norm,
            "batchnorm_source_statistic_divergence": self.batchnorm_source_statistic_divergence,
        }


def validate_adaptation_diagnostics(document: Mapping[str, Any]) -> AdaptationDiagnostics:
    """Reconstruct a strict diagnostics object from a serialized trace."""

    expected_keys = {
        "candidate_id",
        "selected_parameter_names",
        "probe_batches",
        "optimizer_updates",
        "reliable_examples",
        "skipped_empty_reliable_batches",
        "model_recovery_resets",
        "normalized_adapter_update_norm",
        "batchnorm_source_statistic_divergence",
    }
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise IntegrityError("adapter diagnostics have unknown or missing fields")
    names = document.get("selected_parameter_names")
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise IntegrityError("adapter diagnostics parameter names must be nonempty strings")
    return AdaptationDiagnostics(
        candidate_id=str(document.get("candidate_id")),
        selected_parameter_names=tuple(names),
        probe_batches=document.get("probe_batches"),
        optimizer_updates=document.get("optimizer_updates"),
        reliable_examples=document.get("reliable_examples"),
        skipped_empty_reliable_batches=document.get("skipped_empty_reliable_batches"),
        model_recovery_resets=document.get("model_recovery_resets"),
        normalized_adapter_update_norm=document.get("normalized_adapter_update_norm"),
        batchnorm_source_statistic_divergence=document.get("batchnorm_source_statistic_divergence"),
    )


class _SAM(torch.optim.Optimizer):
    """Minimal fixed SAM implementation matching the pinned SAR helper."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        momentum: float,
        rho: float,
    ) -> None:
        parameter_list = list(params)
        if not parameter_list:
            raise IntegrityError("SAR cannot construct SAM without parameters")
        defaults = {"rho": rho, "adaptive": False, "lr": lr, "momentum": momentum}
        super().__init__(parameter_list, defaults)
        self.base_optimizer = torch.optim.SGD(
            self.param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=0.0,
        )
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, *, zero_grad: bool) -> None:
        terms = [
            parameter.grad.norm(p=2)
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if not terms:
            raise IntegrityError("SAR first step found no gradients")
        shared_device = terms[0].device
        norm = torch.norm(torch.stack([term.to(shared_device) for term in terms]), p=2)
        if not torch.isfinite(norm):
            raise IntegrityError("SAR gradient norm is non-finite")
        for group in self.param_groups:
            scale = float(group["rho"]) / (norm + 1.0e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state[parameter]["old_p"] = parameter.detach().clone()
                parameter.add_(parameter.grad * scale.to(parameter))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, *, zero_grad: bool) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                old = self.state[parameter].pop("old_p", None)
                if old is None and parameter.grad is not None:
                    raise IntegrityError("SAR second step lacks the first-step parameter state")
                if old is not None:
                    parameter.copy_(old)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 17:
        raise IntegrityError("adapter model must emit N x 17 logits")
    entropy = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)
    if not torch.isfinite(entropy).all():
        raise IntegrityError("adapter entropy is non-finite")
    return entropy


def _source_batchnorm_buffers(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    buffers: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        if module.running_mean is None or module.running_var is None:
            raise IntegrityError(f"source BatchNorm layer {name!r} lacks running statistics")
        buffers[name] = {
            "running_mean": module.running_mean.detach().clone(),
            "running_var": module.running_var.detach().clone(),
            "num_batches_tracked": module.num_batches_tracked.detach().clone(),
        }
    if not buffers:
        raise IntegrityError("So2Sat adapter requires BatchNorm2d layers")
    return buffers


def _configure_model(
    model: nn.Module,
    *,
    candidate_id: str,
) -> tuple[list[nn.Parameter], tuple[str, ...], dict[str, dict[str, torch.Tensor]]]:
    buffers = _source_batchnorm_buffers(model)
    model.train()
    model.requires_grad_(False)
    parameters: list[nn.Parameter] = []
    names: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        # The official algorithms use batch statistics in every BN layer while
        # adapting. SAR's layer4 exclusion applies only to affine parameters.
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None
        if candidate_id == SAR_CANDIDATE_ID and (module_name == "layer4" or module_name.startswith("layer4.")):
            continue
        module.requires_grad_(True)
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name not in {"weight", "bias"}:
                continue
            parameters.append(parameter)
            names.append(f"{module_name}.{parameter_name}")
    if not parameters or len(parameters) != len(names):
        raise IntegrityError("adapter found no eligible BatchNorm affine parameters")
    if any(not parameter.requires_grad for parameter in parameters):
        raise IntegrityError("adapter parameter configuration failed")
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if name not in set(names)):
        raise IntegrityError("adapter enabled a parameter outside its declared scope")
    return parameters, tuple(names), buffers


def _restore_source_batchnorm_buffers(
    model: nn.Module,
    buffers: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    modules = dict(model.named_modules())
    for name, source in buffers.items():
        module = modules.get(name)
        if not isinstance(module, nn.BatchNorm2d):
            raise IntegrityError(f"BatchNorm layer {name!r} disappeared during adaptation")
        module.track_running_stats = True
        module.running_mean = source["running_mean"].detach().clone().to(module.weight.device)
        module.running_var = source["running_var"].detach().clone().to(module.weight.device)
        module.num_batches_tracked = source["num_batches_tracked"].detach().clone().to(module.weight.device)
    model.eval()
    model.requires_grad_(False)


def _to_cpu_float64(value: torch.Tensor) -> torch.Tensor:
    """Transfer before widening so the operation is valid for MPS tensors."""

    return value.detach().to(device="cpu").to(dtype=torch.float64)


def _parameter_norm(parameters: Sequence[nn.Parameter]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        squared += _to_cpu_float64(parameter).square().sum()
    value = float(torch.sqrt(squared))
    if not math.isfinite(value):
        raise IntegrityError("adapter parameter norm is non-finite")
    return value


class BatchNormDivergenceCollector:
    """Collect source-standardized BN input moment divergence on frozen probe passes."""

    def __init__(self, model: nn.Module) -> None:
        self._handles: list[Any] = []
        self._weighted_sum = 0.0
        self._weight = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            if module.running_mean is None or module.running_var is None:
                raise IntegrityError(f"source BatchNorm layer {name!r} lacks statistics")
            source_mean = _to_cpu_float64(module.running_mean).clone()
            source_var = _to_cpu_float64(module.running_var).clone()
            eps = float(module.eps)

            def hook(
                _module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                *,
                source_mean: torch.Tensor = source_mean,
                source_var: torch.Tensor = source_var,
                eps: float = eps,
            ) -> None:
                value = inputs[0].detach()
                if value.ndim != 4 or value.shape[0] < 1:
                    raise IntegrityError("BatchNorm divergence hook received invalid activations")
                dimensions = (0, 2, 3)
                # The sealed statistic is defined by float64 moments. MPS cannot
                # represent float64, so transfer the full float32 activation first
                # and preserve the same CPU-float64 calculation on every backend.
                statistics = _to_cpu_float64(value)
                observed_mean = statistics.mean(dim=dimensions)
                observed_var = statistics.var(dim=dimensions, unbiased=False)
                mean_term = (observed_mean - source_mean).square() / (source_var + eps)
                variance_term = torch.log(
                    (observed_var + eps) / (source_var + eps)
                ).square()
                divergence = 0.5 * (mean_term + variance_term)
                if not torch.isfinite(divergence).all():
                    raise IntegrityError("BatchNorm source-statistic divergence is non-finite")
                weight = int(value.shape[0])
                self._weighted_sum += float(divergence.mean().cpu()) * weight
                self._weight += weight

            self._handles.append(module.register_forward_pre_hook(hook))
        if not self._handles:
            raise IntegrityError("no BatchNorm layers available for divergence collection")

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def close(self) -> float:
        self.remove_hooks()
        if self._weight < 1:
            raise IntegrityError("BatchNorm divergence collector observed no probe batch")
        result = self._weighted_sum / self._weight
        if not math.isfinite(result) or result < 0.0:
            raise IntegrityError("BatchNorm source-statistic divergence is invalid")
        return result


def frozen_logits_and_bn_divergence(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Evaluate the unmodified source model and collect label-free BN divergence."""

    model.eval()
    collector = BatchNormDivergenceCollector(model)
    outputs: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for images in batches:
                logits = model(images.to(device=device, dtype=torch.float32))
                if logits.ndim != 2 or logits.shape[1] != 17 or not torch.isfinite(logits).all():
                    raise IntegrityError("source model emitted invalid frozen probe logits")
                outputs.append(logits.detach().cpu().double())
    finally:
        # Cleanup must not replace a primary model or hook exception with the
        # secondary "no probe batch" integrity error.
        collector.remove_hooks()
    divergence = collector.close()
    if not outputs:
        raise IntegrityError("frozen probe pass received no batch")
    return torch.cat(outputs, dim=0), divergence


def _tent_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
) -> int:
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = _softmax_entropy(logits).mean()
    if not torch.isfinite(loss):
        raise IntegrityError("Tent entropy loss is non-finite")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return int(images.shape[0])


def _sar_update(
    model: nn.Module,
    optimizer: _SAM,
    images: torch.Tensor,
) -> tuple[int, bool, float | None]:
    optimizer.zero_grad(set_to_none=True)
    first_logits = model(images)
    first_entropy = _softmax_entropy(first_logits)
    reliable = torch.nonzero(first_entropy < SAR_ENTROPY_MARGIN, as_tuple=True)[0]
    if reliable.numel() == 0:
        return 0, True, None
    first_loss = first_entropy[reliable].mean()
    first_loss.backward()
    optimizer.first_step(zero_grad=True)

    second_entropy = _softmax_entropy(model(images))[reliable]
    reliable_second = second_entropy < SAR_ENTROPY_MARGIN
    if not bool(reliable_second.any()):
        # Restore the pre-ascent parameters without applying the base optimizer.
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                old = optimizer.state[parameter].pop("old_p", None)
                if old is not None:
                    parameter.data.copy_(old)
        optimizer.zero_grad(set_to_none=True)
        return int(reliable.numel()), True, None
    loss = second_entropy[reliable_second].mean()
    if not torch.isfinite(loss):
        raise IntegrityError("SAR second entropy loss is non-finite")
    loss.backward()
    optimizer.second_step(zero_grad=True)
    return int(reliable_second.sum().detach().cpu()), False, float(loss.detach().cpu())


def adapt_on_probe(
    source_model: nn.Module,
    probe_batches: Iterable[torch.Tensor],
    *,
    candidate_id: str,
    device: torch.device,
    batchnorm_source_statistic_divergence: float,
) -> tuple[nn.Module, AdaptationDiagnostics]:
    """Adapt a fresh model on probe images and return a fixed deployment model."""

    candidate_spec(candidate_id)
    model = copy.deepcopy(source_model).to(device)
    parameters, names, buffers = _configure_model(model, candidate_id=candidate_id)
    initial = [parameter.detach().clone() for parameter in parameters]
    denominator = _parameter_norm(parameters)
    if candidate_id == TENT_CANDIDATE_ID:
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            parameters,
            lr=TENT_LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
    else:
        optimizer = _SAM(
            parameters,
            lr=SAR_LEARNING_RATE,
            momentum=SAR_MOMENTUM,
            rho=SAR_RHO,
        )

    probe_batch_count = 0
    optimizer_updates = 0
    reliable_examples = 0
    skipped = 0
    recovery_resets = 0
    entropy_ema: float | None = None
    for images in probe_batches:
        images = images.to(device=device, dtype=torch.float32)
        if images.ndim != 4 or images.shape[0] < 1 or tuple(images.shape[1:]) != (10, 32, 32):
            raise IntegrityError("adapter probe batch must have shape N x 10 x 32 x 32")
        probe_batch_count += 1
        if candidate_id == TENT_CANDIDATE_ID:
            reliable_examples += _tent_update(model, optimizer, images)
            optimizer_updates += 1
        else:
            count, no_op, entropy_value = _sar_update(model, optimizer, images)  # type: ignore[arg-type]
            reliable_examples += count
            if no_op:
                skipped += 1
            else:
                optimizer_updates += 1
                if entropy_value is None:
                    raise IntegrityError("SAR update lacks its recovery entropy")
                entropy_ema = (
                    entropy_value
                    if entropy_ema is None
                    else SAR_EMA_DECAY * entropy_ema + (1.0 - SAR_EMA_DECAY) * entropy_value
                )
                if entropy_ema < SAR_RESET_CONSTANT:
                    for parameter, before in zip(parameters, initial, strict=True):
                        parameter.data.copy_(before)
                    optimizer.state.clear()
                    if isinstance(optimizer, _SAM):
                        optimizer.base_optimizer.state.clear()
                    entropy_ema = None
                    recovery_resets += 1
    if probe_batch_count < 1:
        raise IntegrityError("adapter received no probe batches")

    delta_squared = torch.zeros((), dtype=torch.float64)
    for parameter, before in zip(parameters, initial, strict=True):
        delta_squared += (parameter.detach().cpu().double() - before.cpu().double()).square().sum()
    normalized_update = float(torch.sqrt(delta_squared)) / max(denominator, 1.0e-12)
    if not math.isfinite(normalized_update):
        raise IntegrityError("normalized adapter update norm is non-finite")
    _restore_source_batchnorm_buffers(model, buffers)
    diagnostics = AdaptationDiagnostics(
        candidate_id=candidate_id,
        selected_parameter_names=names,
        probe_batches=probe_batch_count,
        optimizer_updates=optimizer_updates,
        reliable_examples=reliable_examples,
        skipped_empty_reliable_batches=skipped,
        model_recovery_resets=recovery_resets,
        normalized_adapter_update_norm=normalized_update,
        batchnorm_source_statistic_divergence=float(batchnorm_source_statistic_divergence),
    )
    return model, diagnostics


@torch.inference_mode()
def fixed_model_logits(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Collect logits without changing model state or using batch statistics."""

    if model.training:
        raise IntegrityError("fixed deployment model must be in eval mode")
    before = clone_cpu_state(model)
    outputs: list[torch.Tensor] = []
    for images in batches:
        logits = model(images.to(device=device, dtype=torch.float32))
        if logits.ndim != 2 or logits.shape[1] != 17 or not torch.isfinite(logits).all():
            raise IntegrityError("fixed adapter model emitted invalid logits")
        outputs.append(logits.detach().cpu().double())
    if not outputs:
        raise IntegrityError("fixed model evaluation received no batch")
    after = clone_cpu_state(model)
    for name in before:
        if not torch.equal(before[name], after[name]):
            raise IntegrityError(f"fixed deployment evaluation changed model state {name!r}")
    return torch.cat(outputs, dim=0)
