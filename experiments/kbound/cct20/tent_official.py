"""Strict binding to the pinned upstream Tent implementation.

This module deliberately imports ``external/tent_official/tent.py`` by file
path.  It does not copy or reimplement Tent.  Every adapter instance is scoped
to exactly one source-checkpoint x camera-location cell, so continual state can
never leak from one natural domain (or checkpoint) into another.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn

from .audit_checkpoints import tensor_state_sha256
from .integrity import IntegrityError, file_sha256, require_sha256

OFFICIAL_TENT_COMMIT = "e9e926a668d85244c66a6d5c006efbd2b82e83e8"
OFFICIAL_TENT_TREE = "f482757a924f4a61651a37b15e75d0eab2fd3c9e"
OFFICIAL_TENT_FILE_SHA256 = "d854e1f65f741aa1632dfb55420ca1bcf5405c978bef890e45caad04980514c8"
TENT_LR = 1.0e-3
TENT_BETAS = (0.9, 0.999)
TENT_WEIGHT_DECAY = 0.0
TENT_STEPS = 1
TENT_EPISODIC = False
LOCKED_BACKEND_STRATEGY = "mps_resnet50_official_tent_cpu_root_bn1_v1"
BN_GAUSSIAN_KL_SCHEMA = "kbound_cct20_frozen_bn_probe_moments_v2"
BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION = "stable_relative_variance_log1p_taylor_v2"
BN_GAUSSIAN_KL_TAYLOR_THRESHOLD = 1.0e-4
BN_GAUSSIAN_KL_TAYLOR_TERMS = 6
BN_GAUSSIAN_KL_NUMERIC_CLIPPING = "none"


def stable_gaussian_kl_probe_to_source(
    probe_mean: torch.Tensor,
    probe_var: torch.Tensor,
    source_mean: torch.Tensor,
    source_var: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the Gaussian KL without subtracting nearly equal unit terms.

    With ``x = (probe_var - source_var) / (source_var + eps)``, the
    variance term is ``x - log1p(x)``.  A six-term Taylor polynomial for
    powers two through seven avoids cancellation when ``|x| <= 1e-4``.
    The formulation is algebraically identical to the protocol definition
    and applies no floor or clipping.
    """

    tensors = (probe_mean, probe_var, source_mean, source_var)
    if any(not torch.is_tensor(value) for value in tensors):
        raise IntegrityError("Gaussian KL moments must be tensors")
    if any(value.dtype != torch.float64 or value.device.type != "cpu" for value in tensors):
        raise IntegrityError("Gaussian KL moments must be CPU float64 tensors")
    if any(value.shape != probe_mean.shape for value in tensors) or probe_mean.ndim != 1:
        raise IntegrityError("Gaussian KL moments must be same-shaped channel vectors")
    if not math.isfinite(eps) or eps <= 0.0:
        raise IntegrityError("Gaussian KL epsilon must be finite and positive")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise IntegrityError("Gaussian KL moments are non-finite")
    if torch.any(probe_var < 0.0) or torch.any(source_var < 0.0):
        raise IntegrityError("Gaussian KL variances must be nonnegative")

    source_scale = source_var + eps
    probe_scale = probe_var + eps
    if torch.any(source_scale <= 0.0) or torch.any(probe_scale <= 0.0):
        raise IntegrityError("Gaussian KL augmented variances must be positive")
    relative_variance = (probe_var - source_var) / source_scale
    if torch.any(relative_variance <= -1.0) or not torch.isfinite(relative_variance).all():
        raise IntegrityError("Gaussian KL relative variance left its valid domain")

    taylor_mask = torch.abs(relative_variance) <= BN_GAUSSIAN_KL_TAYLOR_THRESHOLD
    x = relative_variance
    # x - log1p(x) = x^2/2 - x^3/3 + ... - x^7/7 + O(x^8).
    taylor_gap = x.square() * (
        0.5 + x * (-(1.0 / 3.0) + x * (0.25 + x * (-0.2 + x * ((1.0 / 6.0) + x * (-(1.0 / 7.0))))))
    )
    direct_gap = x - torch.log1p(x)
    variance_gap = torch.where(taylor_mask, taylor_gap, direct_gap)
    mean_gap = (probe_mean - source_mean).square() / source_scale
    kl = 0.5 * (variance_gap + mean_gap)
    if not torch.isfinite(kl).all():
        raise IntegrityError("Gaussian KL is non-finite")
    if torch.any(kl < 0.0):
        raise IntegrityError("Gaussian KL became negative under the stable no-clipping implementation")
    return kl, taylor_mask


class KBoundCPUFallbackBatchNorm2d(nn.BatchNorm2d):
    """Run one affine BatchNorm on CPU while the surrounding model stays on MPS.

    The sealed PyTorch 2.5.1 MPS runtime produces non-finite gradients for the
    ResNet-50 root BatchNorm affine pair.  Tent freezes the upstream convolution,
    so this device transfer preserves the ordinary BatchNorm autograd graph for
    the only trainable values at this layer: its weight and bias.
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.device.type != "mps":
            raise IntegrityError("locked root-BatchNorm fallback requires an MPS input")
        if value.requires_grad:
            raise IntegrityError("locked root-BatchNorm fallback requires a frozen upstream path")
        if self.weight is None or self.bias is None:
            raise IntegrityError("locked root-BatchNorm fallback requires affine parameters")
        if self.weight.device.type != "cpu" or self.bias.device.type != "cpu":
            raise IntegrityError("locked root-BatchNorm fallback affine parameters left CPU")
        return super().forward(value.to(device="cpu")).to(device=value.device)


def install_locked_root_bn_cpu_fallback(model: nn.Module) -> dict[str, Any]:
    """Install and receipt the reviewed CPU fallback on an MPS ResNet root BN."""

    if any(parameter.device.type != "mps" for parameter in model.parameters()):
        raise IntegrityError("root-BatchNorm fallback requires an otherwise all-MPS model")
    original = getattr(model, "bn1", None)
    if not isinstance(original, nn.BatchNorm2d) or isinstance(original, KBoundCPUFallbackBatchNorm2d):
        raise IntegrityError("root-BatchNorm fallback requires one native model.bn1")
    if not original.affine or original.weight is None or original.bias is None:
        raise IntegrityError("root-BatchNorm fallback requires affine model.bn1")
    source_hash = tensor_state_sha256(original.state_dict())
    replacement = KBoundCPUFallbackBatchNorm2d(
        original.num_features,
        eps=original.eps,
        momentum=original.momentum,
        affine=original.affine,
        track_running_stats=original.track_running_stats,
        device="cpu",
        dtype=original.weight.dtype,
    )
    replacement.load_state_dict(
        {name: value.detach().to(device="cpu") for name, value in original.state_dict().items()},
        strict=True,
    )
    installed_hash = tensor_state_sha256(replacement.state_dict())
    if installed_hash != source_hash:
        raise IntegrityError("root-BatchNorm fallback changed the source BN tensor state")
    model.bn1 = replacement
    receipt = {
        "schema": "kbound_cct20_backend_installation_v1",
        "strategy": LOCKED_BACKEND_STRATEGY,
        "fallback_layer": "bn1",
        "source_module_class": "torch.nn.BatchNorm2d",
        "fallback_module_class": "KBoundCPUFallbackBatchNorm2d",
        "fallback_input_device": "mps",
        "fallback_compute_device": "cpu",
        "fallback_parameter_device": "cpu",
        "fallback_output_device": "mps",
        "num_features": int(original.num_features),
        "eps": float(original.eps),
        "momentum": None if original.momentum is None else float(original.momentum),
        "affine": bool(original.affine),
        "preconfigure_track_running_stats": bool(original.track_running_stats),
        "source_bn_state_sha256": source_hash,
        "installed_bn_state_sha256": installed_hash,
        "state_hash_equal": True,
    }
    return receipt


def _validate_locked_backend_installation(
    model: nn.Module,
    receipt: Mapping[str, Any],
) -> None:
    root = getattr(model, "bn1", None)
    expected = {
        "schema": "kbound_cct20_backend_installation_v1",
        "strategy": LOCKED_BACKEND_STRATEGY,
        "fallback_layer": "bn1",
        "source_module_class": "torch.nn.BatchNorm2d",
        "fallback_module_class": "KBoundCPUFallbackBatchNorm2d",
        "fallback_input_device": "mps",
        "fallback_compute_device": "cpu",
        "fallback_parameter_device": "cpu",
        "fallback_output_device": "mps",
    }
    if any(receipt.get(field) != value for field, value in expected.items()):
        raise IntegrityError("root-BatchNorm fallback receipt identity drift")
    if (
        not isinstance(root, KBoundCPUFallbackBatchNorm2d)
        or root.weight is None
        or root.bias is None
        or root.weight.device.type != "cpu"
        or root.bias.device.type != "cpu"
        or receipt.get("num_features") != root.num_features
        or receipt.get("eps") != float(root.eps)
        or receipt.get("momentum") != (None if root.momentum is None else float(root.momentum))
        or receipt.get("affine") is not True
        or receipt.get("preconfigure_track_running_stats") is not True
        or receipt.get("state_hash_equal") is not True
    ):
        raise IntegrityError("root-BatchNorm fallback module differs from its receipt")
    source_hash = require_sha256(receipt.get("source_bn_state_sha256"), field="source_bn_state_sha256")
    installed_hash = require_sha256(receipt.get("installed_bn_state_sha256"), field="installed_bn_state_sha256")
    if installed_hash != source_hash or tensor_state_sha256(root.state_dict()) != installed_hash:
        raise IntegrityError("root-BatchNorm fallback tensor state differs from its receipt")


def _to_cpu_float64(value: torch.Tensor) -> torch.Tensor:
    """Transfer before widening because MPS cannot materialize float64 tensors."""

    return value.to(device="cpu").to(dtype=torch.float64)


def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IntegrityError(f"cannot verify official Tent repository {repo}: {exc}") from exc
    return completed.stdout.strip()


def verify_official_tent(repo_root: str | Path) -> dict[str, Any]:
    """Verify the exact clean upstream implementation frozen by the protocol."""

    repo = Path(repo_root).expanduser().resolve()
    source = repo / "tent.py"
    if not source.is_file():
        raise IntegrityError(f"official Tent source is missing: {source}")
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    tracked_changes = _git(repo, "status", "--porcelain", "--untracked-files=no")
    source_hash = file_sha256(source)
    if commit != OFFICIAL_TENT_COMMIT:
        raise IntegrityError(f"official Tent commit mismatch: expected {OFFICIAL_TENT_COMMIT}, found {commit}")
    if tree != OFFICIAL_TENT_TREE:
        raise IntegrityError(f"official Tent tree mismatch: expected {OFFICIAL_TENT_TREE}, found {tree}")
    if source_hash != OFFICIAL_TENT_FILE_SHA256:
        raise IntegrityError(
            f"official Tent source hash mismatch: expected {OFFICIAL_TENT_FILE_SHA256}, found {source_hash}"
        )
    if tracked_changes:
        raise IntegrityError(f"official Tent repository has tracked changes: {tracked_changes}")
    return {
        "implementation": "DequanWang/tent upstream",
        "repository": str(repo),
        "git_commit": commit,
        "git_tree": tree,
        "tent_py_sha256": source_hash,
        "tracked_worktree_clean": True,
        "configure_function": "tent.configure_model",
        "parameter_function": "tent.collect_params",
        "adapter_class": "tent.Tent",
        "optimizer": {
            "class": "torch.optim.Adam",
            "lr": TENT_LR,
            "betas": list(TENT_BETAS),
            "weight_decay": TENT_WEIGHT_DECAY,
        },
        "steps": TENT_STEPS,
        "episodic": TENT_EPISODIC,
        "reset_scope": "source_checkpoint_x_camera_location",
    }


def _load_module(repo: Path) -> ModuleType:
    source = repo / "tent.py"
    name = "kbound_cct20_pinned_tent"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise IntegrityError(f"cannot import pinned Tent module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for symbol in ("configure_model", "collect_params", "check_model", "Tent"):
        if not hasattr(module, symbol):
            raise IntegrityError(f"pinned Tent module lacks required symbol {symbol}")
    return module


@dataclass
class OfficialTentBinding:
    """One fresh, continual Tent session for one checkpoint-location cell."""

    adapter: nn.Module
    optimizer: torch.optim.Optimizer
    parameter_names: tuple[str, ...]
    checkpoint_tensor_sha256: str
    location_id: str
    provenance: dict[str, Any]
    backend_installation: dict[str, Any] | None
    _parameters: tuple[nn.Parameter, ...] = field(repr=False)
    _initial_parameters: tuple[torch.Tensor, ...] = field(repr=False)

    @property
    def reset_scope(self) -> str:
        return f"{self.checkpoint_tensor_sha256}:{self.location_id}"

    def reset(self) -> None:
        """Restore this cell to its own checkpoint-start state."""

        reset = getattr(self.adapter, "reset", None)
        if not callable(reset):  # pragma: no cover - upstream contract guard
            raise IntegrityError("pinned Tent adapter no longer exposes reset()")
        reset()

    def normalized_update_norm(self) -> float:
        """Relative L2 change over exactly official Tent's collected parameters."""

        squared_difference = 0.0
        squared_initial = 0.0
        for parameter, initial in zip(self._parameters, self._initial_parameters, strict=True):
            current = _to_cpu_float64(parameter.detach())
            reference = initial.to(dtype=torch.float64)
            squared_difference += float(torch.sum((current - reference) ** 2).item())
            squared_initial += float(torch.sum(reference**2).item())
        result = math.sqrt(squared_difference) / max(math.sqrt(squared_initial), 1.0e-12)
        if not math.isfinite(result):  # pragma: no cover - corrupted tensor guard
            raise IntegrityError("Tent relative BN-affine update norm is non-finite")
        return result

    def probe_update_receipt(self) -> dict[str, Any]:
        return {
            "schema": "kbound_cct20_tent_probe_update_v1",
            "checkpoint_tensor_sha256": self.checkpoint_tensor_sha256,
            "location_id": self.location_id,
            "reset_scope": self.reset_scope,
            "parameter_names": list(self.parameter_names),
            "normalized_tent_update_norm": self.normalized_update_norm(),
            "formula": "l2(after_probe-before_probe)/max(l2(before_probe),1e-12)",
        }

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "kbound_cct20_official_tent_binding_v1",
            "checkpoint_tensor_sha256": self.checkpoint_tensor_sha256,
            "location_id": self.location_id,
            "reset_scope": self.reset_scope,
            "parameter_names": list(self.parameter_names),
            "n_parameters": len(self.parameter_names),
            "initial_bn_affine_l2": math.sqrt(
                sum(float(torch.sum(value.to(dtype=torch.float64) ** 2).item()) for value in self._initial_parameters)
            ),
            "update_norm_formula": ("l2(after_probe-before_probe)/max(l2(before_probe),1e-12)"),
            "backend_installation": (None if self.backend_installation is None else dict(self.backend_installation)),
            "provenance": self.provenance,
        }


class FrozenBatchNormMomentAccumulator:
    """Accumulate probe moments at frozen-model BatchNorm inputs.

    Each BatchNorm channel receives equal weight in the final Gaussian KL.  The
    source moments and per-layer epsilon are copied before any probe forward.
    """

    def __init__(self, frozen_model: nn.Module) -> None:
        if frozen_model.training:
            raise IntegrityError("BN probe moments require the frozen model in eval mode")
        self._states: dict[str, dict[str, Any]] = {}
        self._handles: list[Any] = []
        self._finalized = False
        for name, module in frozen_model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            if module.running_mean is None or module.running_var is None:
                raise IntegrityError(f"frozen source BatchNorm {name!r} lacks stored running moments")
            channels = int(module.num_features)
            source_mean = _to_cpu_float64(module.running_mean.detach()).clone()
            source_var = _to_cpu_float64(module.running_var.detach()).clone()
            if (
                source_mean.numel() != channels
                or source_var.numel() != channels
                or not torch.isfinite(source_mean).all()
                or not torch.isfinite(source_var).all()
                or torch.any(source_var < 0.0)
            ):
                raise IntegrityError(f"frozen source BatchNorm {name!r} has invalid moments")
            state = {
                "source_mean": source_mean,
                "source_var": source_var,
                "eps": float(module.eps),
                "probe_mean": torch.zeros(channels, dtype=torch.float64),
                "probe_m2": torch.zeros(channels, dtype=torch.float64),
                "count_per_channel": 0,
            }
            self._states[name] = state

            def capture(
                _module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                *,
                layer_name: str = name,
                expected_channels: int = channels,
            ) -> None:
                if self._finalized:
                    raise IntegrityError("BN probe accumulator received a forward after finalization")
                if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                    raise IntegrityError(f"BatchNorm {layer_name!r} received unexpected inputs")
                value = inputs[0].detach()
                if value.ndim != 4 or value.shape[1] != expected_channels:
                    raise IntegrityError(
                        f"BatchNorm {layer_name!r} probe input has unexpected shape {tuple(value.shape)}"
                    )
                if not torch.isfinite(value).all():
                    raise IntegrityError(f"BatchNorm {layer_name!r} probe input is non-finite")
                current = self._states[layer_name]
                # A ResNet activation can exceed 800 MB if widened before the
                # reduction.  Reduce in the model's float32 dtype on the
                # originating device, transfer only O(channels) statistics,
                # then merge batches in CPU float64.  This also keeps the
                # numerical contract consistent across CPU, CUDA, and MPS.
                working = value.to(dtype=torch.float32)
                batch_variance, batch_mean = torch.var_mean(
                    working,
                    dim=(0, 2, 3),
                    correction=0,
                )
                batch_count = int(value.shape[0] * value.shape[2] * value.shape[3])
                batch_mean = _to_cpu_float64(batch_mean)
                batch_m2 = _to_cpu_float64(batch_variance) * batch_count
                prior_count = int(current["count_per_channel"])
                if prior_count == 0:
                    current["probe_mean"] = batch_mean
                    current["probe_m2"] = batch_m2
                    current["count_per_channel"] = batch_count
                else:
                    total = prior_count + batch_count
                    delta = batch_mean - current["probe_mean"]
                    current["probe_mean"] = current["probe_mean"] + delta * (batch_count / total)
                    current["probe_m2"] = (
                        current["probe_m2"] + batch_m2 + delta**2 * (prior_count * batch_count / total)
                    )
                    current["count_per_channel"] = total

            self._handles.append(module.register_forward_pre_hook(capture))
        if not self._states:
            raise IntegrityError("frozen source model has no BatchNorm2d layers")

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise IntegrityError("BN probe moments were already finalized")
        self._finalized = True
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        total_kl = 0.0
        total_channels = 0
        layer_receipts = []
        for name, state in self._states.items():
            count = int(state["count_per_channel"])
            if count < 1:
                raise IntegrityError(f"BatchNorm {name!r} saw no probe values")
            probe_mean = state["probe_mean"]
            probe_var = state["probe_m2"] / count
            source_mean = state["source_mean"]
            source_var = state["source_var"]
            eps = float(state["eps"])
            kl, taylor_mask = stable_gaussian_kl_probe_to_source(
                probe_mean,
                probe_var,
                source_mean,
                source_var,
                eps=eps,
            )
            channel_count = int(kl.numel())
            total_kl += float(kl.sum().item())
            total_channels += channel_count
            layer_receipts.append(
                {
                    "layer": name,
                    "channels": channel_count,
                    "values_per_channel": count,
                    "bn_eps": eps,
                    "mean_kl": float(kl.mean().item()),
                    "min_kl": float(kl.min().item()),
                    "taylor_branch_channels": int(taylor_mask.sum().item()),
                }
            )
        divergence = total_kl / total_channels
        if not math.isfinite(divergence):  # pragma: no cover
            raise IntegrityError("aggregate BatchNorm source-statistic divergence is non-finite")
        return {
            "schema": BN_GAUSSIAN_KL_SCHEMA,
            "batchnorm_batch_source_statistic_divergence": divergence,
            "channel_count": total_channels,
            "layers": layer_receipts,
            "formula": "channel_weighted_mean_gaussian_kl_probe_to_source",
            "numeric_implementation": BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION,
            "taylor_threshold": BN_GAUSSIAN_KL_TAYLOR_THRESHOLD,
            "taylor_terms": BN_GAUSSIAN_KL_TAYLOR_TERMS,
            "numeric_clipping": BN_GAUSSIAN_KL_NUMERIC_CLIPPING,
            "taylor_branch_channels": sum(row["taylor_branch_channels"] for row in layer_receipts),
            "minimum_channel_kl": min(row["min_kl"] for row in layer_receipts),
        }


def new_checkpoint_location_session(
    model: nn.Module,
    *,
    repo_root: str | Path,
    checkpoint_tensor_sha256: str,
    location_id: str | int,
    backend_installation_receipt: Mapping[str, Any] | None = None,
) -> OfficialTentBinding:
    """Create the locked official Tent adapter for one fresh cell.

    The caller must load a fresh copy of the named checkpoint before each call.
    The returned adapter is continual *within* the location and must be discarded
    afterwards; sharing it across locations violates the protocol.
    """

    checkpoint_hash = require_sha256(checkpoint_tensor_sha256, field="checkpoint_tensor_sha256")
    location = str(location_id)
    if not location:
        raise IntegrityError("location_id cannot be empty")
    repo = Path(repo_root).expanduser().resolve()
    provenance = verify_official_tent(repo)
    module = _load_module(repo)
    backend_installation = None
    if backend_installation_receipt is not None:
        _validate_locked_backend_installation(model, backend_installation_receipt)
        backend_installation = dict(backend_installation_receipt)

    configured = module.configure_model(model)
    params, names = module.collect_params(configured)
    if not params or not names or len(params) != len(names):
        raise IntegrityError("official Tent found no valid BatchNorm affine parameters")
    if len(set(names)) != len(names):
        raise IntegrityError("official Tent returned duplicate parameter names")
    module.check_model(configured)
    parameter_tuple = tuple(params)
    parameter_devices = sorted({parameter.device.type for parameter in parameter_tuple})
    if backend_installation is not None:
        root = getattr(configured, "bn1", None)
        if (
            parameter_devices != ["cpu", "mps"]
            or not isinstance(root, KBoundCPUFallbackBatchNorm2d)
            or root.track_running_stats
            or root.running_mean is not None
            or root.running_var is not None
        ):
            raise IntegrityError("configured official Tent model violates the locked hybrid backend")
        backend_installation.update(
            {
                "official_tent_parameter_devices": parameter_devices,
                "configured_track_running_stats": False,
                "configured_running_moments_absent": True,
            }
        )
    initial_parameters = tuple(parameter.detach().to(device="cpu").clone() for parameter in parameter_tuple)

    optimizer = torch.optim.Adam(
        params,
        lr=TENT_LR,
        betas=TENT_BETAS,
        weight_decay=TENT_WEIGHT_DECAY,
    )
    adapter = module.Tent(
        configured,
        optimizer,
        steps=TENT_STEPS,
        episodic=TENT_EPISODIC,
    )
    group = optimizer.param_groups[0]
    observed = (
        float(group["lr"]),
        tuple(float(value) for value in group["betas"]),
        float(group["weight_decay"]),
    )
    expected = (TENT_LR, TENT_BETAS, TENT_WEIGHT_DECAY)
    if observed != expected:  # pragma: no cover - PyTorch contract guard
        raise IntegrityError(f"Tent optimizer settings drifted: {observed} != {expected}")
    if getattr(adapter, "steps", None) != 1 or getattr(adapter, "episodic", None) is not False:
        raise IntegrityError("Tent adapter settings drifted from steps=1, episodic=False")
    return OfficialTentBinding(
        adapter=adapter,
        optimizer=optimizer,
        parameter_names=tuple(str(name) for name in names),
        checkpoint_tensor_sha256=checkpoint_hash,
        location_id=location,
        provenance=provenance,
        backend_installation=backend_installation,
        _parameters=parameter_tuple,
        _initial_parameters=initial_parameters,
    )
