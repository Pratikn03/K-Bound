"""Authenticated reference-source model construction and preprocessing.

The public inference boundary is a raw float NCHW tensor in ``[0, 1]`` at the
family's declared spatial size.  Model-specific normalization deliberately
lives in the returned module so downstream adaptation operates on the same
module graph as frozen inference.  The preprocessing callable only performs
deterministic image conversion, resizing, and cropping.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch
import torchvision
from torch import nn
from torchvision import models as tv_models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_functional


class ReferenceSourceError(RuntimeError):
    """Base error for a reference-source compatibility failure."""


class CheckpointIdentityError(ReferenceSourceError):
    """A required artifact is absent or its bytes do not match its binding."""


class CheckpointSchemaError(ReferenceSourceError):
    """Authenticated bytes do not have the required learned-state schema."""


class UnsupportedReferenceFamilyError(ReferenceSourceError):
    """No reviewed compatibility implementation exists for this family."""


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_CIFAR_MEAN = [0.5, 0.5, 0.5]
_CIFAR_STD = [0.5, 0.5, 0.5]
_CIFAR_CONSTRUCTOR_REVISION = "4d73272b1c6017b98b33148eaea93e5d71c311e0"
_CIFAR_CONSTRUCTOR_SHA256 = "a43531f9fff226183ba45c67dd2bfa175219ba7ddb1e71cae682b4657cb119e2"

_FAMILIES: dict[str, dict[str, Any]] = {
    "cifar10_augmix_wrn40_2": {
        "architecture": "RobustBench WideResNet-40-2",
        "num_classes": 10,
        "input_size": [32, 32],
        "checkpoint_namespace": "root",
        "classifier_weight": "fc.weight",
        "normalization": {
            "kind": "fixed_channel",
            "mean": _CIFAR_MEAN,
            "std": _CIFAR_STD,
            "location": "model_wrapper",
        },
    },
    "torchvision_resnet50_imagenet1k_v2": {
        "architecture": "torchvision ResNet50",
        "num_classes": 1000,
        "input_size": [224, 224],
        "checkpoint_namespace": "root",
        "classifier_weight": "fc.weight",
        "normalization": {
            "kind": "fixed_channel",
            "mean": _IMAGENET_MEAN,
            "std": _IMAGENET_STD,
            "location": "model_wrapper",
        },
    },
    "wilds_camelyon17_densenet121": {
        "architecture": "torchvision DenseNet121",
        "num_classes": 2,
        "input_size": [96, 96],
        "checkpoint_namespace": "algorithm/model.",
        "classifier_weight": "classifier.weight",
        "normalization": {
            "kind": "fixed_channel",
            "mean": _IMAGENET_MEAN,
            "std": _IMAGENET_STD,
            "location": "model_wrapper",
        },
    },
    "wilds_rxrx1_resnet50": {
        "architecture": "torchvision ResNet50",
        "num_classes": 1139,
        "input_size": [256, 256],
        "checkpoint_namespace": "algorithm/model.",
        "classifier_weight": "fc.weight",
        "normalization": {
            "kind": "per_image_channel",
            "reduction_dimensions": [2, 3],
            "std_correction": 1,
            "zero_std_replacement": 1.0,
            "location": "model_wrapper",
        },
    },
    "wilds_iwildcam_resnet50": {
        "architecture": "torchvision ResNet50",
        "num_classes": 182,
        "input_size": [448, 448],
        "checkpoint_namespace": "algorithm/model.",
        "classifier_weight": "fc.weight",
        "normalization": {
            "kind": "fixed_channel",
            "mean": _IMAGENET_MEAN,
            "std": _IMAGENET_STD,
            "location": "model_wrapper",
        },
    },
}


def _require_family(family: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(family, str) or family not in _FAMILIES:
        raise UnsupportedReferenceFamilyError(f"unsupported reference family: {family}")
    return family, _FAMILIES[family]


def _streaming_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_artifact(spec: Mapping[str, Any], path_key: str, hash_key: str, label: str) -> tuple[Path, str]:
    raw_path = spec.get(path_key)
    expected = spec.get(hash_key)
    if not isinstance(raw_path, str) or not raw_path:
        raise CheckpointIdentityError(f"{label} path is required")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if path.is_symlink():
        raise CheckpointIdentityError(f"{label} must not be a symlink: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise CheckpointIdentityError(f"{label} expected SHA-256 must be 64 hexadecimal characters")
    try:
        int(expected, 16)
    except ValueError as exc:
        raise CheckpointIdentityError(f"{label} expected SHA-256 must be hexadecimal") from exc
    actual = _streaming_sha256(path)
    if actual != expected.lower():
        raise CheckpointIdentityError(f"{label} SHA-256 mismatch: expected {expected.lower()}, observed {actual}")
    return path.resolve(), actual


@contextmanager
def _preserve_torch_rng() -> Iterator[None]:
    cpu_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _load_constructor_module(path: Path, digest: str) -> ModuleType:
    module_spec = importlib.util.spec_from_file_location(f"_kbound_wrn_{digest}", path)
    if module_spec is None or module_spec.loader is None:
        raise CheckpointIdentityError(f"CIFAR constructor cannot be imported: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    constructor = getattr(module, "WideResNet", None)
    if not isinstance(constructor, type) or not issubclass(constructor, nn.Module):
        raise CheckpointSchemaError("CIFAR constructor must expose an nn.Module WideResNet class")
    return module


def _validated_cifar_constructor(spec: Mapping[str, Any]) -> tuple[Path, str]:
    path, digest = _validated_artifact(spec, "constructor_path", "constructor_sha256", "constructor")
    if digest != _CIFAR_CONSTRUCTOR_SHA256:
        raise CheckpointIdentityError(
            "selected pinned CIFAR constructor SHA-256 mismatch: "
            f"required {_CIFAR_CONSTRUCTOR_SHA256}, observed {digest}"
        )
    return path, digest


def _construct_model(family: str, spec: Mapping[str, Any]) -> nn.Module:
    if family == "cifar10_augmix_wrn40_2":
        constructor_path, constructor_sha256 = _validated_cifar_constructor(spec)
        module = _load_constructor_module(constructor_path, constructor_sha256)
        return module.WideResNet(
            depth=40,
            num_classes=10,
            widen_factor=2,
            sub_block1=False,
            dropRate=0.0,
            bias_last=True,
        )
    if family == "torchvision_resnet50_imagenet1k_v2":
        return tv_models.resnet50(weights=None, num_classes=1000)
    if family == "wilds_camelyon17_densenet121":
        return tv_models.densenet121(weights=None, num_classes=2)
    if family == "wilds_rxrx1_resnet50":
        return tv_models.resnet50(weights=None, num_classes=1139)
    if family == "wilds_iwildcam_resnet50":
        return tv_models.resnet50(weights=None, num_classes=182)
    raise UnsupportedReferenceFamilyError(f"unsupported reference family: {family}")


def _checkpoint_state(payload: object, family: str) -> dict[str, torch.Tensor]:
    config = _FAMILIES[family]
    namespace = config["checkpoint_namespace"]
    if namespace == "algorithm/model.":
        if not isinstance(payload, Mapping) or "algorithm" not in payload:
            raise CheckpointSchemaError("WILDS checkpoint root must contain an algorithm state mapping")
        payload = payload["algorithm"]
        if not isinstance(payload, Mapping):
            raise CheckpointSchemaError("WILDS checkpoint algorithm entry must be a state mapping")
        bad_names = [str(name) for name in payload if not isinstance(name, str) or not name.startswith("model.")]
        if bad_names:
            raise CheckpointSchemaError(f"unexpected WILDS algorithm namespace keys: {bad_names[:4]}")
        state = {name[len("model.") :]: value for name, value in payload.items()}
    else:
        if not isinstance(payload, Mapping):
            raise CheckpointSchemaError("checkpoint root must be a state mapping")
        state = dict(payload)

    if not state:
        raise CheckpointSchemaError("checkpoint state mapping is empty")
    non_tensors = [str(name) for name, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        raise CheckpointSchemaError(f"checkpoint state contains non-tensor values: {non_tensors[:4]}")
    for name, value in state.items():
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise CheckpointSchemaError(f"non-finite checkpoint tensor: {name}")

    head_name = config["classifier_weight"]
    head = state.get(head_name)
    expected_classes = int(config["num_classes"])
    if not isinstance(head, torch.Tensor) or head.ndim != 2 or int(head.shape[0]) != expected_classes:
        observed = None if not isinstance(head, torch.Tensor) else list(head.shape)
        raise CheckpointSchemaError(
            f"classifier {head_name} must have {expected_classes} output rows; observed {observed}"
        )
    return state


class _NormalizedReferenceModel(nn.Module):
    def __init__(self, model: nn.Module, family: str) -> None:
        super().__init__()
        self.model = model
        self.family = family
        config = _FAMILIES[family]
        self.input_size = tuple(int(value) for value in config["input_size"])
        normalization = config["normalization"]
        if normalization["kind"] == "fixed_channel":
            self.register_buffer(
                "normalization_mean",
                torch.tensor(normalization["mean"], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "normalization_std",
                torch.tensor(normalization["std"], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("reference model input must be an NCHW tensor")
        if not x.is_floating_point():
            raise ValueError("reference model input must be floating point")
        if x.shape[1] != 3 or tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(f"reference model input must have shape NCHW with C=3 and HW={self.input_size}")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("reference model input contains non-finite values")
        if bool((x < 0).any()) or bool((x > 1).any()):
            raise ValueError("reference model input must be in [0, 1]")

        normalization = _FAMILIES[self.family]["normalization"]
        if normalization["kind"] == "fixed_channel":
            normalized = (x - self.normalization_mean) / self.normalization_std
        else:
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), correction=1, keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            normalized = (x - mean) / std
        return self.model(normalized)


def _wrap_model(model: nn.Module, family: str) -> nn.Module:
    _require_family(family)
    return _NormalizedReferenceModel(model, family)


def _load_state_strict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> dict[str, list[str]]:
    model_state = model.state_dict()
    model_names = set(model_state)
    state_names = set(state)
    missing = sorted(model_names - state_names)
    unexpected = sorted(state_names - model_names)
    buffers = dict(model.named_buffers())
    legacy_counters = sorted(
        name
        for name in missing
        if name.endswith(".num_batches_tracked")
        and name in buffers
        and buffers[name].numel() == 1
        and not buffers[name].is_floating_point()
        and not buffers[name].is_complex()
    )
    missing_required = sorted(set(missing) - set(legacy_counters))
    if missing_required or unexpected:
        raise CheckpointSchemaError(
            f"strict state mismatch: missing={missing_required[:8]}, unexpected={unexpected[:8]}"
        )
    compatible_state = dict(state)
    for name in legacy_counters:
        compatible_state[name] = model_state[name]
    try:
        model.load_state_dict(compatible_state, strict=True)
    except RuntimeError as exc:
        raise CheckpointSchemaError(f"strict state tensor mismatch: {exc}") from exc
    return {"synthesized_missing_num_batches_tracked": legacy_counters}


def load_reference_model(spec: dict[str, Any], device: str) -> tuple[nn.Module, dict[str, Any]]:
    """Construct and strict-load one authenticated published reference model."""

    if not isinstance(spec, dict):
        raise TypeError("reference model spec must be a dictionary")
    family, config = _require_family(spec.get("family"))
    checkpoint_path, checkpoint_sha256 = _validated_artifact(spec, "checkpoint_path", "checkpoint_sha256", "checkpoint")
    if family == "cifar10_augmix_wrn40_2":
        _validated_cifar_constructor(spec)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = _checkpoint_state(payload, family)
    with _preserve_torch_rng():
        model = _construct_model(family, spec)
    state_compatibility = _load_state_strict(model, state)
    try:
        target_device = torch.device(device)
        model = _wrap_model(model, family).to(target_device).eval()
    except (RuntimeError, TypeError) as exc:
        raise ReferenceSourceError(f"cannot place reference model on device {device!r}: {exc}") from exc

    provenance: dict[str, Any] = {
        "schema": "kbound_reference_source_model_v1",
        "family": family,
        "architecture": config["architecture"],
        "num_classes": int(config["num_classes"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_namespace": config["checkpoint_namespace"],
        "strict_state_load": True,
        "state_compatibility": state_compatibility,
        "safe_torch_load": "weights_only=True",
        "device": str(target_device),
        "software": {
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
        },
        "preprocessing": get_reference_preprocessing(family)[1],
    }
    if family == "cifar10_augmix_wrn40_2":
        constructor_path, constructor_sha256 = _validated_cifar_constructor(spec)
        provenance.update(
            {
                "constructor_path": str(constructor_path),
                "constructor_sha256": constructor_sha256,
                "constructor_revision": _CIFAR_CONSTRUCTOR_REVISION,
                "constructor_arguments": {
                    "depth": 40,
                    "num_classes": 10,
                    "widen_factor": 2,
                    "sub_block1": False,
                    "dropRate": 0.0,
                    "bias_last": True,
                },
            }
        )
    json.dumps(provenance)
    return model, provenance


def _raw_float_chw(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        if image.device.type != "cpu":
            raise ValueError("preprocessing requires a CPU tensor; hidden device copies are forbidden")
        if image.ndim != 3:
            raise ValueError("preprocessing tensor input must have CHW shape")
        if image.dtype == torch.uint8:
            tensor = image.to(dtype=torch.float32).div(255.0)
        elif image.is_floating_point():
            tensor = image.to(dtype=torch.float32)
        else:
            raise ValueError("preprocessing tensor must be uint8 or floating point")
    else:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        tensor = tv_functional.to_tensor(image)
    if tensor.shape[0] != 3:
        raise ValueError("preprocessing input must have three channels")
    if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()) or bool((tensor > 1).any()):
        raise ValueError("preprocessing input must convert to finite pixels in [0, 1]")
    return tensor


def _make_preprocessor(family: str) -> Callable[[Any], torch.Tensor]:
    def preprocess(image: Any) -> torch.Tensor:
        # The published torchvision/WILDS pipelines resize PIL images before
        # ToTensor.  Retain that ordering: resizing an already-quantized tensor
        # is close, but not logit-identical.
        if hasattr(image, "convert"):
            working: Any = image.convert("RGB")
        else:
            working = _raw_float_chw(image)
        if family == "torchvision_resnet50_imagenet1k_v2":
            working = tv_functional.resize(
                working,
                232,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            working = tv_functional.center_crop(working, [224, 224])
        elif family in {"cifar10_augmix_wrn40_2", "wilds_rxrx1_resnet50"}:
            expected_size = tuple(_FAMILIES[family]["input_size"])
            observed_size = (
                (int(working.height), int(working.width)) if hasattr(working, "height") else tuple(working.shape[-2:])
            )
            if observed_size != expected_size:
                raise ValueError(
                    f"{family} preprocessing requires native spatial size {expected_size}; "
                    "silent resampling is forbidden"
                )
        else:
            working = tv_functional.resize(
                working,
                _FAMILIES[family]["input_size"],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        return _raw_float_chw(working).contiguous()

    return preprocess


def get_reference_preprocessing(family: str) -> tuple[Callable[[Any], torch.Tensor], dict[str, Any]]:
    """Return deterministic raw-pixel conversion and its JSON contract."""

    family, config = _require_family(family)
    description: dict[str, Any] = {
        "schema": "kbound_reference_preprocessing_v1",
        "family": family,
        "model_input": "raw_nchw_float_0_1",
        "callable_output": "raw_chw_float32_cpu_0_1",
        "input_size": list(config["input_size"]),
        "conversion": "RGB then float32 scaled to [0,1]",
        "normalization": dict(config["normalization"]),
    }
    if family == "torchvision_resnet50_imagenet1k_v2":
        description.update(
            {
                "resize": {
                    "kind": "shorter_side",
                    "size": 232,
                    "interpolation": "bilinear",
                    "antialias": True,
                },
                "crop": {"kind": "center", "size": [224, 224]},
                "lineage": "torchvision.ResNet50_Weights.IMAGENET1K_V2",
            }
        )
    elif family in {"cifar10_augmix_wrn40_2", "wilds_rxrx1_resnet50"}:
        description.update({"resize": {"kind": "none"}, "crop": {"kind": "none"}})
    else:
        description.update(
            {
                "resize": {
                    "kind": "exact",
                    "size": list(config["input_size"]),
                    "interpolation": "bilinear",
                    "antialias": True,
                },
                "crop": {"kind": "none"},
            }
        )
    if family == "cifar10_augmix_wrn40_2":
        description["lineage"] = "RobustBench Hendrycks2020AugMix_WRN input wrapper"
    elif family.startswith("wilds_"):
        description["lineage"] = "WILDS published reference image_base eval transform"
    json.dumps(description)
    return _make_preprocessor(family), description
