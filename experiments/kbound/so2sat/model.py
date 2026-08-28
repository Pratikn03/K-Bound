"""Native-resolution Sentinel-2 source model and tensor identities.

This module intentionally contains no data-loading, adaptation, or target-
scoring code.  The only production architecture is a full-network ResNet-18
whose stem is appropriate for the 10-band, 32-by-32 So2Sat LCZ42 patches.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torchvision.models import resnet18

from .integrity import IntegrityError, canonical_json_bytes


INPUT_CHANNELS = 10
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
NUM_CLASSES = 17
CANONICAL_MODEL_SEEDS = (0, 1, 2, 3, 4)
ARCHITECTURE_ID = "torchvision_resnet18_sen2_10band_native32_v1"


def architecture_spec() -> dict[str, Any]:
    """Return the exact, JSON-serializable model contract."""

    return {
        "architecture_id": ARCHITECTURE_ID,
        "implementation": "torchvision.models.resnet18",
        "torchvision_weights": None,
        "input_shape_chw": [INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH],
        "stem": {
            "in_channels": INPUT_CHANNELS,
            "out_channels": 64,
            "kernel_size": [3, 3],
            "stride": [1, 1],
            "padding": [1, 1],
            "bias": False,
        },
        "maxpool": "identity",
        "outputs": NUM_CLASSES,
        "optimization_scope": "full_network",
        "initialization": "independent_torchvision_kaiming_per_model_seed",
    }


def assert_model_contract(model: nn.Module) -> None:
    """Fail closed if a model differs from the sealed source architecture."""

    conv1 = getattr(model, "conv1", None)
    if not isinstance(conv1, nn.Conv2d):
        raise IntegrityError("So2Sat ResNet-18 must expose a Conv2d conv1 stem")
    observed_stem = {
        "in_channels": conv1.in_channels,
        "out_channels": conv1.out_channels,
        "kernel_size": tuple(conv1.kernel_size),
        "stride": tuple(conv1.stride),
        "padding": tuple(conv1.padding),
        "bias": conv1.bias is not None,
    }
    expected_stem = {
        "in_channels": INPUT_CHANNELS,
        "out_channels": 64,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "bias": False,
    }
    if observed_stem != expected_stem:
        raise IntegrityError(f"So2Sat ResNet-18 stem drift: {observed_stem!r}")
    if not isinstance(getattr(model, "maxpool", None), nn.Identity):
        raise IntegrityError("So2Sat ResNet-18 must remove the ImageNet max-pool")
    classifier = getattr(model, "fc", None)
    if not isinstance(classifier, nn.Linear) or classifier.out_features != NUM_CLASSES:
        raise IntegrityError("So2Sat ResNet-18 must have exactly 17 classifier outputs")
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if frozen:
        raise IntegrityError(f"So2Sat source training must optimize the full network; frozen={frozen}")


def build_so2sat_resnet18() -> nn.Module:
    """Build one randomly initialized 10-band native-resolution ResNet-18.

    ``weights=None`` is deliberate: construction never downloads a checkpoint,
    and each of the five model seeds receives an independent initialization.
    The caller must seed PyTorch before invoking this function.
    """

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        INPUT_CHANNELS,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    assert_model_contract(model)
    return model


def clone_cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detach and clone a model state so later optimization cannot mutate it."""

    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and exact contiguous CPU bytes.

    The byte view also works for dtypes that NumPy cannot represent directly,
    such as bfloat16.  Sorting names makes the identity independent of mapping
    insertion order while retaining every parameter and persistent buffer.
    """

    if not state:
        raise IntegrityError("cannot hash an empty tensor state")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not name:
            raise IntegrityError("tensor-state names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise IntegrityError(f"state entry {name!r} is not a tensor")
        if tensor.is_sparse or tensor.is_quantized:
            raise IntegrityError(f"state entry {name!r} must be a dense, non-quantized tensor")
        cpu = tensor.detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(cpu.dtype),
            "shape": list(cpu.shape),
            "numel": cpu.numel(),
        }
        encoded = canonical_json_bytes(header)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()
