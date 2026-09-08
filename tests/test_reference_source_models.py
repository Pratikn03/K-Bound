from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet50_Weights

from experiments.kbound.reference_source import (
    CheckpointIdentityError,
    CheckpointSchemaError,
    UnsupportedReferenceFamilyError,
    get_reference_preprocessing,
    load_reference_model,
)
from experiments.kbound.reference_source import models as bridge


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _TinyResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=(2, 3)))


class _TinyBatchNormResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(x).mean(dim=(2, 3)))


class _CaptureInputModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x
        return x


def _wilds_checkpoint(path: Path, num_classes: int, **extra_state: torch.Tensor) -> Path:
    model = _TinyResNet(num_classes)
    state = {f"model.{name}": value for name, value in model.state_dict().items()}
    state.update(extra_state)
    torch.save({"algorithm": state, "epoch": 3, "best_val_metric": 0.5}, path)
    return path


def _rxrx_spec(checkpoint: Path) -> dict[str, str]:
    return {
        "family": "wilds_rxrx1_resnet50",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }


def test_missing_checkpoint_is_a_hard_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pth"
    spec = {
        "family": "wilds_rxrx1_resnet50",
        "checkpoint_path": str(missing),
        "checkpoint_sha256": "0" * 64,
    }

    with pytest.raises(FileNotFoundError, match="checkpoint.*missing"):
        load_reference_model(spec, "cpu")


def test_altered_checkpoint_bytes_fail_before_torch_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"authenticated bytes")
    expected = _sha256(checkpoint)
    checkpoint.write_bytes(b"altered bytes")
    monkeypatch.setattr(
        bridge.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("hash mismatch must precede torch.load"),
    )

    with pytest.raises(CheckpointIdentityError, match="SHA-256 mismatch"):
        load_reference_model(
            {
                "family": "wilds_rxrx1_resnet50",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": expected,
            },
            "cpu",
        )


def test_cifar_constructor_hash_is_mandatory_and_verified(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    constructor = tmp_path / "wide_resnet.py"
    constructor.write_text("class WideResNet: pass\n", encoding="utf-8")

    with pytest.raises(CheckpointIdentityError, match="constructor SHA-256 mismatch"):
        load_reference_model(
            {
                "family": "cifar10_augmix_wrn40_2",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "constructor_path": str(constructor),
                "constructor_sha256": "f" * 64,
            },
            "cpu",
        )


def test_cifar_rejects_arbitrary_constructor_even_when_caller_hash_matches(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    constructor = tmp_path / "wide_resnet.py"
    constructor.write_text("class WideResNet: pass\n", encoding="utf-8")

    with pytest.raises(CheckpointIdentityError, match="selected pinned.*constructor"):
        load_reference_model(
            {
                "family": "cifar10_augmix_wrn40_2",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "constructor_path": str(constructor),
                "constructor_sha256": _sha256(constructor),
            },
            "cpu",
        )


def test_wrong_classifier_dimensions_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _wilds_checkpoint(tmp_path / "wrong-head.pth", 1138)
    monkeypatch.setattr(bridge, "_construct_model", lambda family, spec: _TinyResNet(1139))

    with pytest.raises(CheckpointSchemaError, match="classifier.*1139"):
        load_reference_model(_rxrx_spec(checkpoint), "cpu")


def test_nonfinite_learned_weight_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _wilds_checkpoint(tmp_path / "nan.pth", 1139)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["algorithm"]["model.fc.weight"][0, 0] = float("nan")
    torch.save(payload, checkpoint)
    monkeypatch.setattr(bridge, "_construct_model", lambda family, spec: _TinyResNet(1139))

    with pytest.raises(CheckpointSchemaError, match="non-finite.*fc.weight"):
        load_reference_model(_rxrx_spec(checkpoint), "cpu")


def test_unexpected_learned_weight_fails_strict_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _wilds_checkpoint(tmp_path / "unexpected.pth", 1139, **{"model.unreviewed.weight": torch.ones(1)})
    monkeypatch.setattr(bridge, "_construct_model", lambda family, spec: _TinyResNet(1139))

    with pytest.raises(CheckpointSchemaError, match="unexpected.*unreviewed.weight"):
        load_reference_model(_rxrx_spec(checkpoint), "cpu")


def test_missing_learned_weight_fails_strict_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _wilds_checkpoint(tmp_path / "missing-weight.pth", 1139)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del payload["algorithm"]["model.fc.bias"]
    torch.save(payload, checkpoint)
    monkeypatch.setattr(bridge, "_construct_model", lambda family, spec: _TinyResNet(1139))

    with pytest.raises(CheckpointSchemaError, match="missing.*fc.bias"):
        load_reference_model(_rxrx_spec(checkpoint), "cpu")


def test_legacy_missing_batch_norm_counter_is_synthesized_but_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _TinyBatchNormResNet(1139)
    state = {f"model.{name}": value for name, value in source.state_dict().items() if name != "bn.num_batches_tracked"}
    checkpoint = tmp_path / "legacy-bn.pth"
    torch.save({"algorithm": state}, checkpoint)
    monkeypatch.setattr(bridge, "_construct_model", lambda family, spec: _TinyBatchNormResNet(1139))

    _, provenance = load_reference_model(_rxrx_spec(checkpoint), "cpu")

    assert provenance["state_compatibility"] == {"synthesized_missing_num_batches_tracked": ["bn.num_batches_tracked"]}


def test_safe_load_and_wilds_namespace_conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _wilds_checkpoint(tmp_path / "valid.pth", 1139)
    original_load = bridge.torch.load
    calls: list[dict] = []

    def checked_load(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(bridge.torch, "load", checked_load)
    monkeypatch.setattr(bridge, "_construct_model", lambda family, spec: _TinyResNet(1139))

    model, provenance = load_reference_model(_rxrx_spec(checkpoint), "cpu")

    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert model.training is False
    assert next(model.parameters()).device.type == "cpu"
    assert provenance["family"] == "wilds_rxrx1_resnet50"
    assert provenance["checkpoint_sha256"] == _sha256(checkpoint)
    assert provenance["checkpoint_namespace"] == "algorithm/model."
    json.dumps(provenance)


def test_model_construction_preserves_caller_rng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _wilds_checkpoint(tmp_path / "valid.pth", 1139)
    monkeypatch.setattr(bridge, "_construct_model", lambda family, spec: _TinyResNet(1139))
    torch.manual_seed(727)
    before = torch.random.get_rng_state().clone()

    load_reference_model(_rxrx_spec(checkpoint), "cpu")

    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize(
    ("family", "input_size", "normalization"),
    [
        ("cifar10_augmix_wrn40_2", [32, 32], "fixed_channel"),
        ("torchvision_resnet50_imagenet1k_v2", [224, 224], "fixed_channel"),
        ("wilds_camelyon17_densenet121", [96, 96], "fixed_channel"),
        ("wilds_rxrx1_resnet50", [256, 256], "per_image_channel"),
        ("wilds_iwildcam_resnet50", [448, 448], "fixed_channel"),
    ],
)
def test_preprocessing_contract_is_json_serializable_and_returns_raw_cpu_float(
    family: str, input_size: list[int], normalization: str
) -> None:
    preprocess, description = get_reference_preprocessing(family)
    source_size = (
        input_size
        if family
        in {
            "cifar10_augmix_wrn40_2",
            "wilds_rxrx1_resnet50",
        }
        else [513, 617]
    )
    source = Image.fromarray(np.full((*source_size, 3), 128, dtype=np.uint8), mode="RGB")

    tensor = preprocess(source)

    assert list(tensor.shape) == [3, *input_size]
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    assert description["model_input"] == "raw_nchw_float_0_1"
    assert description["input_size"] == input_size
    assert description["normalization"]["kind"] == normalization
    assert description["normalization"]["location"] == "model_wrapper"
    json.dumps(description)


def test_imagenet_v2_preprocessing_is_exactly_declared() -> None:
    _, description = get_reference_preprocessing("torchvision_resnet50_imagenet1k_v2")

    assert description["resize"] == {
        "kind": "shorter_side",
        "size": 232,
        "interpolation": "bilinear",
        "antialias": True,
    }
    assert description["crop"] == {"kind": "center", "size": [224, 224]}
    assert description["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert description["normalization"]["std"] == [0.229, 0.224, 0.225]
    assert description["lineage"] == "torchvision.ResNet50_Weights.IMAGENET1K_V2"


def test_imagenet_v2_split_normalization_matches_upstream_transform() -> None:
    source = Image.fromarray(
        np.random.default_rng(8).integers(0, 256, size=(301, 407, 3), dtype=np.uint8),
        mode="RGB",
    )
    preprocess, description = get_reference_preprocessing("torchvision_resnet50_imagenet1k_v2")

    raw = preprocess(source)
    normalization = description["normalization"]
    mean = torch.tensor(normalization["mean"]).view(3, 1, 1)
    std = torch.tensor(normalization["std"]).view(3, 1, 1)
    observed = (raw - mean) / std
    expected = ResNet50_Weights.IMAGENET1K_V2.transforms()(source)

    assert torch.equal(observed, expected)


@pytest.mark.parametrize("family", ["cifar10_augmix_wrn40_2", "wilds_rxrx1_resnet50"])
def test_native_size_families_do_not_silently_resample(family: str) -> None:
    preprocess, description = get_reference_preprocessing(family)

    with pytest.raises(ValueError, match="native spatial size"):
        preprocess(torch.zeros(3, 19, 23))

    assert description["resize"] == {"kind": "none"}


def test_rxrx1_uses_sample_std_inside_model_not_imagenet_normalization() -> None:
    capture = _CaptureInputModule()
    wrapped = bridge._wrap_model(capture, "wilds_rxrx1_resnet50")
    tile = torch.tensor(
        [[[[0.0, 0.2], [0.4, 0.6]], [[0.1, 0.3], [0.5, 0.7]], [[0.2, 0.4], [0.6, 0.8]]]],
        dtype=torch.float32,
    )
    raw = tile.repeat(1, 1, 128, 128)

    wrapped(raw)

    expected_mean = raw.mean(dim=(2, 3), keepdim=True)
    expected_std = raw.std(dim=(2, 3), correction=1, keepdim=True)
    assert torch.allclose(capture.last_input, (raw - expected_mean) / expected_std)


@pytest.mark.parametrize("family", ["pacs_domainbed", "officehome_shot", "generic_resnet50"])
def test_unimplemented_families_have_no_generic_fallback(family: str) -> None:
    with pytest.raises(UnsupportedReferenceFamilyError, match=family):
        get_reference_preprocessing(family)
