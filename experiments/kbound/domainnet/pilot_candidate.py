"""Label-free development-only source-BN-restored entropy probe (not online Tent).

The caller owns source identity, RGB preprocessing, U/V/E separation and custody.
No API accepts labels, beta or a population radius. Models remain on their supplied
device. Ordered CHW images or NCHW chunks are rebatched into 32 rows, including the
last short batch. Counts are mandatory to prevent silently incomplete windows.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from experiments.kbound.domainnet.train_source import tensor_hash

FEATURE_NAMES = (
    "frozen_entropy_V",
    "candidate_entropy_V",
    "prediction_disagreement_V",
    "normalized_bn_affine_update",
)


@dataclass(frozen=True)
class _Snapshot:
    state_sha256: str
    configuration: tuple[Any, ...]


@dataclass(frozen=True)
class CandidateResult:
    model: nn.Module
    count: int
    batches: int
    class_count: int
    normalized_bn_affine_update: float
    source_state_sha256: str
    candidate_state_sha256: str
    adaptation_seconds: float
    source_snapshot: _Snapshot
    candidate_snapshot: _Snapshot


@dataclass(frozen=True)
class PairPredictions:
    count: int
    batches: int
    class_count: int
    frozen_predictions: tuple[int, ...]
    candidate_predictions: tuple[int, ...]
    frozen_entropy_sum: float
    candidate_entropy_sum: float
    source_state_sha256: str
    candidate_state_sha256: str
    inference_seconds: float

    @property
    def frozen_entropy_mean(self) -> float:
        return self.frozen_entropy_sum / self.count

    @property
    def candidate_entropy_mean(self) -> float:
        return self.candidate_entropy_sum / self.count


def image_transform(image: Image.Image) -> torch.Tensor:
    """The frozen deterministic PIL RGB/ImageNet transform for every window."""
    if not isinstance(image, Image.Image):
        raise TypeError("expected a PIL image")
    value: torch.Tensor = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(image.convert("RGB"))
    return value


def _tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    # named_buffers includes nonpersistent buffers omitted by state_dict.
    return {
        **{f"parameter:{k}": v for k, v in model.named_parameters()},
        **{f"buffer:{k}": v for k, v in model.named_buffers()},
    }


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"nonfinite {name}")


def _snapshot(model: nn.Module) -> _Snapshot:
    configuration = []
    for name, layer in model.named_modules():
        entry: tuple[Any, ...] = (name, type(layer).__module__, type(layer).__qualname__, layer.training)
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            entry += (
                layer.eps,
                layer.momentum,
                layer.track_running_stats,
                layer.affine,
                layer.num_features,
                tuple((k, v is None) for k, v in layer._buffers.items()),
            )
        configuration.append(entry)
    configuration.extend((name, p.requires_grad, str(p.device)) for name, p in model.named_parameters())
    configuration.extend((name, str(b.device)) for name, b in model.named_buffers())
    return _Snapshot(tensor_hash(_tensors(model)), tuple(configuration))


def _unchanged(model: nn.Module, expected: _Snapshot) -> None:
    if _snapshot(model) != expected:
        raise ValueError("model state or mode/BN configuration mutated")


def _validate_model(model: nn.Module) -> dict[str, nn.BatchNorm2d]:
    if not isinstance(model, nn.Module):
        raise TypeError("expected nn.Module")
    tensors = _tensors(model)
    if not tensors or len({t.device for t in tensors.values()}) != 1:
        raise ValueError("model must have tensors on one backend")
    for name, value in tensors.items():
        _finite(value, name)
    if any(layer.training for layer in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("source/deployment model must be frozen in eval mode")
    layers = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            if type(layer) is not nn.BatchNorm2d:
                raise ValueError("only ordinary BatchNorm2d is supported")
            if not layer.affine or layer.weight is None or layer.bias is None or not layer.track_running_stats:
                raise ValueError("BatchNorm2d requires tracked statistics and affine parameters")
            if (
                layer.running_mean is None
                or layer.running_var is None
                or layer.num_batches_tracked is None
                or layer.running_mean.shape != (layer.num_features,)
                or layer.running_var.shape != (layer.num_features,)
                or layer.weight.shape != (layer.num_features,)
                or layer.bias.shape != (layer.num_features,)
                or layer.num_batches_tracked.shape != ()
                or layer.num_batches_tracked.dtype != torch.int64
                or int(layer.num_batches_tracked) < 0
                or bool((layer.running_var < 0).any())
                or not math.isfinite(layer.eps)
                or layer.eps <= 0
                or (layer.momentum is not None and (not math.isfinite(layer.momentum) or not 0 <= layer.momentum <= 1))
            ):
                raise ValueError("invalid source BatchNorm2d statistics/configuration")
            layers[name] = layer
    if not layers:
        raise ValueError("model has no BatchNorm2d affine parameters")
    return layers


def _batches(images: Iterable[torch.Tensor], expected_count: int, device: torch.device) -> Iterator[torch.Tensor]:
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    pending = []
    count = 0
    shape = None
    dtype = None
    for chunk in images:
        if not isinstance(chunk, torch.Tensor):
            raise TypeError("image tensors only; tuple/image-label batches are forbidden")
        if not chunk.is_floating_point() or chunk.ndim not in (3, 4):
            raise ValueError("images must be floating CHW or NCHW tensors")
        chunk = chunk.unsqueeze(0) if chunk.ndim == 3 else chunk
        if chunk.shape[0] == 0 or chunk.shape[1] != 3 or min(chunk.shape[2:]) <= 0:
            raise ValueError("invalid nonempty RGB image shape")
        if chunk.device != device:
            raise ValueError("images must remain on the supplied model backend")
        if shape is None:
            shape, dtype = chunk.shape[1:], chunk.dtype
        if chunk.shape[1:] != shape or chunk.dtype != dtype:
            raise ValueError("image shapes and dtypes must be consistent")
        _finite(chunk, "images")
        count += len(chunk)
        if count > expected_count:
            raise ValueError("image count exceeds expected_count")
        for row in chunk:
            pending.append(row)
            if len(pending) == 32:
                yield torch.stack(pending)
                pending = []
    if count != expected_count:
        raise ValueError("image count differs from expected_count")
    if pending:
        yield torch.stack(pending)


def _entropy(logits: Any, rows: int, classes: int | None) -> tuple[torch.Tensor, int]:
    if (
        not isinstance(logits, torch.Tensor)
        or not logits.is_floating_point()
        or logits.ndim != 2
        or logits.shape[0] != rows
        or logits.shape[1] < 2
    ):
        raise ValueError("invalid logit shape/type/count")
    if classes is not None and logits.shape[1] != classes:
        raise ValueError("logit class count changed or differs between models")
    _finite(logits, "logits")
    entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    _finite(entropy, "entropy")
    if bool((entropy < 0).any()):
        raise ValueError("negative entropy")
    return entropy, logits.shape[1]


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def normalized_bn_affine_update(source: nn.Module, candidate: nn.Module) -> float:
    """BN-affine L2 change / max(source BN-affine L2, 1e-12)."""
    left, right = _validate_model(source), _validate_model(candidate)
    if left.keys() != right.keys():
        raise ValueError("BN affine layout differs")
    source_squared = change_squared = 0.0
    for name, layer in left.items():
        for key in ("weight", "bias"):
            a, b = getattr(layer, key), getattr(right[name], key)
            if a.shape != b.shape:
                raise ValueError("BN affine shape differs")
            # CPU double diagnostics avoid float32 squared-sum overflow and MPS float64.
            a64, b64 = a.detach().cpu().double(), b.detach().cpu().double()
            source_squared += float(a64.square().sum())
            change_squared += float((b64 - a64).square().sum())
    result = math.sqrt(change_squared) / max(math.sqrt(source_squared), 1e-12)
    if not math.isfinite(result) or result < 0:
        raise ValueError("nonfinite/negative BN update diagnostic")
    return result


def create_candidate(source: nn.Module, images: Iterable[torch.Tensor], *, expected_count: int) -> CandidateResult:
    """Fresh source copy and Adam; one U-only entropy pass, then restore source BN."""
    source_layers = _validate_model(source)
    source_snapshot = _snapshot(source)
    device = next(source.parameters()).device
    candidate = copy.deepcopy(source)
    _unchanged(source, source_snapshot)
    source_storage = {t.untyped_storage().data_ptr() for t in _tensors(source).values() if t.numel()}
    if {id(m) for m in source.modules()} & {id(m) for m in candidate.modules()} or any(
        t.numel() and t.untyped_storage().data_ptr() in source_storage for t in _tensors(candidate).values()
    ):
        raise ValueError("candidate deepcopy must have independent modules and tensor storage")
    layers = dict(candidate.named_modules())
    affine = []
    affine_names: set[str] = set()
    for name in source_layers:
        layer = layers[name]
        layer.train()
        layer.requires_grad_(True)
        layer.track_running_stats = False
        layer.running_mean = layer.running_var = layer.num_batches_tracked = None
        affine.extend([layer.weight, layer.bias])
        affine_names.update(f"parameter:{name + '.' if name else ''}{key}" for key in ("weight", "bias"))
    optimizer = torch.optim.Adam(affine, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    count = batches = 0
    classes = None
    _sync(device)
    started = time.perf_counter()
    try:
        for batch in _batches(images, expected_count, device):
            optimizer.zero_grad(set_to_none=True)
            entropy, classes = _entropy(candidate(batch), len(batch), classes)
            loss = entropy.mean()
            _finite(loss, "loss")
            loss.backward()
            for parameter in affine:
                if parameter.grad is None:
                    raise ValueError("missing BN affine gradient")
                _finite(parameter.grad, "gradient")
            optimizer.step()
            for name, value in _tensors(candidate).items():
                _finite(value, name)
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        _finite(value, "optimizer state")
            count += len(batch)
            batches += 1
        for name, original in source_layers.items():
            layer = layers[name]
            for key in ("running_mean", "running_var", "num_batches_tracked"):
                setattr(layer, key, getattr(original, key).detach().clone())
            layer.track_running_stats = original.track_running_stats
            layer.momentum, layer.eps = original.momentum, original.eps
        candidate.requires_grad_(False).eval()
        for parameter in candidate.parameters():
            parameter.grad = None
        _validate_model(candidate)
        source_tensors, final = _tensors(source), _tensors(candidate)
        if source_tensors.keys() != final.keys() or any(
            tensor_hash({name: value}) != tensor_hash({name: final[name]})
            for name, value in source_tensors.items()
            if name not in affine_names
        ):
            raise ValueError("non-affine parameter or buffer mutated during adaptation")
        candidate_snapshot = _snapshot(candidate)
        if candidate_snapshot.configuration != source_snapshot.configuration:
            raise ValueError("model mode/BN configuration mutated during adaptation")
        update = normalized_bn_affine_update(source, candidate)
        _sync(device)
        elapsed = time.perf_counter() - started
    finally:
        _unchanged(source, source_snapshot)
    assert classes is not None
    return CandidateResult(
        candidate,
        count,
        batches,
        classes,
        update,
        source_snapshot.state_sha256,
        candidate_snapshot.state_sha256,
        elapsed,
        source_snapshot,
        candidate_snapshot,
    )


def predict_pair(
    source: nn.Module, candidate: CandidateResult | nn.Module, images: Iterable[torch.Tensor], *, expected_count: int
) -> PairPredictions:
    """Fixed deployment predictions for a separately supplied V or E image window.

    Pass CandidateResult in orchestration: its creation snapshots additionally bind
    both models across calls. Bare frozen modules support independent synthetic tests.
    """
    classes = None
    if isinstance(candidate, CandidateResult):
        _unchanged(source, candidate.source_snapshot)
        _unchanged(candidate.model, candidate.candidate_snapshot)
        if (
            candidate.source_state_sha256 != candidate.source_snapshot.state_sha256
            or candidate.candidate_state_sha256 != candidate.candidate_snapshot.state_sha256
        ):
            raise ValueError("candidate receipt state hashes mutated")
        if type(candidate.class_count) is not int or candidate.class_count < 2:
            raise ValueError("invalid candidate class count")
        classes = candidate.class_count
        candidate = candidate.model
    _validate_model(source)
    _validate_model(candidate)
    snapshots = (_snapshot(source), _snapshot(candidate))
    device = next(source.parameters()).device
    if next(candidate.parameters()).device != device:
        raise ValueError("pair models must use the same backend")
    frozen_predictions: list[int] = []
    candidate_predictions: list[int] = []
    frozen_sum = candidate_sum = 0.0
    count = batches = 0
    _sync(device)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for batch in _batches(images, expected_count, device):
                frozen_logits = source(batch)
                frozen_entropy, classes = _entropy(frozen_logits, len(batch), classes)
                candidate_logits = candidate(batch)
                candidate_entropy, classes = _entropy(candidate_logits, len(batch), classes)
                frozen_predictions.extend(frozen_logits.argmax(1).cpu().tolist())
                candidate_predictions.extend(candidate_logits.argmax(1).cpu().tolist())
                frozen_sum += float(frozen_entropy.cpu().double().sum())
                candidate_sum += float(candidate_entropy.cpu().double().sum())
                count += len(batch)
                batches += 1
                _unchanged(source, snapshots[0])
                _unchanged(candidate, snapshots[1])
        _sync(device)
        elapsed = time.perf_counter() - started
    finally:
        _unchanged(source, snapshots[0])
        _unchanged(candidate, snapshots[1])
    assert classes is not None
    result = PairPredictions(
        count,
        batches,
        classes,
        tuple(frozen_predictions),
        tuple(candidate_predictions),
        frozen_sum,
        candidate_sum,
        snapshots[0].state_sha256,
        snapshots[1].state_sha256,
        elapsed,
    )
    extract_features(result, 0.0)
    return result


def extract_features(
    predictions: PairPredictions, normalized_bn_affine_update: float
) -> tuple[float, float, float, float]:
    """Validate summaries and return the frozen four-feature order, using every row."""
    p = predictions
    if (
        type(p.count) is not int
        or p.count <= 0
        or type(p.batches) is not int
        or p.batches != math.ceil(p.count / 32)
        or type(p.class_count) is not int
        or p.class_count < 2
        or len(p.frozen_predictions) != p.count
        or len(p.candidate_predictions) != p.count
    ):
        raise ValueError("invalid prediction counts")
    if any(type(v) is not int or not 0 <= v < p.class_count for v in (*p.frozen_predictions, *p.candidate_predictions)):
        raise ValueError("invalid argmax predictions")
    values = (p.frozen_entropy_sum, p.candidate_entropy_sum, normalized_bn_affine_update, p.inference_seconds)
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("nonfinite or negative feature diagnostics")
    disagreement = sum(a != b for a, b in zip(p.frozen_predictions, p.candidate_predictions)) / p.count
    return p.frozen_entropy_mean, p.candidate_entropy_mean, disagreement, normalized_bn_affine_update
