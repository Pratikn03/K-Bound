"""kbound_edge.model -- MobileNetV3-Small inspection model (frozen base f0).

A small, CPU-friendly classifier for camera-based inspection: torchvision's
``mobilenet_v3_small`` with the final classifier layer replaced by a 4-class
inspection head.  The base model is the FROZEN reference f0 -- it is never
adapted in place; adaptation always happens on a deep copy (see
:mod:`kbound_edge.tent_adapter`).

Design choices
--------------
* ``weights=None`` by default: no network download, fully offline and
  deterministic given a seed.  Pass ``pretrained=True`` to use torchvision's
  ImageNet weights when you have connectivity and want a stronger init.
* :func:`state_dict_hash` produces a stable content hash of the parameters used
  as the ``model_version`` in every log record and by the candidate-isolation
  test.

torch is imported lazily so that importing the package on a torch-less machine
does not fail; only calling these functions requires torch.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np

#: Default 4-class inspection label set (generic; override via config).
DEFAULT_CLASS_NAMES: tuple[str, ...] = ("ok", "defect", "empty", "misaligned")
NUM_CLASSES: int = 4
IMAGE_SIZE: int = 224


def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "kbound_edge.model requires PyTorch + torchvision. "
            "On the host:  source ~/.venv_wilds/bin/activate  (torch already present)"
        ) from exc


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    seed: int = 0,
    device: str = "cpu",
):
    """Build a MobileNetV3-Small with a fresh ``num_classes`` head.

    Parameters
    ----------
    num_classes : int, default=4
        Size of the inspection head.
    pretrained : bool, default=False
        If True, load torchvision ImageNet weights (requires download);
        otherwise random init (offline, deterministic with ``seed``).
    seed : int, default=0
        Manual seed for reproducible random init / head init.
    device : str, default="cpu"

    Returns
    -------
    model : torch.nn.Module  (in eval mode)
    """
    torch = _torch()
    import torch.nn as nn
    import torchvision

    torch.manual_seed(seed)
    weights = None
    if pretrained:
        try:
            weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        except Exception:
            weights = None
    model = torchvision.models.mobilenet_v3_small(weights=weights)

    # Replace the final Linear with a fresh num_classes head.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    nn.init.xavier_uniform_(model.classifier[-1].weight)
    nn.init.zeros_(model.classifier[-1].bias)

    model.to(device)
    model.eval()
    return model


def predict_logits(model, x):
    """Forward pass returning logits (no grad).  ``x`` is an (N,3,H,W) tensor."""
    torch = _torch()
    model.eval()
    with torch.no_grad():
        return model(x)


def predict_proba(model, x) -> np.ndarray:
    """Forward pass returning softmax probabilities as a numpy (N, C) array."""
    torch = _torch()
    logits = predict_logits(model, x)
    return torch.softmax(logits, dim=1).cpu().numpy()


def state_dict_hash(model) -> str:
    """Stable content hash of all model parameters + buffers (the model_version).

    Deterministic across processes: hashes the sorted (name -> bytes) of every
    tensor in ``state_dict()``.  Used as ``model_version`` in logs and by
    :mod:`tests.test_candidate_isolation` to prove f0 is untouched.
    """
    h = hashlib.sha256()
    sd = model.state_dict()
    for name in sorted(sd.keys()):
        t = sd[name]
        h.update(name.encode("utf-8"))
        arr = t.detach().cpu().numpy()
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def bn_affine_param_names(model) -> Sequence[str]:
    """Names of the BatchNorm affine (weight/bias) parameters -- the TENT params."""
    torch = _torch()
    import torch.nn as nn

    names = []
    for mod_name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:
                names.append(f"{mod_name}.weight")
            if m.bias is not None:
                names.append(f"{mod_name}.bias")
    return names


def train_classifier(model, X, y, epochs: int = 18, lr: float = 2e-3,
                     batch_size: int = 64, seed: int = 0, verbose: bool = False):
    """Standard supervised training loop for f0 (OFFLINE, labelled).  Trains in place.

    ``X`` is an (N,3,H,W) tensor, ``y`` an (N,) long tensor.  Returns the model in
    eval mode.  (Call :func:`recalibrate_bn` afterwards to fix BN running stats.)
    """
    torch = _torch()
    import torch.nn as nn

    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    last = float("nan")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            last = float(loss)
        if verbose:
            print(f"  epoch {ep:2d}  loss={last:.4f}")
    model.eval()
    return model


def recalibrate_bn(model, X, passes: int = 5):
    """Reset and re-estimate BatchNorm running statistics over ``X``.

    A freshly trained small net often classifies perfectly in train mode (batch
    BN stats) yet collapses to chance in eval mode because its running stats are
    poorly estimated.  Running a few no-grad train-mode passes over a
    representative clean batch fixes the running stats so the FROZEN model is a
    competent eval-mode baseline.  Returns the model in eval mode.
    """
    torch = _torch()
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average over the passes
    model.train()
    with torch.no_grad():
        for _ in range(max(1, passes)):
            model(X)
    model.eval()
    return model


class ClipRecord:
    """Represents a metadata record for a captured physical clip."""
    def __init__(self, record_dict):
        self.clip_id = record_dict.get("clip_id")
        self.session_id = record_dict.get("session_id")
        self.phone_id = record_dict.get("phone_id")
        self.object_id = record_dict.get("object_id")
        self.class_id = record_dict.get("class_id")
        self.shift_id = record_dict.get("shift_id")
        self.repetition = record_dict.get("repetition")
        self.sha256 = record_dict.get("sha256")
        self.frame_count = record_dict.get("frame_count")
        self.captured_at = record_dict.get("captured_at")


def source_datasets(manifest_or_path) -> tuple[list[ClipRecord], list[ClipRecord]]:
    """Filter train (S01) and val (S02) datasets from a manifest path or dictionary."""
    import json
    from pathlib import Path
    
    if isinstance(manifest_or_path, (str, Path)):
        with open(manifest_or_path) as f:
            manifest = json.load(f)
    else:
        manifest = manifest_or_path
        
    clips = [ClipRecord(c) for c in manifest.get("clips", [])]
    train = [c for c in clips if c.session_id == "S01"]
    val = [c for c in clips if c.session_id == "S02"]
    return train, val

