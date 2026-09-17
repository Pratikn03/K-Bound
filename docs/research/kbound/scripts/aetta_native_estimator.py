"""Authenticated AETTA estimator function-extraction harness, CPU only.

This is not execution of upstream ``main.py`` or its complete learner loop.
The selected upstream function bodies, TorchScript helpers, normalization,
and four-stage-dropout ResNet-50 architecture execute unchanged. Extracting
definitions avoids upstream DNN constructor side effects and its unconditional
CUDA selection; it does not globally patch CUDA or replace native equations.

Callers own checkpoint authentication, image preprocessing before normalization,
the adaptation trajectory, episode boundaries, RNG, timing, and any conversion
from an accuracy estimate to a routing decision. An estimator is not a gate.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import types

SOURCE_COMMIT = "ce472b20b8d8d8228599a7b93602ea0e89e6d9a3"
SOURCE_SHA256 = {
    "learner/dnn.py": "c410ab64637e82dde27e0cf67898cd9708e0a26f346cb6229f5d8556f1e1dd13",
    "models/ResNet.py": "d57babb5f4dbf906f46e3679d8d25e93468b8c0bec59f9eff68ae855f0e6823e",
    "utils/loss_functions.py": "9ab80487d6136b92cbb5aed93377ba41f0f550adc8024f492ca0af7eb443ed50",
    "utils/normalize_layer.py": "6a50651da1f2540b1dd66a14d1255c25a6c90043dd99c9fc18b27363e38654e0",
    "main.py": "88cbc355afbcb9a972b8c0ea468ab3a0bf14ac3757352c823c828f6c716c5d5e",
}
METHODS = ("model_inference", "evaluate_dropout", "aetta")
STATE_KEYS = (
    "est_dropout", "est_dropout_avg_entropy", "est_dropout_softmax_mean",
    "est_dropout_softmax_std", "aetta",
)


def authenticate_source(source: Path, auth_receipt: Path | None = None) -> dict[str, bytes]:
    """Fail before native imports unless current bytes equal fixed pinned Git bytes."""
    source = Path(source).absolute()
    if source.is_symlink():
        raise ValueError(f"Expected regular, non-symlink source directory: {source}")
    # Canonicalize platform aliases such as macOS /var -> /private/var, while
    # still rejecting a symlink checkout or symlinks within the source tree.
    source = source.resolve()
    payloads = {}
    for name, expected in SOURCE_SHA256.items():
        path = source / name
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError(f"Expected regular, non-symlink source: {path}")
        try:
            metadata = path.stat()
        except FileNotFoundError as exc:
            raise ValueError(f"Required source is missing: {path}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Expected regular, non-symlink source: {path}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"AETTA SHA-256 mismatch: {name}")
        payloads[name] = payload
    if auth_receipt is not None:
        receipt_path = Path(auth_receipt)
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("Expected a regular, non-symlink prepared source authentication receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in {
            "schema": "kbound_pinned_raw_source_auth_v1",
            "origin": "https://github.com/taeckyung/AETTA",
            "commit": SOURCE_COMMIT,
            "method": "https_commit_pinned_raw_files",
        }.items()):
            raise ValueError("AETTA prepared authentication origin, pin or method mismatch")
        files = receipt.get("files")
        if not isinstance(files, dict) or set(files) != set(SOURCE_SHA256):
            raise ValueError("AETTA prepared authentication file inventory mismatch")
        for name, expected in SOURCE_SHA256.items():
            entry = files[name]
            expected_url = f"https://raw.githubusercontent.com/taeckyung/AETTA/{SOURCE_COMMIT}/{name}"
            if (not isinstance(entry, dict) or entry.get("sha256") != expected
                    or entry.get("url") != expected_url):
                raise ValueError(f"AETTA prepared authentication URL or fixed SHA mismatch: {name}")
        # Receipt hashes are not user-selectable trust anchors: every local
        # payload has already matched the independently pinned constants above.
        # Acquisition occurs separately and explicitly, never at inference time.
        return payloads
    git_environment = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0")
    head = subprocess.check_output(
        ["git", "-c", "protocol.allow=never", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=10,
        env=git_environment).strip()
    if head != SOURCE_COMMIT:
        raise ValueError("AETTA source revision mismatch")
    for name, payload in payloads.items():
        pinned = subprocess.check_output(
            ["git", "-c", "protocol.allow=never", "-C", str(source), "show", f"{SOURCE_COMMIT}:{name}"], timeout=10,
            env=git_environment)
        if pinned != payload:
            raise ValueError(f"AETTA working source differs from pinned Git bytes: {name}")
    return payloads


def _execute_selected(payload: bytes, filename: Path, namespace: dict,
                      names: tuple[str, ...], class_name: str | None = None) -> None:
    """Compile original AST nodes, retaining bodies, decorators and source lines."""
    tree = ast.parse(payload, filename=str(filename))
    scope = tree.body
    if class_name is not None:
        classes = [n for n in scope if isinstance(n, ast.ClassDef) and n.name == class_name]
        if len(classes) != 1:
            raise ValueError(f"Expected exactly one native class {class_name}")
        scope = classes[0].body
    selected = []
    for name in names:
        nodes = [n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(nodes) != 1:
            raise ValueError(f"Expected exactly one native function {name}")
        selected.extend(nodes)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(filename), "exec"), namespace)


class NativeAETTASource:
    """Verified source and exact native definitions; construction reads no data."""

    def __init__(self, source: Path, auth_receipt: Path | None = None):
        self.source = Path(source).absolute()
        self.source_bytes = authenticate_source(self.source, auth_receipt=auth_receipt)
        # Lazy imports keep lightweight authentication tests independent of torch.
        import torch
        import torch.nn.functional as F
        import torchvision

        namespace = {
            "__name__": "kbound_native_aetta_extracted",
            "torch": torch, "F": F,
            "conf": types.SimpleNamespace(args=types.SimpleNamespace(
                opt={"indices_in_1k": None}, dataset="imagenetoutdist", dropout_rate=0.5)),
        }
        _execute_selected(self.source_bytes["utils/loss_functions.py"],
                          self.source / "utils/loss_functions.py", namespace,
                          ("calc_energy", "softmax_entropy"))
        _execute_selected(self.source_bytes["learner/dnn.py"],
                          self.source / "learner/dnn.py", namespace, METHODS, "DNN")
        self._native_type = type("ExtractedNativeAETTA", (), {name: namespace[name] for name in METHODS})
        architecture = {"__name__": "kbound_native_aetta_architecture"}
        exec(compile(self.source_bytes["models/ResNet.py"], str(self.source / "models/ResNet.py"),
                     "exec"), architecture)
        normalization = {"__name__": "kbound_native_aetta_normalization"}
        exec(compile(self.source_bytes["utils/normalize_layer.py"],
                     str(self.source / "utils/normalize_layer.py"), "exec"), normalization)
        self._architecture = architecture
        self._normalization = normalization
        self.provenance = {
            "schema": "kbound_native_aetta_extraction_v1",
            "bridge_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": {"python": platform.python_version(),
                        "torch": torch.__version__, "torchvision": torchvision.__version__},
            "upstream_commit": SOURCE_COMMIT,
            "source_sha256": dict(SOURCE_SHA256),
            "source_authentication": "prepared_pinned_raw_source_receipt" if auth_receipt else "pinned_local_git_blobs",
            "local_git_blob_authentication_passed": auth_receipt is None,
            "source_authentication_receipt_sha256": hashlib.sha256(Path(auth_receipt).read_bytes()).hexdigest() if auth_receipt else None,
            "native_functions": ["DNN." + name for name in METHODS],
            "native_helpers": ["calc_energy", "softmax_entropy", "get_normalize_layer"],
            "architecture": "native ResNetDropout50; torchvision BN ResNet-50, 1000 classes",
            "checkpoint_authenticated_by_bridge": False,
            "native_main_executed": False,
            "native_dnn_constructor_executed": False,
            "native_full_learner_loop_executed": False,
            "function_bodies_changed": False,
            "cuda_parity_verified": False,
            "device": "cpu",
            "derivation": "Compile selected unchanged DNN methods and TorchScript helpers; execute native architecture and normalization definitions. Do not import DNN module or invoke its constructor/CUDA selection.",
            "dependency_injection": {"conf.args.dataset": "imagenetoutdist",
                                     "conf.args.dropout_rate": 0.5,
                                     "conf.args.opt.indices_in_1k": None},
            "dropout_draws": 10,
            "dropout_locations": ["layer1", "layer2", "layer3", "layer4"],
            "ema_lifecycle": "retained across estimate calls; a new episode starts with None",
            "input": "raw RGB float32 [0,1], N x 3 x 224 x 224, CPU; native normalization applied once",
            "routing_action_produced": False,
        }

    def create_model(self):
        """Create the full native architecture, with RANDOM weights and no download.

        A benchmark caller must separately authenticate its task-compatible BN
        checkpoint, strictly load it, and record that identity before evaluation.
        """
        return self._architecture["ResNetDropout50"]().cpu().eval()

    def new_episode(self, model):
        """Start fresh estimator state; do not reset or copy the supplied model."""
        import torch

        if (type(model) is not self._architecture["ResNetDropout"]
                or model.fc.out_features != 1000
                or [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)] != [3, 4, 6, 3]
                or model.fc.in_features != 2048):
            raise ValueError("AETTA requires the authenticated native full ResNetDropout50/1000-class model")
        tensors = list(model.parameters()) + list(model.buffers())
        if any(value.device.type != "cpu" for value in tensors):
            raise ValueError("AETTA extraction bridge supports CPU only")
        if any(value.is_floating_point() and value.dtype != torch.float32 for value in tensors):
            raise ValueError("AETTA extraction bridge requires native float32 model tensors")
        native = self._native_type()
        native.net = torch.nn.Sequential(self._normalization["get_normalize_layer"]("imagenetoutdist"), model)
        native.est_ema_dropout = None
        native.acc_est_json = {key: [] for key in STATE_KEYS}
        return _AETTAEpisode(native)


class _AETTAEpisode:
    def __init__(self, native):
        self._native = native

    @property
    def estimator_state(self):
        """Snapshot, not a mutable handle into the native state."""
        return deepcopy(self._native.acc_est_json)

    @property
    def ema_error(self):
        return self._native.est_ema_dropout

    def estimate(self, images):
        """Run exact native AETTA on unlabeled inputs at the caller's model state."""
        import torch

        if (not isinstance(images, torch.Tensor) or images.device.type != "cpu"
                or images.dtype != torch.float32 or images.ndim != 4
                or images.shape[0] == 0 or tuple(images.shape[1:]) != (3, 224, 224)):
            raise ValueError("Expected nonempty CPU float32 RGB images of shape N x 3 x 224 x 224")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Expected finite raw [0,1] pixels, not already-normalized inputs")
        with torch.no_grad():
            # y_pred is unused by this exact pinned method; do not add another
            # forward or a label interface simply to populate an unused argument.
            self._native.aetta(images, None)
        result = {
            "estimated_accuracy_percent": self._native.acc_est_json["aetta"][-1],
            "ema_error": self._native.est_ema_dropout,
            "dropout_agreement": self._native.acc_est_json["est_dropout"][-1],
            "dropout_mean_entropy": self._native.acc_est_json["est_dropout_avg_entropy"][-1],
            "dropout_confidence_mean": self._native.acc_est_json["est_dropout_softmax_mean"][-1],
            "dropout_confidence_std": self._native.acc_est_json["est_dropout_softmax_std"][-1],
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("Native AETTA returned a non-finite estimate")
        return result
