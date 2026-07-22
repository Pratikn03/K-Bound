"""kbound_repro.runtime -- one shared device/runtime selection helper.

Unifies the inconsistent CUDA / MPS / CPU handling scattered across the K-Bound
and AETTA runners.  ``torch`` is imported lazily so this module (and any manifest
tooling that imports it) works in a torch-free environment; a torch device is
only materialized when actually requested.

Selection order (documented and testable):

    1. an explicitly requested device (CLI arg or ``$KBOUND_DEVICE``);
    2. CUDA, if available;
    3. MPS, if available;
    4. CPU fallback.

Reproducibility rule: when a device is *explicitly requested* but unavailable,
we FAIL CLEARLY instead of silently falling back -- silently switching CUDA->CPU
(or MPS->CPU) changes numerics and would corrupt a reproduction.  Automatic
selection (no explicit request) is allowed to fall back to CPU.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

from . import deps

__all__ = ["ResolvedDevice", "resolve_device", "describe_runtime"]

_VALID_TYPES = ("cuda", "mps", "cpu")


@dataclass(frozen=True)
class ResolvedDevice:
    """A resolved compute device plus provenance for the result manifest."""

    type: str            # "cuda" | "mps" | "cpu"
    index: int | None    # cuda ordinal, else None
    requested: str | None
    source: str          # "requested" | "auto:cuda" | "auto:mps" | "auto:cpu"

    @property
    def spec(self) -> str:
        return f"{self.type}:{self.index}" if self.index is not None else self.type

    def torch_device(self):
        """Materialize the ``torch.device`` (imports torch lazily)."""
        torch = deps.require("torch", feature="device selection")
        return torch.device(self.spec)

    def manifest(self) -> dict:
        """Serializable record to embed in every result manifest."""
        return {
            "resolved_device": self.spec,
            "device_type": self.type,
            "device_index": self.index,
            "requested_device": self.requested,
            "selection_source": self.source,
        }


def _parse_request(requested: str | None) -> tuple[str, int | None]:
    req = (requested or "").strip().lower()
    if req in ("gpu",):
        req = "cuda"
    if ":" in req:
        base, _, idx = req.partition(":")
        return base, (int(idx) if idx != "" else None)
    return req, None


def _cuda_available(torch) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive; torch present but backend broken
        return False


def _mps_available(torch) -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(requested: str | None = None, *, allow_fallback: bool = True) -> ResolvedDevice:
    """Resolve a :class:`ResolvedDevice` following the documented order.

    Parameters
    ----------
    requested:
        Explicit device such as ``"cuda"``, ``"cuda:1"``, ``"mps"`` or ``"cpu"``.
        Falls back to ``$KBOUND_DEVICE`` when ``None``.  An explicit request that
        is unavailable raises ``RuntimeError`` (never a silent CPU switch).
    allow_fallback:
        Only affects the *automatic* path; an explicit request is always strict.
    """
    if requested is None:
        requested = os.environ.get("KBOUND_DEVICE")

    torch = deps.require("torch", feature="device selection")

    # (1) explicit request -> strict validation
    if requested:
        base, index = _parse_request(requested)
        if base not in _VALID_TYPES:
            raise ValueError(f"unknown requested device {requested!r}; expected one of {_VALID_TYPES}")
        if base == "cuda":
            if not _cuda_available(torch):
                raise RuntimeError(
                    "CUDA was explicitly requested but torch.cuda.is_available() is False. "
                    "Refusing to silently fall back to CPU (it would change numerics). "
                    "Set KBOUND_DEVICE=cpu to run on CPU intentionally."
                )
            n = torch.cuda.device_count()
            if index is not None and not (0 <= index < n):
                raise RuntimeError(f"requested cuda:{index} but only {n} CUDA device(s) present")
            return ResolvedDevice("cuda", index if index is not None else 0, requested, "requested")
        if base == "mps":
            if not _mps_available(torch):
                raise RuntimeError(
                    "MPS was explicitly requested but torch.backends.mps.is_available() is False. "
                    "Refusing to silently fall back to CPU. Set KBOUND_DEVICE=cpu to force CPU."
                )
            return ResolvedDevice("mps", None, requested, "requested")
        return ResolvedDevice("cpu", None, requested, "requested")

    # (2) auto: CUDA -> (3) MPS -> (4) CPU
    if _cuda_available(torch):
        return ResolvedDevice("cuda", 0, None, "auto:cuda")
    if _mps_available(torch):
        return ResolvedDevice("mps", None, None, "auto:mps")
    if not allow_fallback:
        raise RuntimeError("no CUDA/MPS device available and allow_fallback=False")
    return ResolvedDevice("cpu", None, None, "auto:cpu")


def describe_runtime() -> dict:
    """Version/platform block for release metadata (torch-free safe).

    Records Python, platform, and -- when importable -- torch/torchvision/
    numpy/sklearn versions.  Never raises on a missing optional package.
    """
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for pkg, attr in (("torch", "__version__"), ("torchvision", "__version__"),
                      ("numpy", "__version__"), ("sklearn", "__version__")):
        mod = deps.optional(pkg)
        info[pkg] = getattr(mod, attr, None) if mod is not None else None
    return info
