"""Regression test for Phase-4 import hardening of cifar_tent_mps_v2.

Guards against the reintroduced anti-pattern where the module set ``torch=None``
under a broad ``except`` and then defined ``torch.utils.data.Dataset`` subclasses
at module level -- which crashed pytest *collection* on any box without torch.

The module must now (a) import/collect regardless of whether torch is present,
and (b) fail with a clear, named ``ImportError`` (never ``SystemExit``) when a
torch-only code path is actually entered without torch.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

pytest.importorskip("sklearn", reason="cifar_tent_mps_v2 imports sklearn in the supported env")


@pytest.fixture()
def mod():
    sys.path.insert(0, str(_SCRIPTS))
    m = importlib.import_module("cifar_tent_mps_v2")
    return m


def test_module_collects(mod):
    assert hasattr(mod, "_HAS_TORCH")
    # Dataset subclasses are always defined (object base when torch is absent).
    assert hasattr(mod, "_ICSampledDS")
    assert hasattr(mod, "_ICTarDS")


def test_dataset_base_is_object_without_torch(mod):
    if mod._HAS_TORCH:
        pytest.skip("torch present: base is torch.utils.data.Dataset")
    assert mod._DatasetBase is object


def test_torch_only_path_raises_named_importerror(mod):
    if mod._HAS_TORCH:
        pytest.skip("torch present: the torch-only path runs normally")
    with pytest.raises(ImportError) as ei:
        mod._require_torch()
    msg = str(ei.value).lower()
    assert "torch" in msg and "install" in msg


def test_require_torch_is_not_sys_exit(mod):
    """_require_torch must raise ImportError, not SystemExit (which kills pytest)."""
    if mod._HAS_TORCH:
        pytest.skip("torch present")
    with pytest.raises(ImportError):
        mod.imagenet_c_loader("x", "gaussian_noise", 5, None, 10, "cpu")
