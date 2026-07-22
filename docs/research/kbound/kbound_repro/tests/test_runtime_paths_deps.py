"""Tests for paths / deps / runtime helpers.

These run CPU-only and torch-free: device-selection *logic* is exercised through
an injected fake torch, and the real-torch / MPS paths skip cleanly when the
deep-learning stack (or an MPS backend) is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kbound_repro import deps, paths, runtime  # noqa: E402


# --------------------------- deps ---------------------------
def test_require_missing_names_package():
    with pytest.raises(deps.MissingDependency) as ei:
        deps.require("definitely_not_a_real_pkg_xyz", feature="unit test")
    msg = str(ei.value)
    assert "definitely_not_a_real_pkg_xyz" in msg
    assert "install" in msg.lower()


def test_optional_missing_returns_none():
    assert deps.optional("definitely_not_a_real_pkg_xyz") is None


def test_require_present_returns_module():
    assert deps.require("json") is not None


# --------------------------- paths ---------------------------
def test_find_repo_root_from_module():
    root = paths.find_repo_root()
    assert (root / "pyproject.toml").exists() or (root / ".git").exists()


def test_repo_relative_resolves_under_root():
    p = paths.repo_relative("docs", "research", "kbound")
    assert p.is_dir()


def test_require_dir_missing_is_actionable():
    with pytest.raises(paths.DataRootError) as ei:
        paths.require_dir("/no/such/dir/xyz", what="ImageNet-R", env_var="KBOUND_IMAGENETR_ROOT")
    assert "KBOUND_IMAGENETR_ROOT" in str(ei.value)


def test_env_override_for_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("KBOUND_IMAGENETR_ROOT", str(tmp_path))
    assert paths.imagenetr_root() == tmp_path.resolve()


# ----------------------- runtime (fake torch) -----------------------
class _FakeBackendMPS:
    def __init__(self, avail):
        self._a = avail

    def is_available(self):
        return self._a


class _FakeBackends:
    def __init__(self, mps_avail):
        self.mps = _FakeBackendMPS(mps_avail)


class _FakeCuda:
    def __init__(self, avail, count):
        self._a, self._c = avail, count

    def is_available(self):
        return self._a

    def device_count(self):
        return self._c


class _FakeTorch:
    def __init__(self, cuda=False, cuda_count=0, mps=False):
        self.cuda = _FakeCuda(cuda, cuda_count)
        self.backends = _FakeBackends(mps)

    def device(self, spec):
        return f"torch.device({spec})"


@pytest.fixture
def fake_torch(monkeypatch):
    def _install(**kw):
        ft = _FakeTorch(**kw)
        monkeypatch.setattr(runtime.deps, "require", lambda pkg, feature=None: ft)
        return ft
    return _install


def test_auto_prefers_cuda_then_mps_then_cpu(fake_torch):
    fake_torch(cuda=True, cuda_count=2, mps=True)
    assert runtime.resolve_device().type == "cuda"
    fake_torch(cuda=False, mps=True)
    assert runtime.resolve_device().type == "mps"
    fake_torch(cuda=False, mps=False)
    r = runtime.resolve_device()
    assert r.type == "cpu"
    assert r.source == "auto:cpu"


def test_explicit_unavailable_cuda_fails_loudly(fake_torch):
    fake_torch(cuda=False, mps=False)
    with pytest.raises(RuntimeError, match="CUDA was explicitly requested"):
        runtime.resolve_device("cuda")


def test_explicit_unavailable_mps_fails_loudly(fake_torch):
    fake_torch(cuda=False, mps=False)
    with pytest.raises(RuntimeError, match="MPS was explicitly requested"):
        runtime.resolve_device("mps")


def test_explicit_cpu_always_ok_and_manifest(fake_torch):
    fake_torch(cuda=True, cuda_count=1, mps=False)
    r = runtime.resolve_device("cpu")
    assert r.type == "cpu" and r.source == "requested"
    m = r.manifest()
    assert m["resolved_device"] == "cpu"
    assert m["requested_device"] == "cpu"


def test_cuda_index_out_of_range(fake_torch):
    fake_torch(cuda=True, cuda_count=1, mps=False)
    with pytest.raises(RuntimeError, match="only 1 CUDA"):
        runtime.resolve_device("cuda:3")


def test_describe_runtime_is_torch_safe():
    info = runtime.describe_runtime()
    assert "python" in info and "platform" in info
    # torch may legitimately be None in a torch-free CI environment
    assert "torch" in info


def test_resolve_device_defers_missing_torch(monkeypatch):
    # With no torch installed, resolution raises a *named* deferred error.
    if deps.optional("torch") is not None:
        pytest.skip("torch is installed in this environment")
    with pytest.raises(deps.MissingDependency, match="torch"):
        runtime.resolve_device("cpu")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
