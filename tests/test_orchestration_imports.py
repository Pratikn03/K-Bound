"""Import guards for optional orchestration dependencies."""

from __future__ import annotations

import builtins
import importlib
import sys


def test_submodule_imports_do_not_require_tensorflow(monkeypatch):
    for name in list(sys.modules):
        if name == "orchestration" or name.startswith("orchestration."):
            del sys.modules[name]

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise ModuleNotFoundError("No module named 'tensorflow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("orchestration.fraud_flow")
    assert hasattr(module, "fraud_pipeline")
