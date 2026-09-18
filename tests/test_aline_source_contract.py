"""Default source/provenance contracts; not the required native ALine stage."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/aline_native_estimator.py"


@pytest.fixture
def interface():
    spec = importlib.util.spec_from_file_location("aline_source_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_method_runtime_fails_before_arrays_or_source(interface, monkeypatch):
    monkeypatch.delenv("ALINE_PYTHON", raising=False)
    with pytest.raises(ValueError, match="ALINE_PYTHON must explicitly"):
        interface.estimate({}, expected_bank_sha256="0"*64, source="/not-opened")


def test_wrong_method_runtime_fails_before_arrays_or_source(interface, monkeypatch):
    monkeypatch.setenv("ALINE_PYTHON", "/not-the-running-interpreter")
    with pytest.raises(ValueError, match="ALINE_PYTHON does not match"):
        interface.estimate({}, expected_bank_sha256="0"*64, source="/not-opened")


def test_unsupported_package_version_fails_before_arrays_or_source(interface, monkeypatch):
    monkeypatch.setenv("ALINE_PYTHON", sys.executable)
    monkeypatch.setattr(interface.sys, "version_info", (3, 12, 12))
    monkeypatch.setattr(interface.importlib.metadata, "version", lambda name: {
        "numpy": "0.0.0", "scipy": "1.13.1", "statsmodels": "0.14.6"}[name])
    with pytest.raises(ValueError, match="unsupported ALine method runtime"):
        interface.estimate({}, expected_bank_sha256="0"*64, source="/not-opened")


def test_source_schema_rejects_target_outcomes(interface):
    with pytest.raises(ValueError, match="fields"):
        interface._validate({"target_labels": [1, 2]})


def test_duplicate_source_receipt_fields_rejected(interface):
    with pytest.raises(ValueError, match="duplicate"):
        interface._unique_object([("origin", "first"), ("origin", "second")])


def test_bank_binding_changes_when_order_changes(interface):
    assert interface.bank_sha256({"model_ids": ["a", "b"]}) != interface.bank_sha256({"model_ids": ["b", "a"]})
