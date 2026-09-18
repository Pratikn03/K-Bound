"""Required explicit native stage: ALINE_NATIVE_STAGE=1 and declared ALINE_PYTHON.

Default discovery excludes this method-runtime module (no native skip/pass).
The separate test_aline_source_contract.py remains in default verification.
"""
import ast
import json
import os
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/research/kbound/scripts/aline_native_estimator.py"
SOURCE = ROOT / "external/ttaline_official"
__test__ = os.environ.get("ALINE_NATIVE_STAGE") == "1"


@pytest.fixture
def interface():
    assert os.environ.get("ALINE_PYTHON"), "explicit native stage requires ALINE_PYTHON; no skip"
    assert SCRIPT.exists(), "authenticated ALine callable is missing"
    spec = importlib.util.spec_from_file_location("aline_interface_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bank():
    rng = np.random.default_rng(713)
    def predictions():
        shared = rng.integers(0, 3, 240)
        return [np.where(rng.random(240) < p, rng.integers(0, 3, 240), shared).tolist()
                for p in (.15, .22, .31, .39)]
    return dict(model_ids=[f"model-{i}" for i in range(4)],
                checkpoint_sha256=[hashlib.sha256(f"fixture-{i}".encode()).hexdigest() for i in range(4)],
                clean_sample_ids=[f"clean-{i}" for i in range(240)],
                target_sample_ids=[f"target-{i}" for i in range(240)],
                clean_predictions=predictions(), target_predictions=predictions(),
                clean_accuracies=[.64, .68, .74, .79], class_count=3)


def test_interface_matches_native_with_synthetic_git_transport(interface, bank, monkeypatch):
    import json
    import statsmodels.api as sm
    from scipy.stats import norm
    payload = (SOURCE / "agreement_trajectory.ipynb").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == "cef78f7d196c2f86a96d466a30a6967c31eaa84cb176edd072d5356fa89344fc"
    notebook = json.loads(payload)
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    tree = ast.parse(text)
    wanted = {"rescale", "compute_linear_fit", "aline"}
    namespace = dict(np=np, sm=sm, norm=norm)
    exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                                 and node.name in wanted], type_ignores=[]), "native-independent", "exec"), namespace)
    expected, bias, slope = namespace["aline"](np.array(bank["clean_predictions"]),
                                               np.array(bank["clean_accuracies"]), np.array(bank["target_predictions"]))
    # Transport fixture only: real source bytes/functions and calculations.
    # This is not a successful live Git authentication claim (local pack damaged).
    def git(source, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return b"99cb46ee1b50960c04b11d8e2f27943e68cbfbec\n"
        assert arguments == ("show", "99cb46ee1b50960c04b11d8e2f27943e68cbfbec:agreement_trajectory.ipynb")
        return payload
    monkeypatch.setattr(interface, "_git", git)
    result = interface.estimate(bank, expected_bank_sha256=interface.bank_sha256(bank), source=SOURCE)
    np.testing.assert_allclose(result["aline_s_probit"], expected[0], rtol=0, atol=0)
    np.testing.assert_allclose(result["aline_d_accuracy"], expected[1], rtol=0, atol=0)
    assert result["bias"] == bias and result["slope"] == slope
    assert result["method"] == "Baek_Agreement_on_the_Line"
    assert result["benchmark_completed"] is False
    assert result["runtime"]["numpy"] in ("2.0.2", "2.4.4")
    assert result["runtime"]["scipy"] == "1.13.1"
    assert result["runtime"]["statsmodels"] == "0.14.6"
    assert result["wrapper_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def source_receipt(tmp_path):
    path = tmp_path / "source.json"
    data = {"schema": "aline-fixed-commit-source-v1", "status": "EXACT_SOURCE_HASH_VERIFIED",
            "origin": "https://github.com/kebaek/Agreement-on-the-line.git",
            "url": "https://raw.githubusercontent.com/kebaek/Agreement-on-the-line/99cb46ee1b50960c04b11d8e2f27943e68cbfbec/agreement_trajectory.ipynb",
            "commit": "99cb46ee1b50960c04b11d8e2f27943e68cbfbec",
            "path": str(SOURCE / "agreement_trajectory.ipynb"), "bytes": 7702,
            "sha256": "cef78f7d196c2f86a96d466a30a6967c31eaa84cb176edd072d5356fa89344fc"}
    path.write_text(json.dumps(data))
    return path, data


def test_explicit_fixed_source_receipt_runs_offline(interface, bank, tmp_path, monkeypatch):
    path, _ = source_receipt(tmp_path)
    def forbidden(*args):
        raise AssertionError("explicit source receipt must not request Git or network")
    monkeypatch.setattr(interface, "_git", forbidden)
    result = interface.estimate(bank, expected_bank_sha256=interface.bank_sha256(bank), source=SOURCE, source_receipt=path)
    assert result["source_authority"]["method"] == "fixed_commit_public_source_receipt"
    assert len(result["aline_d_accuracy"]) == 4


@pytest.mark.parametrize("field,value", [("origin", "https://example.com"), ("commit", "0"*40),
                                         ("sha256", "0"*64), ("path", "/different/notebook"), ("bytes", 7)])
def test_source_receipt_identity_substitution_fails(interface, bank, tmp_path, field, value):
    path, data = source_receipt(tmp_path)
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        interface.estimate(bank, expected_bank_sha256=interface.bank_sha256(bank), source=SOURCE, source_receipt=path)


def test_missing_pinned_git_blob_fails_instead_of_using_working_copy(interface, bank, monkeypatch):
    def missing(*args):
        raise RuntimeError("pinned Git object unavailable")
    monkeypatch.setattr(interface, "_git", missing)
    with pytest.raises(RuntimeError, match="Git object"):
        interface.estimate(bank, expected_bank_sha256=interface.bank_sha256(bank), source=SOURCE)


def test_identity_substitution_fails_binding(interface, bank):
    locked = interface.bank_sha256(bank)
    bank["checkpoint_sha256"][0] = "f" * 64
    with pytest.raises(ValueError, match="binding"):
        interface.estimate(bank, expected_bank_sha256=locked, source=SOURCE)


@pytest.mark.parametrize("change", ["target_labels", "alignment", "class_id", "clean_endpoint", "model_count", "empty_pairs"])
def test_invalid_or_outcome_bearing_bank_rejected_before_native(interface, bank, change):
    if change == "target_labels": bank["target_labels"] = [0] * 240
    if change == "alignment": bank["target_predictions"][1].pop()
    if change == "class_id": bank["target_predictions"][0][0] = True
    if change == "clean_endpoint": bank["clean_accuracies"][0] = 1.0
    if change == "model_count":
        for key in ("model_ids", "checkpoint_sha256", "clean_predictions", "target_predictions", "clean_accuracies"):
            bank[key] = bank[key][:2]
    if change == "empty_pairs":
        bank["clean_predictions"] = [bank["clean_predictions"][0]] * 4
    with pytest.raises(ValueError):
        interface.estimate(bank, expected_bank_sha256=interface.bank_sha256(bank), source=Path("/unavailable-native-source"))
