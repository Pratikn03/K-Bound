"""Executor contracts fail before images or models are loaded."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/run_poem_real_image_smoke.py"


@pytest.fixture
def executor(monkeypatch):
    assert SCRIPT.exists(), "authenticated real-image executor is missing"
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("poem_real_image_executor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def manifest():
    def row(i, source):
        name = f"n00000001/ILSVRC2012_val_{i:08d}.JPEG"
        return {"relative_path": name if source else "gaussian_noise/5/" + name,
                "sample_id": f"ILSVRC2012_val_{i:08d}", "sha256": "a" * 64,
                "size_bytes": 100, "width": 224, "height": 224, "decoded_rgb": True}
    return {"schema": "kbound-real-image-smoke-panel-v1", "scope": "ENGINEERING_SMOKE_NOT_BENCHMARK",
            "outcomes_read": False, "upstream_full_dataset_protocol": False,
            "official_image_bytes_authenticated": False, "seed": 0, "batch_size": 4,
            "condition": "gaussian_noise/5", "clean_root": "/clean", "corruption_root": "/corrupt",
            "source": [row(i, True) for i in range(1, 33)],
            "target": [row(i, False) for i in range(33, 137)]}


def test_manifest_requires_exact_scope_ids_roles_and_counts(executor, manifest):
    executor.validate_panel(manifest)
    bad = copy.deepcopy(manifest)
    bad["target"][0]["sample_id"] = bad["source"][0]["sample_id"]
    with pytest.raises(ValueError):
        executor.validate_panel(bad)
    bad = copy.deepcopy(manifest)
    bad["target"][0]["relative_path"] = "gaussian_noise/5/" + bad["source"][0]["relative_path"]
    with pytest.raises(ValueError):
        executor.validate_panel(bad)
    bad = copy.deepcopy(manifest)
    bad["target"].pop()
    with pytest.raises(ValueError):
        executor.validate_panel(bad)


@pytest.mark.parametrize("field,value", [("scope", "BENCHMARK"), ("outcomes_read", True),
                                        ("batch_size", 1), ("seed", 7),
                                        ("upstream_full_dataset_protocol", True)])
def test_cannot_silently_change_locked_smoke(executor, manifest, field, value):
    manifest[field] = value
    with pytest.raises(ValueError):
        executor.validate_panel(manifest)


def test_labels_cannot_be_smuggled_into_executor(executor, manifest):
    manifest["target"][0]["label"] = 5
    with pytest.raises(ValueError, match="fields"):
        executor.validate_panel(manifest)


def test_device_derivation_is_constructor_only_and_declared(executor):
    raw = b"self.temperature = nn.Parameter(torch.ones(1) * temp).cuda()\nself.cuda()\n"
    new, receipt = executor.derive_temperature(raw, "cpu")
    assert b".to('cpu')" in new and b"\nself.cuda()" in new
    assert receipt["original_sha256"] != receipt["derived_sha256"]
    with pytest.raises(ValueError):
        executor.derive_temperature(raw + raw, "cpu")
    with pytest.raises(ValueError):
        executor.derive_temperature(raw, "anything")


def test_binding_rejects_wrong_manifest_hash_without_loading_models(executor, manifest, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest"):
        executor.read_panel(path, "0" * 64)


def test_unknown_or_duplicate_json_fields_do_not_override_contract(executor, tmp_path):
    data = b'{"scope":"BENCHMARK","scope":"ENGINEERING_SMOKE_NOT_BENCHMARK"}'
    path = tmp_path / "manifest.json"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="duplicate"):
        executor.read_panel(path, executor.digest(data))


def test_replaced_output_directory_never_redirects_receipt(executor, manifest, tmp_path, monkeypatch):
    import sys
    clean, corrupt = tmp_path / "clean", tmp_path / "corrupt"
    clean.mkdir()
    corrupt.mkdir()
    manifest.update(clean_root=str(clean), corruption_root=str(corrupt))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "output"
    original = tmp_path / "original-output"
    def swap(args, panel, record):
        output.rename(original)
        output.symlink_to(clean, target_is_directory=True)
        record["status"] = "PASS_REAL_IMAGE_ENGINEERING_SMOKE_ONLY"
    monkeypatch.setattr(executor, "run", swap)
    monkeypatch.setenv("POEM_PYTHON", sys.executable)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--manifest", str(path),
                        "--manifest-sha256", executor.digest(path.read_bytes()),
                        "--checkpoint", str(tmp_path / "not-loaded.pth"),
                        "--poem-source", str(tmp_path / "source"), "--output", str(output)])
    assert executor.main() == 2
    assert not (clean / "receipt.json").exists()


def test_preprocessing_reference_is_hash_bound(executor):
    assert "dataset/selectedRotateImageFolder.py" in executor.NATIVE_HASHES
    assert len(executor.NATIVE_HASHES["dataset/selectedRotateImageFolder.py"]) == 64
