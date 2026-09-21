from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _bootstrap():
    path = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/poem_dependency_bootstrap.py"
    spec = importlib.util.spec_from_file_location("poem_dependency_bootstrap_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changed_dependency_is_rejected_before_execution(tmp_path):
    module = _bootstrap()
    source = tmp_path / "Res.py"
    source.write_text("raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match="SHA-256"):
        module.authenticated_bytes(source, "0" * 64)


def test_authenticated_binding_uses_exact_source_not_local_shim(tmp_path, monkeypatch):
    module = _bootstrap()
    source = tmp_path / "Res.py"
    payload = b"MODEL_ID = 'authenticated-full-model'\n"
    source.write_bytes(payload)
    monkeypatch.delitem(sys.modules, "models", raising=False)
    monkeypatch.delitem(sys.modules, "models.Res", raising=False)
    try:
        module.bind_model(source, hashlib.sha256(payload).hexdigest())
        import models.Res as model
        assert model.MODEL_ID == "authenticated-full-model"
        assert model.__file__ == str(source)
        assert sys.modules["models"].__path__ == []
    finally:
        sys.modules.pop("models.Res", None)
        sys.modules.pop("models", None)


def test_existing_model_binding_is_not_silently_replaced(tmp_path, monkeypatch):
    module = _bootstrap()
    original = object()
    monkeypatch.setitem(sys.modules, "models", original)
    with pytest.raises(RuntimeError, match="already loaded"):
        module.bind_model(tmp_path / "Res.py", "0" * 64)
    assert sys.modules["models"] is original


NATIVE_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
NATIVE_LITERAL = repr(NATIVE_CORRUPTIONS).encode("utf-8")


@pytest.fixture
def synthetic_bootstrap(tmp_path, monkeypatch):
    """Execute the real bootstrap against authenticated, harmless Python sources."""
    module = _bootstrap()
    poem, sar = tmp_path / "poem", tmp_path / "sar"
    poem.mkdir()
    (sar / "models").mkdir(parents=True)
    model_marker, driver_marker = tmp_path / "model-executed", tmp_path / "driver-executed"
    model_payload = (
        "from pathlib import Path\n"
        f"Path({str(model_marker)!r}).write_text('model executed')\n"
    ).encode()
    payload = (
        "# Preserve this non-ASCII prelude: café.\n"
        "import argparse, json\nfrom pathlib import Path\n"
        f"Path({str(driver_marker)!r}).write_text('driver executed')\n"
        "if __name__ == '__main__':\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--exp_type', default='normal')\n"
        "    parser.add_argument('--corruption', default='gaussian_noise')\n"
        "    parser.add_argument('--output', required=True)\n"
        "    parser.add_argument('--seed', type=int, default=0)\n"
        "    args = parser.parse_args()\n"
        f"    common_corruptions = {NATIVE_CORRUPTIONS!r}  # preserve suffix\n"
        "    if args.exp_type == 'continual':\n"
        "        common_corruptions = ['continual']\n"
        "    output = Path(args.output)\n"
        "    output.mkdir(parents=True, exist_ok=True)\n"
        "    (output / 'dispatch.json').write_text(json.dumps(common_corruptions))\n"
    ).encode()
    (poem / "main.py").write_bytes(payload)
    (sar / "models/Res.py").write_bytes(model_payload)
    monkeypatch.setattr(module, "MAIN_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(module, "MODEL_SHA256", hashlib.sha256(model_payload).hexdigest())

    def revision(command, **kwargs):
        assert command in (
            ["git", "-C", str(poem), "rev-parse", "HEAD"],
            ["git", "-C", str(sar), "rev-parse", "HEAD"],
        )
        return module.POEM_COMMIT if command[2] == str(poem) else module.SAR_COMMIT

    monkeypatch.setattr(module.subprocess, "check_output", revision)
    monkeypatch.delitem(sys.modules, "models", raising=False)
    monkeypatch.delitem(sys.modules, "models.Res", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    output = tmp_path / "output with spaces"

    def run(*, single="shot_noise", upstream=None, wrapper_extra=()):
        override = [] if single is None else ["--single-corruption", single]
        arguments = upstream if upstream is not None else [
            "--exp_type", "normal", "--corruption", "shot_noise", "--output", str(output),
        ]
        monkeypatch.setattr(sys, "argv", [
            "bootstrap", "--poem-source", str(poem), "--sar-source", str(sar),
            *override, *wrapper_extra, "--", *arguments,
        ])
        module.main()

    yield module, run, poem, sar, output, payload, model_marker, driver_marker
    sys.modules.pop("models.Res", None)
    sys.modules.pop("models", None)


def test_single_corruption_executes_only_selected_upstream_loop(synthetic_bootstrap):
    _, run, poem, _, output, payload, model_marker, driver_marker = synthetic_bootstrap
    run()
    assert json.loads((output / "dispatch.json").read_text()) == ["shot_noise"]
    assert (poem / "main.py").read_bytes() == payload
    assert model_marker.exists() and driver_marker.exists()
    receipt = json.loads((output / "POEM_PROTOCOL_DERIVATION_RECEIPT.json").read_text())
    expected = payload.replace(NATIVE_LITERAL, b"['shot_noise']", 1)
    assert receipt["original_main_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["derived_main_sha256"] == hashlib.sha256(expected).hexdigest()
    assert receipt["status"] == "DERIVATION_RECORDED_BEFORE_EXECUTION"
    assert receipt["benchmark_attestation"] is False
    assert receipt["override"]["corruption"] == "shot_noise"
    start, end = receipt["override"]["literal_byte_span"]
    assert payload[start:end] == NATIVE_LITERAL
    assert payload[:start] + b"['shot_noise']" + payload[end:] == expected


def test_default_bootstrap_preserves_native_dispatch_without_derivation(synthetic_bootstrap):
    _, run, poem, _, output, payload, _, _ = synthetic_bootstrap
    run(single=None)
    assert json.loads((output / "dispatch.json").read_text()) == NATIVE_CORRUPTIONS
    assert (poem / "main.py").read_bytes() == payload
    assert not (output / "POEM_PROTOCOL_DERIVATION_RECEIPT.json").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--exp_type", "continual"],
        ["--corruption", "gaussian_noise"],
        ["--exp_type=normal"],
        ["--corruption=shot_noise"],
        ["--output", "another-output"],
        ["--seed", "0", "--seed", "1"],
        ["--exp_typ", "normal"],
    ],
)
def test_single_corruption_rejects_duplicate_or_abbreviated_flags_before_execution(synthetic_bootstrap, extra):
    _, run, _, _, output, _, model_marker, driver_marker = synthetic_bootstrap
    with pytest.raises((ValueError, SystemExit)):
        run(upstream=[
            "--exp_type", "normal", "--corruption", "shot_noise", "--output", str(output), *extra,
        ])
    assert not model_marker.exists() and not driver_marker.exists()
    assert not output.exists()


@pytest.mark.parametrize("mode", ["continual", "severity_shift", "in_dist"])
def test_single_corruption_rejects_non_normal_mode_before_execution(synthetic_bootstrap, mode):
    _, run, _, _, output, _, model_marker, driver_marker = synthetic_bootstrap
    with pytest.raises((ValueError, SystemExit)):
        run(upstream=["--exp_type", mode, "--corruption", "shot_noise", "--output", str(output)])
    assert not model_marker.exists() and not driver_marker.exists()
    assert not output.exists()


@pytest.mark.parametrize("single", ["invalid_noise", "gaussian_noise"])
def test_single_corruption_rejects_unknown_or_mismatched_name_before_execution(synthetic_bootstrap, single):
    _, run, _, _, output, _, model_marker, driver_marker = synthetic_bootstrap
    with pytest.raises((ValueError, SystemExit)):
        run(single=single)
    assert not model_marker.exists() and not driver_marker.exists()
    assert not output.exists()


def test_duplicate_bootstrap_override_is_rejected_before_execution(synthetic_bootstrap):
    _, run, _, _, output, _, model_marker, driver_marker = synthetic_bootstrap
    with pytest.raises((ValueError, SystemExit)):
        run(wrapper_extra=["--single-corruption", "shot_noise"])
    assert not model_marker.exists() and not driver_marker.exists()
    assert not output.exists()


def test_single_corruption_preserves_occupied_output_before_execution(synthetic_bootstrap):
    _, run, _, _, output, _, model_marker, driver_marker = synthetic_bootstrap
    output.mkdir()
    receipt = output / "POEM_PROTOCOL_DERIVATION_RECEIPT.json"
    receipt.write_text("prior receipt\n")
    with pytest.raises((ValueError, FileExistsError)):
        run()
    assert receipt.read_text() == "prior receipt\n"
    assert not model_marker.exists() and not driver_marker.exists()


def test_single_corruption_authenticates_dependency_before_derivation_output(synthetic_bootstrap):
    _, run, _, sar, output, _, model_marker, driver_marker = synthetic_bootstrap
    (sar / "models/Res.py").write_text("raise RuntimeError('unauthenticated source')\n")
    with pytest.raises(ValueError, match="SHA-256"):
        run()
    assert not model_marker.exists() and not driver_marker.exists()
    assert not output.exists()


def test_pinned_driver_derivation_changes_only_the_native_list_literal():
    module = _bootstrap()
    source = Path(__file__).resolve().parents[1] / "external/poem_official/main.py"
    payload = module.authenticated_bytes(source, module.MAIN_SHA256)
    assert payload.count(NATIVE_LITERAL) == 1
    # Preserve the authenticated upstream bytes, including legacy docstring escapes.
    with pytest.warns((SyntaxWarning, DeprecationWarning), match="invalid escape sequence"):
        derived, override = module.derive_single_corruption(payload, "impulse_noise")
    assert derived == payload.replace(NATIVE_LITERAL, b"['impulse_noise']", 1)
    start, end = override["literal_byte_span"]
    assert payload[start:end] == NATIVE_LITERAL
    assert source.read_bytes() == payload


@pytest.mark.parametrize("initializer", ["call", "duplicate"])
def test_adapter_rejects_unreviewed_initializer_shape_before_execution(synthetic_bootstrap, initializer):
    module, run, poem, _, output, payload, model_marker, driver_marker = synthetic_bootstrap
    native_line = b"    common_corruptions = " + NATIVE_LITERAL
    if initializer == "call":
        replacement = b"    common_corruptions = list(" + NATIVE_LITERAL + b")"
    else:
        replacement = native_line + b"\n" + native_line
    changed = payload.replace(native_line, replacement, 1)
    (poem / "main.py").write_bytes(changed)
    module.MAIN_SHA256 = hashlib.sha256(changed).hexdigest()
    with pytest.raises(ValueError, match="common_corruptions"):
        run()
    assert not model_marker.exists() and not driver_marker.exists()
    assert not output.exists()
