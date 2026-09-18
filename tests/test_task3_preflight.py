from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/research/kbound/scripts/task3_natural_preflight.py"


def load_module():
    spec = importlib.util.spec_from_file_location("task3_natural_preflight", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_path_resolved_python_runs_dependency_and_entrypoint_checks(tmp_path, monkeypatch):
    module = load_module()
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "fixture-python").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("import argparse\nargparse.ArgumentParser().parse_args()\n")
    details = {}
    assert module._runtime_status("poem", "fixture-python", ("json",)) == []
    assert module._entrypoint_status("poem", source, "fixture-python", details=details) == []
    assert details["poem_entrypoint"]["returncode"] == 0
    panel = module.check_task3_inputs(
        repo=tmp_path, imagenet_root=None, imagenetc_root=None,
        poem_source=None, aetta_source=None, ttaline_source=None,
        python_executable="fixture-python",
    )
    assert not any(b.startswith("python_executable_") for b in panel["blockers"])
    assert panel["status"] == "OPEN"  # Missing datasets remain blockers.


def test_missing_command_is_reported_without_starting_a_process():
    module = load_module()
    assert module._runtime_status("poem", "kbound-nonexistent-python-987654", ("json",)) == ["poem_python_not_found"]
    assert module._runtime_capabilities("kbound-nonexistent-python-987654") is None


def test_interpreter_resolution_preserves_selected_virtual_environment(tmp_path):
    module = load_module()
    environment = tmp_path / "selected-environment"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    command, error = module._python_command(environment / "bin/python")
    assert error is None
    probe = subprocess.run([command, "-c", "import sys; print(sys.prefix)"],
                           capture_output=True, text=True, check=True)
    assert Path(probe.stdout.strip()) == environment


def test_poem_dependency_probe_uses_pinned_bootstrap_not_ignored_shim(tmp_path, monkeypatch):
    module = load_module()
    source = tmp_path / "poem"
    source.mkdir()
    (source / "main.py").write_text("raise RuntimeError('direct entry is not the repaired path')")
    sar = tmp_path / "sar"
    sar.mkdir()
    calls = []
    def probe(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "Pinned revision mismatch")
    monkeypatch.setattr(module.subprocess, "run", probe)
    details = {}
    assert module._entrypoint_status("poem", source, sys.executable,
        details=details, sar_source=sar) == ["poem_entrypoint_not_runnable"]
    assert calls[0][1:] == [str(MODULE_PATH.with_name("poem_dependency_bootstrap.py")),
        "--poem-source", str(source), "--sar-source", str(sar), "--", "--help"]
    assert "Pinned revision mismatch" in details["poem_entrypoint"]["stderr_tail"]


def test_runbook_passes_explicit_sar_dependency_to_preflight(tmp_path):
    capture = tmp_path / "args.json"
    interpreter = tmp_path / "capture-python"
    interpreter.write_text(f"#!{sys.executable}\nimport sys,json\nfrom pathlib import Path\nPath({str(capture)!r}).write_text(json.dumps(sys.argv[1:]))\n")
    interpreter.chmod(0o755)
    result = subprocess.run(["bash", str(ROOT / "docs/research/kbound/runbooks/run_task3_natural.sh"), "preflight-task3"],
        env={**os.environ, "KBOUND_PYTHON": str(interpreter), "POEM_PYTHON": sys.executable,
             "AETTA_PYTHON": sys.executable, "IMAGENET_ROOT": str(tmp_path / "clean"),
             "IMAGENETC_ROOT": str(tmp_path / "corrupt"), "TTALINE_SOURCE": str(tmp_path / "ttaline"),
             "SAR_SOURCE": str(tmp_path / "sar"), "TASK3_OUTPUT": str(tmp_path / "output")},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    assert args[args.index("--poem-sar-source") + 1] == str(tmp_path / "sar")


def test_mac_metadata_does_not_make_an_external_checkout_dirty(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(
        ["git", "-C", str(source), "init", "-q"], check=True, capture_output=True
    )
    (source / ".DS_Store").write_bytes(b"machine metadata")

    assert module._source_status("poem", source, require_clean=True) == []


def test_native_cuda_source_fails_closed_on_mps_runtime(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    base = tmp_path / "inputs"
    clean = base / "imagenet" / "val" / "class_000"
    clean.mkdir(parents=True)
    corrupt = base / "imagenetc"
    for name in module.OFFICIAL_CORRUPTIONS:
        (corrupt / name / "5" / "class_000").mkdir(parents=True)

    poem = base / "poem"
    poem.mkdir()
    (poem / ".git").mkdir()
    (poem / "main.py").write_text("model.cuda()\n", encoding="utf-8")
    aetta = base / "aetta"
    aetta.mkdir()
    (aetta / ".git").mkdir()
    (aetta / "main.py").write_text("print('help')\n", encoding="utf-8")
    ttaline = base / "ttaline"
    ttaline.mkdir()
    (ttaline / ".git").mkdir()

    monkeypatch.setattr(module, "_runtime_status", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "_runtime_capabilities", lambda _executable: {"cuda": False, "mps": True})
    result = module.check_task3_inputs(
        repo=tmp_path,
        imagenet_root=base / "imagenet",
        imagenetc_root=corrupt,
        poem_source=poem,
        aetta_source=aetta,
        ttaline_source=ttaline,
        python_executable=Path(__file__).resolve(),
        poem_python=Path(__file__).resolve(),
        aetta_python=Path(__file__).resolve(),
        expected_classes=1,
        require_clean_sources=False,
    )
    assert result["status"] == "OPEN"
    assert "poem_native_cuda_unavailable" in result["blockers"]
    assert result["details"]["poem_native_runtime"]["native_cuda_required"] is True


def test_native_cuda_check_uses_fallback_python(tmp_path, monkeypatch):
    module = load_module()
    source = tmp_path / "poem"
    source.mkdir()
    (source / ".git").mkdir()
    (source / "main.py").write_text("model.cuda()\n")
    observed = []
    def capabilities(executable):
        observed.append(executable)
        return {"cuda": False, "mps": True} if executable else None
    monkeypatch.setattr(module, "_runtime_capabilities", capabilities)
    monkeypatch.setattr(module, "_runtime_status", lambda *args: [])
    monkeypatch.setattr(module, "_entrypoint_status", lambda *args, **kwargs: [])
    result = module.check_task3_inputs(repo=tmp_path, imagenet_root=None,
        imagenetc_root=None, poem_source=source, aetta_source=None, ttaline_source=None,
        python_executable=sys.executable, require_clean_sources=False)
    assert "poem_native_cuda_unavailable" in result["blockers"]
    assert observed == [sys.executable, sys.executable]
    monkeypatch.setattr(module, "_runtime_capabilities", lambda executable: None)
    unknown = module.check_task3_inputs(repo=tmp_path, imagenet_root=None,
        imagenetc_root=None, poem_source=source, aetta_source=None, ttaline_source=None,
        python_executable=sys.executable, require_clean_sources=False)
    assert "poem_native_runtime_unverified" in unknown["blockers"]
