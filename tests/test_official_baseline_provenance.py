from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPO / "docs/research/kbound/scripts/audit_official_baselines.py"
CONVERTER = REPO / "docs/research/kbound/runbooks/convert_official_logs_to_decisions.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_official_baselines", AUDIT_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tree_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    module = load_audit_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("x = 1\n", encoding="utf-8")
    first = module.tree_hash(source)
    assert first == module.tree_hash(source)
    (source / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert module.tree_hash(source) != first


def test_tree_hash_excludes_local_data_and_run_logs(tmp_path: Path) -> None:
    module = load_audit_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("x = 1\n", encoding="utf-8")
    expected = module.tree_hash(source)
    for directory, name in (
        ("cached_data", "cache.pkl"),
        ("dataset", "sample.npy"),
        ("raw_logs", "run.log"),
    ):
        local_dir = source / directory
        local_dir.mkdir()
        (local_dir / name).write_bytes(b"machine-local payload")
    assert module.tree_hash(source) == expected


def test_tree_hash_excludes_static_public_assets(tmp_path: Path) -> None:
    module = load_audit_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("x = 1\n", encoding="utf-8")
    expected = module.tree_hash(source)
    public = source / "public/fonts"
    public.mkdir(parents=True)
    (public / "font.woff2").write_bytes(b"static UI asset")
    assert module.tree_hash(source) == expected


def test_native_log_traceback_fails_closed(tmp_path: Path) -> None:
    module = load_audit_module()
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.log").write_text("Traceback (most recent call last)\n", encoding="utf-8")
    result = module.native_logs(logs, repo=tmp_path)
    assert result["count"] == 1
    assert result["sha256"].keys() == {"logs/run.log"}
    assert result["failure_markers"] == ["logs/run.log"]
    assert result["successful"] is False


def test_native_logs_reject_paths_outside_root_binding(tmp_path: Path) -> None:
    module = load_audit_module()
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run.log").write_text("complete\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the repository-root binding"):
        module.native_logs(outside, repo=repository)


def test_native_logs_fail_closed_when_a_log_cannot_be_hashed(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_audit_module()
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.log").write_text("complete\n", encoding="utf-8")

    def unavailable(_path: Path) -> str:
        raise OSError(89, "operation canceled")

    monkeypatch.setattr(module, "sha256_file", unavailable)
    result = module.native_logs(logs, repo=tmp_path)

    assert result["sha256"] == {}
    assert result["unavailable"] == ["logs/run.log"]
    assert result["successful"] is False


def test_saved_audit_uses_only_repo_relative_provenance_paths() -> None:
    artifact_path = (
        REPO
        / "experiments/kbound/results/official_repro_v1/OFFICIAL_BASELINE_AUDIT.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 2
    assert artifact["provenance_path_binding"] == {
        "schema": "git-repository-relative-posix-v1",
        "root": ".",
        "root_role": "git_repository_root",
        "content_scope": "working_tree_at_generation",
        "generation_base_git_head": artifact["provenance_path_binding"][
            "generation_base_git_head"
        ],
    }
    assert len(artifact["provenance_path_binding"]["generation_base_git_head"]) == 40

    absolute_strings: list[str] = []

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.startswith("/"):
                    absolute_strings.append(key)
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
        elif isinstance(value, str) and value.startswith("/"):
            absolute_strings.append(value)

    inspect(artifact)
    assert absolute_strings == []


def test_converter_writes_unverified_provenance_wrapper(tmp_path: Path) -> None:
    stream = tmp_path / "stream.json"
    logs = tmp_path / "logs.json"
    output = tmp_path / "decisions.json"
    stream.write_text(
        json.dumps({"records": [{"condition": "c1"}, {"condition": "c2"}]}),
        encoding="utf-8",
    )
    logs.write_text(
        json.dumps(
            {
                "c1": {"est_acc_adapted": 0.8, "est_acc_frozen": 0.7},
                "c2": {"est_acc_adapted": 0.6, "est_acc_frozen": 0.7},
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--method",
            "aetta",
            "--logs",
            str(logs),
            "--stream",
            str(stream),
            "--out",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["official_label_allowed"] is False
    assert result["label"] == "external_protocol_adapter_unverified"
    assert result["decisions"] == {"c1": "adapt", "c2": "freeze"}
    assert len(result["source_log_sha256"]) == 64
    assert len(result["locked_stream_sha256"]) == 64


def test_converter_rejects_incomplete_locked_stream(tmp_path: Path) -> None:
    stream = tmp_path / "stream.json"
    logs = tmp_path / "logs.json"
    output = tmp_path / "decisions.json"
    stream.write_text(
        json.dumps({"records": [{"condition": "c1"}, {"condition": "c2"}]}),
        encoding="utf-8",
    )
    logs.write_text(json.dumps({"c1": "adapt"}), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--method",
            "poem",
            "--logs",
            str(logs),
            "--stream",
            str(stream),
            "--out",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "missing" in proc.stderr
    assert not output.exists()
