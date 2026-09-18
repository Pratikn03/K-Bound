"""Behavioral contract for the minimal, source-only production API image."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

from docs.research.kbound.scripts import run_repository_verification as runner

ROOT = Path(__file__).resolve().parents[1]


def _copy_declared_release_sources(destination: Path) -> None:
    for relative in runner.DOCKER_CONTEXT_PATHSPECS:
        source = ROOT / relative
        target = destination / relative
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "._*"),
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _logical_dockerfile_lines(path: Path) -> tuple[str, ...]:
    logical: list[str] = []
    pending = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical.append(pending)
        pending = ""
    assert not pending, "Dockerfile ends with an incomplete continuation"
    return tuple(logical)


def _copy_instructions(dockerfile: Path) -> tuple[tuple[str, str], ...]:
    instructions: list[tuple[str, str]] = []
    for line in _logical_dockerfile_lines(dockerfile):
        tokens = shlex.split(line)
        if tokens[0].upper() != "COPY":
            continue
        operands = [token for token in tokens[1:] if not token.startswith("--")]
        assert len(operands) == 2, f"test simulator requires one-source COPY: {line}"
        instructions.append((operands[0], operands[1]))
    return tuple(instructions)


def _materialize_copy_instructions(context: Path, image_root: Path) -> None:
    for source_name, destination_name in _copy_instructions(context / "Dockerfile"):
        source = context / source_name
        assert source.exists(), f"Dockerfile COPY source is absent from staged context: {source_name}"
        destination = image_root.joinpath(*PurePosixPath(destination_name).parts[1:])
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def _materialize_declared_directories(dockerfile: Path, image_root: Path) -> None:
    for line in _logical_dockerfile_lines(dockerfile):
        if not line.upper().startswith("RUN "):
            continue
        for segment in line[4:].split("&&"):
            tokens = shlex.split(segment.strip())
            if tokens[:2] != ["mkdir", "-p"]:
                continue
            for directory in tokens[2:]:
                assert directory.startswith("/"), f"expected absolute image directory: {directory}"
                image_root.joinpath(*PurePosixPath(directory).parts[1:]).mkdir(parents=True, exist_ok=True)


def _pip_install_tokens(dockerfile: Path) -> tuple[str, ...]:
    segments = [
        segment.strip()
        for line in _logical_dockerfile_lines(dockerfile)
        if line.upper().startswith("RUN ")
        for segment in line[4:].split("&&")
    ]
    installs = [tuple(shlex.split(segment)) for segment in segments if segment.startswith("pip install ")]
    assert len(installs) == 1
    return installs[0]


def test_exact_staged_context_materializes_locked_source_only_api_and_serves_health(
    tmp_path: Path,
) -> None:
    """Removing a required context file or adding an undeclared COPY must fail."""

    source_repo = tmp_path / "source-repo"
    source_repo.mkdir()
    _copy_declared_release_sources(source_repo)
    subprocess.run(["git", "init", "-q"], cwd=source_repo, check=True)
    subprocess.run(
        ["git", "add", "-f", "--", *runner.DOCKER_CONTEXT_PATHSPECS],
        cwd=source_repo,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Contract Test",
            "-c",
            "user.email=release-contract@example.invalid",
            "commit",
            "-qm",
            "synthetic release source",
        ],
        cwd=source_repo,
        check=True,
    )
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_repo, text=True).strip()

    context = tmp_path / "staged-context"
    context.mkdir()
    runner._stage_docker_context(repo=source_repo, source_commit=source_commit, destination=context)

    image_root = tmp_path / "image-root"
    _materialize_copy_instructions(context, image_root)
    _materialize_declared_directories(context / "Dockerfile", image_root)
    app_root = image_root / "app"
    assert (app_root / "requirements-api-py311-linux.lock.txt").is_file()
    assert not (app_root / "src").exists()
    assert not (app_root / "experiments").exists()
    assert not (app_root / "docs").exists()

    pip_install = _pip_install_tokens(context / "Dockerfile")
    assert "--require-hashes" in pip_install
    assert "--only-binary=:all:" in pip_install
    requirement_index = pip_install.index("-r") + 1
    assert pip_install[requirement_index] == "/app/requirements-api-py311-linux.lock.txt"

    probe = """
import json
from fastapi.testclient import TestClient
from deploy.api.main import app

with TestClient(app) as client:
    health = client.get('/health')
    decision = client.post(
        '/decide',
        json={'calib_scores': [0.1, 0.2, 0.3], 'test_scores': [0.4, 0.5, 0.6]},
        headers={'X-API-Key': 'synthetic-test-key'},
    )
assert health.status_code == 200, health.text
assert health.json()['status'] == 'ok'
assert decision.status_code == 200, decision.text
assert decision.json()['model_action'] == 'retain_frozen'
print(json.dumps({'health': health.json()['status'], 'model_action': decision.json()['model_action']}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(app_root)
    environment["KGA_API_KEYS"] = "synthetic-test-key"
    environment.pop("UAIS_API_KEYS", None)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=app_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"health": "ok", "model_action": "retain_frozen"}
