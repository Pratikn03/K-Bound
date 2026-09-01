from __future__ import annotations

import hashlib
import shlex
import sys
from pathlib import Path

import pytest
import yaml

from docs.research.kbound.scripts import verify_release_checksums as verifier
from docs.research.kbound.scripts import build_release_source_seal as seal


_CURRENT_REGRESSION_TESTS = {
    "tests/test_kga_package.py",
    "tests/test_kga_benefit_estimator.py",
    "tests/test_kga_routing.py",
    "tests/test_kga_masked_inputs.py",
    "tests/test_certificate_drift_guard.py",
    "tests/test_kga_unavailable_runtime.py",
    "tests/test_kga_api_routes.py",
    "tests/test_kga_unavailable_api.py",
    "tests/test_kbound_theory_scope.py",
    "tests/test_kbound_formal_audit.py",
    "tests/test_kbound_current_policy_bindings.py",
    "tests/test_kbound_bibliography.py",
    "tests/test_kbound_estimand_inference_wording.py",
    "tests/test_kbound_dashboard_metadata.py",
    "tests/test_reconcile_no_implicit_cleanup.py",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_checksum_verifier_accepts_exact_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"release bytes\n")
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text(f"{_digest(artifact)}  artifact.bin\n", encoding="utf-8")
    assert verifier.verify_checksum_file(
        checksums,
        root=tmp_path,
        required_paths=("artifact.bin",),
    ) == 1


def test_release_checksum_verifier_rejects_mismatch_and_missing_entry(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"release bytes\n")
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text(f"{'0' * 64}  artifact.bin\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verifier.verify_checksum_file(checksums, root=tmp_path)

    checksums.write_text(f"{_digest(artifact)}  artifact.bin\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required checksum entries are missing"):
        verifier.verify_checksum_file(
            checksums,
            root=tmp_path,
            required_paths=("artifact.bin", "missing.json"),
        )


def test_release_runbook_verifies_temp_file_before_atomic_publish() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "docs/research/kbound/runbooks/release_candidate.sh"
    ).read_text(encoding="utf-8")
    assert "verify_release_checksums.py" in runbook
    assert 'mv -f "$checksum_tmp" "$output"' in runbook
    assert 'tee -a "$checksum_tmp"' in runbook
    assert '"$KB/scripts/verify_release_checksums.py" --list-required' in runbook
    verify_temp = '"$KB/scripts/verify_release_checksums.py" "$checksum_tmp" --root "$REPO"'
    assert runbook.index(verify_temp) < runbook.index('mv -f "$checksum_tmp" "$output"')


def _executed_pytest_targets(source: str) -> set[str]:
    """Read explicit pytest commands, excluding collection and echoed text."""
    targets: set[str] = set()
    for line in source.replace("\\\n", " ").splitlines():
        if not line.lstrip().startswith(("pytest", '"$PY"')):
            continue
        tokens = shlex.split(line, comments=True)
        if tokens[:3] != ["$PY", "-m", "pytest"] and tokens[:1] != ["pytest"]:
            continue
        if any(token == "--co" or token.startswith("--collect-only") for token in tokens):
            continue
        targets.update(
            token for token in tokens if token.startswith("tests/") and token.endswith(".py")
        )
    return targets


def test_release_test_mode_executes_current_safety_and_manuscript_regressions() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "docs/research/kbound/runbooks/release_candidate.sh"
    ).read_text(encoding="utf-8")
    test_body = runbook.split("step_test() {", 1)[1].split("\n}", 1)[0]
    required = _CURRENT_REGRESSION_TESTS | {
        "tests/test_kga_frontier_api.py",
        "tests/test_kga_canonical_rule.py",
        "tests/test_release_checksum_verifier.py",
        "tests/test_kbound_formal_audit.py",
        "tests/test_kbound_current_policy_bindings.py",
        "tests/test_kga_masked_inputs.py",
    }
    assert not (required - _executed_pytest_targets(test_body))


@pytest.mark.parametrize(
    "source",
    [
        '"$PY" -m pytest --collect-only -q tests/test_runtime.py',
        'pytest --co tests/test_runtime.py',
        'echo "pytest tests/test_runtime.py"',
        '# pytest tests/test_runtime.py',
    ],
)
def test_collection_comments_or_echoes_do_not_satisfy_execution_guard(source: str) -> None:
    assert _executed_pytest_targets(source) == set()


def test_execution_guard_accepts_runbook_line_continuations() -> None:
    source = '"$PY" -m pytest -q \\\n    tests/test_runtime.py\n'
    assert _executed_pytest_targets(source) == {"tests/test_runtime.py"}


def _workflow_jobs(name: str) -> dict:
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]


def _job_run_text(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _pip_install_arguments(job: dict) -> set[str]:
    arguments: set[str] = set()
    for line in _job_run_text(job).replace("\\\n", " ").splitlines():
        if not line.lstrip().startswith("python -m pip install"):
            continue
        tokens = shlex.split(line, comments=True)
        if tokens[:4] == ["python", "-m", "pip", "install"]:
            arguments.update(tokens[4:])
    return arguments


def test_research_ci_executes_current_regressions_under_the_locked_profile() -> None:
    job = _workflow_jobs("kbound-ci.yml")["kbound-research-tests"]
    assert not (_CURRENT_REGRESSION_TESTS - _executed_pytest_targets(_job_run_text(job)))
    assert {"--require-hashes", "-r", "requirements-research-ci.lock.txt"} <= _pip_install_arguments(job)


def test_api_ci_executes_unavailable_regressions_under_the_production_profile() -> None:
    job = _workflow_jobs("kbound-ci.yml")["unit-tests"]
    required = {
        "tests/test_kga_api_routes.py",
        "tests/test_kga_masked_inputs.py",
        "tests/test_kga_unavailable_runtime.py",
        "tests/test_kga_unavailable_api.py",
    }
    assert not (required - _executed_pytest_targets(_job_run_text(job)))
    assert {"-r", "requirements-api.txt", "pytest", "pytest-cov", "pandas"} <= _pip_install_arguments(job)


def test_core_ci_installs_the_package_and_legacy_drift_guard_dependency() -> None:
    job = _workflow_jobs("ci.yml")["kbound-core"]
    assert {".", "pytest", "scikit-learn", "ruff", "mypy"} <= _pip_install_arguments(job)


def test_installed_cli_smoke_directory_is_defined_in_the_same_ci_step() -> None:
    for workflow in ("ci.yml", "kbound-ci.yml"):
        for job in _workflow_jobs(workflow).values():
            for step in job.get("steps", []):
                command = step.get("run", "")
                if '"$smoke_dir"' in command:
                    assert "smoke_dir=" in command
                    assert command.index("smoke_dir=") < command.index('"$smoke_dir"')


def test_default_cli_rejects_truncated_release_but_generic_mode_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"valid but not a complete release\n")
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text(f"{_digest(artifact)}  artifact.bin\n")
    argv = ["verify_release_checksums.py", str(checksums), "--root", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as failure:
        verifier.main()
    assert failure.value.code == 1
    assert "required checksum entries are missing" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [*argv, "--generic"])
    assert verifier.main() == 0
    assert "generic checksums: PASS (1 files)" in capsys.readouterr().out


def test_release_inventory_is_unique_complete_and_excludes_self_hash() -> None:
    paths = verifier.REQUIRED_RELEASE_PATHS
    assert len(paths) == len(set(paths))
    checksum_self = "docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt"
    assert checksum_self not in paths
    assert seal.GENERATED_OUTPUT_ALLOWLIST - {checksum_self} <= set(paths)
    assert {
        "docs/research/kbound/kbound_short_final_draft.pdf",
        "docs/research/kbound/kbound_tmlr.pdf",
        "docs/research/kbound/kbound_short_final_draft.docx",
        "docs/research/kbound/audits/release_source_seal_2026_08_29.json",
        "docs/research/kbound/audits/formal_foundations_2026_08_31.json",
    } <= set(paths)


def test_default_cli_verifies_complete_release_and_lists_required_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = []
    for relative in verifier.REQUIRED_RELEASE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
        lines.append(f"{_digest(path)}  {relative}")
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(
        sys, "argv", ["verify_release_checksums.py", str(checksums), "--root", str(tmp_path)]
    )
    assert verifier.main() == 0
    assert f"release checksums: PASS ({len(lines)} files)" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["verify_release_checksums.py", "--list-required"])
    assert verifier.main() == 0
    assert capsys.readouterr().out.splitlines() == list(verifier.REQUIRED_RELEASE_PATHS)


@pytest.mark.parametrize("omitted", verifier.REQUIRED_RELEASE_PATHS)
def test_each_required_release_entry_is_enforced(tmp_path: Path, omitted: str) -> None:
    checksums = tmp_path / "SHA256SUMS.txt"
    # The inventory check must fail before trying to read any artifact bytes.
    lines = [f"{'0' * 64}  {path}" for path in verifier.REQUIRED_RELEASE_PATHS if path != omitted]
    checksums.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="required checksum entries are missing"):
        verifier.verify_checksum_file(
            checksums, root=tmp_path, required_paths=verifier.REQUIRED_RELEASE_PATHS
        )


@pytest.mark.parametrize("relative", ["../escape.txt", "/absolute.txt", "a/../b.txt", "./file.txt"])
def test_checksum_verifier_rejects_unsafe_paths(tmp_path: Path, relative: str) -> None:
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text(f"{'0' * 64}  {relative}\n")
    with pytest.raises(ValueError, match="unsafe checksum path"):
        verifier.verify_checksum_file(checksums, root=tmp_path)


def test_checksum_verifier_rejects_symlinked_parent(tmp_path: Path) -> None:
    root = tmp_path / "release"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact = outside / "artifact.bin"
    artifact.write_bytes(b"outside release root\n")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text(f"{_digest(artifact)}  linked/artifact.bin\n")
    with pytest.raises(FileNotFoundError, match="missing or a symlink"):
        verifier.verify_checksum_file(checksums, root=root)
