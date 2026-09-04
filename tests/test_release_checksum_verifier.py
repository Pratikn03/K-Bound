from __future__ import annotations

import hashlib
import shlex
import sys
from pathlib import Path

import pytest
import yaml

from docs.research.kbound.scripts import build_release_source_seal as seal
from docs.research.kbound.scripts import verify_release_checksums as verifier

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


def test_release_checksum_writer_emits_complete_verified_inventory(tmp_path: Path) -> None:
    (tmp_path / "b.bin").write_bytes(b"b\n")
    (tmp_path / "a.bin").write_bytes(b"a\n")
    checksums = tmp_path / "SHA256SUMS.txt"
    assert verifier.write_checksum_file(
        checksums,
        root=tmp_path,
        required_paths=("b.bin", "a.bin"),
    ) == 2
    assert checksums.read_text().splitlines() == [
        f"{_digest(tmp_path / 'a.bin')}  a.bin",
        f"{_digest(tmp_path / 'b.bin')}  b.bin",
    ]
    assert verifier.verify_checksum_file(
        checksums, root=tmp_path, required_paths=("a.bin", "b.bin")
    ) == 2


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


def test_release_test_mode_uses_the_complete_tracked_repository_runner() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "docs/research/kbound/runbooks/release_candidate.sh"
    ).read_text(encoding="utf-8")
    test_body = runbook.split("step_test() {", 1)[1].split("\n}", 1)[0]
    assert '"$KB/scripts/run_repository_verification.py"' in test_body
    assert '--output "$KB/audits/repository_test_inventory.json"' in test_body
    assert _executed_pytest_targets(test_body) == set()


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
    assert {
        "--only-binary=:all:",
        "--require-hashes",
        "-r",
        "requirements-ci-py312-linux.lock.txt",
    } <= _pip_install_arguments(job)


def test_api_ci_executes_unavailable_regressions_under_the_production_profile() -> None:
    job = _workflow_jobs("kbound-ci.yml")["unit-tests"]
    required = {
        "tests/test_kga_api_routes.py",
        "tests/test_kga_masked_inputs.py",
        "tests/test_kga_unavailable_runtime.py",
        "tests/test_kga_unavailable_api.py",
    }
    assert not (required - _executed_pytest_targets(_job_run_text(job)))
    assert {
        "--only-binary=:all:",
        "--require-hashes",
        "-r",
        "requirements-ci-py312-linux.lock.txt",
    } <= _pip_install_arguments(job)


def test_core_ci_installs_the_package_and_legacy_drift_guard_dependency() -> None:
    job = _workflow_jobs("ci.yml")["kbound-core"]
    assert {
        "--only-binary=:all:",
        "--require-hashes",
        "-r",
        "requirements-ci-py312-linux.lock.txt",
    } <= _pip_install_arguments(job)


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
    assert set(verifier.POST_CHECKSUM_RELEASE_PATHS).isdisjoint(paths)
    assert (
        seal.GENERATED_OUTPUT_ALLOWLIST
        - {checksum_self}
        - set(verifier.POST_CHECKSUM_RELEASE_PATHS)
        - seal.NONRELEASE_BUILD_OUTPUT_ALLOWLIST
    ) <= set(paths)
    assert {
        "docs/research/kbound/audits/release_source_seal_2026_08_29.json",
        "docs/research/kbound/audits/formal_foundations_2026_08_31.json",
    } <= set(paths)


def test_checksum_inventory_publishes_exactly_the_four_current_pdf_roles() -> None:
    expected = {
        "docs/research/kbound/release/current/kbound_short_main.pdf",
        "docs/research/kbound/release/current/kbound_short_supplement.pdf",
        "docs/research/kbound/release/current/kbound_tmlr.pdf",
        "docs/research/kbound/release/current/kbound_full_report.pdf",
    }
    current_documents = {
        path
        for path in verifier.REQUIRED_RELEASE_PATHS
        if path.startswith("docs/research/kbound/release/current/")
        and path.endswith((".pdf", ".docx"))
    }

    assert current_documents == expected
    assert (
        "docs/research/kbound/release/current/KBOUND_CURRENT_SHA256SUMS.txt"
        in verifier.REQUIRED_RELEASE_PATHS
    )
    assert not any(path.endswith(".docx") for path in verifier.REQUIRED_RELEASE_PATHS)
    assert {
        "docs/research/kbound/kbound_short_final_draft.pdf",
        "docs/research/kbound/kbound_short_final_draft.docx",
        "docs/research/kbound/kbound_tmlr.pdf",
    }.isdisjoint(verifier.REQUIRED_RELEASE_PATHS)


def test_release_pdf_phase_does_not_build_the_nonrelease_docx() -> None:
    runbook = (
        verifier.ROOT / "docs/research/kbound/runbooks/release_candidate.sh"
    ).read_text(encoding="utf-8")
    pdf_phase = runbook.split("step_pdf() {", 1)[1].split("\n}", 1)[0]

    assert "BUILD_DOCX=1" not in pdf_phase


def test_default_checksum_inventory_never_preauthorizes_so2sat_gate_or_target_outputs() -> None:
    """Protected natural-shift outputs need an explicit authorized release mode."""

    protected_markers = (
        "so2sat_lcz42_prospective",
        "so2sat_numbers",
        "SO2SAT_DEVELOPMENT_RUNTIME",
        "prospective_protocol_v",
        "tent_citymean",
    )
    assert not any(marker in path for marker in protected_markers for path in verifier.REQUIRED_RELEASE_PATHS)


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


def test_checksum_verifier_rejects_symlinked_checksum_receipt(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact\n")
    external = tmp_path / "external-sums.txt"
    external.write_text(f"{_digest(artifact)}  artifact.bin\n", encoding="utf-8")
    linked = tmp_path / "SHA256SUMS.txt"
    linked.symlink_to(external)

    with pytest.raises(FileNotFoundError, match="checksum.*symlink"):
        verifier.verify_checksum_file(linked, root=tmp_path)
