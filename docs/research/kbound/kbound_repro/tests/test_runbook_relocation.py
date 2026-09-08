"""Phase-2 relocation test.

Copies the reproducibility toolkit + release runbook into a DIFFERENT absolute
location (a temporary fake repo) and verifies that the runbook still resolves
its root there and runs `preflight` -- i.e. no current command depends on the
original user-home or external-volume checkout path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kbound_repro import storage  # noqa: E402

KB = Path(__file__).resolve().parents[2]  # .../docs/research/kbound
REPO = Path(__file__).resolve().parents[5]  # repo root
RUNBOOK = KB / "runbooks" / "release_candidate.sh"
TOOL_OVERRIDES = (
    "KBOUND_TOOL_LATEXMK",
    "KBOUND_TOOL_LATEXPAND",
    "KBOUND_TOOL_PANDOC",
    "KBOUND_TOOL_PERL",
    "KBOUND_TOOL_PDFLATEX",
    "KBOUND_TOOL_PDFINFO",
    "KBOUND_TOOL_PDFTOPPM",
    "KBOUND_TOOL_PDFTOTEXT",
    "KBOUND_TOOL_PDFDETACH",
    "KBOUND_TOOL_SOFFICE",
)


def _write_synthetic_verifier_interfaces(kb_dst: Path, *, fail_python: bool = False) -> None:
    """Provide the real runbook's verifier CLIs without claiming release verification."""
    scripts = kb_dst / "scripts"
    scripts.mkdir()
    (kb_dst.parent.parent.parent / "requirements-release-macos-arm64.lock.txt").write_text(
        "SYNTHETIC RELOCATION FIXTURE ONLY\n", encoding="utf-8"
    )
    (kb_dst / "release_python_environment_macos_arm64.json").write_text(
        '{"synthetic_fixture": true, "role": "historical_v1"}\n', encoding="utf-8"
    )
    (kb_dst / "release_python_environment_macos_arm64_v2.json").write_text(
        '{"synthetic_fixture": true, "role": "active_v2"}\n', encoding="utf-8"
    )
    (kb_dst / "release_toolchain_macos_arm64.json").write_text('{"synthetic_fixture": true}\n', encoding="utf-8")
    (kb_dst / "release_toolchain_macos_arm64_v2.json").write_text('{"synthetic_fixture": true}\n', encoding="utf-8")
    (kb_dst / "audits").mkdir(exist_ok=True)
    (kb_dst / "audits/release_toolchain_2026_09_02.json").write_text("historical fixture\n", encoding="utf-8")
    (kb_dst / "audits/release_toolchain_2026_09_05_v2.json").write_text('{"synthetic_fixture": true}\n', encoding="utf-8")

    python_failure = "True" if fail_python else "False"
    (scripts / "verify_python_environment.py").write_text(
        """import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--lock", required=True)
parser.add_argument("--content-profile", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
assert Path(args.lock).is_file()
assert Path(args.content_profile).name == "release_python_environment_macos_arm64_v2.json"
assert Path(args.content_profile).is_file()
if PYTHON_FAILURE:
    raise SystemExit("synthetic locked-runtime mismatch")
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"synthetic_fixture": True}) + "\\n", encoding="utf-8")
""".replace("PYTHON_FAILURE", python_failure),
        encoding="utf-8",
    )
    (scripts / "verify_release_toolchain.py").write_text(
        """import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--profile", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--resolved-tools-output", required=True)
args = parser.parse_args()
assert Path(args.profile).name == "release_toolchain_macos_arm64_v2.json"
assert Path(args.profile).is_file()
root = Path(args.profile).resolve().parent
assert Path(args.output).parent != root / "audits"
(root / "toolchain-invoked.txt").write_text("synthetic fixture\\n", encoding="utf-8")
tools = root / "synthetic-tools"
tools.mkdir(exist_ok=True)
overrides = TOOL_OVERRIDES
rows = []
for override in overrides:
    executable = tools / override.lower()
    executable.write_text("#!/bin/sh\\nexit 0\\n", encoding="utf-8")
    executable.chmod(0o755)
    rows.append(f"{override}\\t{executable.resolve()}\\n")
Path(args.resolved_tools_output).write_text("".join(rows), encoding="utf-8")
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"synthetic_fixture": True}) + "\\n", encoding="utf-8")
""".replace("TOOL_OVERRIDES", repr(TOOL_OVERRIDES)),
        encoding="utf-8",
    )


def _relocated_repo(tmp_path: Path, *, fail_python: bool = False) -> tuple[Path, Path]:
    reloc = tmp_path / "relocated_repo"
    kb_dst = reloc / "docs" / "research" / "kbound"
    (kb_dst / "runbooks").mkdir(parents=True)
    (reloc / "pyproject.toml").write_text("[project]\nname='reloc'\n", encoding="utf-8")
    (reloc / ".git").mkdir()
    shutil.copytree(KB / "kbound_repro", kb_dst / "kbound_repro")
    shutil.copy(RUNBOOK, kb_dst / "runbooks" / "release_candidate.sh")
    shutil.copy(KB / "claim_ledger.json", kb_dst / "claim_ledger.json")
    _write_synthetic_verifier_interfaces(kb_dst, fail_python=fail_python)
    return reloc, kb_dst


def _run_preflight(tmp_path: Path, kb_dst: Path, *, mode: str = "preflight") -> subprocess.CompletedProcess[str]:
    env = dict(
        os.environ,
        KBOUND_PYTHON=sys.executable,
        # The actual runbook must override contrary inherited build switches.
        BUILD_LONG_TMLR="0",
        BUILD_SHORT_MAIN="0",
        BUILD_SHORT_SUPPLEMENT="0",
        BUILD_FULL_REPORT="0",
        BUILD_DOCX="1",
    )
    return subprocess.run(
        ["bash", str(kb_dst / "runbooks" / "release_candidate.sh"), mode],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=30,
    )


def _write_synthetic_pdf_tools(kb_dst: Path, *, build_fails: bool = False) -> None:
    """Observe the real runbook's child contract without building a manuscript."""
    (kb_dst / "scripts/build_pdfs.sh").write_text(
        """#!/bin/bash
set -eu
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
keys = ("BUILD_LONG_TMLR", "BUILD_SHORT_MAIN", "BUILD_SHORT_SUPPLEMENT",
        "BUILD_FULL_REPORT", "BUILD_DOCX", "PYTHON")
Path("docs/research/kbound/build-observed.json").write_text(
    json.dumps({key: os.environ.get(key) for key in keys}), encoding="utf-8")
PY
"""
        + ("exit 19\n" if build_fails else "exit 0\n"),
        encoding="utf-8",
    )
    (kb_dst / "scripts/render_pdf_pages.py").write_text(
        """from pathlib import Path
root = Path("docs/research/kbound")
assert (root / "build-observed.json").is_file(), "renderer ran before build"
(root / "render-observed.txt").write_text("synthetic render only\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )


def test_runbook_has_no_hardcoded_machine_paths():
    # The current commands (runbook + release gate) must be fully portable.
    files = [
        str(RUNBOOK.relative_to(REPO)),
        "docs/research/kbound/kbound_repro/release_checks.py",
    ]
    flagged = storage.scan_absolute_paths(files, root=REPO)
    assert flagged == [], f"hard-coded machine paths found: {flagged}"


def test_toolkit_clean_under_self_allowlist():
    # The detector's own files contain the patterns by necessity; with the
    # documented self-allowlist the whole toolkit scans clean.
    files = [str(p.relative_to(REPO)) for p in (KB / "kbound_repro").glob("*.py")]
    flagged = storage.scan_absolute_paths(files, root=REPO, provenance_allowlist=storage.SELF_ALLOWLIST)
    assert flagged == [], f"unexpected machine paths: {flagged}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_runbook_resolves_relocated_root_and_runs_preflight(tmp_path):
    # Build a minimal relocated repo at a brand-new absolute path.
    reloc, kb_dst = _relocated_repo(tmp_path)
    proc = _run_preflight(tmp_path, kb_dst)
    assert proc.returncode == 0, proc.stderr
    # It resolved the RELOCATED root, not the original checkout.
    assert f"repo root: {reloc}" in proc.stdout, proc.stdout
    assert str(REPO) not in proc.stdout
    assert (kb_dst / "audits/python_environment_2026_09_02.json").is_file()
    assert (kb_dst / "audits/release_toolchain_2026_09_02.json").is_file()
    assert (kb_dst / "audits/release_toolchain_2026_09_02.json").read_text() == "historical fixture\n"
    assert (kb_dst / "audits/release_toolchain_2026_09_05_v2.json").read_text() == '{"synthetic_fixture": true}\n'


@pytest.mark.parametrize("fault", ["profile_missing", "receipt_missing", "receipt_changed", "receipt_symlink", "verifier_failure"])
def test_toolchain_v2_failures_preserve_maintained_receipts_and_stop_children(tmp_path, fault):
    _, kb_dst = _relocated_repo(tmp_path)
    _write_synthetic_pdf_tools(kb_dst)
    historical = kb_dst / "audits/release_toolchain_2026_09_02.json"
    current = kb_dst / "audits/release_toolchain_2026_09_05_v2.json"
    before = historical.read_bytes()
    if fault == "profile_missing":
        (kb_dst / "release_toolchain_macos_arm64_v2.json").unlink()
    elif fault == "receipt_missing":
        current.unlink()
    elif fault == "receipt_changed":
        current.write_text("previously accepted different receipt\n")
    elif fault == "receipt_symlink":
        current.unlink()
        current.symlink_to(historical)
    else:
        (kb_dst / "scripts/verify_release_toolchain.py").write_text("raise SystemExit(9)\n")
    expected_current = current.read_bytes() if current.exists() else None
    proc = _run_preflight(tmp_path, kb_dst, mode="pdf")
    assert proc.returncode != 0
    assert not (kb_dst / "build-observed.json").exists()
    assert historical.read_bytes() == before
    assert (current.read_bytes() if current.exists() else None) == expected_current


def test_toolchain_success_then_mismatch_never_reuses_prior_paths(tmp_path):
    _, kb_dst = _relocated_repo(tmp_path)
    first = _run_preflight(tmp_path, kb_dst)
    assert first.returncode == 0, first.stderr
    current = kb_dst / "audits/release_toolchain_2026_09_05_v2.json"
    current.write_text("different maintained authority\n")
    _write_synthetic_pdf_tools(kb_dst)
    second = _run_preflight(tmp_path, kb_dst, mode="pdf")
    assert second.returncode != 0
    assert current.read_text() == "different maintained authority\n"
    assert not (kb_dst / "build-observed.json").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_runbook_preflight_stops_before_toolchain_after_python_verifier_failure(tmp_path):
    _, kb_dst = _relocated_repo(tmp_path, fail_python=True)

    proc = _run_preflight(tmp_path, kb_dst)

    assert proc.returncode != 0
    assert "synthetic locked-runtime mismatch" in proc.stderr
    assert not (kb_dst / "toolchain-invoked.txt").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_runbook_missing_v2_does_not_fall_back_to_historical_profile(tmp_path):
    _, kb_dst = _relocated_repo(tmp_path)
    (kb_dst / "release_python_environment_macos_arm64_v2.json").unlink()
    assert (kb_dst / "release_python_environment_macos_arm64.json").is_file()

    proc = _run_preflight(tmp_path, kb_dst)

    assert proc.returncode != 0
    assert not (kb_dst / "toolchain-invoked.txt").exists()
    assert not (kb_dst / "audits/python_environment_2026_09_02.json").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_runbook_pdf_executes_all_four_roles_before_rendering(tmp_path):
    _, kb_dst = _relocated_repo(tmp_path)
    _write_synthetic_pdf_tools(kb_dst)

    proc = _run_preflight(tmp_path, kb_dst, mode="pdf")

    assert proc.returncode == 0, proc.stderr
    observed = json.loads((kb_dst / "build-observed.json").read_text(encoding="utf-8"))
    assert observed == {
        "BUILD_LONG_TMLR": "1",
        "BUILD_SHORT_MAIN": "1",
        "BUILD_SHORT_SUPPLEMENT": "1",
        "BUILD_FULL_REPORT": "1",
        "BUILD_DOCX": "0",
        "PYTHON": sys.executable,
    }
    assert (kb_dst / "render-observed.txt").read_text() == "synthetic render only\n"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_runbook_pdf_propagates_build_failure_before_rendering(tmp_path):
    _, kb_dst = _relocated_repo(tmp_path)
    _write_synthetic_pdf_tools(kb_dst, build_fails=True)

    proc = _run_preflight(tmp_path, kb_dst, mode="pdf")

    assert proc.returncode != 0
    assert (kb_dst / "build-observed.json").is_file()
    assert not (kb_dst / "render-observed.txt").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_runbook_pdf_requires_both_build_and_render_tools(tmp_path):
    _, kb_dst = _relocated_repo(tmp_path)
    proc = _run_preflight(tmp_path, kb_dst, mode="pdf")
    assert proc.returncode != 0
    assert not (kb_dst / "build-observed.json").exists()

    _write_synthetic_pdf_tools(kb_dst)
    (kb_dst / "scripts/render_pdf_pages.py").unlink()
    proc = _run_preflight(tmp_path, kb_dst, mode="pdf")
    assert proc.returncode != 0
    assert (kb_dst / "build-observed.json").is_file()
    assert not (kb_dst / "render-observed.txt").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
