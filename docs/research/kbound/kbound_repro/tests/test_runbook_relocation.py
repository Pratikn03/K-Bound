"""Phase-2 relocation test.

Copies the reproducibility toolkit + release runbook into a DIFFERENT absolute
location (a temporary fake repo) and verifies that the runbook still resolves
its root there and runs `preflight` -- i.e. no current command depends on the
original user-home or external-volume checkout path.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kbound_repro import storage  # noqa: E402

KB = Path(__file__).resolve().parents[2]            # .../docs/research/kbound
REPO = Path(__file__).resolve().parents[5]          # repo root
RUNBOOK = KB / "runbooks" / "release_candidate.sh"


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
    flagged = storage.scan_absolute_paths(files, root=REPO,
                                          provenance_allowlist=storage.SELF_ALLOWLIST)
    assert flagged == [], f"unexpected machine paths: {flagged}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_runbook_resolves_relocated_root_and_runs_preflight(tmp_path):
    # Build a minimal relocated repo at a brand-new absolute path.
    reloc = tmp_path / "relocated_repo"
    kb_dst = reloc / "docs" / "research" / "kbound"
    (kb_dst / "runbooks").mkdir(parents=True)
    (reloc / "pyproject.toml").write_text("[project]\nname='reloc'\n")
    (reloc / ".git").mkdir()
    shutil.copytree(KB / "kbound_repro", kb_dst / "kbound_repro")
    shutil.copy(RUNBOOK, kb_dst / "runbooks" / "release_candidate.sh")
    shutil.copy(KB / "claim_ledger.json", kb_dst / "claim_ledger.json")

    proc = subprocess.run(
        ["bash", str(kb_dst / "runbooks" / "release_candidate.sh"), "preflight"],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    # It resolved the RELOCATED root, not the original checkout.
    assert f"repo root: {reloc}" in proc.stdout, proc.stdout
    assert str(REPO) not in proc.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
