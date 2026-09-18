from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/audit_official_baselines.py"


def invoke(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo), "--out-dir", str(repo / "results"), *arguments],
        capture_output=True, text=True, check=False,
    )


def test_required_audit_cannot_silently_succeed_without_evidence(tmp_path: Path) -> None:
    result = invoke(tmp_path, "--require-promotable")
    assert result.returncode == 2
    report = json.loads((tmp_path / "results/OFFICIAL_BASELINE_VERIFICATION.json").read_text())
    assert report["status"] == "OPEN"
    assert report["official_label_allowed"] is False
    assert set(report["methods"]) == {"aetta", "poem"}
    assert all(item["blockers"] for item in report["methods"].values())


def test_diagnostic_mode_reports_missing_evidence_without_claiming_a_benchmark(tmp_path: Path) -> None:
    result = invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "OPEN"
    assert report["native_execution_launched"] is False
    assert report["benchmark_complete"] is False


def test_existing_evidence_and_report_are_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    audit = output / "OFFICIAL_BASELINE_AUDIT.json"
    audit.write_text('{"historical":true}\n')
    report = output / "OFFICIAL_BASELINE_VERIFICATION.json"
    report.write_bytes(b"preserve this earlier verification\n")
    before = audit.read_bytes(), report.read_bytes()
    result = invoke(tmp_path, "--require-promotable")
    assert result.returncode == 2
    assert (audit.read_bytes(), report.read_bytes()) == before


def test_audit_output_cannot_escape_repository_via_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "results").symlink_to(outside, target_is_directory=True)
    result = invoke(repo, "--require-promotable")
    assert result.returncode == 2
    assert list(outside.iterdir()) == []


def test_require_promotable_option_is_available() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--require-promotable" in result.stdout
