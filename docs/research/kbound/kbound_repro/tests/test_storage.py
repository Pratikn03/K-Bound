"""Storage-policy / guardrail tests (Phase 8)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kbound_repro import storage, check_repo  # noqa: E402


@pytest.mark.parametrize("path,expected", [
    ("AETTA/cached_data/cifar10/x.pkl", "cache"),
    ("AETTA/dataset/CIFAR-10-C/fog.npy", "dataset"),
    ("experiments/kbound/results/x/model.pt", "checkpoint"),
    ("experiments/kbound/results/x/run.log", "raw_log"),
    ("data/archive.tar.gz", "archive"),
    (".venv/lib/python3.11/site.py", "virtualenv"),
    ("external/DomainBed/foo.py", "external_repo"),
    ("docs/research/kbound/._foo", "macos_junk"),
    ("docs/research/kbound/claim_ledger.json", "tracked_evidence"),
    ("docs/research/kbound/kbound_repro/metrics.py", "tracked_source"),
])
def test_path_class(path, expected):
    assert storage.path_class(path) == expected


def test_scan_forbidden_paths():
    files = [
        "AETTA/cached_data/x.pkl",
        "AETTA/dataset/CIFAR-10-C/fog.npy",
        "docs/research/kbound/claim_ledger.json",  # OK
        "experiments/kbound/results/x/model.pt",
    ]
    flagged = {r["path"] for r in storage.scan_forbidden_paths(files)}
    assert "docs/research/kbound/claim_ledger.json" not in flagged
    assert "AETTA/cached_data/x.pkl" in flagged
    assert "experiments/kbound/results/x/model.pt" in flagged


def test_scan_large_files_with_allowlist(tmp_path):
    big_json = tmp_path / "big.json"
    big_json.write_bytes(b"x" * (6 * 1024 * 1024))
    big_pdf = tmp_path / "fig.pdf"
    big_pdf.write_bytes(b"x" * (6 * 1024 * 1024))
    flagged = storage.scan_large_files(["big.json", "fig.pdf"], root=tmp_path)
    paths = {r["path"] for r in flagged}
    assert "big.json" in paths       # oversized non-figure -> flagged
    assert "fig.pdf" not in paths     # allowlisted figure -> ok


def test_scan_absolute_paths(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("from kbound_repro.paths import find_repo_root\nroot = find_repo_root()\n")
    bad = tmp_path / "bad.sh"
    bad.write_text("#!/bin/bash\ncd /" + "Volumes/T9/uav/AutoML_Flagship_V8 || exit 1\n")
    bad2 = tmp_path / "bad2.py"
    bad2.write_text("DATA = '/" + "Users/pratik_n/kbound_data'\n")
    flagged = storage.scan_absolute_paths(["good.py", "bad.sh", "bad2.py"], root=tmp_path)
    hits = {r["path"] for r in flagged}
    assert hits == {"bad.sh", "bad2.py"}


def test_scan_absolute_paths_provenance_allowlist(tmp_path):
    prov = tmp_path / "provenance.yaml"
    prov.write_text("original_capture_path: /" + "Volumes/T9/uav/raw\n")
    flagged = storage.scan_absolute_paths(
        ["provenance.yaml"], root=tmp_path, provenance_allowlist=["provenance.yaml"]
    )
    assert flagged == []


def test_check_repo_cli_fails_closed(tmp_path, capsys):
    bad = tmp_path / "bad.sh"
    bad.write_text("cd /" + "Volumes/T9/uav/AutoML_Flagship_V8\n")
    rc = check_repo.main(["--files", "bad.sh", "--root", str(tmp_path), "--check", "abspaths"])
    assert rc == 1


def test_check_repo_cli_passes_clean(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text("x = 1\n")
    rc = check_repo.main(["--files", "ok.py", "--root", str(tmp_path)])
    assert rc == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
