"""Keep maintained DomainNet code discoverable without admitting private data."""

import subprocess
from pathlib import Path

from docs.research.kbound.scripts import run_repository_verification as runner

SOURCE_FILES = (
    "experiments/kbound/domainnet/__init__.py",
    "experiments/kbound/domainnet/source_data.py",
    "experiments/kbound/domainnet/train_source.py",
    "experiments/kbound/domainnet/pilot_data.py",
    "experiments/kbound/domainnet/pilot_data_v2.py",
    "experiments/kbound/domainnet/pilot_candidate.py",
    "experiments/kbound/domainnet/pilot_analysis.py",
    "experiments/kbound/domainnet/pilot_runner.py",
)

PRIVATE_FILES = (
    "experiments/kbound/domainnet/clipart.zip",
    "experiments/kbound/domainnet/data/clipart/image.jpg",
    "experiments/kbound/domainnet/seed-0/final.pt",
    "experiments/kbound/domainnet/labels.txt",
    "experiments/kbound/domainnet/results.json",
    "experiments/kbound/domainnet/__pycache__/source_data.cpython-312.pyc",
    "experiments/kbound/domainnet/test_unreviewed.py",
    "experiments/kbound/domainnet/unreviewed.py",
    "experiments/kbound/domainnet/data/test_label_trap.py",
    "experiments/kbound/domainnet/pilot_raw.py",
    "experiments/kbound/domainnet/nested/pilot_runner.py",
    "experiments/kbound/domainnet_sibling/pilot_runner.py",
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


def test_ignore_rules_include_only_maintained_domainnet_sources(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    ignore = Path(__file__).resolve().parents[1] / ".gitignore"
    (tmp_path / ".gitignore").write_bytes(ignore.read_bytes())
    for name in (*SOURCE_FILES, *PRIVATE_FILES):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == (1 if name in SOURCE_FILES else 0), name


def test_committed_inventory_includes_source_but_not_raw_domainnet(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Synthetic Inventory Test")
    _git(tmp_path, "config", "user.email", "inventory@example.invalid")
    ignore = Path(__file__).resolve().parents[1] / ".gitignore"
    (tmp_path / ".gitignore").write_bytes(ignore.read_bytes())
    for name in (*SOURCE_FILES, *PRIVATE_FILES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic bytes; never imported\n", encoding="utf-8")
    # Force-tracked private artifacts must remain outside the positive inventory.
    _git(tmp_path, "add", "--force", "--", *SOURCE_FILES, *PRIVATE_FILES)
    _git(tmp_path, "commit", "-qm", "synthetic source freeze")
    assert runner.tracked_paths(tmp_path) == sorted(SOURCE_FILES)
