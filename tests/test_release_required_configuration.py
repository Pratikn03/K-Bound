"""Pinned runtime configurations must be eligible for exact-path recording."""

import subprocess
from pathlib import Path

import pytest

from docs.research.kbound.scripts import build_release_source_seal as seal

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONFIGURATIONS = (
    "requirements-api-py311-linux.lock.txt",
    "requirements-ci-py312-linux.lock.txt",
    "requirements-ci.txt",
    "requirements-release-macos-arm64.lock.txt",
    "requirements-release.txt",
)


@pytest.mark.parametrize("relative", REQUIRED_CONFIGURATIONS)
def test_required_release_configuration_is_not_ignored(relative):
    assert relative in seal.EXPLICIT_FILES["configuration"]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", relative],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 1, "required release authority is hidden by ignore policy"
