"""Declared offline build dependencies must be satisfiable by the release lock."""

import re
from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def test_release_lock_satisfies_declared_build_requirements() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock = (root / "requirements-release-macos-arm64.lock.txt").read_text()
    locked = {
        canonicalize_name(name): version
        for name, version in re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", lock, re.MULTILINE)
    }
    target = {
        "python_version": "3.12",
        "python_full_version": "3.12.13",
        "sys_platform": "darwin",
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "implementation_name": "cpython",
    }
    for text in project["build-system"]["requires"]:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate(target):
            continue
        version = locked.get(canonicalize_name(requirement.name))
        assert version is not None, f"Offline release build dependency is missing: {requirement}"
        assert requirement.specifier.contains(version), (str(requirement), version)
