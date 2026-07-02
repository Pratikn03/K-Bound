"""Resolve canonical K-Bound script paths (SSoT: docs/research/kbound/scripts/)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
KBOUND_SCRIPTS = REPO_ROOT / "docs" / "research" / "kbound" / "scripts"


def canonical_script(name: str) -> Path:
    path = KBOUND_SCRIPTS / name
    if not path.is_file():
        raise FileNotFoundError(f"Canonical script missing: {path}")
    return path


def run_canonical(name: str) -> None:
    path = canonical_script(name)
    sys.argv[0] = str(path)
    runpy.run_path(str(path), run_name="__main__")
