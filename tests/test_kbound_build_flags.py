"""Behavioral checks for the maintained K-Bound PDF build interface."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "docs/research/kbound/scripts/build_pdfs.sh"
MANUSCRIPT_SOURCES = ROOT / "docs/research/kbound/kbound_repro/manuscript_sources.py"


def test_build_rejects_invalid_full_flag_before_dependency_checks() -> None:
    """A malformed full-build request must fail before invoking build tools."""

    env = os.environ.copy()
    env.update({"PATH": "/usr/bin:/bin", "BUILD_FULL": "2"})
    result = subprocess.run(
        ["/bin/bash", str(BUILD_SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "ERROR: BUILD_FULL must be 0 or 1" in result.stderr


def test_full_supplement_is_inside_the_maintained_claim_scan() -> None:
    """The claim firewall must traverse the full driver's complete input closure."""

    spec = importlib.util.spec_from_file_location("kbound_manuscript_sources", MANUSCRIPT_SOURCES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    relative_paths = {
        path.relative_to(ROOT).as_posix() for path in module.active_source_paths(ROOT)
    }
    assert "docs/research/kbound/kbound_full.tex" in relative_paths
    assert "docs/research/kbound/kbound_full_supplement.tex" in relative_paths
