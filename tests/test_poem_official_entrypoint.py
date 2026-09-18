"""The pinned POEM source must at least expose its documented CLI."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_pinned_poem_help_is_runnable() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "external" / "poem_official"
    python = Path(os.environ.get("POEM_PYTHON", "/opt/anaconda3/envs/poem/bin/python"))
    if not python.exists():
        return
    result = subprocess.run(
        [str(python), str(source / "main.py"), "--help"],
        cwd=source,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "--method" in result.stdout
