"""Boundary contracts for the four maintained K-Bound PDF artifacts."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "docs/research/kbound"
BUILD_SCRIPT = PAPER / "scripts/build_pdfs.sh"


def _active_tex_inputs(path: Path) -> set[str]:
    """Return active direct inputs, ignoring TeX comments."""
    source = path.read_text(encoding="utf-8")
    uncommented = "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in source.splitlines())
    return set(re.findall(r"\\input\s*\{([^}]+)\}", uncommented))


def _run_with_flag(name: str, value: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    for flag in ("BUILD_SHORT_MAIN", "BUILD_SHORT_SUPPLEMENT", "BUILD_FULL_REPORT"):
        environment.pop(flag, None)
    environment.update(
        {
            name: value,
            # This invalid override would produce a distinct discovery error if
            # validation did not happen before release-tool discovery.
            "KBOUND_TOOL_LATEXMK": "relative-tool",
        }
    )
    return subprocess.run(
        ["bash", str(BUILD_SCRIPT)],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_four_artifact_drivers_preserve_their_input_boundaries() -> None:
    """A supplement leak or a retired input must fail this publication boundary."""
    short_main_driver = _active_tex_inputs(PAPER / "kbound_short_main.tex")
    standalone_supplement_driver = _active_tex_inputs(PAPER / "kbound_short_supplement.tex")
    full_report_driver = _active_tex_inputs(PAPER / "kbound_full_report.tex")

    assert "kbound_submission_supplement" not in short_main_driver
    assert "kbound_submission_supplement" in standalone_supplement_driver
    assert "kbound_full_report_extensions" in full_report_driver
    assert "legacy_publication_surfaces" not in full_report_driver


def test_release_identity_is_generated_before_claim_validation() -> None:
    """A clean checkout must create the shared include before scanning TeX."""
    build = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert build.index("scripts/generate_release_identity.py") < build.index(
        "src/scripts/validate_manuscript_claims.py"
    )


def test_active_tex_input_parser_ignores_commented_directives(tmp_path: Path) -> None:
    """A commented boundary input cannot make a publication-driver test pass."""
    driver = tmp_path / "driver.tex"
    driver.write_text("% \\input{retired}\n\\input{current}\n", encoding="utf-8")

    assert _active_tex_inputs(driver) == {"current"}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("BUILD_SHORT_MAIN", "invalid"),
        ("BUILD_SHORT_MAIN", ""),
        ("BUILD_SHORT_SUPPLEMENT", "invalid"),
        ("BUILD_SHORT_SUPPLEMENT", ""),
        ("BUILD_FULL_REPORT", "invalid"),
        ("BUILD_FULL_REPORT", ""),
    ],
)
def test_invalid_four_artifact_flag_fails_before_release_tool_discovery(name: str, value: str) -> None:
    """A malformed opt-in flag must not reach release-tool discovery."""
    result = _run_with_flag(name, value)

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert f"{name} must be 0 or 1" in output
    assert "KBOUND_TOOL_LATEXMK" not in output


def test_flag_subprocess_ignores_ambient_opt_in_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the explicitly supplied malformed flag controls this subprocess."""
    monkeypatch.setenv("BUILD_SHORT_MAIN", "1")
    monkeypatch.setenv("BUILD_SHORT_SUPPLEMENT", "1")
    monkeypatch.setenv("BUILD_FULL_REPORT", "1")

    result = _run_with_flag("BUILD_SHORT_MAIN", "")

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "BUILD_SHORT_MAIN must be 0 or 1" in output
    assert "KBOUND_TOOL_LATEXMK" not in output
