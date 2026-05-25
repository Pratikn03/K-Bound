"""Phase 2.1 — Family-D v1 must remain marked INVALID_FOR_EXECUTION and
its execution-commands file must not become an active scripted entry
point."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVAL_NOTICE = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_V1_INVALIDATION_NOTICE.md"
V1_EXEC = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md"


def test_invalidation_notice_exists():
    assert INVAL_NOTICE.exists(), "FAMILY_D_V1_INVALIDATION_NOTICE.md missing"


def test_invalidation_notice_marks_v1_as_invalid_for_execution():
    t = INVAL_NOTICE.read_text()
    assert "INVALID_FOR_EXECUTION" in t, (
        "invalidation notice must state INVALID_FOR_EXECUTION"
    )


def test_invalidation_notice_lists_grounds():
    t = INVAL_NOTICE.read_text().lower()
    for required in (
        "placeholder",
        "mpdd",
        "visa",
        "eyecandies",
        "audited-reanalysis",
    ):
        assert required in t, (
            f"invalidation notice missing required ground keyword: {required!r}"
        )


def test_v1_execution_commands_remain_marked_not_run():
    """The v1 execution-commands file must continue to be marked as
    not-yet-run, and must not be referenced as an active script entry
    point from a scripts/ shim."""
    assert V1_EXEC.exists()
    t = V1_EXEC.read_text()
    assert "NOT RUN" in t or "not run" in t.lower(), (
        "v1 execution commands must remain labelled NOT RUN"
    )

    # No active shim script should treat the v1 commands as live.
    scripts_dir = ROOT / "src" / "scripts"
    if scripts_dir.exists():
        for p in scripts_dir.glob("**/*.py"):
            try:
                txt = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            assert "FAMILY_D_EXECUTION_COMMANDS_NOT_RUN" not in txt, (
                f"{p} imports / references the v1 NOT_RUN command set as active"
            )
