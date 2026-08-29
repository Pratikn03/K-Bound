"""Static and dry-run guards for the development-only multi-seed orchestrator."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNBOOK = REPO / "docs/research/kbound/scripts/run_multiseed.sh"
RXRX1_SUPERVISOR = REPO / "docs/research/kbound/scripts/supervise_rxrx1_9plus.sh"
OFFICEHOME_SUPERVISOR = REPO / "experiments/kbound/officehome/supervise_oh.sh"


def test_multiseed_runbook_is_valid_bash_and_all_is_development_only():
    syntax = subprocess.run(
        ["bash", "-n", str(RUNBOOK)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    environment = dict(os.environ, KBOUND_DRY_RUN="1")
    dry_run = subprocess.run(
        ["bash", str(RUNBOOK), "all"],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "development-safe Office-Home, PACS, and iWildCam only" in dry_run.stdout
    assert "Camelyon17 and RxRx1 held-out paths remain disabled" in dry_run.stdout


def test_active_runbook_has_no_target_test_or_stale_aggregate_execution_path():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "for role in target_val target_test" not in text
    assert "for split in val test" not in text
    assert "--split test" not in text
    assert "json_has_keys" not in text
    assert '"$K/scripts/multiseed_natural.py"' not in text
    assert '"$OUT/*/extracted/multiseed_*.json"' not in text
    assert "checkpoint_log_matches" in text
    assert "CURRENT_AGGREGATES" in text
    assert "KBOUND_ALLOW_MISSING_SOURCES" in text


def test_retired_target_tracks_exit_before_any_runner_command(tmp_path):
    for track in ("camelyon", "rxrx1"):
        environment = dict(os.environ, OUT=str(tmp_path / track))
        result = subprocess.run(
            ["bash", str(RUNBOOK), track],
            cwd=REPO,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 64
        assert "No target data were opened" in result.stderr
        assert not (tmp_path / track).exists() or not any((tmp_path / track).iterdir())


def test_rxrx1_multimodel_supervisor_verifies_bound_completion_receipts():
    syntax = subprocess.run(
        ["zsh", "-n", str(RXRX1_SUPERVISOR)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    text = RXRX1_SUPERVISOR.read_text(encoding="utf-8")
    assert "--verify-completion" in text
    assert text.count("completion_receipt_valid") >= 4
    assert "removed invalid/stale completion receipt" in text
    assert 'rm -f "$ALLDONE_PATH"' in text
    assert '"${RUN_ARGS[@]}"' in text
    assert '--model-seed "$S"' in text
    assert 'set_seed_context "$S"' in text
    assert '.done present before attempt' not in text
    assert '.done after attempt' not in text


def test_officehome_supervisor_clears_stale_marker_before_attempting_run():
    syntax = subprocess.run(
        ["bash", "-n", str(OFFICEHOME_SUPERVISOR)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    text = OFFICEHOME_SUPERVISOR.read_text(encoding="utf-8")
    assert 'DONE="$OUT/.${ROLE}.done"' in text
    assert 'rm -f "$DONE"' in text
    assert 'touch "$DONE"' in text
