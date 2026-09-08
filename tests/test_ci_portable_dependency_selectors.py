"""Require direct CI execution of the four portable dependency regressions."""

import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/kbound-ci.yml"
SAFE_COMMAND_TESTS = {
    "tests/test_historical_diagnostic_recovery.py",
    "tests/test_project_historical_diagnostic_receipt.py",
    "tests/test_domainnet_pilot_data_v2.py",
    "tests/test_project_domainnet_stop.py",
    "tests/test_ci_portable_dependency_selectors.py",
}


def assert_portable_regressions_executed(workflow):
    """Accept one dedicated pytest command in the research job, with failures fatal."""
    job = workflow["jobs"].get("kbound-research-tests", {})
    commands = []
    for step in job.get("steps", []):
        block = step.get("run", "").replace("\\\n", " ")
        if not block.lstrip().startswith("pytest "):
            continue
        args = shlex.split(block, comments=True)
        targets = [arg for arg in args[1:] if arg in SAFE_COMMAND_TESTS]
        if not targets:
            continue
        assert "if" not in job and "if" not in step
        assert not job.get("continue-on-error") and not step.get("continue-on-error")
        assert len(targets) == len(SAFE_COMMAND_TESTS) and set(targets) == SAFE_COMMAND_TESTS
        assert [arg for arg in args[1:] if arg not in SAFE_COMMAND_TESTS] == ["-q", "--tb=short"]
        commands.append(targets)
    assert len(commands) == 1, "require one executed dedicated portable dependency regression command"


def fixture(command):
    return {"jobs": {"kbound-research-tests": {"steps": [{"run": command}]}}}


def exact_command():
    return "pytest " + " ".join(sorted(SAFE_COMMAND_TESTS)) + " -q --tb=short"


def test_ci_executes_all_portable_dependency_regressions():
    # Removing the dedicated executed step must fail even when collection remains.
    assert_portable_regressions_executed(yaml.safe_load(WORKFLOW.read_text()))


def test_exact_executed_selector_set_is_accepted():
    assert_portable_regressions_executed(fixture(exact_command()))


@pytest.mark.parametrize(
    ("relative", "should_be_ignored"),
    [
        ("experiments/audit/historical_diagnostic_recovery_receipt.portable.json", False),
        ("experiments/audit/unreviewed_private_result.json", True),
    ],
)
def test_only_the_reviewed_portable_companion_is_visible_to_git(tmp_path, relative, should_be_ignored):
    """A source dependency must not disappear behind a broad artifact ignore rule."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", relative],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == (0 if should_be_ignored else 1), result.stdout + result.stderr


@pytest.mark.parametrize("missing", sorted(SAFE_COMMAND_TESTS))
def test_omitting_any_required_selector_fails(missing):
    with pytest.raises(AssertionError):
        assert_portable_regressions_executed(fixture(exact_command().replace(missing, "")))


@pytest.mark.parametrize(
    "mutation",
    ["collect_only", "comment", "keyword_subset", "ignored_failure", "shell_skip",
     "expanded", "duplicate", "job_conditional", "step_conditional",
     "job_continue_on_error", "step_continue_on_error", "wrong_job"],
)
def test_skipped_expanded_or_fail_open_execution_is_rejected(mutation):
    # Collection, subset filters, shell/YAML skips and ignored failures cannot count as execution.
    command = exact_command()
    if mutation == "collect_only":
        command += " --collect-only"
    elif mutation == "comment":
        command = "# " + command
    elif mutation == "keyword_subset":
        command += " -k never_matches"
    elif mutation == "ignored_failure":
        command += " || true"
    elif mutation == "shell_skip":
        command = "if false; then\n" + command + "\nfi"
    elif mutation == "expanded":
        command += " tests/test_unreviewed_reader.py"
    elif mutation == "duplicate":
        command += " tests/test_domainnet_pilot_data_v2.py"
    workflow = fixture(command)
    job = workflow["jobs"]["kbound-research-tests"]
    if mutation in {"job_conditional", "step_conditional"}:
        target = job if mutation == "job_conditional" else job["steps"][0]
        target["if"] = False
    elif mutation in {"job_continue_on_error", "step_continue_on_error"}:
        target = job if mutation == "job_continue_on_error" else job["steps"][0]
        target["continue-on-error"] = True
    elif mutation == "wrong_job":
        workflow["jobs"] = {"other-job": job}
    with pytest.raises(AssertionError):
        assert_portable_regressions_executed(workflow)
