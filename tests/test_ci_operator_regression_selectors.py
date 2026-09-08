"""Require actual execution of the bounded synthetic operator/dependency repairs."""

import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/kbound-ci.yml"
SAFE_COMMAND_TESTS = {
    "tests/test_operator_wrapper_guards.py",
    "tests/test_repository_bandit_compatibility.py",
    "tests/test_release_bandit_contract.py",
    "tests/test_release_build_dependencies.py",
    "tests/test_ci_operator_regression_selectors.py",
}


def assert_operator_regressions_executed(workflow):
    """Recognize a direct dedicated command, not comments/collection/shell indirection."""
    job = workflow["jobs"].get("kbound-research-tests", {})
    commands = []
    for step in job.get("steps", []):
        for line in step.get("run", "").replace("\\\n", " ").splitlines():
            if not line.lstrip().startswith("pytest "):
                continue
            args = shlex.split(line, comments=True)
            targets = [arg for arg in args[1:] if arg in SAFE_COMMAND_TESTS]
            if not targets:
                continue
            assert "if" not in job and "if" not in step, "require an executed dedicated operator regression command"
            assert not job.get("continue-on-error") and not step.get("continue-on-error"), (
                "require an executed dedicated operator regression command"
            )
            assert len(targets) == len(SAFE_COMMAND_TESTS) and set(targets) == SAFE_COMMAND_TESTS, (
                "require an executed dedicated operator regression command"
            )
            options = [arg for arg in args[1:] if arg not in SAFE_COMMAND_TESTS]
            assert options == ["-q", "--tb=short", "-W", "error"], (
                "require an executed dedicated operator regression command"
            )
            commands.append(targets)
    assert len(commands) == 1, "require one executed dedicated operator regression command"


def fixture(command):
    return {"jobs": {"kbound-research-tests": {"steps": [{"run": command}]}}}


def exact_command():
    return "pytest " + " ".join(sorted(SAFE_COMMAND_TESTS)) + " -q --tb=short -W error"


def test_ci_executes_all_reviewed_operator_and_dependency_regressions():
    assert_operator_regressions_executed(yaml.safe_load(WORKFLOW.read_text()))


def test_exact_executed_selector_set_is_accepted():
    assert_operator_regressions_executed(fixture(exact_command()))


@pytest.mark.parametrize("missing", sorted(SAFE_COMMAND_TESTS))
def test_missing_any_required_selector_fails(missing):
    workflow = fixture(exact_command().replace(missing, ""))
    with pytest.raises(AssertionError, match="executed dedicated"):
        assert_operator_regressions_executed(workflow)


@pytest.mark.parametrize(
    "mutation",
    [
        "collect_only",
        "protected_reader",
        "comment",
        "duplicate",
        "keyword_subset",
        "ignored_failure",
        "conditional",
        "continue_on_error",
    ],
)
def test_nonexecuting_or_expanded_or_fail_open_command_is_rejected(mutation):
    command = exact_command()
    if mutation == "collect_only":
        command += " --collect-only"
    elif mutation == "protected_reader":
        command += " tests/test_canonical_release_data.py"
    elif mutation == "comment":
        command = "# " + command
    elif mutation == "duplicate":
        command += " tests/test_operator_wrapper_guards.py"
    elif mutation == "keyword_subset":
        command += " -k never_matches"
    elif mutation == "ignored_failure":
        command += " || true"
    workflow = fixture(command)
    if mutation == "conditional":
        workflow["jobs"]["kbound-research-tests"]["steps"][0]["if"] = False
    elif mutation == "continue_on_error":
        workflow["jobs"]["kbound-research-tests"]["steps"][0]["continue-on-error"] = True
    with pytest.raises(AssertionError, match="executed dedicated"):
        assert_operator_regressions_executed(workflow)
