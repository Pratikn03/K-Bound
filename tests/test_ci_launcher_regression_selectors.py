"""Require direct CI execution of the bounded launcher regression modules."""

import shlex
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/kbound-ci.yml"
SAFE_COMMAND_TESTS = {
    "tests/test_camelyon_results_root_isolation.py",
    "tests/test_cifar_imagenetc_grid_contract.py",
    "tests/test_cifar_percell_output_isolation.py",
    "tests/test_kbtrain_dispatch_safety.py",
    "tests/test_final_showcase_dispatch.py",
    "tests/test_rxrx1_9plus_launcher.py",
    "tests/test_ci_launcher_regression_selectors.py",
}


def assert_launcher_regressions_executed(workflow):
    """Accept one fail-closed pytest command in the research job only."""
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
            assert "if" not in job and "if" not in step
            assert not job.get("continue-on-error") and not step.get("continue-on-error")
            assert len(targets) == len(SAFE_COMMAND_TESTS)
            assert set(targets) == SAFE_COMMAND_TESTS
            options = [arg for arg in args[1:] if arg not in SAFE_COMMAND_TESTS]
            assert options == ["-q", "--tb=short"]
            commands.append(targets)
    assert len(commands) == 1, "require one executed dedicated launcher regression command"


def fixture(command):
    return {"jobs": {"kbound-research-tests": {"steps": [{"run": command}]}}}


def exact_command():
    return "pytest " + " ".join(sorted(SAFE_COMMAND_TESTS)) + " -q --tb=short"


def test_ci_executes_all_reviewed_launcher_regressions():
    assert_launcher_regressions_executed(yaml.safe_load(WORKFLOW.read_text()))


def test_exact_executed_selector_set_is_accepted():
    assert_launcher_regressions_executed(fixture(exact_command()))


@pytest.mark.parametrize("missing", sorted(SAFE_COMMAND_TESTS))
def test_missing_any_required_selector_fails(missing):
    with pytest.raises(AssertionError):
        assert_launcher_regressions_executed(fixture(exact_command().replace(missing, "")))


@pytest.mark.parametrize(
    "mutation",
    ["collect_only", "comment", "non_pytest", "conditional", "continue_on_error"],
)
def test_nonexecuting_or_fail_open_command_is_rejected(mutation):
    command = exact_command()
    if mutation == "collect_only":
        command += " --collect-only"
    elif mutation == "comment":
        command = "# " + command
    elif mutation == "non_pytest":
        command = "python -m " + command
    workflow = fixture(command)
    if mutation == "conditional":
        workflow["jobs"]["kbound-research-tests"]["steps"][0]["if"] = False
    elif mutation == "continue_on_error":
        workflow["jobs"]["kbound-research-tests"]["steps"][0]["continue-on-error"] = True
    with pytest.raises(AssertionError):
        assert_launcher_regressions_executed(workflow)
