"""Require execution of the reviewed synthetic SAR/AETTA regression modules."""
from pathlib import Path
import shlex

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/kbound-ci.yml"
METHOD_REGRESSIONS = {
    "tests/test_aetta_estimate_pair_validation.py",
    "tests/test_aetta_mc_image_cap_regression.py",
    "tests/test_sar_reliable_rollback_regression.py",
}
SAFE_COMMAND_TESTS = METHOD_REGRESSIONS | {
    "tests/test_ci_method_regression_selectors.py",
}


def assert_method_regressions_executed(workflow):
    """Check the dedicated command, not arbitrary shell or transitive imports."""
    commands = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            block = step.get("run", "")
            for line in block.replace("\\\n", " ").splitlines():
                if not line.lstrip().startswith("pytest "):
                    continue
                arguments = shlex.split(line)
                if "--collect-only" in arguments:
                    continue
                targets = {arg for arg in arguments[1:] if arg.startswith("tests/")}
                if SAFE_COMMAND_TESTS <= targets:
                    commands.append(targets)
    assert commands == [SAFE_COMMAND_TESTS], (
        "require one executed dedicated synthetic method-regression command"
    )


def workflow_for(command):
    return {"jobs": {"synthetic": {"steps": [{"run": command}]}}}


def test_ci_executes_reviewed_synthetic_method_regressions():
    assert_method_regressions_executed(yaml.safe_load(WORKFLOW.read_text()))


def test_exact_executed_module_set_is_accepted():
    assert_method_regressions_executed(
        workflow_for("pytest " + " ".join(sorted(SAFE_COMMAND_TESTS)) + " -q")
    )


@pytest.mark.parametrize("missing", sorted(SAFE_COMMAND_TESTS))
def test_omitting_any_required_module_fails(missing):
    command = "pytest " + " ".join(sorted(SAFE_COMMAND_TESTS - {missing}))
    with pytest.raises(AssertionError, match="executed dedicated"):
        assert_method_regressions_executed(workflow_for(command))


def test_collection_is_not_execution():
    command = "pytest --collect-only " + " ".join(sorted(SAFE_COMMAND_TESTS))
    with pytest.raises(AssertionError, match="executed dedicated"):
        assert_method_regressions_executed(workflow_for(command))


def test_adding_protected_reader_to_synthetic_step_fails():
    command = "pytest " + " ".join(sorted(SAFE_COMMAND_TESTS))
    command += " tests/test_canonical_release_data.py"
    with pytest.raises(AssertionError, match="executed dedicated"):
        assert_method_regressions_executed(workflow_for(command))


def test_commented_command_is_not_execution():
    command = "# pytest " + " ".join(sorted(SAFE_COMMAND_TESTS))
    with pytest.raises(AssertionError, match="executed dedicated"):
        assert_method_regressions_executed(workflow_for(command))
