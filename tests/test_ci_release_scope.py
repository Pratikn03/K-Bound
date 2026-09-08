"""Static CI authority/selector contracts: no workflow, inventory or data execution."""
import ast
from pathlib import Path
import shlex

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/kbound-ci.yml'
INVENTORY = ROOT / 'docs/research/kbound/scripts/run_repository_verification.py'
SAFE_RELEASE_TESTS = {
    'tests/test_release_provenance_integration.py',
    'tests/test_ci_release_scope.py',
}
NATURAL_WRITER = 'docs/research/kbound/scripts/audit_natural_target_provenance.py'


def protected_paths():
    # Read the maintained policy as literal syntax, never import/run the
    # inventory or any protected test/validator module.
    tree = ast.parse(INVENTORY.read_text())
    wanted = {'PROTECTED_SO2SAT_TEST_MODULES', 'PROTECTED_SO2SAT_VALIDATORS'}
    found = set()
    paths = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        value = node.value
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
        assert value.func.id == 'frozenset' and len(value.args) == 1 and not value.keywords
        paths.update(ast.literal_eval(value.args[0]))
        found.add(target.id)
    assert found == wanted and paths
    return paths


def run_blocks(workflow):
    return [step.get('run', '') for job in workflow['jobs'].values()
            for step in job.get('steps', []) if 'run' in step]


def assert_no_unscoped_paths(workflow, forbidden):
    # The current automatic workflow has no reviewed protected-data authority
    # profile. Reject explicit forbidden paths anywhere in its run code; this
    # intentionally does not purport to audit arbitrary transitive shell code.
    for block in run_blocks(workflow):
        code = '\n'.join(line for line in block.splitlines() if not line.lstrip().startswith('#'))
        for path in forbidden:
            assert path not in code, f'default CI contains unscoped protected reader/writer: {path}'


def assert_release_tests_executed(workflow):
    selected = []
    for block in run_blocks(workflow):
        for line in block.replace('\\\n', ' ').splitlines():
            if not line.lstrip().startswith('pytest '):
                continue
            arguments = shlex.split(line)
            if '--collect-only' in arguments:
                continue
            targets = {arg for arg in arguments[1:] if arg.startswith('tests/')}
            if SAFE_RELEASE_TESTS <= targets:
                selected.append(targets)
    assert selected == [SAFE_RELEASE_TESTS], 'require one executed dedicated safe release command'


def test_default_ci_preserves_repository_protected_reader_policy():
    assert_no_unscoped_paths(yaml.safe_load(WORKFLOW.read_text()), protected_paths())


def test_default_ci_does_not_run_broad_natural_provenance_writer():
    assert_no_unscoped_paths(yaml.safe_load(WORKFLOW.read_text()), {NATURAL_WRITER})


def test_ci_executes_safe_release_scope_and_interface_regressions():
    assert_release_tests_executed(yaml.safe_load(WORKFLOW.read_text()))


@pytest.mark.parametrize('path', [
    'tests/test_canonical_release_data.py',
    'tests/test_so2sat_numbers_builder.py',
    'docs/research/kbound/scripts/validate_canonical_release_data.py',
    NATURAL_WRITER,
])
def test_injected_protected_reader_or_writer_fails_static_guard(path):
    workflow = {'jobs': {'synthetic': {'steps': [{'run': f'python -m pytest {path}'}]}}}
    with pytest.raises(AssertionError, match='unscoped protected'):
        assert_no_unscoped_paths(workflow, protected_paths() | {NATURAL_WRITER})


def test_collection_alone_cannot_substitute_for_release_test_execution():
    command = 'pytest --collect-only ' + ' '.join(sorted(SAFE_RELEASE_TESTS))
    workflow = {'jobs': {'synthetic': {'steps': [{'run': command}]}}}
    with pytest.raises(AssertionError, match='executed dedicated'):
        assert_release_tests_executed(workflow)


def test_extra_protected_target_cannot_hide_in_dedicated_safe_command():
    command = 'pytest ' + ' '.join(sorted(SAFE_RELEASE_TESTS)) + ' tests/test_canonical_release_data.py'
    workflow = {'jobs': {'synthetic': {'steps': [{'run': command}]}}}
    with pytest.raises(AssertionError, match='executed dedicated'):
        assert_release_tests_executed(workflow)
