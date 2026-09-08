"""The source-stage unit tests must execute in CI, not merely collect."""
from pathlib import Path
import shlex
import subprocess

import pytest
import yaml


TARGETS = {
    'tests/test_reference_source_models.py',
    'tests/test_reference_source_population.py',
    'tests/test_reference_source_preflight.py',
    'tests/test_reference_rxrx1_training.py',
    'tests/test_reference_officehome_training.py',
    'tests/test_reference_cifar_population.py',
    'tests/test_reference_imagenet_archives.py',
    'tests/test_ci_reference_source_selectors.py',
}


def assert_executed(workflow):
    commands = []
    for job in workflow['jobs'].values():
        for step in job.get('steps', []):
            for line in step.get('run', '').replace('\\\n', ' ').splitlines():
                if not line.lstrip().startswith('pytest '):
                    continue
                args = shlex.split(line)
                targets = {arg for arg in args[1:] if arg.startswith('tests/')}
                if '--collect-only' not in args and TARGETS <= targets:
                    commands.append(targets)
    assert commands == [TARGETS], 'require one executed exact source-regression command'


def test_ci_executes_all_reference_source_tests():
    path = Path(__file__).parents[1] / '.github/workflows/kbound-ci.yml'
    assert_executed(yaml.safe_load(path.read_text()))


@pytest.mark.parametrize('prefix', ['pytest --collect-only ', '# pytest '])
def test_collection_and_comments_are_not_execution(prefix):
    workflow = {'jobs': {'test': {'steps': [{'run': prefix + ' '.join(sorted(TARGETS))}]}}}
    with pytest.raises(AssertionError):
        assert_executed(workflow)


def test_reference_source_code_is_included_but_raw_outputs_stay_ignored():
    root = Path(__file__).parents[1]
    paths = [
        'experiments/kbound/reference_source/cifar_population.py',
        'experiments/kbound/reference_source/train_officehome.py',
        'experiments/kbound/reference_source/imagenet_archives.py',
        'experiments/kbound/reference_source/model.pth',
        'experiments/kbound/reference_source/run.log',
        'experiments/kbound/reference_source/._models.py',
    ]
    result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'],
                            input='\n'.join(paths) + '\n', text=True, capture_output=True, cwd=root)
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == set(paths[3:])
