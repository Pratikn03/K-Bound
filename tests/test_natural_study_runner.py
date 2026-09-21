"""Terminal stage bindings and access traps on synthetic records only."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tests.test_natural_study_stages import panel, score_inputs

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'docs/research/kbound/scripts/natural_study_runner.py'


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def invocation(phase, dest, inputs):
    args = [phase, '--output-dir', str(dest)]
    for key, (path, digest) in inputs.items():
        args += ['--' + key, str(path), '--' + key + '-sha256', digest]
    return args


def module():
    assert SCRIPT.is_file(), 'stage CLI not implemented'
    return importlib.import_module('natural_study_runner')


def pipeline(root, rows=None, fresh_process=False):
    m = module(); root.mkdir()
    p, default = panel(); rows = default if rows is None else rows
    values = {'protocol': p, 'fit-rows': rows['fit'], 'tune-rows': rows['tune'],
              'calibration-rows': rows['calibrate']}
    values['score-features'], values['score-outcomes'] = score_inputs(rows['score'])
    bound = {k: put(root / (k + '.json'), v) for k, v in values.items()}
    receipt = {}
    for phase, keys, output in [
        ('develop', ['protocol', 'fit-rows', 'tune-rows'], 'developed'),
        ('calibrate', ['protocol', 'developed', 'calibration-rows'], 'gate'),
        ('decide', ['protocol', 'gate', 'score-features'], 'decisions'),
        ('score', ['protocol', 'gate', 'score-features', 'decisions', 'score-outcomes'], 'metrics'),
        ('verify', ['protocol', 'gate', 'score-features', 'decisions', 'score-outcomes', 'metrics'], 'verification'),
    ]:
        args = invocation(phase, root / phase, {k: bound[k] for k in keys})
        if fresh_process:
            result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
        else:
            assert m.main(args) == 0
        path = root / phase / (output + '.json')
        bound[output] = path, hashlib.sha256(path.read_bytes()).hexdigest()
        receipt[phase] = json.loads((root / phase / 'receipt.json').read_text())
    return bound, receipt


@pytest.mark.parametrize('indices', [[0], [0, 1]], ids=['single_cell', 'whole_environment'])
def test_terminal_full_reconstruction_excludes_outcomes(tmp_path, indices):
    p, rows = panel()
    original, receipts = pipeline(tmp_path / 'before', rows)
    changed = deepcopy(rows)
    for i in indices:
        changed['score'][i]['frozen_correct'] = [0] * 4
        changed['score'][i]['candidate_correct'] = [1] * 4
    after, _ = pipeline(tmp_path / 'after', changed)
    for key in ['developed', 'gate', 'decisions']:
        assert original[key][0].read_bytes() == after[key][0].read_bytes()
    assert original['metrics'][1] != after['metrics'][1]
    assert receipts['develop']['timing']['predictor_fit_call_count'] == 1
    assert all(receipts[phase]['timing']['predictor_fit_call_count'] == 0
               for phase in ['calibrate', 'decide', 'score', 'verify'])
    assert receipts['verify']['status'] == 'VERIFIED_SAVED_RECORDS_NOT_POPULATION_OR_PROSPECTIVE_EVIDENCE'
    assert receipts['decide']['scope']['native_execution_complete'] is False


def test_parser_rejects_outcome_or_selection_arguments_before_any_file_read(monkeypatch, tmp_path):
    m = module()
    def forbidden(*a, **kw): raise AssertionError('file accessed before parser rejection')
    monkeypatch.setattr(m, '_read_bound', forbidden)
    inputs = {k: (tmp_path / k, 'a' * 64) for k in ['protocol', 'gate', 'score-features']}
    for argument in ['--score-outcomes', '--fit-rows', '--select-candidate', '--feature-selection']:
        with pytest.raises(SystemExit) as error:
            m.main(invocation('decide', tmp_path / 'out', inputs) + [argument, 'forbidden'])
        assert error.value.code == 2


@pytest.mark.parametrize('scope', ['single_cell', 'whole_environment'])
def test_decision_cannot_read_scored_outcome_archive(monkeypatch, tmp_path, scope):
    m = module(); bound, _ = pipeline(tmp_path / 'initial')
    forbidden_path = bound['score-outcomes'][0]
    original = m._read_bound; accessed = []
    def trap(path, digest):
        assert Path(path) != forbidden_path, scope + ' scored archive accessed'
        accessed.append(Path(path)); return original(path, digest)
    monkeypatch.setattr(m, '_read_bound', trap)
    assert m.main(invocation('decide', tmp_path / ('trap-' + scope),
        {k: bound[k] for k in ['protocol', 'gate', 'score-features']})) == 0
    assert len(accessed) == 3


@pytest.mark.parametrize('fault', ['hash', 'collision', 'symlink', 'gate', 'metrics'])
def test_cli_rejects_tampering_and_reuse(tmp_path, monkeypatch, fault):
    m = module(); bound, _ = pipeline(tmp_path / 'first')
    inputs = {k: bound[k] for k in ['protocol', 'gate', 'score-features', 'decisions', 'score-outcomes', 'metrics']}
    out = tmp_path / 'verify-again'
    if fault == 'hash': inputs['decisions'] = inputs['decisions'][0], '0' * 64
    if fault == 'collision': out = tmp_path / 'first' / 'verify'
    if fault == 'symlink':
        link = tmp_path / 'link.json'; link.symlink_to(inputs['decisions'][0]); inputs['decisions'] = link, inputs['decisions'][1]
    if fault in ('gate', 'metrics'):
        value = json.loads(inputs[fault][0].read_text())
        if fault == 'gate': value['development']['margin'] = .5
        else: value['policies']['always_adapt']['mean_accuracy'] = .99
        inputs[fault] = put(tmp_path / 'changed.json', value)
    original = m._read_bound
    if fault != 'metrics':
        def trap(path, digest):
            assert Path(path) != bound['score-outcomes'][0], 'outcomes opened before decision identity rejection'
            return original(path, digest)
        monkeypatch.setattr(m, '_read_bound', trap)
    with pytest.raises(SystemExit) as error: m.main(invocation('verify', out, inputs))
    assert error.value.code == 2


def test_preflight_lock_and_fresh_process(tmp_path):
    m = module(); p, _ = panel(); p['status'] = 'DRAFT'
    bound = put(tmp_path / 'draft.json', p)
    for phase in ('preflight', 'lock'):
        result = subprocess.run([sys.executable, str(SCRIPT), *invocation(phase, tmp_path / phase, {'protocol': bound})],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    locked = json.loads((tmp_path / 'lock' / 'protocol.json').read_text())
    assert locked['status'] == 'LOCKED_COMPUTATIONAL'
    assert locked['claim_scope'] == 'synthetic_validation'


def test_full_fresh_process_chain_and_no_refit_after_development(tmp_path, monkeypatch):
    m = module(); bound, _ = pipeline(tmp_path / 'processes', fresh_process=True)
    from sklearn.ensemble import GradientBoostingRegressor
    def forbidden(*a, **kw): raise AssertionError('fit outside development')
    monkeypatch.setattr(GradientBoostingRegressor, 'fit', forbidden)
    for phase, keys in m.STAGES.items():
        if phase in ('preflight', 'lock', 'develop'): continue
        assert m.main(invocation(phase, tmp_path / ('trap-' + phase), {k: bound[k] for k in keys})) == 0
