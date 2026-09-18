"""The maintained head-to-head must exclude scored outcomes from gate fitting."""
import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def test_every_scored_outcome_is_excluded_from_full_gate_pipeline():
    path = Path(__file__).resolve().parents[1] / 'docs/research/kbound/scripts/official_baselines_headtohead.py'
    spec = importlib.util.spec_from_file_location('headtohead_exclusion', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(117)
    features = rng.normal(size=(40, 11))
    benefits = np.clip(features[:, 0] * 0.15 + 0.2, -1, 1)
    ids = [f'cell-{i}' for i in range(40)]
    original = module.kga_exact_rank(features, benefits, sample_ids=ids, return_details=True)
    assert np.isfinite(original.prediction).all()
    assert np.isfinite(original.radius).all()
    for i in range(len(benefits)):
        perturbed = benefits.copy()
        perturbed[i] = -1.0 if benefits[i] >= 0 else 1.0
        replay = module.kga_exact_rank(features, perturbed, sample_ids=ids, return_details=True)
        assert replay.prediction[i] == original.prediction[i]
        assert replay.radius[i] == original.radius[i]
        assert replay.action[i] == original.action[i]
        assert replay.prediction[i] - replay.radius[i] == original.prediction[i] - original.radius[i]
        assert replay.prediction[i] + replay.radius[i] == original.prediction[i] + original.radius[i]
    decisions = module.kga_exact_rank(features, benefits, sample_ids=ids)
    assert decisions.tolist() == [str(a).lower() for a in original.action]


def test_cli_records_shared_point_predictions_and_keeps_external_unverified(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'docs/research/kbound/scripts/official_baselines_headtohead.py'
    spec = importlib.util.spec_from_file_location('headtohead_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(25)
    records = []
    for i in range(40):
        benefit = float(rng.uniform(-0.2, 0.2))
        records.append(dict(condition=f'c{i}', Z=rng.normal(size=11).tolist(), B=benefit,
                            a0=0.5, a_adapted=0.5+benefit, a_oracle=0.5+max(benefit,0)))
    (tmp_path / 'per_condition_cifar10c_tent_seed0.json').write_text(json.dumps({'records':records}))
    external = tmp_path / 'external.json'
    external.write_text(json.dumps({r['condition']:'freeze' for r in records}))
    output = tmp_path / 'output.json'
    monkeypatch.setattr(module, 'RES', str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['headtohead', '--decisions', f'poem={external}', '--out', str(output)])
    module.main()
    result = json.loads(output.read_text())
    assert result['official'] == []
    assert 'POEM_external_unverified' in result['rows']
    assert result['decision_threshold'] == 0.0
    assert result['conditions'] == [r['condition'] for r in records]
    expected = ['adapt' if v > 0 else 'freeze' for v in result['gate']['prediction']]
    assert result['policy_actions']['benefit_regression'] == expected


def _checkpoint_cli_case(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'docs/research/kbound/scripts/official_baselines_headtohead.py'
    spec = importlib.util.spec_from_file_location('headtohead_checkpoint_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(117)
    records = [dict(condition=f'cell-{i}', Z=rng.normal(size=11).tolist(),
                    a0=0.5, aa=0.6 if i % 2 == 0 else 0.4) for i in range(27)]
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps({'rows': {'sar': records}, 'cells_done': 27,
                                     'cells_total': 27, 'done': [r['condition'] for r in records]}))
    expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = tmp_path / 'fresh-replay'
    argv = ['headtohead', '--candidate', 'sar', '--checkpoint', str(checkpoint),
            '--checkpoint-sha256', expected_sha, '--out-dir', str(output)]
    monkeypatch.setattr(module, 'RES', str(tmp_path / 'no-default-data'))
    monkeypatch.setattr(sys, 'argv', argv)
    return module, checkpoint, output, records, argv


def test_checkpoint_cli_binds_saved_input_and_serializes_shared_policy_arithmetic(tmp_path, monkeypatch):
    """Catch fallback to CIFAR files, unauthenticated inputs, or separately fitted point outputs."""
    module, checkpoint, output, records, argv = _checkpoint_cli_case(tmp_path, monkeypatch)
    before = checkpoint.read_bytes()
    external = tmp_path / 'external.json'
    external.write_text(json.dumps({r['condition']: 'freeze' for r in records}))
    monkeypatch.setattr(sys, 'argv', argv + ['--decisions', f'poem={external}'])
    module.main()
    result = json.loads((output / 'headtohead.json').read_text())
    receipt = json.loads((output / 'replay_receipt.json').read_text())
    assert checkpoint.read_bytes() == before
    assert result['input_sha256'] == hashlib.sha256(before).hexdigest()
    assert receipt['expected_input_sha256'] == receipt['input_sha256_before'] == receipt['input_sha256_after']
    assert receipt['input_sha256_before'] == result['input_sha256']
    assert receipt['implementation_sha256'] and receipt['python_executable']
    assert receipt['scope'] == 'retrospective_saved_feature_cell_outcome_disjoint'
    assert result['official'] == []
    assert 'POEM_external_unverified' in result['rows']
    assert result['rows']['always_adapt']['regret_exact'] == pytest.approx(1.3 / 27)
    assert result['rows']['always_freeze']['regret_exact'] == pytest.approx(1.4 / 27)
    assert result['rows']['always_adapt']['mean_accuracy'] == pytest.approx(13.6 / 27)
    assert result['gate']['protocol']['requested_n_folds'] == 8
    assert all(fold['residual_calibration_count'] == 9 for fold in result['gate']['protocol']['folds'])
    assert len(result['per_condition']) == 27
    for index, row in enumerate(result['per_condition']):
        assert row['condition'] == records[index]['condition']
        assert row['prediction'] == result['gate']['prediction'][index]
        assert row['radius'] == result['gate']['radius'][index]
        assert row['B'] == pytest.approx(records[index]['aa'] - 0.5)
        assert row['point_action'] == ('adapt' if row['prediction'] > 0 else 'freeze')
        assert row['KGA_action'] == result['policy_actions']['KGA'][index]
        assert row['threshold'] == 0.0
        assert row['point_accuracy'] == (records[index]['aa'] if row['point_action'] == 'adapt' else 0.5)
        assert row['KGA_accuracy'] == (records[index]['aa'] if row['KGA_action'] == 'adapt' else 0.5)
    # A retrospective saved grid must not acquire fresh bootstrap significance claims.
    assert all('p_better' not in str(row) and 'ci95' not in str(row) for row in result['rows'].values())


def test_checkpoint_cli_rejects_hash_mismatch_before_creating_output(tmp_path, monkeypatch):
    """Catch accepting bytes different from the user's expected source identity."""
    module, checkpoint, output, records, argv = _checkpoint_cli_case(tmp_path, monkeypatch)
    checkpoint.write_text(checkpoint.read_text() + '\n')
    with pytest.raises(SystemExit, match='SHA-256 mismatch'):
        module.main()
    assert not output.exists()


def test_checkpoint_cli_requires_explicit_hash(tmp_path, monkeypatch):
    """Catch substituting a freshly calculated digest for an expected identity."""
    module, checkpoint, output, records, argv = _checkpoint_cli_case(tmp_path, monkeypatch)
    index = argv.index('--checkpoint-sha256')
    monkeypatch.setattr(sys, 'argv', argv[:index] + argv[index + 2:])
    with pytest.raises(SystemExit, match='checkpoint-sha256'):
        module.main()
    assert not output.exists()


def test_checkpoint_cli_refuses_existing_output_directory(tmp_path, monkeypatch):
    """Catch overwriting any previous analysis, even when its directory is empty."""
    module, checkpoint, output, records, argv = _checkpoint_cli_case(tmp_path, monkeypatch)
    output.mkdir()
    marker = output / 'existing.txt'
    marker.write_text('keep')
    with pytest.raises(SystemExit, match='fresh|exists'):
        module.main()
    assert marker.read_text() == 'keep'
    assert sorted(p.name for p in output.iterdir()) == ['existing.txt']


@pytest.mark.parametrize('defect', ['duplicate_condition', 'nonfinite', 'overflow', 'boolean',
                                  'string', 'ragged', 'invalid_accuracy', 'incomplete'])
def test_checkpoint_cli_rejects_malformed_saved_rows_without_output(tmp_path, monkeypatch, defect):
    """Catch publishing a partial, ambiguous, or physically invalid checkpoint replay."""
    module, checkpoint, output, records, argv = _checkpoint_cli_case(tmp_path, monkeypatch)
    payload = json.loads(checkpoint.read_text())
    if defect == 'duplicate_condition':
        payload['rows']['sar'][1]['condition'] = payload['rows']['sar'][0]['condition']
    elif defect == 'nonfinite':
        payload['rows']['sar'][0]['Z'][0] = float('nan')
    elif defect == 'overflow':
        payload['rows']['sar'][0]['Z'][0] = 1.2345
    elif defect == 'boolean':
        payload['rows']['sar'][0]['aa'] = True
    elif defect == 'string':
        payload['rows']['sar'][0]['aa'] = '0.5'
    elif defect == 'ragged':
        payload['rows']['sar'][0]['Z'].pop()
    elif defect == 'invalid_accuracy':
        payload['rows']['sar'][0]['aa'] = 1.1
    else:
        payload['cells_done'] = 26
    serialized = json.dumps(payload)
    if defect == 'overflow':
        serialized = serialized.replace('1.2345', '1e400')
    checkpoint.write_text(serialized)
    argv[argv.index('--checkpoint-sha256') + 1] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match='checkpoint|condition|finite|accuracy|complete'):
        module.main()
    assert not output.exists()


def test_checkpoint_cli_does_not_accept_self_attested_official_decisions(tmp_path, monkeypatch):
    """Catch weakening external provenance when using an explicit checkpoint input."""
    module, checkpoint, output, records, argv = _checkpoint_cli_case(tmp_path, monkeypatch)
    external = tmp_path / 'external.json'
    external.write_text(json.dumps({'schema_version': 3, 'official_label_allowed': True,
                                   'label': 'official_implementation_under_protocol_adapter',
                                   'decisions': {r['condition']: 'adapt' for r in records}}))
    monkeypatch.setattr(sys, 'argv', argv + ['--decisions', f'poem={external}'])
    with pytest.raises(SystemExit, match='audit|provenance'):
        module.main()
    assert not output.exists()


def test_checkpoint_cli_preserves_scored_outcome_exclusion_through_ingestion(tmp_path, monkeypatch):
    """Catch allowing the scored aa to affect split identity or either gate during CLI ingestion."""
    module, checkpoint, output, records, argv = _checkpoint_cli_case(tmp_path, monkeypatch)
    module.main()
    original = json.loads((output / 'headtohead.json').read_text())
    payload = json.loads(checkpoint.read_text())
    payload['rows']['sar'][7]['aa'] = 1.0
    checkpoint.write_text(json.dumps(payload))
    changed_output = tmp_path / 'changed-replay'
    argv[argv.index('--checkpoint-sha256') + 1] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    argv[argv.index('--out-dir') + 1] = str(changed_output)
    module.main()
    changed = json.loads((changed_output / 'headtohead.json').read_text())
    assert changed['per_condition'][7]['B'] == 0.5
    for field in ('prediction', 'radius', 'lower', 'upper', 'threshold', 'point_action', 'KGA_action'):
        assert changed['per_condition'][7][field] == original['per_condition'][7][field]


def test_checkpoint_cli_serializes_infeasible_calibration_as_abstention(tmp_path, monkeypatch):
    """Catch converting canonical unavailable predictions/radii into fabricated finite decisions."""
    module, checkpoint, output, records, argv = _checkpoint_cli_case(tmp_path, monkeypatch)
    records = records[:8]
    checkpoint.write_text(json.dumps({'rows': {'sar': records}, 'cells_done': 8,
                                     'cells_total': 8, 'done': [r['condition'] for r in records]}))
    argv[argv.index('--checkpoint-sha256') + 1] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    module.main()
    serialized = (output / 'headtohead.json').read_text()
    result = json.loads(serialized, parse_constant=lambda value: pytest.fail(f'nonstandard JSON {value}'))
    assert result['gate']['protocol']['status'] == 'fail_closed'
    assert result['gate']['prediction'] == [None] * 8
    assert result['gate']['radius'] == [None] * 8
    assert result['policy_actions']['KGA'] == ['abstain'] * 8
    assert result['policy_actions']['benefit_regression'] == ['abstain'] * 8
    assert result['rows']['KGA']['FA_c'] is None
    assert all(row['radius_status'] == 'positive_infinity' for row in result['per_condition'])
