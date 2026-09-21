"""Synthetic full-pipeline outcome-exclusion checks, not natural benchmark evidence."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'docs/research/kbound/scripts'))


def api():
    module = importlib.import_module('natural_study_contract')
    assert callable(getattr(module, 'prepare_gate', None)), 'four-role natural stages missing'
    return module


def panel(ncal=9):
    cells, rows = [], {r: [] for r in ('fit', 'tune', 'calibrate', 'score')}
    for role, n in [('fit', 8), ('tune', 4), ('calibrate', ncal), ('score', 4)]:
        for i in range(n):
            cell = {'cell_id': f'{role}:{i}', 'environment_id': f'{role}:env:{i // 2}',
                    'sample_ids': [f'{role}:{i}:eval:{j}' for j in range(4)],
                    'adaptation_ids': [f'{role}:{i}:adapt:{j}' for j in range(4)],
                    'stream_seed': i, 'role': role}
            cells.append(cell)
            rows[role].append({'cell_id': cell['cell_id'], 'sample_ids': cell['sample_ids'][:],
                'candidate_state_sha256': 'b' * 64,
                'Z': [.2, .5, .1, .1, .6, .1, 0., .1, .7, .1, float(i % 2)],
                'frozen_correct': [1, 1, 0, 0],
                'candidate_correct': [1, 1, 1, 0] if i % 2 else [1, 0, 0, 0]})
    p = {'schema': 'natural-calibration-value-v1', 'status': 'LOCKED_COMPUTATIONAL',
         'claim_scope': 'synthetic_validation', 'metric': 'paired_accuracy',
         'calibration_unit': 'cell_residual_not_domain_certificate', 'alpha': .1,
         'checkpoint_sha256': 'a' * 64, 'class_map_sha256': 'c' * 64,
         'candidate_recipe_sha256': 'd' * 64, 'source_hashes': {'adapter': 'e' * 64},
         'feature_names': ['pre_entropy', 'pre_conf', 'pre_pbal', 'post_entropy', 'post_conf',
                          'post_pbal', 'pbal_drop', 'entropy_drop', 'frac_highconf', 'marginal_KL', 'update_norm'],
         'margin_grid': [0., .1, .5], 'harmful_cost': 1., 'cells': cells,
         'prior_access': {c['environment_id']: 'SYNTHETIC' for c in cells},
         'warnings': ['synthetic fixture'], 'unavailable_methods': {'AETTA': '126-class adapter unsupported'}}
    return p, rows


def score_inputs(rows):
    features, outcomes = [], []
    for row in rows:
        common = {k: deepcopy(row[k]) for k in ('cell_id', 'sample_ids', 'candidate_state_sha256')}
        features.append({**common, 'Z': deepcopy(row['Z'])})
        outcomes.append({**common, 'frozen_correct': row['frozen_correct'][:],
                         'candidate_correct': row['candidate_correct'][:]})
    return features, outcomes


def run(p, rows):
    m = api()
    gate = m.prepare_gate(p, rows['fit'], rows['tune'], rows['calibrate'])
    features, outcomes = score_inputs(rows['score'])
    decisions = m.decide_rows(gate, features)
    return gate, decisions, m.score_rows(p, decisions, outcomes)


def test_four_roles_share_predictions_and_identical_oracles():
    p, rows = panel()
    gate, packet, scores = run(p, rows)
    assert gate['calibration_n'] == 9 and gate['calibration_rank'] == 9
    assert scores['policies']['always_freeze']['mean_accuracy'] == .5
    assert scores['policies']['always_adapt']['mean_accuracy'] == .5
    assert scores['policies']['point_benefit']['mean_accuracy'] == .625
    assert scores['policies']['always_freeze']['regret_vs_oracle'] == .125
    assert scores['policies']['always_adapt']['regret_vs_oracle'] == .125
    assert [r['action']['point_benefit'] for r in packet['rows']] == ['FREEZE', 'ADAPT', 'FREEZE', 'ADAPT']
    assert packet['unavailable_methods'] == p['unavailable_methods']
    assert packet['warnings'] == p['warnings']
    assert 'coverage' not in scores['policies']['KGA']
    for row in packet['rows']:
        assert row['action']['point_benefit'] == ('ADAPT' if row['prediction'] > 0 else 'FREEZE')
        assert row['action']['fixed_margin'] == ('ADAPT' if row['prediction'] > row['threshold']['fixed_margin'] else 'FREEZE')


@pytest.mark.parametrize('indices', [[0], [0, 1]], ids=['single_cell', 'whole_environment'])
def test_full_refit_tune_calibrate_and_decide_excludes_scored_outcomes(indices):
    p, rows = panel()
    before = run(p, rows)
    changed = deepcopy(rows)
    for i in indices:
        changed['score'][i]['frozen_correct'] = [0, 0, 0, 0]
        changed['score'][i]['candidate_correct'] = [1, 1, 1, 1]
    after = run(p, changed)
    assert before[0] == after[0]  # every fit/tune/calibration artifact, not just the final action
    assert before[1] == after[1]
    for old, new in zip(before[1]['rows'], after[1]['rows']):
        for field in ('prediction', 'radius', 'threshold', 'action'):
            assert old[field] == new[field]
    assert before[2]['outcomes_sha256'] != after[2]['outcomes_sha256']
    assert before[2]['policies'] != after[2]['policies']


def test_insufficient_calibration_and_no_adapt_rates():
    gate, decisions, scores = run(*panel(8))
    assert gate['radius'] == 'infinity' and gate['calibration_rank'] == 9
    assert all(r['action']['KGA'] == 'ABSTAIN' for r in decisions['rows'])
    assert scores['policies']['KGA']['conditional_strict_harm_rate'] is None
    assert scores['policies']['KGA']['adapt_exposure'] == 0.


@pytest.mark.parametrize('fault', ['environment_overlap', 'sample_overlap', 'adapt_eval_overlap',
    'missing_class_map', 'missing_source', 'prospective', 'population', 'feature_selection', 'alpha'])
def test_protocol_rejects_invalid_roles_or_unsupported_claims(fault):
    m = api(); p, _ = panel()
    if fault == 'environment_overlap': p['cells'][-1]['environment_id'] = p['cells'][0]['environment_id']
    if fault == 'sample_overlap': p['cells'][-1]['sample_ids'][0] = p['cells'][0]['sample_ids'][0]
    if fault == 'adapt_eval_overlap': p['cells'][-1]['adaptation_ids'][0] = p['cells'][0]['sample_ids'][0]
    if fault == 'missing_class_map': del p['class_map_sha256']
    if fault == 'missing_source': p['source_hashes'] = {}
    if fault == 'prospective': p['claim_scope'] = 'prospective'
    if fault == 'population': p['calibration_unit'] = 'population_certified'
    if fault == 'feature_selection': p['feature_names'][0] = 'frozen_accuracy'
    if fault == 'alpha': p['alpha'] = .5
    with pytest.raises(ValueError): m.validate_protocol(p)


class Forbidden:
    def __deepcopy__(self, memo): raise AssertionError('scored outcome accessed')
    def __iter__(self): raise AssertionError('scored outcome accessed')
    def __len__(self): raise AssertionError('scored outcome accessed')


@pytest.mark.parametrize('stage', ['fit', 'tune', 'calibrate', 'decide'])
def test_wrong_role_outcome_trap_rejected_before_payload_read(stage):
    m = api(); p, rows = panel()
    injected = deepcopy(rows['score'][0])
    injected['frozen_correct'] = injected['candidate_correct'] = Forbidden()
    if stage == 'decide':
        gate = m.prepare_gate(p, rows['fit'], rows['tune'], rows['calibrate'])
        features, _ = score_inputs(rows['score'])
        features[0]['frozen_correct'] = Forbidden()
        with pytest.raises(ValueError): m.decide_rows(gate, features)
    else:
        rows[stage][0] = injected
        with pytest.raises(ValueError): m.prepare_gate(p, rows['fit'], rows['tune'], rows['calibrate'])


def test_calibration_cannot_reselect_model_or_margin():
    m = api(); p, rows = panel()
    developed = m.develop_gate(p, rows['fit'], rows['tune'])
    original = deepcopy(developed)
    first = m.calibrate_gate(p, developed, rows['calibrate'])
    changed = deepcopy(rows['calibrate'])
    for row in changed: row['candidate_correct'] = [0, 0, 0, 0]
    second = m.calibrate_gate(p, developed, changed)
    assert developed == original
    assert first['development'] == second['development']
    assert first['calibration_binding'] != second['calibration_binding']


@pytest.mark.parametrize('fault', ['packet', 'protocol', 'sample', 'candidate', 'nonfinite'])
def test_tampered_or_mismatched_score_inputs_fail_closed(fault):
    m = api(); p, rows = panel(); _, packet, _ = run(p, rows)
    features, outcomes = score_inputs(rows['score'])
    if fault == 'packet': packet['rows'][0]['prediction'] = .9
    if fault == 'protocol': p['candidate_recipe_sha256'] = 'f' * 64
    if fault == 'sample': outcomes[0]['sample_ids'].reverse()
    if fault == 'candidate': outcomes[0]['candidate_state_sha256'] = 'f' * 64
    if fault == 'nonfinite': outcomes[0]['candidate_correct'][0] = float('nan')
    with pytest.raises(ValueError): m.score_rows(p, packet, outcomes)


def test_score_stage_never_refits(monkeypatch):
    m = api(); p, rows = panel(); _, packet, _ = run(p, rows)
    from sklearn.ensemble import GradientBoostingRegressor
    def forbidden(*a, **kw): raise AssertionError('scoring called fit')
    monkeypatch.setattr(GradientBoostingRegressor, 'fit', forbidden)
    _, outcomes = score_inputs(rows['score'])
    assert m.score_rows(p, packet, outcomes)['policies']['always_adapt']['n'] == 4


def test_safe_model_matches_clipped_sklearn_and_fits_exactly_once(monkeypatch):
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    m = api(); p, rows = panel(); trained = []
    original = GradientBoostingRegressor.fit
    def capture(self, *a, **kw):
        result = original(self, *a, **kw); trained.append(self); return result
    monkeypatch.setattr(GradientBoostingRegressor, 'fit', capture)
    gate, packet, scores = run(p, rows)
    assert len(trained) == 1
    z = [r['Z'] for role in rows.values() for r in role]
    # Cython may fuse multiply/add; shared exported predictions, not sklearn's
    # backend accumulation, are the authoritative values for every controller.
    np.testing.assert_allclose(m._predict(gate['development']['predictor'], z),
                               np.clip(trained[0].predict(z), -1, 1), rtol=0, atol=1e-15)


def test_score_returns_per_cell_arithmetic_and_ties_separately():
    m = api(); p, rows = panel()
    gate, packet, _ = run(p, rows)
    _, outcomes = score_inputs(rows['score'])
    outcomes[1]['candidate_correct'] = outcomes[1]['frozen_correct'][:]
    scores = m.score_rows(p, packet, outcomes)
    assert len(scores['cells']) == 4
    assert scores['cells'][1]['benefit'] == 0
    assert scores['cells'][0]['oracle_accuracy'] == .5
    assert scores['cells'][0]['policies']['always_adapt']['regret'] == .25
    assert scores['policies']['point_benefit']['n_adapt_ties'] == 1
    assert scores['policies']['point_benefit']['n_strict_harmful_adapt'] == 0


@pytest.mark.parametrize('fault', ['mixed_radius', 'mixed_margin', 'tree_cycle', 'rank_bool'])
def test_resigned_semantic_malformations_rejected(fault):
    m = api(); p, rows = panel(); gate, packet, _ = run(p, rows)
    features, outcomes = score_inputs(rows['score'])
    if fault.startswith('mixed'):
        row = packet['rows'][0]
        if fault == 'mixed_radius': row['radius'] = row['threshold']['KGA'] = .9
        else: row['threshold']['fixed_margin'] = .5
        row['action'] = m._actions(row['prediction'], row['radius'], row['threshold']['fixed_margin'])
        packet = m._seal({k: v for k, v in packet.items() if k != 'artifact_sha256'})
        with pytest.raises(ValueError): m.score_rows(p, packet, outcomes)
    else:
        if fault == 'tree_cycle':
            gate['development']['predictor']['trees'][0]['left'][0] = 0
            gate['development'] = m._seal({k: v for k, v in gate['development'].items() if k != 'artifact_sha256'})
        else: gate['calibration_rank'] = float(gate['calibration_rank'])
        gate = m._seal({k: v for k, v in gate.items() if k != 'artifact_sha256'})
        with pytest.raises(ValueError): m.decide_rows(gate, features)


def test_protocol_is_immutable_and_unknown_does_not_mean_prospective():
    m = api(); p, rows = panel(); original = deepcopy((p, rows))
    run(p, rows)
    assert (p, rows) == original
    p['claim_scope'] = 'retrospective_unknown_history'
    p['prior_access'] = dict.fromkeys(p['prior_access'], 'UNKNOWN')
    run(p, rows)
    p['status'] = 'DRAFT'
    with pytest.raises(ValueError): run(p, rows)


def test_gate_stages_reject_changed_implementation(monkeypatch):
    m = api(); p, rows = panel()
    developed = m.develop_gate(p, rows['fit'], rows['tune'])
    assert 'implementation_sha256' in developed
    identity = m._implementation(); identity['natural_study_contract.py'] = '0' * 64
    monkeypatch.setattr(m, '_implementation', lambda: identity)
    with pytest.raises(ValueError, match='implementation'):
        m.calibrate_gate(p, developed, rows['calibrate'])


@pytest.mark.parametrize('indices', [[0], [0, 1]], ids=['single_cell_access_trap', 'whole_environment_access_trap'])
def test_every_forbidden_member_rejected_at_each_upstream_boundary(indices):
    m = api(); p, rows = panel()
    for role in ['fit', 'tune', 'calibrate']:
        poisoned = deepcopy(rows)
        for i in indices:
            poisoned[role][i] = deepcopy(rows['score'][i])
            poisoned[role][i]['frozen_correct'] = Forbidden()
            poisoned[role][i]['candidate_correct'] = Forbidden()
        with pytest.raises(ValueError): m.prepare_gate(p, poisoned['fit'], poisoned['tune'], poisoned['calibrate'])


def test_exported_trees_match_sklearn_at_float32_split_boundaries():
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    m = api()
    x = np.zeros((8, 11)); x[:, 0] = [.1, .2] * 4
    y = np.array([-.25, .25] * 4)
    model = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=.05,
                                      subsample=.8, random_state=0).fit(x, y)
    threshold = model.estimators_[0, 0].tree_.threshold[0]
    midpoint = np.float32(threshold)
    test = np.zeros((3, 11))
    test[:, 0] = [np.nextafter(midpoint, np.float32(-np.inf)), midpoint,
                  np.nextafter(midpoint, np.float32(np.inf))]
    actual = m._predict(m._export_model(model), test)
    expected = np.clip(model.predict(test), -1, 1)
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
    assert (actual > 0).tolist() == (expected > 0).tolist()


@pytest.mark.parametrize('fault', ['source_bytes', 'wrong_radius', 'fallback'])
def test_active_calibration_backend_is_bound_and_checked(monkeypatch, fault):
    import kbound_decide as kb
    m = api(); p, rows = panel(); developed = m.develop_gate(p, rows['fit'], rows['tune'])
    if fault == 'source_bytes':
        original = Path.read_bytes
        def changed(path):
            data = original(path)
            return data + b'\n# changed implementation\n' if path.name == 'certificate.py' else data
        monkeypatch.setattr(Path, 'read_bytes', changed)
    elif fault == 'wrong_radius': monkeypatch.setattr(kb, '_kga_radius', lambda *args: 0.)
    else: monkeypatch.setattr(kb, 'BACKEND', 'local-fallback')
    with pytest.raises(ValueError): m.calibrate_gate(p, developed, rows['calibrate'])


def test_large_finite_cost_and_unsupported_integer_failures_are_controlled():
    m = api()
    assert m.select_margin([1., 1.], [-1., -1.], [0., 2.], 1e308) == 2.
    with pytest.raises(ValueError): m.select_margin([0.], [0.], [0.], 10**1000)
    with pytest.raises(ValueError): m._number(10**1000)
