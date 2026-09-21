"""Four-role saved-record analysis for a natural-shift study.

No image access, adapter execution, source authentication or prospective claim.
Development fits once and selects a margin; calibration consumes that immutable
artifact; decisions accept features only; scoring never fits. Callers must bind
feature-production provenance and retain artifact hashes before outcome access.
Safe JSON trees replace executable model pickles between stage runtimes.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import numpy as np
from task3_common_scoring import EVIDENCE_NAMES, exact_radius, policy_metrics

ROLES = ('fit', 'tune', 'calibrate', 'score')
POLICIES = ('always_freeze', 'always_adapt', 'point_benefit', 'fixed_margin', 'KGA')


def _implementation():
    import kbound_decide as kb
    import kga.certificate as certificate
    root = Path(__file__).resolve().parents[4]
    if (kb.backend() != 'kga-library' or kb._kga_radius is not certificate.split_conformal_rank_radius
            or Path(certificate.__file__).resolve() != root / 'kga/certificate.py'):
        raise ValueError('unbound or fallback calibration implementation')
    names = ('natural_study_contract.py', 'natural_study_runner.py', 'task3_common_analysis.py',
             'task3_common_scoring.py', 'run_decision_baselines.py', 'kbound_decide.py')
    identity = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    # Package initialization imports additional local modules. Bind the whole
    # small local source package, including added/removed Python dependencies.
    for path in sorted((root / 'kga').rglob('*.py')):
        identity[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _seal(value):
    value = json.loads(_canonical(value))
    value['artifact_sha256'] = _hash(value)
    return value


def _exact(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('unexpected fields in ' + label)


def _text(value):
    if type(value) is not str or not value or value != value.strip():
        raise ValueError('nonempty canonical text required')


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('lowercase SHA256 required')


def _number(value, low=-math.inf, high=math.inf):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('finite numeric value outside permitted range')
    return float(value)


def _verify(value, keys, schema):
    _exact(value, set(keys) | {'schema', 'artifact_sha256'}, schema)
    _sha(value['artifact_sha256'])
    if value['schema'] != schema or value['artifact_sha256'] != _hash({k: v for k, v in value.items() if k != 'artifact_sha256'}):
        raise ValueError('artifact schema or digest mismatch')


def validate_protocol(p):
    """Validate declarations, not independent source identity or historical access."""
    _exact(p, {'schema', 'status', 'claim_scope', 'metric', 'calibration_unit', 'alpha',
               'checkpoint_sha256', 'class_map_sha256', 'candidate_recipe_sha256',
               'source_hashes', 'feature_names', 'margin_grid', 'harmful_cost',
               'cells', 'prior_access', 'warnings', 'unavailable_methods'}, 'protocol')
    if p['schema'] != 'natural-calibration-value-v1' or p['status'] not in ('DRAFT', 'LOCKED_COMPUTATIONAL'):
        raise ValueError('unsupported protocol status/schema')
    if p['claim_scope'] not in ('synthetic_validation', 'retrospective_unknown_history'):
        raise ValueError('prospective or population claims require another independently justified design')
    if p['metric'] != 'paired_accuracy' or p['calibration_unit'] != 'cell_residual_not_domain_certificate':
        raise ValueError('only empirical paired cell accuracy is supported')
    if type(p['alpha']) is not float or p['alpha'] != .1 or p['feature_names'] != EVIDENCE_NAMES:
        raise ValueError('fixed alpha and outcome-free feature specification required')
    for k in ('checkpoint_sha256', 'class_map_sha256', 'candidate_recipe_sha256'): _sha(p[k])
    if type(p['source_hashes']) is not dict or not p['source_hashes']:
        raise ValueError('source bindings required')
    for name, digest in p['source_hashes'].items(): _text(name); _sha(digest)
    if type(p['margin_grid']) is not list or not p['margin_grid']:
        raise ValueError('predeclared finite margin grid required')
    for margin in p['margin_grid']: _number(margin, 0)
    if p['margin_grid'] != sorted(set(p['margin_grid'])):
        raise ValueError('margin grid must be unique and sorted')
    if _number(p['harmful_cost']) <= 0: raise ValueError('positive cost required')
    if type(p['warnings']) is not list or type(p['unavailable_methods']) is not dict:
        raise ValueError('explicit warnings and method availability required')
    for value in p['warnings']: _text(value)
    for name, reason in p['unavailable_methods'].items(): _text(name); _text(reason)
    if type(p['cells']) is not list or not p['cells']: raise ValueError('cells required')
    ids, samples, environments = set(), set(), {}
    counts = dict.fromkeys(ROLES, 0)
    for cell in p['cells']:
        _exact(cell, {'cell_id', 'environment_id', 'sample_ids', 'adaptation_ids', 'stream_seed', 'role'}, 'cell')
        for k in ('cell_id', 'environment_id', 'role'): _text(cell[k])
        role, env = cell['role'], cell['environment_id']
        if role not in ROLES or cell['cell_id'] in ids: raise ValueError('unknown role or duplicate cell')
        ids.add(cell['cell_id']); counts[role] += 1
        if env in environments and environments[env] != role: raise ValueError('environment role overlap')
        environments[env] = role
        if type(cell['stream_seed']) is not int or not 0 <= cell['stream_seed'] < 2**32:
            raise ValueError('invalid stream seed')
        for key in ('sample_ids', 'adaptation_ids'):
            if type(cell[key]) is not list or not cell[key]: raise ValueError('nonempty explicit image IDs required')
            for sample in cell[key]:
                _text(sample)
                if sample in samples: raise ValueError('overlapping adaptation/evaluation image IDs')
                samples.add(sample)
    if counts['fit'] < 2 or any(counts[r] == 0 for r in ROLES):
        raise ValueError('all four roles and at least two fitting cells required')
    if type(p['prior_access']) is not dict or set(p['prior_access']) != set(environments):
        raise ValueError('exact environment access-history declarations required')
    allowed = {'SYNTHETIC'} if p['claim_scope'] == 'synthetic_validation' else {'UNKNOWN', 'OPENED'}
    if any(history not in allowed for history in p['prior_access'].values()):
        raise ValueError('unsupported access-history assertion')
    return json.loads(_canonical(p))


def _locked(p):
    p = validate_protocol(p)
    if p['status'] != 'LOCKED_COMPUTATIONAL': raise ValueError('computational protocol must be locked first')
    return p


def _rows(p, rows, role, *, features, outcomes):
    """Check role/key boundary before reading any features or correctness arrays."""
    if type(rows) is not list: raise ValueError('explicit row list required')
    expected = {c['cell_id']: c for c in p['cells'] if c['role'] == role}
    found = {}
    fields = {'cell_id', 'sample_ids', 'candidate_state_sha256'}
    if features: fields.add('Z')
    if outcomes: fields.update(('frozen_correct', 'candidate_correct'))
    for row in rows:
        _exact(row, fields, 'role row')
        name = row['cell_id']; _text(name)
        if name not in expected or name in found: raise ValueError('row outside required role or duplicate')
        if row['sample_ids'] != expected[name]['sample_ids']: raise ValueError('ordered sample identity mismatch')
        _sha(row['candidate_state_sha256'])
        found[name] = row
    if set(found) != set(expected): raise ValueError('incomplete role rows')
    ordered = [found[name] for name in expected]
    for row in ordered:
        if features:
            z = row['Z']
            if type(z) is not list or len(z) != len(EVIDENCE_NAMES): raise ValueError('feature shape mismatch')
            for value in z: _number(value)
            for i in (1, 4, 8): _number(z[i], 0, 1)
            for i in (0, 2, 3, 5, 9, 10): _number(z[i], 0)
            if any(abs(v) > np.finfo(np.float32).max for v in z): raise ValueError('features overflow prediction dtype')
        if outcomes:
            for key in ('frozen_correct', 'candidate_correct'):
                values = row[key]
                if type(values) is not list or len(values) != len(row['sample_ids']): raise ValueError('correctness shape mismatch')
                for value in values:
                    if type(value) is not int or value not in (0, 1): raise ValueError('binary integer correctness required')
    return json.loads(_canonical(ordered))


def _benefits(rows):
    return np.asarray([np.mean(r['candidate_correct']) - np.mean(r['frozen_correct']) for r in rows])


def _export_model(model):
    import sklearn
    trees = []
    for estimator in model.estimators_[:, 0]:
        tree = estimator.tree_
        trees.append({'left': tree.children_left.tolist(), 'right': tree.children_right.tolist(),
                      'feature': tree.feature.tolist(), 'threshold': tree.threshold.tolist(),
                      'value': tree.value[:, 0, 0].tolist()})
    return {'schema': 'natural-safe-gbr-v1', 'sklearn_version': sklearn.__version__,
            'init': float(model.init_.constant_[0, 0]), 'learning_rate': .05, 'trees': trees,
            'projection': 'clip prediction to [-1,1] before tuning/calibration/decisions'}


def _predict(model, z):
    _exact(model, {'schema', 'sklearn_version', 'init', 'learning_rate', 'trees', 'projection'}, 'predictor')
    if (model['schema'] != 'natural-safe-gbr-v1' or model['learning_rate'] != .05
            or model['projection'] != 'clip prediction to [-1,1] before tuning/calibration/decisions'
            or type(model['trees']) is not list or len(model['trees']) != 250):
        raise ValueError('unsupported fixed predictor')
    _text(model['sklearn_version'])
    x = np.asarray(z, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(EVIDENCE_NAMES) or not np.isfinite(x).all():
        raise ValueError('finite feature matrix required')
    result = np.full(len(x), _number(model['init'], -1, 1), dtype=float)
    for tree in model['trees']:
        _exact(tree, {'left', 'right', 'feature', 'threshold', 'value'}, 'tree')
        if any(type(v) is not list for v in tree.values()): raise ValueError('invalid tree arrays')
        n = len(tree['value'])
        if not 1 <= n <= 7 or any(len(v) != n for v in tree.values()): raise ValueError('tree dimensions invalid')
        reachable, stack = set(), [(0, 0)]
        while stack:
            node, depth = stack.pop()
            if node in reachable or depth > 2: raise ValueError('tree cycle, sharing or excessive depth')
            reachable.add(node)
            left, right, feature = (tree[k][node] for k in ('left', 'right', 'feature'))
            if any(type(v) is not int for v in (left, right, feature)): raise ValueError('tree integer fields required')
            _number(tree['value'][node]); _number(tree['threshold'][node])
            if left == right == -1:
                if feature != -2: raise ValueError('invalid leaf feature')
            else:
                if not (0 <= feature < len(EVIDENCE_NAMES) and node < left < n and node < right < n and left != right):
                    raise ValueError('invalid tree edge')
                stack.extend(((left, depth + 1), (right, depth + 1)))
        if len(reachable) != n: raise ValueError('unreachable tree nodes')
        for i, row in enumerate(x):
            node = 0
            while tree['left'][node] != -1:
                # sklearn promotes its float32 feature to double for comparing
                # the double split threshold. NumPy2 scalar promotion does not.
                node = tree['left'][node] if float(row[tree['feature'][node]]) <= tree['threshold'][node] else tree['right'][node]
            result[i] += .05 * tree['value'][node]
    if not np.isfinite(result).all(): raise ValueError('nonfinite prediction')
    return np.clip(result, -1, 1)


def develop_gate(protocol, fit_rows, tune_rows):
    """Exactly one supervised fit, then development-only finite-grid selection."""
    from sklearn.ensemble import GradientBoostingRegressor
    p = _locked(protocol); implementation = _implementation()
    fit = _rows(p, fit_rows, 'fit', features=True, outcomes=True)
    tune = _rows(p, tune_rows, 'tune', features=True, outcomes=True)
    model = GradientBoostingRegressor(n_estimators=250, max_depth=2,
        learning_rate=.05, subsample=.8, random_state=0)
    model.fit([r['Z'] for r in fit], _benefits(fit))
    frozen = _export_model(model)
    prediction = _predict(frozen, [r['Z'] for r in tune])
    margin = select_margin(prediction.tolist(), _benefits(tune).tolist(), p['margin_grid'], p['harmful_cost'])
    if implementation != _implementation(): raise ValueError('implementation changed during development')
    return _seal({'schema': 'natural-developed-v1', 'protocol_sha256': _hash(p),
                  'implementation_sha256': implementation,
                  'fit_binding': _hash(fit), 'tune_binding': _hash(tune),
                  'predictor': frozen, 'margin': margin})


def _development(p, developed):
    _verify(developed, {'protocol_sha256', 'implementation_sha256', 'fit_binding', 'tune_binding', 'predictor', 'margin'}, 'natural-developed-v1')
    if developed['implementation_sha256'] != _implementation(): raise ValueError('implementation identity mismatch')
    for key in ('protocol_sha256', 'fit_binding', 'tune_binding'): _sha(developed[key])
    if developed['protocol_sha256'] != _hash(p) or developed['margin'] not in p['margin_grid']:
        raise ValueError('development protocol/margin mismatch')
    _predict(developed['predictor'], [[0.] * len(EVIDENCE_NAMES)])


def calibrate_gate(protocol, developed, calibration_rows):
    """Consume the already frozen fit/tuning artifact; never fit or select here."""
    p = _locked(protocol); _development(p, developed)
    rows = _rows(p, calibration_rows, 'calibrate', features=True, outcomes=True)
    residuals = np.abs(_predict(developed['predictor'], [r['Z'] for r in rows]) - _benefits(rows)).tolist()
    radius = _checked_radius(residuals)
    return _seal({'schema': 'natural-gate-v1', 'protocol': p, 'development': developed,
                  'calibration_binding': _hash(rows), 'residuals': residuals,
                  'calibration_n': len(rows), 'calibration_rank': math.ceil((len(rows) + 1) * .9),
                  'radius': 'infinity' if math.isinf(radius) else radius})


def prepare_gate(protocol, fit_rows, tune_rows, calibration_rows):
    return calibrate_gate(protocol, develop_gate(protocol, fit_rows, tune_rows), calibration_rows)


def _checked_radius(residuals):
    if type(residuals) is not list: raise ValueError('residual list required')
    values = sorted(_number(value, 0, 2) for value in residuals)
    k = math.ceil((len(values) + 1) * .9)
    expected = math.inf if k > len(values) else values[k - 1]
    radius = exact_radius(residuals)
    if radius != expected: raise ValueError('calibration backend disagrees with exact order statistic')
    return radius


def _gate(gate):
    _verify(gate, {'protocol', 'development', 'calibration_binding', 'residuals', 'calibration_n',
                   'calibration_rank', 'radius'}, 'natural-gate-v1')
    p = _locked(gate['protocol']); _development(p, gate['development']); _sha(gate['calibration_binding'])
    n = sum(c['role'] == 'calibrate' for c in p['cells'])
    if (type(gate['calibration_n']) is not int or gate['calibration_n'] != n
            or type(gate['calibration_rank']) is not int
            or gate['calibration_rank'] != math.ceil((n + 1) * .9)
            or type(gate['residuals']) is not list or len(gate['residuals']) != n):
        raise ValueError('calibration count/rank mismatch')
    r = _checked_radius(gate['residuals'])
    if gate['radius'] != ('infinity' if math.isinf(r) else r): raise ValueError('calibration radius mismatch')
    return p


def _actions(prediction, radius, margin):
    return {'always_freeze': 'FREEZE', 'always_adapt': 'ADAPT',
            'point_benefit': 'ADAPT' if prediction > 0 else 'FREEZE',
            'fixed_margin': 'ADAPT' if prediction > margin else 'FREEZE',
            'KGA': 'ADAPT' if prediction > radius else 'FREEZE' if prediction < -radius else 'ABSTAIN'}


def decide_rows(gate, feature_rows):
    p = _gate(gate)
    features = _rows(p, feature_rows, 'score', features=True, outcomes=False)
    predictions = _predict(gate['development']['predictor'], [r['Z'] for r in features])
    r = math.inf if gate['radius'] == 'infinity' else gate['radius']
    margin = gate['development']['margin']
    rows = []
    for feature, prediction in zip(features, predictions):
        rows.append({k: feature[k] for k in ('cell_id', 'sample_ids', 'candidate_state_sha256')})
        rows[-1].update(prediction=float(prediction), radius=gate['radius'],
                        threshold={'point_benefit': 0., 'fixed_margin': margin, 'KGA': gate['radius']},
                        action=_actions(float(prediction), r, margin))
    return _seal({'schema': 'natural-decisions-v1', 'protocol_sha256': _hash(p),
                  'gate_sha256': gate['artifact_sha256'], 'features_sha256': _hash(features),
                  'rows': rows, 'warnings': p['warnings'], 'unavailable_methods': p['unavailable_methods']})


def score_rows(protocol, packet, outcomes):
    """No fitting; externally bind packet bytes before permitting this stage."""
    p = _locked(protocol)
    _verify(packet, {'protocol_sha256', 'gate_sha256', 'features_sha256', 'rows', 'warnings',
                     'unavailable_methods'}, 'natural-decisions-v1')
    for key in ('protocol_sha256', 'gate_sha256', 'features_sha256'): _sha(packet[key])
    if (packet['protocol_sha256'] != _hash(p) or packet['warnings'] != p['warnings']
            or packet['unavailable_methods'] != p['unavailable_methods']):
        raise ValueError('decision protocol/warnings/availability mismatch')
    observed = _rows(p, outcomes, 'score', features=False, outcomes=True)
    if type(packet['rows']) is not list or len(packet['rows']) != len(observed): raise ValueError('decision rows mismatch')
    common_threshold = None
    for row, outcome in zip(packet['rows'], observed):
        _exact(row, {'cell_id', 'sample_ids', 'candidate_state_sha256', 'prediction', 'radius', 'threshold', 'action'}, 'decision')
        if any(row[k] != outcome[k] for k in ('cell_id', 'sample_ids', 'candidate_state_sha256')):
            raise ValueError('decision/outcome identity mismatch')
        pred = _number(row['prediction'], -1, 1)
        r = math.inf if row['radius'] == 'infinity' else _number(row['radius'], 0)
        _exact(row['threshold'], {'KGA', 'point_benefit', 'fixed_margin'}, 'thresholds')
        margin = _number(row['threshold']['fixed_margin'], 0)
        if margin not in p['margin_grid'] or row['threshold']['KGA'] != row['radius'] or row['threshold']['point_benefit'] != 0.:
            raise ValueError('threshold mismatch')
        if row['action'] != _actions(pred, r, margin): raise ValueError('action inconsistent with shared prediction')
        if common_threshold is not None and row['threshold'] != common_threshold:
            raise ValueError('every scored cell must use the same frozen thresholds')
        common_threshold = row['threshold']
    frozen = np.asarray([np.mean(r['frozen_correct']) for r in observed])
    candidate = np.asarray([np.mean(r['candidate_correct']) for r in observed])
    benefit = candidate - frozen
    policies = {}
    for policy in POLICIES:
        action = np.asarray([r['action'][policy] for r in packet['rows']])
        result = policy_metrics(action, frozen, candidate, benefit)
        result['mean_accuracy'] = result.pop('mean_acc')
        result['commitment_rate'] = result.pop('coverage')
        adapt = action == 'ADAPT'; count = int(adapt.sum())
        strict = int((adapt & (benefit < 0)).sum())
        result.update(adapt_exposure=float(adapt.mean()), n_strict_harmful_adapt=strict,
                      n_adapt_ties=int((adapt & (benefit == 0)).sum()),
                      conditional_strict_harm_rate=strict / count if count else None,
                      unconditional_strict_harm_rate=strict / len(adapt),
                      positive_benefit_captured=float(np.maximum(benefit, 0)[adapt].sum()),
                      positive_benefit_available=float(np.maximum(benefit, 0).sum()),
                      false_adapt_definition='ADAPT and measured benefit <= 0; no unconditional empirical coverage claim')
        policies[policy] = result
    cells = []
    for i, (decision, outcome) in enumerate(zip(packet['rows'], observed)):
        oracle = float(max(frozen[i], candidate[i]))
        cell = {k: outcome[k] for k in ('cell_id', 'sample_ids', 'candidate_state_sha256')}
        cell.update(frozen_accuracy=float(frozen[i]), candidate_accuracy=float(candidate[i]),
                    benefit=float(benefit[i]), oracle_accuracy=oracle, policies={})
        for policy in POLICIES:
            action = decision['action'][policy]
            accuracy = float(candidate[i] if action == 'ADAPT' else frozen[i])
            cell['policies'][policy] = {'action': action, 'accuracy': accuracy, 'regret': oracle - accuracy}
        cells.append(cell)
    return _seal({'schema': 'natural-scores-v1', 'protocol_sha256': _hash(p),
                  'decisions_sha256': packet['artifact_sha256'], 'outcomes_sha256': _hash(observed),
                  'aggregation': 'equal-cell paired oracle; ABSTAIN retains frozen; cells are not independent domains',
                  'cells': cells, 'policies': policies, 'warnings': p['warnings'], 'unavailable_methods': p['unavailable_methods']})


def select_margin(predictions, outcomes, margins, harmful_cost):
    """Minimize declared tuning cost; on exact ties choose larger margin.

    ADAPT iff prediction > margin. Harmful adaptation costs cost*(-benefit);
    retaining the frozen model costs max(benefit, 0). Uses the same predictor
    outputs as KGA/point gating; this function fits or calibrates nothing.
    """
    if any(type(values) not in (list, tuple) for values in (predictions, outcomes, margins)):
        raise ValueError('explicit sequences required')
    if not predictions or len(predictions) != len(outcomes) or not margins:
        raise ValueError('nonempty aligned tuning data and margin grid required')
    def finite_number(value):
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False
    if not finite_number(harmful_cost) or harmful_cost <= 0:
        raise ValueError('harmful_cost must be positive and finite')
    if any(not finite_number(x) for values in (predictions, outcomes, margins) for x in values):
        raise ValueError('finite numeric tuning values required')
    if any(abs(b) > 1 for b in outcomes) or any(m < 0 for m in margins):
        raise ValueError('benefit or margin outside permitted range')
    def objective(margin):
        return math.fsum((harmful_cost * max(-b, 0.) if p > margin else max(b, 0.)) / len(predictions)
                         for p, b in zip(predictions, outcomes))
    return float(min(set(margins), key=lambda margin: (objective(margin), -margin)))
