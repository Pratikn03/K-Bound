#!/usr/bin/env python3
"""Exact same-input arithmetic replay, not a refit or native-method evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import stat
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CANON = 'experiments/kbound/results/reconciled_panels_v1/'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def array_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def verify_array_hashes(cells, expected_hashes):
    for name in ['prediction', 'radius', 'action']:
        if array_sha([cell[name] for cell in cells]) != expected_hashes[name]:
            raise ValueError(f'canonical {name} array digest mismatch')


def render_latex(scores):
    labels = {'always_adapt': 'Always Adapt', 'always_freeze': 'Always Freeze',
              'confidence_increase': 'Confidence increase', 'entropy_decrease': 'Entropy decrease',
              'point_benefit': 'Point benefit (same prediction)', 'current_kga': 'Current KGA'}
    lines = [r'% Generated arithmetic-only opened-record replay; no native methods or significance claims.',
             r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule',
             r'Policy & Regret & Mean acc. & $\mathrm{FA}_{\mathrm u}$ & Commitment & ADAPT & FREEZE & ABSTAIN \\', r'\midrule']
    for key, label in labels.items():
        value = scores[key]
        count = value['action_counts']
        lines.append(f"{label} & {value['regret']:.4f} & {value['mean_accuracy']:.4f} & {value['fa_u']:.3f} & "
                     f"{value['directional_commitment']:.3f} & {count['ADAPT']} & {count['FREEZE']} & {count['ABSTAIN']} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def build_policy_actions(records, cells):
    if len(records) != len(cells) or not records:
        raise ValueError('records and cell authority length must match and be nonzero')
    if len({cell['sample_id'] for cell in cells}) != len(cells):
        raise ValueError('sample IDs must be unique')
    for record, cell in zip(records, cells):
        values = [record['a0'], record['a_adapted'], record['B'], cell['prediction'], cell['radius'], *record['Z']]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError('all input values must be finite')
        if cell['radius'] < 0:
            raise ValueError('radius must be nonnegative')
        if len(record['Z']) != 11:
            raise ValueError('evidence length must be 11')
        if not 0 <= record['a0'] <= 1 or not 0 <= record['a_adapted'] <= 1:
            raise ValueError('paired accuracies must lie in [0, 1]')
        if abs(record['B'] - (record['a_adapted'] - record['a0'])) > 1e-12:
            raise ValueError('benefit must equal adapted minus frozen accuracy')
        prediction, radius = cell['prediction'], cell['radius']
        expected = 'ADAPT' if prediction - radius > 0 else ('FREEZE' if prediction + radius < 0 else 'ABSTAIN')
        if cell['action'] != expected:
            raise ValueError('stored action disagrees with stored prediction/radius gate')
    return {
        'always_adapt': ['ADAPT'] * len(records),
        'always_freeze': ['FREEZE'] * len(records),
        'confidence_increase': ['ADAPT' if record['Z'][4] > record['Z'][1] else 'FREEZE' for record in records],
        'entropy_decrease': ['ADAPT' if record['Z'][7] > 0 else 'FREEZE' for record in records],
        'point_benefit': ['ADAPT' if cell['prediction'] > 0 else 'FREEZE' for cell in cells],
        'current_kga': [cell['action'] for cell in cells],
    }


def score_actions(records, actions):
    if len(records) != len(actions) or set(actions) - {'ADAPT', 'FREEZE', 'ABSTAIN'}:
        raise ValueError('actions must align to records and use declared labels')
    n = len(records)
    chosen = [record['a_adapted'] if action == 'ADAPT' else record['a0'] for record, action in zip(records, actions)]
    oracle = [max(record['a0'], record['a_adapted']) for record in records]
    counts = {action: actions.count(action) for action in ['ADAPT', 'FREEZE', 'ABSTAIN']}
    false_adapt = sum(action == 'ADAPT' and record['B'] <= 0 for record, action in zip(records, actions))
    return {
        'n': n, 'mean_accuracy': math.fsum(chosen) / n,
        'regret': math.fsum(o - a for o, a in zip(oracle, chosen)) / n,
        'action_counts': counts, 'directional_commitment': (counts['ADAPT'] + counts['FREEZE']) / n,
        'false_adapt_nonpositive_count': false_adapt, 'fa_u': false_adapt / n,
        'fa_c': false_adapt / counts['ADAPT'] if counts['ADAPT'] else None,
        'action_sha256': array_sha(actions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--audit', type=Path, required=True, help='Prior paired-array provenance audit JSON')
    parser.add_argument('--audit-sha256', required=True, help='Expected SHA-256 of the prior audit bytes')
    parser.add_argument('--out-dir', type=Path, required=True, help='New output directory; must not exist')
    args = parser.parse_args()
    root = args.repo.resolve()
    prior_path = args.audit
    prior_bytes = prior_path.read_bytes()
    if sha(prior_bytes) != args.audit_sha256:
        raise ValueError('prior audit SHA-256 mismatch')
    prior = json.loads(prior_bytes, object_pairs_hook=unique_pairs)
    expected = {entry['relative_path']: entry['sha256'] for entry in prior['source_inventory']
                if entry['root_role'] == 'active_checkout' and entry['status'] == 'READ'}
    input_hashes = []

    def authenticated_read(relative):
        path = root / relative
        if getattr(path.stat(), 'st_flags', 0) & getattr(stat, 'SF_DATALESS', 0):
            raise ValueError(f'input is dataless and was not opened: {relative}')
        payload = path.read_bytes()
        digest = sha(payload)
        if digest != expected[relative]:
            raise ValueError(f'input changed since paired-array audit: {relative}')
        input_hashes.append({'path': relative, 'sha256': digest})
        return json.loads(payload, object_pairs_hook=unique_pairs)

    canonical = authenticated_read(CANON + 'canonical_panel_results.json')
    manifest = authenticated_read(CANON + 'source_manifest.json')
    assert canonical['source_manifest_sha256'] == input_hashes[-1]['sha256']
    authority = canonical['panels']['cifar10c']['panel']['candidates']['tent']
    records, cells, seed_metadata = [], [], []
    for seed in range(5):
        relative = CANON + f'source/cifar10c/per_condition_cifar10c_tent_seed{seed}.json'
        source = authenticated_read(relative)
        manifest_row = next(row for row in manifest['files'] if row['destination'] == relative)
        assert input_hashes[-1]['sha256'] == manifest_row['compact_sha256']
        seed_records = source['records']
        saved = next(item for item in authority['per_file'] if item['seed'] == seed)
        seed_cells = saved['current_cell_authority']['cells']
        assert len(seed_records) == len(seed_cells) == 432
        assert [cell['sample_id'] for cell in seed_cells] == [f"tent|seed={seed}|condition={record['condition']}" for record in seed_records]
        verify_array_hashes(seed_cells, {'prediction': saved['current_prediction_sha256'],
                                        'radius': saved['current_radius_sha256'], 'action': saved['current_action_sha256']})
        records.extend(seed_records)
        cells.extend(seed_cells)
        seed_metadata.append({'seed': seed, 'n': 432, 'prediction_sha256': saved['current_prediction_sha256'],
                              'radius_sha256': saved['current_radius_sha256'], 'action_sha256': saved['current_action_sha256']})
    actions = build_policy_actions(records, cells)
    scores = {name: score_actions(records, values) for name, values in actions.items()}
    for name, old_name in [('always_adapt', 'always_adapt'), ('always_freeze', 'always_freeze'), ('current_kga', 'kga')]:
        assert abs(scores[name]['regret'] - authority['regret'][old_name]) < 1e-12
    zero_margin = next(row for row in authority['kappa_sweep'] if row['kappa'] == 0)
    # kappa=0 may ABSTAIN on exactly zero prediction, but its fallback equals point-gate FREEZE.
    assert abs(scores['point_benefit']['regret'] - zero_margin['regret']) < 1e-12
    prediction_hash = array_sha([cell['prediction'] for cell in cells])
    formulas = {
        'always_adapt': 'ADAPT for every cell', 'always_freeze': 'FREEZE for every cell',
        'confidence_increase': 'ADAPT iff Z[post_conf] > Z[pre_conf], else FREEZE; fixed untuned rule',
        'entropy_decrease': 'ADAPT iff Z[entropy_drop] > 0, else FREEZE; fixed untuned rule',
        'point_benefit': 'ADAPT iff canonical current prediction > 0, else FREEZE',
        'current_kga': 'Use canonical current stored action; verify ADAPT iff prediction-radius>0, FREEZE iff prediction+radius<0, else ABSTAIN',
    }
    result = {
        'schema': 'kbound-current-cifar-same-prediction-arithmetic-panel-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'CURRENT_POLICY_REPLACEMENT_CANDIDATE_NOT_AUTOMATIC_MANUSCRIPT_PROMOTION',
        'claim_scope': 'Retrospective opened dependent constructed CIFAR records; arithmetic-only rescore; not prospective evidence and no native-method claims.',
        'generation': {'script_sha256': sha(Path(__file__).read_bytes()), 'source_hash_contract': 'explicit prior-audit SHA-256 plus each canonical input SHA-256',
                       'prior_audit_sha256': sha(prior_path.read_bytes()), 'python': platform.python_version(),
                       'model_imports': False, 'fit_performed': False, 'new_folds_or_sample_ids': False,
                       'bootstrap_or_significance_tests': False, 'native_methods': []},
        'input_hashes': input_hashes,
        'canonical_stored_fit_runtime': canonical['runtime'],
        'identity_contract': 'Exactly the existing canonical literal tent|seed=N|condition=... IDs and stored prediction/radius/action arrays; no v4 hashed-ID substitution.',
        'weight_definition': {'n': 2160, 'per_seed_n': 432, 'seeds': [0, 1, 2, 3, 4], 'each_row_weight_numerator': 1, 'each_row_weight_denominator': 2160},
        'metric_definitions': {'paired_oracle': 'max(a0, a_adapted)', 'selected_accuracy': 'a_adapted after ADAPT; a0 after FREEZE or ABSTAIN',
                               'regret': 'equal-row-weight mean(paired_oracle - selected_accuracy)',
                               'fa_u': 'count(ADAPT and B <= 0) / 2160', 'directional_commitment': '(ADAPT + FREEZE) / 2160'},
        'same_prediction_check': {'point_benefit_prediction_sha256': prediction_hash, 'current_kga_prediction_sha256': prediction_hash,
                                  'all_2160_predictions_shared': True, 'current_kga_matches_canonical_regret': True,
                                  'point_gate_matches_canonical_kappa_zero_regret': True},
        'per_seed_authority': seed_metadata, 'policy_formulas': formulas, 'policies': scores,
        'historical_table_binding': 'Neither the synchronized table (0.0089/0.1229, actions1100/367/693) nor same-predictor threshold table (0.00892/0.12290, actions1101/372/687) has identified matching paired-array authority in the audited trees. This new panel does not reconstruct those tables.',
    }
    row_payload = ''.join(json.dumps({'sample_id': cell['sample_id'], 'seed': record['seed'], 'condition': record['condition'],
                                      'a0': record['a0'], 'a_adapted': record['a_adapted'], 'paired_oracle': max(record['a0'], record['a_adapted']),
                                      'B': record['B'], 'Z': record['Z'], 'current_prediction': cell['prediction'], 'current_radius': cell['radius'],
                                      'policy_actions': {name: values[index] for name, values in actions.items()}},
                                     sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
                          for index, (record, cell) in enumerate(zip(records, cells)))
    result['cell_record_file'] = {'path': 'cells.jsonl', 'sha256': sha(row_payload.encode()), 'n': len(cells)}
    latex = render_latex(scores)
    result['latex_file'] = {'path': 'current_arithmetic_panel.tex', 'sha256': sha(latex.encode())}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / 'cells.jsonl').write_text(row_payload)
    (args.out_dir / 'current_arithmetic_panel.tex').write_text(latex)
    (args.out_dir / 'CURRENT_ARITHMETIC_PANEL.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.out_dir), 'policies': scores}, indent=2))


if __name__ == '__main__':
    main()
