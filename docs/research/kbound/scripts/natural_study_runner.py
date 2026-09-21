#!/usr/bin/env python3
"""Separate SHA-bound natural-study saved-record stages (no image execution).

preflight -> lock -> develop -> calibrate -> decide -> score -> verify.
Develop alone fits and selects a margin. Calibrate consumes the frozen model.
Decide accepts no scored outcomes. Score first reconstructs the bound decision
from features and gate, then opens outcomes; verify also checks saved metrics.
Lock means computational immutability, not approval, historical nonaccess,
external source authentication or evidence of a population guarantee.

Every input requires a raw-byte SHA256 supplied independently by the caller.
Continue at a completed stage by supplying its saved digest to the next stage;
never overwrite a completed stage. No automatic downloads or device fallback.
"""
from __future__ import annotations

import argparse
import cProfile
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
import time

from task3_common_analysis import _read_bound, _preflight_output, _write_output

STAGES = {
    'preflight': ('protocol',),
    'lock': ('protocol',),
    'develop': ('protocol', 'fit-rows', 'tune-rows'),
    'calibrate': ('protocol', 'developed', 'calibration-rows'),
    'decide': ('protocol', 'gate', 'score-features'),
    'score': ('protocol', 'gate', 'score-features', 'decisions', 'score-outcomes'),
    'verify': ('protocol', 'gate', 'score-features', 'decisions', 'score-outcomes', 'metrics'),
}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='phase', required=True)
    for phase, keys in STAGES.items():
        sub = subs.add_parser(phase)
        for key in keys:
            sub.add_argument('--' + key, type=Path, required=True)
            sub.add_argument('--' + key + '-sha256', required=True, help='Expected SHA256 of raw input bytes')
        sub.add_argument('--output-dir', type=Path, required=True, help='Fresh directory under an existing parent')
    return parser


def _source_identity():
    import natural_study_contract as m
    return m._implementation()


def _develop(m, values):
    from sklearn.ensemble import GradientBoostingRegressor
    profiler = cProfile.Profile()
    artifact = profiler.runcall(m.develop_gate, values['protocol'], values['fit-rows'], values['tune-rows'])
    code = inspect.unwrap(GradientBoostingRegressor.fit).__code__
    entries = [entry for entry in profiler.getstats() if entry.code is code]
    if len(entries) != 1 or entries[0].callcount != 1:
        raise RuntimeError('could not isolate exactly one predictor fit')
    return artifact, {'predictor_fit_seconds': entries[0].totaltime, 'predictor_fit_call_count': 1,
        'predictor_fit_scope': 'profiled cumulative GBR.fit only; excludes tuning, calibration, adaptation and inference'}


def main(argv=None):
    started = time.perf_counter(); parser = _parser(); args = parser.parse_args(argv)
    try:
        _preflight_output(args.output_dir)
        sources = _source_identity()
        import natural_study_contract as m
        import sklearn
        bindings, values = {}, {}
        def read(key):
            attr = key.replace('-', '_')
            values[key], bindings[key] = _read_bound(getattr(args, attr), getattr(args, attr + '_sha256'))
        # Score correctness is deliberately not read by this loop.
        for key in STAGES[args.phase]:
            if key not in ('score-outcomes', 'metrics'): read(key)
        p = m.validate_protocol(values['protocol'])
        timing = {'predictor_fit_seconds': 0., 'predictor_fit_call_count': 0,
                  'predictor_fit_scope': 'no predictor fitting in this stage'}
        if args.phase == 'preflight':
            artifact = {'schema': 'natural-preflight-v1', 'protocol_sha256': m._hash(p),
                        'role_counts': {r: sum(c['role'] == r for c in p['cells']) for r in m.ROLES},
                        'claim_scope': p['claim_scope'], 'status': 'SCHEMA_VALID_NOT_EXECUTION_AUTHORIZATION'}
            name = 'preflight'; status = artifact['status']
        elif args.phase == 'lock':
            if p['status'] != 'DRAFT': raise ValueError('lock requires a draft; never relock a completed protocol')
            p['status'] = 'LOCKED_COMPUTATIONAL'
            artifact = m.validate_protocol(p); name = 'protocol'; status = 'LOCKED_COMPUTATIONAL_NOT_PROSPECTIVE'
        elif args.phase == 'develop':
            artifact, timing = _develop(m, values)
            name = 'developed'; status = 'FIT_AND_MARGIN_FROZEN_NOT_CALIBRATED'
        elif args.phase == 'calibrate':
            artifact = m.calibrate_gate(p, values['developed'], values['calibration-rows'])
            name = 'gate'; status = 'CALIBRATED_CELL_GATE_NOT_POPULATION_CERTIFICATE'
        else:
            if values['gate']['protocol'] != p: raise ValueError('gate protocol mismatch')
            decision = m.decide_rows(values['gate'], values['score-features'])
            if args.phase == 'decide':
                artifact = decision; name = 'decisions'; status = 'DECISIONS_BOUND_NOT_SCORED'
            else:
                if decision != values['decisions']:
                    raise ValueError('saved decisions differ from outcome-free gate/features reconstruction')
                read('score-outcomes')  # Only after complete outcome-free decision validation.
                artifact = m.score_rows(p, decision, values['score-outcomes'])
                name = 'metrics'; status = 'SCORED_SAVED_RECORDS_NOT_NATIVE_AUTHENTICATION'
                if args.phase == 'verify':
                    read('metrics')
                    if artifact != values['metrics']: raise ValueError('saved metrics differ from paired record arithmetic')
                    artifact = {'schema': 'natural-verification-v1', 'protocol_sha256': m._hash(p),
                                'gate_sha256': values['gate']['artifact_sha256'],
                                'decisions_sha256': decision['artifact_sha256'],
                                'metrics_sha256': artifact['artifact_sha256'],
                                'verified_cells': len(decision['rows']), 'warnings': p['warnings'],
                                'unavailable_methods': p['unavailable_methods']}
                    name = 'verification'; status = 'VERIFIED_SAVED_RECORDS_NOT_POPULATION_OR_PROSPECTIVE_EVIDENCE'
        if sources != _source_identity(): raise ValueError('analysis source changed during stage')
        receipt = {'schema': 'natural-stage-receipt-v1', 'phase': args.phase, 'status': status,
                   'created_at_utc': datetime.now(timezone.utc).isoformat(), 'inputs': bindings,
                   'implementation_source_sha256_bytes': sources, 'timing': timing,
                   'runtime': {'python': sys.version, 'executable': sys.executable,
                               'numpy': m.np.__version__, 'sklearn': sklearn.__version__},
                   'scope': {'native_execution_complete': False, 'release_sealed': False,
                             'population_guarantee': False, 'prospective_nonaccess_verified': False,
                             'stage_completed': args.phase},
                   'warnings': p['warnings'], 'unavailable_methods': p['unavailable_methods'],
                   'binding_scope': 'caller supplied raw-byte digests; not independently authenticated source or access chronology',
                   'sealing_scope': 'exclusive fresh files0444/directory0555; no overwrite or privileged tamper protection'}
        _write_output(args.output_dir, name + '.json', artifact, receipt, started)
        print(json.dumps({'status': status, 'output': str(args.output_dir), 'artifact': name + '.json'}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.exit(2, 'error: ' + str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
