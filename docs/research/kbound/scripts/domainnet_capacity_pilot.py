#!/usr/bin/env python3
"""One approved CPU capacity pilot, with separate prediction and label processes.

Not an N2 benchmark, source training, reference reproduction or release gate.
No reusable CLI for changing domains, checkpoints, devices or sample budgets.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import warnings

from task3_common_analysis import _directory, _encode, _preflight_output, _read_bound

APPROVED_SHA = '8580ce1a2ac494ee12f25716edf5d8bdc4deb5eac974d36e6466463a6e41a072'
HERE = Path(__file__).resolve().parent
SOURCES = ('domainnet_capacity_pilot.py', 'domainnet_pilot_images.py',
           'domainnet_reference_adapter.py', 'domainnet_reference_source.py',
           'task3_image_panel.py', 'task3_common_analysis.py')


def implementation():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCES}


def read_approved(path, expected_sha):
    if expected_sha != APPROVED_SHA:
        raise ValueError('only the explicitly approved proposal is executable')
    proposal, _ = _read_bound(path, expected_sha)
    return proposal


def write_json(path, value):
    """Exclusive descriptor-relative write; no permission-seal claim on ExFAT."""
    path = Path(path)
    data = _encode(value)
    if len(data) > 1024**2:
        raise ValueError('pilot artifact exceeds one MiB')
    parent = _directory(path.parent)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
    finally:
        os.close(parent)
    return hashlib.sha256(data).hexdigest()


def disk_bytes(root):
    total = 0
    for folder, dirs, names in os.walk(root, followlinks=False):
        if any((Path(folder) / n).is_symlink() for n in dirs + names):
            raise ValueError('symlink appeared in pilot output')
        total += sum((Path(folder) / n).stat().st_size for n in names)
    return total


def run_bounded(command, output_root, budget):
    """Watch and reap only a new private process group. Never attach to old jobs.

    RSS/disk ceilings are sampled every 50ms, not kernel memory quotas; transient
    overshoot is possible. Workers also check peak RSS at phase boundaries.
    """
    import psutil
    output_root = Path(output_root)
    directory = _directory(output_root)
    start, peak, size, status = time.monotonic(), 0, 0, None
    process = None
    try:
        fd = os.open('worker.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        with os.fdopen(fd, 'wb') as log:
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=output_root, env=env, start_new_session=True)
            observed = psutil.Process(process.pid)
            while True:
                try:
                    procs = [observed] + observed.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in procs if p.is_running())
                except psutil.NoSuchProcess:
                    rss = 0
                peak = max(peak, rss)
                size = disk_bytes(output_root)
                if time.monotonic() - start > budget['hard_wall_ceiling_seconds']:
                    status = 'STOP_WALL'
                elif peak > budget['peak_rss_stop_bytes']:
                    status = 'STOP_RSS'
                elif size > budget['max_new_disk_bytes']:
                    status = 'STOP_DISK'
                if status:
                    break
                if process.poll() is not None:
                    status = 'PROCESS_EXIT_0' if process.returncode == 0 else 'PROCESS_FAILED'
                    break
                time.sleep(.05)
    finally:
        if process is not None:
            # Reap and terminate this group's descendants even after parent exit.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        os.close(directory)
    return {'status': status, 'returncode': process.returncode, 'child_reaped': True,
            'wall_seconds': time.monotonic() - start, 'sampled_peak_rss_bytes': peak,
            'observed_output_bytes': size, 'sampling_interval_seconds': .05,
            'ceiling_qualification': 'polled RSS/disk stop limits; transient overshoot possible'}


def score_predictions(packet, labels, *, classes):
    ids = packet['sample_ids']
    names = ('frozen', 'candidate_direct', 'candidate_refined')
    if (not ids or len(ids) != len(set(ids)) or type(classes) is not int or classes <= 0
            or any(not isinstance(packet[n], list) or len(packet[n]) != len(ids) for n in names)):
        raise ValueError('unaligned predictions')
    if any(type(v) is not int or not 0 <= v < classes for n in names for v in packet[n]):
        raise ValueError('invalid predicted class')
    if any(i not in labels or type(labels[i]) is not int or not 0 <= labels[i] < classes for i in ids):
        raise ValueError('invalid/missing selected truth')
    correct = {n: [int(v == labels[i]) for i, v in zip(ids, packet[n])] for n in names}
    means = {n + '_accuracy': sum(correct[n]) / len(ids) for n in names}
    return {**means, 'direct_benefit': means['candidate_direct_accuracy'] - means['frozen_accuracy'],
            'refined_benefit': means['candidate_refined_accuracy'] - means['frozen_accuracy'],
            'correctness': correct, 'n': len(ids),
            'scope': '32-image development capacity diagnostic; no model/recipe selection or routing conclusion'}


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


@contextmanager
def visible_warnings():
    """Record warnings and emit immediately, including on exception or hard stop."""
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        captured = []
        def show(message, category, filename, lineno, file=None, line=None):
            captured.append(warnings.WarningMessage(message, category, filename, lineno))
            print(warnings.formatwarning(message, category, filename, lineno, line),
                  file=sys.stderr, end='', flush=True)
        warnings.showwarning = show
        yield captured


def collect_payload(proposal):
    """No outcome-list access; images supplied only by the selected-member broker."""
    started = time.perf_counter()
    timing = {}
    with visible_warnings() as emitted:
        import torch
        from domainnet_reference_adapter import build_session, _plain_args, _state_digest
        from domainnet_pilot_images import load_selected
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        def measure(name, function):
            tick = time.perf_counter()
            result = function()
            timing[name] = time.perf_counter() - tick
            print(name, round(timing[name], 3), flush=True)
            if peak_rss() > proposal['budget']['peak_rss_stop_bytes']:
                raise MemoryError('phase peak RSS ceiling')
            return result
        config = {'reference_root': proposal['reference']['root'], 'epochs': 1, 'steps_per_epoch': 4, 'seed': 2020}
        session = measure('source_session_seconds', lambda: build_session(proposal['checkpoint_path'], config, 'cpu'))
        if _plain_args(session.args) != proposal['execution']['resolved_args']:
            raise ValueError('resolved native arguments differ from approved proposal')
        cache, inventory = measure('archive_and_decode_seconds', lambda: load_selected(
            proposal['archive'], proposal['selection']['members'],
            cache_limit=proposal['budget']['decoded_rgb_cache_stop_bytes']))
        adapt = [m['image_id'] for m in proposal['selection']['members'] if m['role'] == 'adaptation']
        evaluate = [m['image_id'] for m in proposal['selection']['members'] if m['role'] == 'diagnostic_evaluation']
        if len(adapt) != 16 or len(evaluate) != 32 or len(cache) != 48:
            raise ValueError('approved role counts mismatch')
        test_transform = session.ref.get_augmentation('test')
        augmentation = session.ref.get_augmentation_versions(session.args)
        def batches(ids):
            for i in range(0, len(ids), 4):
                yield torch.stack([test_transform(cache[n]) for n in ids[i:i+4]])
        frozen = measure('frozen_inference_seconds', lambda: session.predict(batches(evaluate)).tolist())
        measure('bank_seconds', lambda: session.initialize_bank(batches(adapt)))
        class Epoch:
            def __len__(self): return 4
            def __iter__(self):
                for i in range(0, 16, 4):
                    images = [augmentation(cache[n]) for n in adapt[i:i+4]]
                    views = [torch.stack([v[j] for v in images]) for j in range(3)]
                    yield views, torch.arange(i, i+4)
        measure('four_updates_seconds', lambda: session.train_epoch(Epoch(), 0))
        direct = measure('candidate_direct_seconds', lambda: session.predict(batches(evaluate)).tolist())
        refined = measure('candidate_refined_seconds', lambda: session.predict(batches(evaluate), refine=True).tolist())
        state_digest = _state_digest({'model': session.model.state_dict(), 'banks': session.banks,
                                      'queue_ptr': session.model.queue_ptr, 'completed_steps': session.completed_steps})
        if session.completed_steps != 4 or session.model.queue_ptr != 16:
            raise ValueError('incomplete native update budget')
        timing['collection_total_seconds'] = time.perf_counter() - started
        report = {'schema': 'domainnet-capacity-predictions-v1', 'status': 'COLLECTED_NOT_SCORED',
            'proposal_sha256': APPROVED_SHA, 'sample_ids': evaluate, 'adaptation_ids': adapt,
            'frozen': frozen, 'candidate_direct': direct, 'candidate_refined': refined,
            'candidate_state_sha256': state_digest, 'source_identity': session.identity,
            'source_provenance': session.provenance, 'inventory': inventory,
            'completed_updates': session.completed_steps, 'queue_slots_replaced': session.model.queue_ptr,
            'implementation': implementation(), 'runtime': {'python': platform.python_version(),
                'executable': sys.executable, 'torch': str(torch.__version__), 'device': 'cpu',
                'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads()},
            'stream_order': 'fixed predeclared proposal order, no loader workers, no stream shuffle',
            'unused_native_data_defaults': 'real/sketch, batch128/workers4 retained in args but native data launcher never called; broker uses painting batch4/workers0',
            'peak_rss_bytes': peak_rss(), 'timing': timing, 'prior_access': proposal['prior_access'],
            'scope': proposal['scope'], 'queue_caveat': proposal['execution']['queue_caveat'],
            'study_locked': False, 'benchmark_complete': False}
    report['warnings'] = [{'category': w.category.__name__, 'message': str(w.message),
                           'filename': w.filename, 'lineno': w.lineno} for w in emitted]
    return report


def validate_collection(packet, proposal, expected_implementation):
    expected = [m['image_id'] for m in proposal['selection']['members'] if m['role'] == 'diagnostic_evaluation']
    adapt = [m['image_id'] for m in proposal['selection']['members'] if m['role'] == 'adaptation']
    if (packet['schema'] != 'domainnet-capacity-predictions-v1' or packet['status'] != 'COLLECTED_NOT_SCORED'
            or packet['proposal_sha256'] != APPROVED_SHA or packet['implementation'] != expected_implementation
            or packet['sample_ids'] != expected or packet['adaptation_ids'] != adapt or packet['completed_updates'] != 4
            or packet['study_locked'] is not False or packet['benchmark_complete'] is not False):
        raise ValueError('prediction packet source/scope/selection identity mismatch')
    # Validate class arrays before any actual outcomes become available.
    score_predictions(packet, dict.fromkeys(expected, 0), classes=126)


def score_payload(proposal, predictions_path, predictions_sha, expected_implementation):
    packet, binding = _read_bound(predictions_path, predictions_sha)
    validate_collection(packet, proposal, expected_implementation)
    from domainnet_reference_source import read_verified_checkpoint
    path = Path(proposal['reference']['list_path'])
    # This is the only ground-truth read; collect_payload never calls it.
    raw = read_verified_checkpoint(path, proposal['reference']['list_sha256'], path.stat().st_size)
    mapping, rows = {}, {}
    for line in raw.decode().splitlines():
        image_id, label = line.split(); label = int(label)
        if image_id in rows:
            raise ValueError('duplicate native list row')
        rows[image_id] = label
        category = image_id.split('/')[1]
        if category in mapping and mapping[category] != label:
            raise ValueError('nonfunctional class mapping')
        mapping[category] = label
    class_hash = hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (len(rows) != proposal['reference']['list_rows'] or set(mapping.values()) != set(range(126))
            or class_hash != proposal['reference']['class_map_sha256']):
        raise ValueError('native class mapping mismatch')
    return {'schema': 'domainnet-capacity-score-v1', 'status': 'CAPACITY_SMOKE_SCORED_NOT_BENCHMARK',
            'prediction_binding': binding, 'proposal_sha256': APPROVED_SHA,
            'label_authority_sha256': proposal['reference']['list_sha256'],
            'metrics': score_predictions(packet, rows, classes=126), 'warnings': packet['warnings'],
            'scope': packet['scope'], 'queue_caveat': packet['queue_caveat'],
            'study_locked': False, 'benchmark_complete': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', '_collect-worker', '_score-worker'):
        sub = commands.add_parser(name)
        sub.add_argument('--proposal', required=True)
        sub.add_argument('--proposal-sha256', required=True)
        if name != 'run': sub.add_argument('--authorization-sha256', required=True)
        if name == '_score-worker': sub.add_argument('--predictions-sha256', required=True)
    args = parser.parse_args(argv)
    proposal = read_approved(args.proposal, args.proposal_sha256)
    root = Path(proposal['fresh_output_required'])
    current = implementation()
    if args.command != 'run':
        authorization, _ = _read_bound(root / 'authorization.json', args.authorization_sha256)
        if authorization['implementation'] != current or authorization['proposal_sha256'] != APPROVED_SHA:
            raise ValueError('authorized implementation changed')
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
        if args.command == '_collect-worker':
            result = collect_payload(proposal)
            write_json(root / 'collect' / 'predictions.json', result)
        else:
            result = score_payload(proposal, root / 'collect' / 'predictions.json', args.predictions_sha256, current)
            write_json(root / 'score' / 'metrics.json', result)
        return 0
    _preflight_output(root)
    parent = _directory(root.parent)
    try: os.mkdir(root.name, mode=0o700, dir_fd=parent)
    finally: os.close(parent)
    approval_sha = write_json(root / 'authorization.json', {
        'schema': 'domainnet-capacity-authorization-v1', 'proposal_sha256': APPROVED_SHA,
        'approval': 'User explicitly approved the bounded pilot in the current conversation',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'implementation': current,
        'budget': proposal['budget'], 'scope': proposal['scope']})
    started, reports, status = time.monotonic(), {}, 'NOT_STARTED'
    try:
        for stage in ('collect', 'score'):
            destination = root / stage; destination.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), '_' + stage + '-worker',
                       '--proposal', str(Path(args.proposal).absolute()), '--proposal-sha256', APPROVED_SHA,
                       '--authorization-sha256', approval_sha]
            if stage == 'score':
                data = (root / 'collect' / 'predictions.json').read_bytes()
                command += ['--predictions-sha256', hashlib.sha256(data).hexdigest()]
            budget = dict(proposal['budget'])
            budget['hard_wall_ceiling_seconds'] -= time.monotonic() - started
            budget['max_new_disk_bytes'] -= disk_bytes(root)
            reports[stage] = run_bounded(command, destination, budget)
            write_json(root / (stage + '-process.json'), reports[stage])
            if reports[stage]['status'] != 'PROCESS_EXIT_0':
                status = 'FAILED_' + stage.upper(); break
        else:
            status = 'CAPACITY_SMOKE_COMPLETE_NOT_NATURAL_STUDY'
    finally:
        result = {'schema': 'domainnet-capacity-run-v1', 'status': status, 'processes': reports,
                  'wall_seconds': time.monotonic() - started, 'new_disk_bytes': disk_bytes(root),
                  'proposal_sha256': APPROVED_SHA, 'scope': proposal['scope'],
                  'study_locked': False, 'benchmark_complete': False}
        write_json(root / 'status.json', result)
        print(json.dumps(result, indent=2))
    return 0 if status == 'CAPACITY_SMOKE_COMPLETE_NOT_NATURAL_STUDY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
