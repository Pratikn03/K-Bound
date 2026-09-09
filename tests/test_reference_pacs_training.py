"""Source-only PACS contracts; tiny fixtures never certify a full source run."""
import copy
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from experiments.kbound.reference_source import train_pacs as p


def test_pinned_hparams_and_job_grid():
    assert p.seed_hash('PACS', 'ERM', [0], 1, 2) == 1553601838
    assert p.hparams(0, 0)['batch_size'] == 32
    assert p.hparams(1, 0) == dict(data_augmentation=True, resnet18=False,
        class_balanced=False, resnet_dropout=.1, lr=5.0781288859686544e-05,
        batch_size=44, weight_decay=.00046410133598234803)
    assert p.hparams(19, 2)['lr'] == 7.509081014584126e-05
    assert p.hparams(19, 2)['batch_size'] == 8
    jobs = p.jobs()
    assert len(jobs) == 240
    assert len({(j['target'], j['trial_seed'], j['hparams_seed']) for j in jobs}) == 240
    for args in [(20, 0), (0, 3), (-1, 0), (True, 0)]:
        with pytest.raises(p.SourceError):
            p.hparams(*args)


def test_split_exact_pinned_numpy_order_without_global_rng():
    torch.manual_seed(77)
    before = torch.get_rng_state().clone()
    indices = list(range(23))
    np.random.RandomState(p.seed_hash(2, 1)).shuffle(indices)
    assert p.partition(23, 2, 1) == {'out': indices[:4], 'in': indices[4:]}
    assert torch.equal(before, torch.get_rng_state())


class Infinite:
    def __init__(self, sampler): self.sampler = sampler
    def __iter__(self):
        while True:
            yield from self.sampler


@pytest.mark.parametrize('replacement,drop_last', [(True, True), (False, False)])
def test_sampler_matches_pinned_workers_zero_and_resume(replacement, drop_last):
    torch.manual_seed(123)
    loader = iter(DataLoader(range(11), batch_sampler=Infinite(BatchSampler(
        RandomSampler(range(11), replacement=replacement), 4, drop_last)), num_workers=0))
    expected = []
    for _ in range(12):
        expected.append(next(loader).tolist())
        torch.rand(3)  # Interleave augmentation/model RNG between batches.
    end = torch.get_rng_state()
    torch.manual_seed(123)
    sampler = p.ReplaySampler(11, 4, replacement=replacement, drop_last=drop_last)
    actual = []
    for i in range(12):
        actual.append(sampler.next())
        torch.rand(3)
        if i == 4:
            state, rng = sampler.state_dict(), torch.get_rng_state()
            sampler = p.ReplaySampler(11, 4, replacement=replacement, drop_last=drop_last)
            sampler.load_state_dict(state)
            torch.set_rng_state(rng)
    assert actual == expected
    assert torch.equal(torch.get_rng_state(), end)
    bad = copy.deepcopy(sampler.state_dict()); bad['order'][0] = -1
    with pytest.raises(p.SourceError): sampler.load_state_dict(bad)
    bad = copy.deepcopy(sampler.state_dict()); bad['cursor'] = 1
    with pytest.raises(p.SourceError): sampler.load_state_dict(bad)


def fixture_population(tmp_path, target='sketch'):
    root = tmp_path / 'images'; root.mkdir()
    domains = {}
    for domain in p.DOMAINS:
        if domain == target: continue
        lines = []
        for cls in range(2):
            folder = root / domain / f'c{cls}'; folder.mkdir(parents=True)
            for i in range(5):
                path = folder / f'{i}.png'
                Image.new('RGBA', (9, 7), (10, 20, 30, 255)).save(path)
                lines.append(f'{domain}/c{cls}/{i}.png {cls}')
        listing = tmp_path / f'{domain}.txt'; listing.write_text('\n'.join(lines) + '\n')
        domains[domain] = {'list': str(listing), 'sha256': p.file_sha(listing)}
    # Target is deliberately an inaccessible/non-dataset object, never opened.
    (root / target).symlink_to(tmp_path / 'nonexistent-target')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema': 'pacs_source_manifest_v1', 'target': target, 'domains': domains}))
    return root, manifest


def accept_fixture(root, manifest):
    return p.accept_population(root, manifest, p.file_sha(manifest), 'sketch',
                               expected_counts={d: 10 for d in p.DOMAINS}, classes=2)


def test_population_source_only_exact_order_and_source_validation(tmp_path):
    root, manifest = fixture_population(tmp_path)
    rows, identity = accept_fixture(root, manifest)
    assert list(rows) == list(p.DOMAINS[:3])
    assert all(len(x) == 10 for x in rows.values())
    assert identity['source_validation_ineligible_for_independent_kga_check'] is True
    ds = p.SourceDataset(root, rows['photo'], [0, 1])
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and y == 0
    names = [type(t).__name__ for t in ds.transform.transforms]
    assert names == ['RandomResizedCrop', 'RandomHorizontalFlip', 'ColorJitter',
                     'RandomGrayscale', 'ToTensor', 'Normalize']


@pytest.mark.parametrize('damage', ['order', 'extra_file', 'empty_dir', 'symlink', 'corrupt', 'wrong_hash', 'target_manifest'])
def test_population_rejects_damaged_or_extra_source_input(tmp_path, damage):
    root, manifest = fixture_population(tmp_path)
    value = json.loads(manifest.read_text())
    image = root / 'photo/c0/0.png'
    if damage == 'order':
        listing = Path(value['domains']['photo']['list'])
        listing.write_text('\n'.join(reversed(listing.read_text().splitlines())) + '\n')
        value['domains']['photo']['sha256'] = p.file_sha(listing)
    elif damage == 'extra_file': (root / 'photo/c0/extra.txt').write_text('x')
    elif damage == 'empty_dir': (root / 'photo/c0/nested').mkdir()
    elif damage == 'symlink': image.unlink(); image.symlink_to(root / 'photo/c0/1.png')
    elif damage == 'corrupt': image.write_bytes(b'not an image')
    elif damage == 'wrong_hash': value['domains']['photo']['sha256'] = '0' * 64
    elif damage == 'target_manifest': value['domains']['sketch'] = value['domains']['photo']
    manifest.write_text(json.dumps(value))
    with pytest.raises(p.SourceError): accept_fixture(root, manifest)


def test_appledouble_explicit_receipt_and_orphan_rejected(tmp_path):
    import struct
    root, manifest = fixture_population(tmp_path)
    data = struct.pack('>II16sHIII', 0x51607, 0x20000, bytes(16), 1, 9, 38, 1) + b'x'
    sidecar = root / 'photo/c0/._0.png'; sidecar.write_bytes(data)
    _, identity = accept_fixture(root, manifest)
    assert identity['domains']['photo']['metadata_sidecars'][0]['sha256'] == p.sha(data)
    (root / 'photo/c0/._orphan.png').write_bytes(data)
    with pytest.raises(p.SourceError): accept_fixture(root, manifest)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__(); self.bn = nn.BatchNorm2d(3); self.fc = nn.Linear(3, 1000)
    def forward(self, x): return self.fc(self.bn(x).mean((2, 3)))


def test_model_bn_eval_affine_learns_dropout_and_actual_optimizer():
    model = p.ERM(Backbone(), .5, features=3, classes=2)
    model.train()
    assert not model.featurizer.network.bn.training
    assert model.featurizer.network.bn.weight.requires_grad
    assert model.featurizer.dropout.training
    optimizer = p.optimizer_for(model, p.hparams(0, 0))
    before = model.classifier.weight.detach().clone()
    batches = [(torch.randn(4, 3, 5, 5), torch.tensor([0, 1, 0, 1])) for _ in range(3)]
    result = p.update(model, batches, optimizer, 'cpu')
    assert result['examples'] == 12
    assert not torch.equal(model.classifier.weight, before)
    assert model.featurizer.network.bn.num_batches_tracked.item() == 0
    assert model.featurizer.network.bn.weight.grad is not None
    model.eval(); assert not model.featurizer.dropout.training
    assert optimizer.defaults['lr'] == 5e-5 and optimizer.defaults['weight_decay'] == 0


def test_pinned_cadence_and_selection_ties():
    assert p.selection_steps() == list(range(0, 5001, 300))
    assert len(p.selection_steps()) == 17 and 5000 not in p.selection_steps()
    records = [{'step': 0, 'out': {'a': .7, 'b': .5, 'c': .9}},
               {'step': 300, 'out': {'a': .8, 'b': .4, 'c': .9}}]
    assert p.best_checkpoint(records)['step'] == 0
    assert p.best_cell([{'hparams_seed': 0, 'score': .7}, {'hparams_seed': 19, 'score': .7}])['hparams_seed'] == 19


def test_snapshot_restores_actual_cpu_optimizer_rng_and_sampler(tmp_path):
    torch.manual_seed(9)
    model = p.ERM(Backbone(), .1, features=3, classes=2)
    opt = p.optimizer_for(model, p.hparams(0, 0))
    samplers = {'a': p.ReplaySampler(11, 4, replacement=True, drop_last=True)}
    samplers['a'].next()
    batch = [(torch.randn(4, 3, 5, 5), torch.tensor([0, 1, 0, 1]))] * 3
    p.update(model, batch, opt, 'cpu')
    path = tmp_path / 'state.pt'; identity = {'example': 'toy-not-complete'}
    history = [{'step': 0}]
    p.save_state(path, model, opt, samplers, identity, history, [], {}, 'cpu')
    expected_rng = torch.rand(4); expected_order = samplers['a'].next()
    p.update(model, batch, opt, 'cpu'); expected = copy.deepcopy(model.state_dict())
    restored = p.load_state(path, p.file_sha(path), model, opt, samplers, identity, 'cpu')
    assert restored['history'] == history
    assert torch.equal(torch.rand(4), expected_rng)
    assert samplers['a'].next() == expected_order
    p.update(model, batch, opt, 'cpu')
    assert all(torch.equal(model.state_dict()[k], v) for k, v in expected.items())
    with pytest.raises(p.SourceError): p.load_state(path, '0' * 64, model, opt, samplers, identity, 'cpu')
    with pytest.raises(p.SourceError): p.load_state(path, p.file_sha(path), model, opt, samplers, {}, 'cpu')


@pytest.mark.parametrize('damage', [
    'empty', 'partial', 'extra_parameter', 'string_parameter', 'missing_moment',
    'extra_moment', 'moment_shape', 'moment_dtype', 'moment_nan', 'moment_inf',
    'negative_second_moment', 'fractional_step', 'wrong_step', 'nan_step',
    'infinite_step', 'vector_step', 'boolean_step', 'numeric_step',
    'reordered_parameters', 'duplicate_parameter', 'learning_rate', 'betas',
    'epsilon', 'weight_decay', 'amsgrad', 'maximize', 'extra_group_key',
    'tensor_beta', 'vector_parameter_id',
])
def test_rehashed_checkpoint_rejects_incomplete_or_changed_adam_state(tmp_path, damage):
    """Byte authentication must not substitute for exact Adam-state validation."""
    model = p.ERM(Backbone(), .1, features=3, classes=2)
    optimizer = p.optimizer_for(model, p.hparams(0, 0))
    batches = [(torch.randn(4, 3, 5, 5), torch.tensor([0, 1, 0, 1]))] * 3
    p.update(model, batches, optimizer, 'cpu')
    path = tmp_path / 'valid.pt'; identity = {'example': 'toy-not-complete'}
    p.save_state(path, model, optimizer, {}, identity, [{'step': 0}], [], {}, 'cpu')
    state = p.authenticated_torch_load(path, p.file_sha(path))
    adam = state['optimizer']; first = next(iter(adam['state']))
    moment = adam['state'][first]; group = adam['param_groups'][0]
    if damage == 'empty': adam['state'] = {}
    elif damage == 'partial': del adam['state'][first]
    elif damage == 'extra_parameter': adam['state'][999] = copy.deepcopy(moment)
    elif damage == 'string_parameter': adam['state'][str(first)] = adam['state'].pop(first)
    elif damage == 'missing_moment': del moment['exp_avg']
    elif damage == 'extra_moment': moment['max_exp_avg_sq'] = moment['exp_avg_sq'].clone()
    elif damage == 'moment_shape': moment['exp_avg'] = torch.zeros(1)
    elif damage == 'moment_dtype': moment['exp_avg'] = moment['exp_avg'].double()
    elif damage == 'moment_nan': moment['exp_avg'].fill_(float('nan'))
    elif damage == 'moment_inf': moment['exp_avg_sq'].fill_(float('inf'))
    elif damage == 'negative_second_moment': moment['exp_avg_sq'].fill_(-1.)
    elif damage == 'fractional_step': moment['step'].fill_(1.5)
    elif damage == 'wrong_step': moment['step'].fill_(2.)
    elif damage == 'nan_step': moment['step'].fill_(float('nan'))
    elif damage == 'infinite_step': moment['step'].fill_(float('inf'))
    elif damage == 'vector_step': moment['step'] = torch.tensor([1.])
    elif damage == 'boolean_step': moment['step'] = torch.tensor(True)
    elif damage == 'numeric_step': moment['step'] = 1
    elif damage == 'reordered_parameters': group['params'].reverse()
    elif damage == 'duplicate_parameter': group['params'][1] = group['params'][0]
    elif damage == 'learning_rate': group['lr'] = .001
    elif damage == 'betas': group['betas'] = (.8, .999)
    elif damage == 'epsilon': group['eps'] = 1e-7
    elif damage == 'weight_decay': group['weight_decay'] = .001
    elif damage == 'amsgrad': group['amsgrad'] = True
    elif damage == 'maximize': group['maximize'] = True
    elif damage == 'extra_group_key': group['unexpected'] = True
    elif damage == 'tensor_beta': group['betas'] = (torch.tensor(.9), .999)
    elif damage == 'vector_parameter_id': group['params'][0] = torch.tensor([0, 0])
    damaged = tmp_path / 'damaged.pt'
    with damaged.open('xb') as f: torch.save(state, f)
    assert p.file_sha(damaged) != p.file_sha(path)
    # Recompute the hash: this must fail on semantics, not byte authentication.
    with pytest.raises(p.SourceError, match='Adam|optimizer'):
        p.load_state(damaged, p.file_sha(damaged), model, optimizer, {}, identity, 'cpu')


def test_plan_fresh_output_and_cli_forbids_recipe_reduction(tmp_path):
    out = tmp_path / 'plan'
    assert p.main(['plan', '--output-dir', str(out)]) == 0
    assert len(p.read_json(out / 'plan.json')['jobs']) == 240
    snapshot = p.read_bytes(out / 'plan.json')
    assert p.main(['plan', '--output-dir', str(out)]) != 0
    assert p.read_bytes(out / 'plan.json') == snapshot
    with pytest.raises(SystemExit): p.parser().parse_args(['train', '--steps', '1'])


def test_reference_completion_rejects_toy_partial_and_wrong_geometry():
    assert not p.reference_complete([], [], {}, 32)
    assert not p.reference_complete([{'step': 0, 'examples': 96}], [], {}, 32)


def test_selector_rejects_incomplete_grid_without_touching_output(tmp_path):
    manifest = tmp_path / 'runs.json'; manifest.write_text('[]')
    out = tmp_path / 'selected'
    assert p.main(['select', '--runs-manifest', str(manifest), '--runs-sha256', p.file_sha(manifest),
                   '--target', 'sketch', '--trial-seed', '0', '--output-dir', str(out)]) != 0
    assert not out.exists()


def test_real_v1_model_constructor_and_bn_cpu():
    path = Path('/Volumes/T9/kbound-full-nine-campaign.IfCs0L/resnet50-v1-initialization-001.pth')
    if not path.exists(): pytest.skip('authenticated V1 asset not supplied')
    model = p.load_model(path, .1)
    assert model.classifier.weight.shape == (7, 2048)
    assert isinstance(model.featurizer.network.fc, nn.Identity)
    model.train()
    assert all(not m.training and m.weight.requires_grad for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    assert 'network.0.network.conv1.weight' in model.state_dict()


def test_execute_failure_resume_exact_next_batch_and_preservation(tmp_path, monkeypatch):
    root, manifest = fixture_population(tmp_path)
    original_accept = p.accept_population
    monkeypatch.setattr(p, 'accept_population', lambda *a: original_accept(*a,
                        expected_counts={d: 10 for d in p.DOMAINS}, classes=2))
    monkeypatch.setattr(p, 'load_model', lambda *a: p.ERM(Backbone(), .1, features=3, classes=2))
    out = tmp_path / 'run'
    argv = ['train', '--data-root', str(root), '--source-manifest', str(manifest),
            '--source-sha256', p.file_sha(manifest), '--initialization', str(tmp_path / 'toy-init'),
            '--output-dir', str(out), '--target', 'sketch', '--hparams-seed', '19',
            '--trial-seed', '2', '--device', 'cpu']
    original_update, observed = p.update, []
    def stop_second(model, batches, optimizer, device):
        observed.append([(x.clone(), y.clone()) for x, y in batches])
        if len(observed) == 2: raise RuntimeError('test interruption before second update')
        return original_update(model, batches, optimizer, device)
    monkeypatch.setattr(p, 'update', stop_second)
    assert p.main(argv) == 1
    assert not (out / 'completion.json').exists()
    assert len(list(out.glob('failure-*.json'))) == 1
    pointer = p.read_bytes(out / 'resume.json')
    pending_batch = observed[-1]
    def stop_first(model, batches, optimizer, device):
        assert all(torch.equal(x, ex) and torch.equal(y, ey)
                   for (x, y), (ex, ey) in zip(batches, pending_batch))
        raise RuntimeError('test interruption at replayed second update')
    monkeypatch.setattr(p, 'update', stop_first)
    assert p.main(argv + ['--resume']) == 1
    assert len(list(out.glob('failure-*.json'))) == 2
    assert p.read_bytes(out / 'resume.json') == pointer
    assert p.main(argv) == 1  # Never implicitly resume/overwrite an existing run.
    assert p.read_bytes(out / 'resume.json') == pointer
    (out / 'eligible-source.json').write_text('{}')
    assert p.main(argv + ['--resume']) == 1
    failure = sorted(out.glob('failure-*.json'), key=lambda x: x.stat().st_mtime_ns)[-1]
    assert 'eligible source' in p.read_json(failure)['message']


def test_execute_resume_records_failure_for_rehashed_empty_adam_state(tmp_path, monkeypatch):
    root, manifest = fixture_population(tmp_path)
    original_accept = p.accept_population
    monkeypatch.setattr(p, 'accept_population', lambda *a: original_accept(*a,
                        expected_counts={d: 10 for d in p.DOMAINS}, classes=2))
    monkeypatch.setattr(p, 'load_model', lambda *a: p.ERM(Backbone(), .1, features=3, classes=2))
    out = tmp_path / 'run'
    argv = ['train', '--data-root', str(root), '--source-manifest', str(manifest),
            '--source-sha256', p.file_sha(manifest), '--initialization', str(tmp_path / 'toy-init'),
            '--output-dir', str(out), '--target', 'sketch', '--hparams-seed', '19',
            '--trial-seed', '2', '--device', 'cpu']
    original_update, calls = p.update, []
    def stop_after_one(model, batches, optimizer, device):
        calls.append(1)
        if len(calls) > 1: raise RuntimeError('controlled test interruption')
        return original_update(model, batches, optimizer, device)
    monkeypatch.setattr(p, 'update', stop_after_one)
    assert p.main(argv) == 1
    journal = p.read_json(out / 'resume.json')
    original = out / journal['checkpoint']; original_bytes = p.read_bytes(original)
    state = p.authenticated_torch_load(original, journal['sha256'])
    state['optimizer']['state'] = {}
    damaged = out / 'rehashed-empty-adam.pt'
    with damaged.open('xb') as f: torch.save(state, f)
    journal['checkpoint'], journal['sha256'] = damaged.name, p.file_sha(damaged)
    journal['checkpoints'][0] = {k: journal[k] for k in ('checkpoint', 'sha256', 'step')}
    p.write_json(out / 'resume.json', journal, replace=True)
    before_failures = set(out.glob('failure-*.json'))
    assert p.main(argv + ['--resume']) == 1
    new_failures = set(out.glob('failure-*.json')) - before_failures
    assert len(new_failures) == 1
    failure = p.read_json(new_failures.pop())
    assert failure['error'] == 'SourceError'
    assert 'Adam' in failure['message'] or 'optimizer' in failure['message']
    assert failure['source_training_complete'] is False
    assert p.read_bytes(original) == original_bytes
    assert p.read_json(out / 'resume.json') == journal
    assert not (out / 'completion.json').exists()


def test_resume_sampler_geometry_cannot_skip_or_rewind_cycles():
    memberships = {'a': {'in': list(range(11)), 'out': list(range(3))}}
    samplers = {'train/a': p.ReplaySampler(11, 4, replacement=True, drop_last=True),
                'a/in': p.ReplaySampler(11, 64, replacement=False, drop_last=False),
                'a/out': p.ReplaySampler(3, 64, replacement=False, drop_last=False)}
    for _ in range(3): samplers['train/a'].next()
    samplers['a/in'].next(); samplers['a/out'].next()
    p.validate_sampler_progress(samplers, memberships, 3, 1, 4)
    samplers['train/a'].next()
    with pytest.raises(p.SourceError): p.validate_sampler_progress(samplers, memberships, 3, 1, 4)


def test_full_completion_gate_checks_actual_pacs_membership_not_only_steps():
    domains = p.DOMAINS[:3]
    history = [dict(step=i, examples=96, loss=.7,
                    sampled_source_in_indices={d: [0] * 32 for d in domains}) for i in range(5001)]
    evaluations = [dict(step=i, **{'in': {d: .5 for d in domains}, 'out': {d: .5 for d in domains}})
                   for i in p.selection_steps()]
    memberships = {d: p.partition(p.COUNTS[d], 0, p.DOMAINS.index(d)) for d in domains}
    visits = {d: [5001 * 32] + [0] * (len(memberships[d]['in']) - 1) for d in domains}
    # Synthetic accounting exercises the gate only; this is not a training result.
    assert p.reference_complete(history, evaluations, visits, 32, memberships)
    bad = copy.deepcopy(memberships); bad['photo']['in'][0] = bad['photo']['out'][0]
    assert not p.reference_complete(history, evaluations, visits, 32, bad)
    bad_history = copy.deepcopy(history); bad_history[-1]['examples'] = 95
    assert not p.reference_complete(bad_history, evaluations, visits, 32, memberships)
    bad_visits = copy.deepcopy(visits); bad_visits['photo'][0] -= 1; bad_visits['photo'][1] += 1
    assert not p.reference_complete(history, evaluations, bad_visits, 32, memberships)


def test_checkpoint_provenance_rejects_changed_prefix_history(tmp_path):
    model = p.ERM(Backbone(), 0, features=3, classes=2)
    optimizer = p.optimizer_for(model, p.hparams(0, 0))
    path = tmp_path / 'state.pt'; identity = {'toy': True}
    history, evaluations = [{'step': 0, 'loss': .5}], [{'step': 0}]
    p.save_state(path, model, optimizer, {}, identity, history, evaluations, {}, 'cpu')
    pointer = dict(step=0, checkpoint=path.name, sha256=p.file_sha(path))
    p.verify_checkpoints(tmp_path, [pointer], evaluations, identity, history)
    with pytest.raises(p.SourceError):
        p.verify_checkpoints(tmp_path, [pointer], evaluations, identity, [{'step': 0, 'loss': .9}])


def test_selector_policy_identity_rejects_other_revision_or_membership(tmp_path):
    from argparse import Namespace
    domains = p.DOMAINS[:3]
    population = {'domains': {d: {'count': p.COUNTS[d],
        'class_mapping': {f'c{i}': i for i in range(7)}} for d in domains}}
    memberships = {d: p.partition(p.COUNTS[d], 0, p.DOMAINS.index(d)) for d in domains}
    args = Namespace(target='sketch', hparams_seed=0, trial_seed=0,
                     initialization=str(tmp_path / 'init.pth'), device='cpu')
    identity = p.configuration(args, population, memberships)
    p.validate_reference_identity(identity)
    for field, value in [('reference_revision', 'wrong'), ('workers', 8), ('steps', 3)]:
        bad = copy.deepcopy(identity); bad[field] = value
        with pytest.raises(p.SourceError): p.validate_reference_identity(bad)
    bad = copy.deepcopy(identity); bad['memberships']['photo']['in'].reverse()
    with pytest.raises(p.SourceError): p.validate_reference_identity(bad)
