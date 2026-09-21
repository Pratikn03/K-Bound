"""Synthetic adapter checks. An explicit native checkout enables equivalence tests.

No released checkpoint or dataset is loaded. Tiny networks are test fixtures,
never substitutes for the production full ResNet-50 factory.
"""
import importlib
import copy
import ast
import logging
import math
import random
import time
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image, ImageFilter

SCRIPTS = Path(__file__).resolve().parents[1] / 'docs/research/kbound/scripts'
sys.path.insert(0, str(SCRIPTS))


def adapter():
    return importlib.import_module('domainnet_reference_adapter')


@pytest.fixture
def reference_root():
    value = os.environ.get('KBOUND_ADACONTRAST_REFERENCE')
    if value is None:
        pytest.skip('explicit authenticated native source checkout not supplied')
    return Path(value)


class TinyClassifier(nn.Module):
    output_dim = 5
    num_classes = 3

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 5), nn.BatchNorm1d(5))
        self.fc = nn.Linear(5, 3)

    def forward(self, x, return_feats=False):
        feats = self.encoder(x)
        logits = self.fc(feats)
        return (feats, logits) if return_feats else logits

    def get_params(self):
        return list(self.encoder.parameters()), list(self.fc.parameters())


def tiny_args():
    return SimpleNamespace(distributed=False,
        optim=SimpleNamespace(name='sgd', lr=2e-4, momentum=.9, weight_decay=1e-4,
                              nesterov=True, cos=True, exp=False),
        data=SimpleNamespace(aug_type='moco-v2'),
        learn=SimpleNamespace(alpha=1., beta=1., eta=1., aug_versions='wss',
            dist_type='cosine', num_neighbors=3, refine_method='nearest_neighbors',
            ce_sup_type='weak_strong', ce_type='standard', contrast_type='class_aware',
            epochs=2, full_progress=2, queue_size=-1, print_freq=1000),
        model_tta=SimpleNamespace(queue_size=16, m=.999, T_moco=.07))


def batches():
    generator = torch.Generator().manual_seed(99)
    return [([torch.randn(4, 8, generator=generator) for _ in range(3)], torch.arange(4))]


def test_maintained_adapter_interface_exists():
    module = adapter()
    assert callable(module.build_session)
    assert callable(module.load_reference)


def test_streaming_yield_update_interleaving():
    module = adapter()
    trace = []
    class Source:
        def __len__(self):
            return 2
        def __iter__(self):
            for i in range(2):
                trace.append('yield-' + str(i))
                yield batches()[0]
    stream = module.StreamingEpoch(Source(), bank_size=6, queue_size=16)
    assert len(stream) == 2
    for i, (_, labels, _) in enumerate(stream):
        assert labels is None
        trace.append('update-' + str(i))
    assert trace == ['yield-0', 'update-0', 'yield-1', 'update-1']


@pytest.mark.parametrize('fault', ['views', 'singleton', 'shape', 'nan', 'indices', 'capacity'])
def test_streaming_rejects_invalid_batch_before_update(fault):
    module = adapter()
    views, indices = batches()[0]
    if fault == 'views': views = views[:2]
    if fault == 'singleton':
        views, indices = [v[:1] for v in views], indices[:1]
    if fault == 'shape': views[1] = views[1][:3]
    if fault == 'nan': views[0][0, 0] = float('nan')
    if fault == 'indices': indices = indices.float()
    with pytest.raises(ValueError):
        list(module.StreamingEpoch([(views, indices)], bank_size=3 if fault == 'capacity' else 6,
                                   queue_size=16))


@pytest.mark.parametrize('device', ['cuda', 'bogus', 'cpu:1'])
def test_invalid_device_rejected_before_checkpoint_access(device, monkeypatch):
    module = adapter()
    def forbidden(*args, **kwargs):
        pytest.fail('invalid device opened a checkpoint')
    monkeypatch.setattr(module, 'load_clipart2020', forbidden)
    with pytest.raises(ValueError):
        module.build_session(Path('not-a-checkpoint'), {}, device)


def test_reference_recipe_includes_authenticated_shell_override(reference_root):
    module = adapter()
    args, identity = module.reference_args(reference_root, epochs=15, steps_per_epoch=2)
    assert args.optim.lr == 2e-4
    assert args.optim.weight_decay == 1e-4
    assert args.learn.full_progress == 30
    assert args.model_tta.queue_size == 16384
    assert identity['source_hashes']['train_domainnet-126_target.sh']
    assert identity['source_hashes']['configs/optim/sgd.yaml']


def test_recipe_tampering_rejected(reference_root, tmp_path):
    module = adapter()
    for name in module.RECIPE_HASHES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((reference_root / name).read_bytes())
    (tmp_path / 'train_domainnet-126_target.sh').write_text('optim.lr=99\n')
    with pytest.raises(ValueError, match='integrity'):
        module.reference_args(tmp_path, epochs=15, steps_per_epoch=2)


def test_native_step_and_reset_preserve_full_state(reference_root):
    module = adapter()
    session = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    initial = {k: v.clone() for k, v in session.model.state_dict().items()}
    session.initialize_bank([torch.randn(6, 8)])
    session.train_epoch(batches(), 0)
    assert session.model.queue_ptr == 4 and session.banks['ptr'] == 4
    assert session.optimizer.state and session.completed_steps == 1
    assert not torch.equal(initial['src_model.fc.weight'], session.model.src_model.fc.weight)
    session.reset()
    assert session.banks is None and session.completed_steps == 0
    assert not session.optimizer.state and session.model.queue_ptr == 0
    for name, value in initial.items():
        torch.testing.assert_close(session.model.state_dict()[name], value, rtol=0, atol=0)


def test_epoch_boundary_resume_matches_uninterrupted_state_and_rng(reference_root):
    module = adapter()
    session = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    session.initialize_bank([torch.randn(6, 8)])
    session.train_epoch(batches(), 0)
    saved = session.snapshot_state()
    session.train_epoch(batches(), 1)
    expected = session.snapshot_state()
    resumed = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    resumed.restore_state(saved)
    # Saved buffers must not alias either live session.
    resumed.train_epoch(batches(), 1)
    actual = resumed.snapshot_state()
    assert actual['digest'] == expected['digest']
    assert saved['completed_steps'] == 1 and actual['completed_steps'] == 2
    assert actual['queue_ptr'] == 8 and actual['banks']['ptr'] == 2


@pytest.mark.parametrize('fault', ['tensor', 'optimizer', 'identity', 'rng', 'missing'])
def test_corrupt_snapshot_rejected_without_changing_live_state(reference_root, fault):
    module = adapter()
    session = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    session.initialize_bank([torch.randn(6, 8)])
    session.train_epoch(batches(), 0)
    before = session.snapshot_state()
    bad = copy.deepcopy(before)
    if fault == 'tensor': bad['model']['src_model.fc.weight'][0, 0] = float('nan')
    if fault == 'optimizer': bad['optimizer']['param_groups'][0]['lr'] = 10.
    if fault == 'identity': bad['identity']['device'] = 'mps'
    if fault == 'rng': bad['rng']['torch'] = torch.zeros(4, dtype=torch.uint8)
    if fault == 'missing': del bad['model']
    with pytest.raises(ValueError): session.restore_state(bad)
    assert session.snapshot_state()['digest'] == before['digest']


def test_different_config_snapshot_rejected(reference_root):
    module = adapter()
    left = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    saved = left.snapshot_state()
    args = tiny_args()
    args.optim.lr *= 2
    right = module.ReferenceSession(TinyClassifier, args, reference_root=reference_root)
    before = right.snapshot_state()
    with pytest.raises(ValueError, match='identity'): right.restore_state(saved)
    assert right.snapshot_state()['digest'] == before['digest']


def test_partial_epoch_failure_cannot_be_saved_or_used_for_prediction(reference_root):
    module = adapter()
    args = tiny_args()
    args.learn.full_progress = 4  # two batches per epoch
    session = module.ReferenceSession(TinyClassifier, args, reference_root=reference_root)
    session.initialize_bank([torch.randn(6, 8)])
    invalid = (batches()[0][0][:2], torch.arange(4))
    with pytest.raises(ValueError): session.train_epoch([batches()[0], invalid], 0)
    with pytest.raises(ValueError, match='failed'): session.snapshot_state()
    with pytest.raises(ValueError, match='failed'): session.predict([torch.randn(2, 8)])


def test_resume_refreshes_weightnorm_derived_weights_without_forward(reference_root):
    module = adapter()
    class WeightNormTiny(TinyClassifier):
        def __init__(self):
            super().__init__()
            with pytest.warns(FutureWarning, match="weight_norm.*deprecated"):
                self.fc = torch.nn.utils.weight_norm(self.fc, dim=0)
    session = module.ReferenceSession(WeightNormTiny, tiny_args(), reference_root=reference_root)
    session.initialize_bank([torch.randn(6, 8)])
    session.train_epoch(batches(), 0)
    saved = session.snapshot_state()
    restored = module.ReferenceSession(WeightNormTiny, tiny_args(), reference_root=reference_root)
    restored.restore_state(saved)
    for model in (restored.model.src_model, restored.model.momentum_model):
        expected = torch._weight_norm(model.fc.weight_v, model.fc.weight_g, 0)
        torch.testing.assert_close(model.fc.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize('fault', ['tensor', 'momentum', 'bank', 'rng', 'mode'])
def test_resigned_malformed_state_is_validated_before_live_swap(reference_root, fault):
    module = adapter()
    session = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    session.initialize_bank([torch.randn(6, 8)])
    session.train_epoch(batches(), 0)
    before = session.snapshot_state()
    bad = copy.deepcopy(before)
    if fault == 'tensor': bad['model']['src_model.fc.weight'][0, 0] = float('nan')
    if fault == 'momentum': bad['optimizer']['state'][0]['momentum_buffer'] = torch.ones(9)
    if fault == 'bank': bad['banks']['probs'][0, 0] = -1.
    if fault == 'rng': bad['rng']['torch'] = torch.zeros(4, dtype=torch.uint8)
    if fault == 'mode': bad['training']['src_model'] = 'yes'
    bad['digest'] = module._state_digest({k: v for k, v in bad.items() if k != 'digest'})
    with pytest.raises(ValueError): session.restore_state(bad)
    assert session.snapshot_state()['digest'] == before['digest']


def raw_oracle(root):
    module = adapter()
    ns = dict(torch=torch, nn=nn, F=F, math=math, time=time, logging=logging,
              random=random, Image=Image, ImageFilter=ImageFilter,
              transforms=transforms, concat_all_gather=lambda x: x,
              use_wandb=lambda args: False)
    definitions = {
        'moco/loader.py': {'GaussianBlur', 'NCropsTransform'},
        'utils.py': {'get_augmentation', 'get_distances', 'adjust_learning_rate', 'AverageMeter', 'ProgressMeter'},
        'target.py': {'soft_k_nearest_neighbors', 'update_labels', 'refine_predictions',
            'get_augmentation_versions', 'get_target_optimizer', 'train_epoch',
            'calculate_acc', 'instance_loss', 'classification_loss', 'div',
            'diversification_loss', 'cross_entropy_loss'},
        'moco/builder.py': {'AdaMoCo'},
    }
    for filename, names in definitions.items():
        raw = module._verified_source(root, filename, module.SOURCE_HASHES[filename])
        tree = ast.parse(raw)
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert {n.name for n in tree.body} == names
        exec(compile(tree, filename, 'exec'), ns)
    return SimpleNamespace(**ns)


@pytest.mark.parametrize('supervision', ['weak_strong', 'weak_weak'])
@pytest.mark.parametrize('steps', [1, 2])
def test_two_steps_and_loss_components_match_raw_native_definitions(reference_root, monkeypatch, supervision, steps):
    module = adapter()
    oracle = raw_oracle(reference_root)
    args = tiny_args()
    args.learn.full_progress = 2 * steps
    args.learn.ce_sup_type = supervision
    portable = module.ReferenceSession(TinyClassifier, args, reference_root=reference_root)
    portable.initialize_bank([torch.randn(6, 8)])
    official = oracle.AdaMoCo(TinyClassifier(), TinyClassifier(), K=16, m=.999, T_moco=.07)
    official.load_state_dict(portable.model.state_dict())
    banks = copy.deepcopy(portable.banks)
    optimizer = oracle.get_target_optimizer(official, args)
    losses = {'portable': [], 'official': []}
    native_meters = []
    native_meter = oracle.train_epoch.__globals__['AverageMeter']

    class ObservedNativeMeter(native_meter):
        def __init__(self, name, fmt=':f'):
            super().__init__(name, fmt)
            if name == 'Loss':
                native_meters.append(self)

    monkeypatch.setitem(oracle.train_epoch.__globals__, 'AverageMeter', ObservedNativeMeter)
    for label, function in [('portable', portable.ref.train_epoch), ('official', oracle.train_epoch)]:
        for name in ('instance_loss', 'classification_loss', 'diversification_loss'):
            original = function.__globals__[name]
            def capture(*a, _f=original, _name=name, _label=label, **kw):
                value = _f(*a, **kw)
                scalar = value[0] if isinstance(value, tuple) else value
                losses[_label].append((_name, scalar.detach().clone()))
                return value
            monkeypatch.setitem(function.__globals__, name, capture)
    original_to = torch.Tensor.to
    def cpu_to(self, *values, **kwargs):
        if values and isinstance(values[0], str) and values[0] == 'cuda':
            values = ('cpu',) + values[1:]
        return original_to(self, *values, **kwargs)
    # Allocation/one-process communication shims only in the raw test oracle.
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    monkeypatch.setattr(torch.distributed, 'broadcast', lambda tensor, src: None)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    for epoch in range(2):
        random.seed(101 + epoch); torch.manual_seed(101 + epoch)
        oracle.train_epoch([(views, None, ids) for views, ids in batches() * steps], official, banks, optimizer, epoch, args)
        native_rng = torch.get_rng_state().clone()
        native_python_rng = random.getstate()
        random.seed(101 + epoch); torch.manual_seed(101 + epoch)
        summary = portable.train_epoch(batches() * steps, epoch)
        meter = native_meters[-1]
        assert summary == {'count': steps, 'sum': meter.sum, 'mean': meter.avg, 'last': meter.val}
        assert torch.equal(torch.get_rng_state(), native_rng)
        assert random.getstate() == native_python_rng
        for name, tensor in official.state_dict().items():
            torch.testing.assert_close(portable.model.state_dict()[name], tensor, rtol=0, atol=0)
        assert module._state_digest(portable.banks) == module._state_digest(banks)
        assert module._state_digest(portable.optimizer.state_dict()) == module._state_digest(optimizer.state_dict())
        assert portable.model.queue_ptr == official.queue_ptr
    assert len(losses['official']) == 6 * steps
    for (name, value), (expected_name, expected) in zip(losses['portable'], losses['official']):
        assert name == expected_name
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_native_schedule_rejects_extra_epochs_and_wrong_epoch_length(reference_root):
    module = adapter()
    session = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    session.initialize_bank([torch.randn(6, 8)])
    before = session.snapshot_state()['digest']
    with pytest.raises(ValueError, match='schedule'):
        session.train_epoch(batches() * 2, 0)
    assert session.snapshot_state()['digest'] == before
    session.train_epoch(batches(), 0)
    session.train_epoch(batches(), 1)
    before = session.snapshot_state()['digest']
    with pytest.raises(ValueError, match='schedule'):
        session.train_epoch(batches(), 2)
    assert session.snapshot_state()['digest'] == before


@pytest.mark.parametrize('operation', ['train', 'bank', 'predict', 'reset'])
def test_changed_configuration_rejected_before_execution(reference_root, operation):
    module = adapter()
    session = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    if operation != 'bank': session.initialize_bank([torch.randn(6, 8)])
    before = session.snapshot_state()['digest']
    session.args.learn.alpha = 42.
    actions = {'train': lambda: session.train_epoch(batches(), 0),
               'bank': lambda: session.initialize_bank([torch.ones(6, 8)]),
               'predict': lambda: session.predict([torch.ones(2, 8)]),
               'reset': session.reset}
    with pytest.raises(ValueError, match='configuration'): actions[operation]()
    session.args.learn.alpha = 1.
    assert session.snapshot_state()['digest'] == before


@pytest.mark.parametrize('fault', ['nan', 'empty', 'dtype'])
def test_bad_bank_never_becomes_an_usable_candidate(reference_root, fault):
    module = adapter()
    session = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    images = torch.randn(6, 8)
    if fault == 'nan': images[0, 0] = float('nan')
    if fault == 'empty': images = images[:0]
    if fault == 'dtype': images = images.double()
    with pytest.raises(ValueError): session.initialize_bank([images])
    assert session.banks is None
    with pytest.raises(ValueError): session.train_epoch(batches(), 0)


def test_trained_resume_requires_all_sgd_momentum_slots(reference_root):
    module = adapter()
    session = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    session.initialize_bank([torch.randn(6, 8)])
    session.train_epoch(batches(), 0)
    before = session.snapshot_state()
    bad = copy.deepcopy(before)
    bad['optimizer']['state'] = {}
    bad['digest'] = module._state_digest({k: v for k, v in bad.items() if k != 'digest'})
    with pytest.raises(ValueError, match='momentum'): session.restore_state(bad)
    assert session.snapshot_state()['digest'] == before['digest']


def test_initialized_bank_cannot_be_replaced_without_reset(reference_root):
    module = adapter()
    session = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    session.initialize_bank([torch.ones(6, 8)])
    before = session.snapshot_state()['digest']
    with pytest.raises(ValueError, match='reset'): session.initialize_bank([torch.zeros(6, 8)])
    assert session.snapshot_state()['digest'] == before


def test_every_declared_native_source_is_authenticated(reference_root, tmp_path):
    module = adapter()
    for name in module.SOURCE_HASHES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((reference_root / name).read_bytes())
    (tmp_path / 'classifier.py').write_text('changed classifier identity\n')
    with pytest.raises(ValueError, match='integrity'): module.load_reference(tmp_path)


def test_prediction_rejects_nonfinite_inputs(reference_root):
    module = adapter()
    session = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    with pytest.raises(ValueError): session.predict([torch.full((2, 8), float('nan'))])


def test_nonfinite_native_update_cannot_be_saved_or_predicted(reference_root, monkeypatch):
    module = adapter()
    session = module.ReferenceSession(TinyClassifier, tiny_args(), reference_root=reference_root)
    session.initialize_bank([torch.randn(6, 8)])
    def broken_update(*args, **kwargs):
        with torch.no_grad(): session.model.src_model.fc.weight[0, 0] = float('nan')
    monkeypatch.setattr(session.ref, 'train_epoch', broken_update)
    with pytest.raises(ValueError, match='nonfinite'): session.train_epoch(batches(), 0)
    assert session.failed
    with pytest.raises(ValueError): session.snapshot_state()
