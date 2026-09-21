"""Offline source contract: no images, released weights or training in unit tests."""
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import subprocess
import textwrap

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'docs/research/kbound/scripts'
sys.path.insert(0, str(SCRIPTS))


def load_module():
    path = SCRIPTS / 'domainnet_reference_source.py'
    assert path.exists(), 'strict offline DomainNet source loader is missing'
    spec = importlib.util.spec_from_file_location('domainnet_reference_source', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_model():
    return torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3))


def payload(state):
    out = io.BytesIO()
    torch.save({'state_dict': state, 'epoch': 60}, out)
    return out.getvalue()


def write_payload(tmp_path, state):
    data = payload(state)
    path = tmp_path / 'source.pt'
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest(), len(data)


def test_verified_buffer_loads_exact_state_not_random_fallback(tmp_path):
    loader = load_module()
    model = small_model()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state['0.weight'].fill_(0.25)
    path, digest, size = write_payload(tmp_path, state)
    raw = loader.read_verified_checkpoint(path, digest, size)
    loader.apply_strict_state(model, raw)
    assert torch.equal(model[0].weight, torch.full((3, 2), 0.25))


@pytest.mark.parametrize('fault', ['hash', 'size', 'empty'])
def test_invalid_identity_rejected_without_deserialization(tmp_path, monkeypatch, fault):
    loader = load_module()
    path, digest, size = write_payload(tmp_path, small_model().state_dict())
    if fault == 'hash': digest = '0' * 64
    if fault == 'size': size += 1
    if fault == 'empty': path.write_bytes(b'')
    def forbidden(*args, **kwargs):
        pytest.fail('invalid checkpoint reached torch deserialization')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError):
        loader.read_verified_checkpoint(path, digest, size)


@pytest.mark.parametrize('kind', ['leaf', 'parent', 'directory', 'fifo'])
def test_nonregular_or_symlink_inputs_rejected(tmp_path, kind):
    loader = load_module()
    path, digest, size = write_payload(tmp_path, small_model().state_dict())
    if kind == 'leaf':
        alias = tmp_path / 'alias.pt'; alias.symlink_to(path); path = alias
    elif kind == 'parent':
        alias = tmp_path / 'alias'; alias.symlink_to(tmp_path, target_is_directory=True)
        path = alias / path.name
    elif kind == 'directory': path = tmp_path
    else:
        import os
        path = tmp_path / 'pipe'; os.mkfifo(path)
    with pytest.raises((ValueError, OSError)):
        loader.read_verified_checkpoint(path, digest, size)


@pytest.mark.parametrize('fault', ['missing', 'extra', 'shape', 'dtype', 'nan',
                                  'bn_counter', 'negative_counter', 'negative_variance',
                                  'mixed_prefix', 'double_prefix', 'nontensor'])
def test_malformed_state_fails_before_any_parameter_changes(fault):
    loader = load_module()
    model = small_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = {k: v.clone() for k, v in before.items()}
    state['0.bias'].fill_(9.0)
    if fault == 'missing': del state['0.weight']
    if fault == 'extra': state['extra'] = torch.zeros(1)
    if fault == 'shape': state['0.weight'] = torch.zeros(2, 3)
    if fault == 'dtype': state['0.weight'] = state['0.weight'].double()
    if fault == 'nan': state['0.weight'][0, 0] = float('nan')
    if fault == 'bn_counter': del state['1.num_batches_tracked']
    if fault == 'negative_counter': state['1.num_batches_tracked'].fill_(-1)
    if fault == 'negative_variance': state['1.running_var'][0] = -1
    if fault == 'mixed_prefix': state['module.0.weight'] = state.pop('0.weight')
    if fault == 'double_prefix': state = {'module.module.' + k: v for k, v in state.items()}
    if fault == 'nontensor': state['0.weight'] = 'not tensor'
    with pytest.raises(ValueError): loader.apply_strict_state(model, payload(state))
    for k, v in model.state_dict().items(): assert torch.equal(v, before[k]), k


def test_uniform_native_ddp_prefix_is_accepted():
    loader = load_module()
    model = small_model()
    state = {'module.' + k: v.clone() for k, v in model.state_dict().items()}
    state['module.0.bias'].fill_(3)
    result = loader.apply_strict_state(model, payload(state))
    assert result['prefix_removed'] == 'module.'
    assert torch.equal(model[0].bias, torch.full((3,), 3.0))


def test_zero_weight_norm_direction_rejected_before_mutation():
    loader = load_module()
    model = torch.nn.Sequential(torch.nn.utils.weight_norm(torch.nn.Linear(2, 3), dim=0))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = {k: v.clone() for k, v in before.items()}
    state['0.weight_v'][1].zero_()
    with pytest.raises(ValueError): loader.apply_strict_state(model, payload(state))
    for k, v in model.state_dict().items(): assert torch.equal(v, before[k])


def test_weight_norm_cached_effective_weight_is_refreshed_without_forward():
    loader = load_module()
    model = torch.nn.Sequential(torch.nn.utils.weight_norm(torch.nn.Linear(2, 3), dim=0))
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state['0.weight_v'] = torch.tensor([[3., 4.], [0., 2.], [1., 0.]])
    state['0.weight_g'] = torch.tensor([[10.], [6.], [4.]])
    loader.apply_strict_state(model, payload(state))
    assert torch.allclose(model[0].weight, torch.tensor([[6., 8.], [0., 6.], [4., 0.]]))


def test_full_reference_factory_is_cpu_fp32_and_preserves_rng_and_default_dtype():
    loader = load_module()
    old_dtype = torch.get_default_dtype()
    torch.manual_seed(728)
    rng = torch.get_rng_state().clone()
    try:
        torch.set_default_dtype(torch.float64)
        model = loader.build_reference_classifier()
        assert torch.get_default_dtype() == torch.float64
        assert torch.equal(torch.get_rng_state(), rng)
        assert model.encoder[0].layer3.__len__() == 6
        assert model.encoder[0].fc.weight.shape == (256, 2048)
        assert model.fc.weight_v.shape == (126, 256)
        assert len(model.state_dict()) == 328
        assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters())
        assert not any(m.training for m in model.modules())
    finally: torch.set_default_dtype(old_dtype)


def test_factory_restores_rng_and_dtype_on_constructor_failure(monkeypatch):
    loader = load_module()
    import torchvision.models
    rng = torch.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    def broken(*args, **kwargs):
        assert kwargs.get('weights') is None
        torch.rand(8)
        raise RuntimeError('synthetic constructor failure')
    monkeypatch.setattr(torchvision.models, 'resnet50', broken)
    with pytest.raises(RuntimeError, match='synthetic'):
        loader.build_reference_classifier()
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.get_default_dtype() == dtype


def test_reference_model_exposes_native_optimizer_groups_without_state_changes():
    model = load_module().build_reference_classifier()
    original_keys = tuple(model.state_dict())
    frozen = model.encoder[0].conv1.weight
    frozen.requires_grad_(False)
    assert model.output_dim == 256
    assert model.num_classes == 126
    backbone, extra = model.get_params()
    expected_extra = {id(p) for name, p in model.named_parameters()
                      if name.startswith(('encoder.0.fc.', 'encoder.1.', 'fc.'))
                      and p.requires_grad}
    expected_all = {id(p) for p in model.parameters() if p.requires_grad}
    assert {id(p) for p in extra} == expected_extra
    assert {id(p) for p in backbone} == expected_all - expected_extra
    assert len(backbone) + len(extra) == len(expected_all)
    assert id(frozen) not in {id(p) for p in backbone + extra}
    assert tuple(model.state_dict()) == original_keys


def test_public_official_loader_rejects_fixture_without_model_construction(tmp_path, monkeypatch):
    loader = load_module()
    path, _, _ = write_payload(tmp_path, small_model().state_dict())
    monkeypatch.setattr(loader, 'build_reference_classifier', lambda: pytest.fail('unverified input built a model'))
    with pytest.raises(ValueError): loader.load_clipart2020(path)


@pytest.mark.parametrize('entrypoint', ['factory', 'public_first_import'])
def test_fresh_import_is_cpu_isolated_and_restores_caller_state(entrypoint):
    # A fresh interpreter is essential: already-imported torchvision hides the bug.
    # The public case isolates imports after authentication with deliberately bad
    # state bytes; it neither authenticates a fake release nor accepts a model.
    script = textwrap.dedent('''
        import io, sys, torch
        from importlib.metadata import version
        sys.path.insert(0, sys.argv[1])
        import domainnet_reference_source as loader
        assert 'torchvision' not in sys.modules
        buffer = io.BytesIO()
        torch.save({'state_dict': {'invalid': torch.ones(1)}}, buffer)
        torch.set_default_device('meta')
        torch.set_default_dtype(torch.float64)
        before = torch.get_rng_state().clone()
        try:
            if sys.argv[2] == 'factory':
                model = loader.build_reference_classifier()
                assert all(p.device.type == 'cpu' and p.dtype == torch.float32
                           for p in model.parameters())
                assert not any(m.training for m in model.modules())
            else:
                loader.read_verified_checkpoint = lambda *args: buffer.getvalue()
                try:
                    loader.load_clipart2020('not-a-release')
                except ValueError as error:
                    # The release CI runtime intentionally differs from the
                    # reviewed natural-study runtime. It must fail at that
                    # version guard, not pretend that tensor validation ran.
                    if (torch.__version__, version('torchvision')) == ('2.8.0', '0.23.0'):
                        assert 'exact state keys required' in str(error), str(error)
                    else:
                        assert 'reviewed source loader requires torch2.8.0/torchvision0.23.0' in str(error), str(error)
                else:
                    raise AssertionError('invalid state was accepted')
            assert torch.get_default_device().type == 'meta'
            assert torch.get_default_dtype() == torch.float64
            assert torch.equal(before, torch.get_rng_state())
        finally:
            torch.set_default_device('cpu')
            torch.set_default_dtype(torch.float32)
    ''')
    result = subprocess.run([sys.executable, '-c', script, str(SCRIPTS), entrypoint],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
