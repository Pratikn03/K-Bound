"""Contracts for the shared full-BN native POEM/AETTA executor."""
import json
from pathlib import Path
import sys
import types

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'docs/research/kbound/scripts'
sys.path.insert(0, str(SCRIPTS))
import task3_native_bn as bn


def panel():
    def row(n, role):
        prefix = '' if role == 'source' else 'gaussian_noise/5/'
        return dict(relative_path=f'{prefix}n01440764/ILSVRC2012_val_{n:08}.JPEG',
                    sample_id=f'ILSVRC2012_val_{n:08}', sha256='a'*64,
                    size_bytes=50, width=224, height=224, decoded_rgb=True)
    return dict(schema='kbound-native-bn-cell-v1', scope='ENGINEERING_SMOKE_NOT_BENCHMARK',
                clean_root='/clean/val', corruption_root='/corrupt', condition='gaussian_noise/5',
                seed=0, batch_size=4, source=[row(n, 'source') for n in range(1,33)],
                target=[row(n, 'target') for n in range(33,137)])


def test_shared_panel_rejects_reused_clean_target_identity():
    p = panel()
    p['target'][0]['sample_id'] = p['source'][0]['sample_id']
    p['target'][0]['relative_path'] = 'gaussian_noise/5/' + p['source'][0]['relative_path']
    with pytest.raises(ValueError, match='overlap|duplicate'):
        bn.validate_panel(p)


def test_native_boundary_rejects_outcomes_and_unknown_fields():
    p = panel()
    p['target'][0]['correct'] = True
    with pytest.raises(ValueError, match='fields|outcome'):
        bn.validate_panel(p)
    p = panel()
    p['labels'] = [0]*104
    with pytest.raises(ValueError, match='fields|outcome'):
        bn.validate_panel(p)


@pytest.mark.parametrize('change', ['warmup', 'condition', 'bool_seed', 'traversal'])
def test_invalid_cell_never_reaches_model_execution(change):
    p = panel()
    if change == 'warmup': p['target'] = p['target'][:100]
    if change == 'condition': p['condition'] = 'fog/5'
    if change == 'bool_seed': p['seed'] = True
    if change == 'traversal': p['target'][0]['relative_path'] = '../image.JPEG'
    with pytest.raises(ValueError): bn.validate_panel(p)


def test_valid_full_architecture_cell_contract():
    p = panel()
    bn.validate_panel(p)
    recipe = bn.bn_recipe(4)
    assert recipe['temperature'] == 1.0
    assert recipe['lr'] == 0.00003125
    assert recipe['native_warmup_samples'] == 100
    assert recipe['dropout_draws'] == 10


def test_interleaved_estimator_does_not_change_adapter_random_stream():
    torch = pytest.importorskip('torch')
    torch.manual_seed(19)
    expected = torch.rand(4)
    torch.manual_seed(19)
    def noisy():
        return torch.rand(100).sum().item()
    first = bn.isolated_estimate(noisy, seed=70)
    assert torch.equal(torch.rand(4), expected)
    assert first == bn.isolated_estimate(noisy, seed=70)


def test_byte_bound_input_rejects_changed_manifest_before_model_load(tmp_path):
    path = tmp_path/'manifest.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='digest'):
        bn.read_panel(path, 'a'*64)


def test_private_estimator_stream_continues_and_restores_rng_on_error():
    torch = pytest.importorskip('torch')
    class Episode:
        def estimate(self, batch):
            values = torch.rand(4)
            if batch == 'raise': raise ValueError('fixture error')
            return values
    torch.manual_seed(123)
    outer = torch.get_rng_state().clone()
    stream = bn.EstimatorStream(Episode(), 456)
    other = bn.EstimatorStream(Episode(), 789)
    generator = torch.Generator().manual_seed(456)
    assert torch.equal(torch.get_rng_state(), outer)
    assert torch.equal(stream.estimate(None), torch.rand(4, generator=generator))
    other.estimate(None)
    assert torch.equal(stream.estimate(None), torch.rand(4, generator=generator))
    with pytest.raises(ValueError): stream.estimate('raise')
    assert torch.equal(torch.get_rng_state(), outer)


def test_output_path_replacement_cannot_report_success(tmp_path, monkeypatch):
    tmp_path = tmp_path.resolve()
    p = panel()
    manifest = tmp_path/'cell.json'
    manifest.write_text(json.dumps(p))
    digest = bn.images.digest(manifest.read_bytes())
    out = tmp_path/'out'
    moved = tmp_path/'preserved'
    replacement = tmp_path/'different'
    replacement.mkdir()
    def execute(args, panel, record):
        args.output.rename(moved)
        args.output.symlink_to(replacement, target_is_directory=True)
        record['status'] = 'PASS_SHARED_NATIVE_BN_EXECUTION'
    monkeypatch.setattr(bn, 'execute', execute)
    monkeypatch.setenv('POEM_PYTHON', sys.executable)
    monkeypatch.setattr(sys, 'argv', ['bn', '--manifest', str(manifest), '--manifest-sha256', digest,
        '--checkpoint','/weights.pt','--poem-source','/poem','--aetta-source','/aetta',
        '--auth-receipt','/auth.json','--output',str(out)])
    assert bn.main() == 2
    receipt = json.loads((moved/'receipt.json').read_text())
    assert receipt['status'] == 'FAILED_SHARED_NATIVE_BN_EXECUTION'
    assert list(replacement.iterdir()) == []


def test_native_empty_filter_loss_is_not_counted_as_success():
    torch = pytest.importorskip('torch')
    def native_update():
        empty = torch.empty(0, requires_grad=True)
        return torch.ones(4, 1000), torch.nn.functional.mse_loss(empty, empty)
    module = types.SimpleNamespace(forward_and_adapt=native_update)
    traces = []
    bn.observe_updates(module, traces)
    with pytest.raises(ValueError, match='nonfinite native update loss'):
        module.forward_and_adapt()
    assert traces[0]['finite'] is False and traces[0]['loss'] is None


def test_only_classified_empty_selection_can_continue_without_changing_native_update():
    torch = pytest.importorskip('torch')
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.SGD([parameter], lr=.1, momentum=.9)
    def update(x, model, optimizer, **kwargs):
        output = torch.zeros(4,1000)
        empty = output.reshape(-1)[:0] + parameter
        loss = torch.nn.functional.mse_loss(empty, empty)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        return output, loss
    native = types.SimpleNamespace(forward_and_adapt=update)
    traces=[]
    bn.observe_updates(native,traces,allow_empty_selection=True)
    output, loss = native.forward_and_adapt(None,None,optimizer,vanilla_loss=True,e_margin=2.7)
    assert torch.isnan(loss) and torch.isfinite(output).all()
    assert optimizer.state[parameter]['momentum_buffer'].item() == 0.
    assert traces[0]['classification'] == 'empty_reliable_selection_native_nan'
    assert traces[0]['selected_count'] == 0


def test_other_nonfinite_losses_remain_hard_failures():
    torch = pytest.importorskip('torch')
    def update(**kwargs):
        logits = torch.zeros(4,1000); logits[:,0] = 20
        return logits, torch.tensor(float('nan'))
    native=types.SimpleNamespace(forward_and_adapt=update)
    bn.observe_updates(native,[],allow_empty_selection=True)
    with pytest.raises(ValueError,match='nonfinite'):
        native.forward_and_adapt(vanilla_loss=True,e_margin=2.7)


def test_excluded_model_gradients_must_also_remain_finite():
    torch = pytest.importorskip('torch')
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD([model.weight], lr=.1)
    def update(x, model, optimizer, **kwargs):
        model.bias.grad = torch.full_like(model.bias, float('nan'))
        return torch.zeros(4,1000), torch.tensor(0.)
    native = types.SimpleNamespace(forward_and_adapt=update)
    bn.observe_updates(native, [], allow_empty_selection=True)
    with pytest.raises(ValueError, match='gradient'):
        native.forward_and_adapt(None, model, optimizer, vanilla_loss=True)


@pytest.mark.parametrize('bad', ['count', 'normalization', None])
def test_estimator_forwards_match_the_shared_prediction(bad):
    torch = pytest.importorskip('torch')
    class Model(torch.nn.Module):
        def forward(self, x, dropout=0.): return x * 2
    model = Model()
    raw = torch.ones(4,3)
    class Episode:
        def estimate(self, batch):
            model(batch + (1 if bad == 'normalization' else 0))
            for _ in range(9 if bad == 'count' else 10): model(batch, dropout=.5)
            return {'estimated_accuracy_percent': 40.}
    stream = bn.EstimatorStream(Episode(), 8)
    if bad:
        with pytest.raises(ValueError, match='forward|prediction'):
            bn.checked_estimate(stream, model, raw, raw*2)
    else:
        assert bn.checked_estimate(stream, model, raw, raw*2)['estimated_accuracy_percent'] == 40.
