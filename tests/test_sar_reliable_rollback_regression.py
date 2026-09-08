"""Synthetic SAR branch contracts; never import a data-loading experiment harness."""
import ast
import copy
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
ROOT = Path(__file__).resolve().parents[1]
ROUTES = ["docs/research/kbound/scripts/cifar_tent_mps_v2.py", "experiments/kbound/wilds/tta_methods.py"]


def load_route(relative):
    path = ROOT / relative
    names = {"_entropy", "_bn_affine_params", "_clone_for_tta", "_upd_norm", "sar_adapt"}
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    scope = {"torch": torch, "nn": nn, "copy": copy, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


class EntropyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        z = (2 + self.bn.bias[0]).expand(len(x))
        return torch.stack([z, torch.zeros_like(z)], 1)


@pytest.mark.parametrize("route", ROUTES)
def test_empty_second_filter_restores_parameters_and_clears_gradients(route):
    model = EntropyModel().eval().requires_grad_(False)
    adapted, norm = load_route(route)["sar_adapt"](
        model, [torch.zeros(2, 1)], 1, .1, 2, rho=.5, margin_e0=.4, reset_constant_em=0.
    )
    # H([2,0])=.3653 < .4; SAM produces H([1.5,0])=.4751 > .4.
    assert float(adapted.bn.bias.detach()[0]) == 0.
    assert norm == 0.
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in adapted.parameters())


@pytest.mark.parametrize("route", ROUTES)
def test_empty_first_filter_remains_noop(route):
    adapted, norm = load_route(route)["sar_adapt"](
        EntropyModel().eval().requires_grad_(False), [torch.zeros(2, 1)], 1, .1, 2, margin_e0=.1
    )
    assert norm == 0.
    assert float(adapted.bn.bias.detach()[0]) == 0.


@pytest.mark.parametrize("route", ROUTES)
def test_nonempty_second_filter_applies_sam_gradient_at_perturbed_parameters(route):
    adapted, norm = load_route(route)["sar_adapt"](
        EntropyModel().eval().requires_grad_(False), [torch.zeros(2, 1)], 1, .1, 2,
        rho=.05, margin_e0=.6, reset_constant_em=0.
    )
    # Analytic binary entropy derivative is -z*sigmoid(z)*(1-sigmoid(z)).
    probability = 1 / (1 + math.exp(-1.95))
    expected = .1 * 1.95 * probability * (1 - probability)
    assert float(adapted.bn.bias.detach()[0]) == pytest.approx(expected, abs=1e-7)
    assert norm > 0.


def test_layer4_exclusion_does_not_retain_source_running_statistics():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.early = nn.BatchNorm2d(1)
            self.layer4 = nn.Sequential(nn.BatchNorm2d(1))

    model, parameters, _ = load_route(ROUTES[0])["_clone_for_tta"](
        Model().eval().requires_grad_(False), freeze_layer4=True
    )
    model.layer4(torch.arange(8.).reshape(2, 1, 2, 2))
    assert not model.layer4[0].track_running_stats
    assert model.layer4[0].running_mean is None
    assert model.layer4[0].running_var is None
    assert not any(p is model.layer4[0].weight or p is model.layer4[0].bias for p in parameters)


class StreamEntropyModel(EntropyModel):
    def forward(self, x):
        z = x[:, 0] + self.bn.bias[0]
        return torch.stack([z, torch.zeros_like(z)], 1)


def run_observed_sar(route, values, reset_threshold):
    """Observe real function locals at return; do not substitute optimizer/EMA."""
    function = load_route(route)['sar_adapt']
    observed = {}
    prior_trace = sys.gettrace()

    def trace(frame, event, arg):
        if frame.f_code is function.__code__ and event == 'return':
            observed['ema'] = frame.f_locals['ema']
            optimizer = frame.f_locals['opt']
            observed['momentum'] = [state['momentum_buffer'].detach().clone()
                                    for state in optimizer.state.values()
                                    if 'momentum_buffer' in state]
        return trace

    try:
        sys.settrace(trace)
        adapted, norm = function(
            StreamEntropyModel().eval().requires_grad_(False),
            [torch.full((2, 1), value) for value in values],
            1, .1, 2, rho=.5, margin_e0=.4, reset_constant_em=reset_threshold,
        )
    finally:
        sys.settrace(prior_trace)
    return adapted, norm, observed


@pytest.mark.parametrize('route', ROUTES)
def test_rejected_middle_batch_preserves_existing_momentum_and_ema(route):
    # First accepted SAM point z=2 has H=.3653. Middle point near z=1.52
    # is rejected (H>.4). Final accepted point near z=3.52 has H~.13.
    # Preserved EMA stays >.2; resetting EMA at the rejection would instead
    # trigger recovery on the last batch. Thus final weights test that branch.
    baseline, _, before = run_observed_sar(route, [2.5, 4.], .2)
    adapted, norm, after = run_observed_sar(route, [2.5, 2., 4.], .2)
    probability = 1 / (1 + math.exp(-2.))
    first_gradient = -2. * probability * (1 - probability)
    first_bias = -.1 * first_gradient
    final_z = 3.5 + first_bias
    final_probability = 1 / (1 + math.exp(-final_z))
    final_gradient = -final_z * final_probability * (1 - final_probability)
    expected_momentum = .9 * first_gradient + final_gradient
    expected_bias = first_bias - .1 * expected_momentum
    assert float(adapted.bn.bias.detach()[0]) == pytest.approx(expected_bias, abs=1e-7)
    assert torch.equal(adapted.bn.bias, baseline.bn.bias)
    assert norm > 0.
    assert after['ema'] == pytest.approx(before['ema'], abs=1e-8)
    assert after['ema'] > .2
    assert len(after['momentum']) == len(before['momentum']) == 1
    assert float(after['momentum'][0][0]) == pytest.approx(expected_momentum, abs=1e-7)
    assert torch.equal(after['momentum'][0], before['momentum'][0])
