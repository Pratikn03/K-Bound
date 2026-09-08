"""Synthetic mechanism tests; no archive, labels or checkpoint inputs."""

import copy
import importlib
import inspect
import math

import pytest
import torch
from PIL import Image
from torch import nn


def api():
    # A missing implementation is a RED assertion, not a collection error.
    assert importlib.util.find_spec("experiments.kbound.domainnet.pilot_candidate") is not None
    return importlib.import_module("experiments.kbound.domainnet.pilot_candidate")


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3, eps=0.003, momentum=0.17)
        self.head = nn.Linear(3, 2)
        with torch.no_grad():
            self.bn.running_mean.copy_(torch.tensor([0.2, -0.1, 0.3]))
            self.bn.running_var.copy_(torch.tensor([0.7, 1.3, 1.7]))
            self.bn.num_batches_tracked.fill_(7)
            self.head.weight.copy_(torch.tensor([[1.0, 0.2, -0.3], [-0.4, 0.8, 0.1]]))
            self.head.bias.copy_(torch.tensor([0.2, -0.1]))
        self.requires_grad_(False).eval()

    def forward(self, images):
        return self.head(self.bn(images).mean((2, 3)))


def images(n=35):
    return torch.linspace(-2, 3, n * 3 * 4 * 4).reshape(n, 3, 4, 4)


def test_real_adam_bn_only_reset_restoration_and_short_batch():
    # Catches wrong optimizer, dropped short batch, batch-stat omission, nonaffine updates and stale resets.
    module = api()
    source = Tiny()
    before = copy.deepcopy(source.state_dict())
    data = images()
    result = module.create_candidate(source, [data[:10], data[10:]], expected_count=35)
    assert (result.count, result.batches) == (35, 2)
    assert result.normalized_bn_affine_update > 0
    assert result.adaptation_seconds >= 0
    assert all(torch.equal(source.state_dict()[k], v) for k, v in before.items())
    candidate = result.model
    assert all(not p.requires_grad for p in candidate.parameters())
    assert all(not m.training for m in candidate.modules())
    assert (candidate.bn.track_running_stats, candidate.bn.momentum, candidate.bn.eps) == (True, 0.17, 0.003)
    for name, value in before.items():
        if name not in {"bn.weight", "bn.bias"}:
            assert torch.equal(candidate.state_dict()[name], value)
    # Independent explicit two-step Adam reference makes the frozen recipe observable.
    reference = copy.deepcopy(source)
    reference.bn.train().requires_grad_(True)
    reference.bn.track_running_stats = False
    reference.bn.running_mean = reference.bn.running_var = None
    optimizer = torch.optim.Adam(reference.bn.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    for batch in [data[:32], data[32:]]:
        optimizer.zero_grad()
        logits = reference(batch)
        loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        optimizer.step()
    assert torch.equal(reference.bn.weight, candidate.bn.weight)
    assert torch.equal(reference.bn.bias, candidate.bn.bias)
    again = module.create_candidate(source, iter(data), expected_count=35)
    assert again.candidate_state_sha256 == result.candidate_state_sha256
    assert result.source_state_sha256 != result.candidate_state_sha256


class Literal(nn.Module):
    def __init__(self, flip=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.flip = flip
        self.requires_grad_(False).eval()

    def forward(self, x):
        first = x[:, 0, 0, 0]
        return torch.stack((torch.zeros_like(first), -first if self.flip else first), 1)


def test_literal_entropy_disagreement_first_ties_and_row_weighted_features():
    # 32 equiprobable rows + one [1/4, 3/4] row; candidate reverses the last winner.
    module = api()
    source = Literal()
    candidate = Literal(True)
    data = torch.zeros(33, 3, 2, 2)
    data[-1, 0, 0, 0] = math.log(3)
    pair = module.predict_pair(source, candidate, [data], expected_count=33)
    assert pair.count == 33 and pair.batches == 2
    assert pair.frozen_predictions == (0,) * 32 + (1,)
    assert pair.candidate_predictions == (0,) * 33
    assert pair.frozen_entropy_sum == pytest.approx(22.74304492253656, abs=2e-6)
    assert pair.candidate_entropy_mean == pytest.approx(0.6891831794708049, abs=1e-7)
    assert module.extract_features(pair, 0.25) == pytest.approx((0.6891831794708049, 0.6891831794708049, 1 / 33, 0.25))


def test_normalized_bn_update_literal_and_zero_denominator():
    module = api()
    source, candidate = Tiny(), Tiny()
    with torch.no_grad():
        source.bn.weight.copy_(torch.tensor([3.0, 4.0, 0.0]))
        candidate.bn.weight.copy_(torch.tensor([6.0, 8.0, 0.0]))
    assert module.normalized_bn_affine_update(source, candidate) == 1.0
    with torch.no_grad():
        source.bn.weight.zero_()
    assert module.normalized_bn_affine_update(source, candidate) == 1e13


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [torch.zeros(0, 3, 4, 4)],
        [torch.zeros(2, 4)],
        [torch.zeros(2, 1, 4, 4)],
        [torch.ones(2, 3, 4, 4, dtype=torch.int64)],
        [torch.ones(2, 3, 4, 4, dtype=torch.bool)],
        [torch.full((2, 3, 4, 4), float("nan"))],
        [torch.full((2, 3, 4, 4), float("inf"))],
        [(images(2), torch.zeros(2))],
    ],
)
def test_reject_invalid_images_without_dropping_or_replacing(bad):
    module = api()
    with pytest.raises((ValueError, TypeError)):
        module.create_candidate(Tiny(), bad, expected_count=2)


@pytest.mark.parametrize("count", [0, -1, True, 2.0, 34, 36])
def test_reject_invalid_or_mismatched_counts(count):
    with pytest.raises((ValueError, TypeError)):
        api().create_candidate(Tiny(), [images()], expected_count=count)


@pytest.mark.parametrize("fault", ["tracking", "mean", "negative_variance", "affine", "train", "grad", "nan"])
def test_reject_incompatible_source(fault):
    source = Tiny()
    if fault == "tracking":
        source.bn.track_running_stats = False
    elif fault == "mean":
        source.bn.running_mean = None
    elif fault == "negative_variance":
        source.bn.running_var[0] = -1
    elif fault == "affine":
        source.bn.affine = False
    elif fault == "train":
        source.train()
    elif fault == "grad":
        source.head.weight.requires_grad_(True)
    else:
        source.head.weight[0, 0] = float("nan")
    with pytest.raises(ValueError):
        api().create_candidate(source, [images()], expected_count=35)


@pytest.mark.parametrize("fault", ["parameter", "buffer", "mode", "eps", "tracking"])
def test_inference_rejects_mutation(fault):
    class Mutating(Tiny):
        def forward(self, x):
            out = super().forward(x)
            if fault == "parameter":
                self.head.bias.add_(1)
            elif fault == "buffer":
                self.bn.num_batches_tracked.add_(1)
            elif fault == "mode":
                self.bn.train()
            elif fault == "eps":
                self.bn.eps *= 2
            else:
                self.bn.track_running_stats = False
            return out

    with pytest.raises(ValueError, match="mutat"):
        api().predict_pair(Tiny(), Mutating(), [images()], expected_count=35)


@pytest.mark.parametrize("fault", ["nan", "inf", "rank", "rows", "classes"])
def test_invalid_logits_fail_in_both_paths(fault):
    class Invalid(Tiny):
        def forward(self, x):
            out = super().forward(x)
            if fault == "nan":
                return out * float("nan")
            if fault == "inf":
                return out * float("inf")
            if fault == "rank":
                return out[:, 0]
            if fault == "rows":
                return out[:1]
            return out[:, :1]

    module = api()
    for action in [
        lambda: module.create_candidate(Invalid(), [images()], expected_count=35),
        lambda: module.predict_pair(Tiny(), Invalid(), [images()], expected_count=35),
    ]:
        with pytest.raises(ValueError):
            action()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_gradients_rejected(bad):
    class BadGradient(Tiny):
        def forward(self, x):
            out = super().forward(x)
            if self.bn.weight.requires_grad:
                self.bn.weight.register_hook(lambda g: g * bad)
            return out

    with pytest.raises(ValueError, match="gradient"):
        api().create_candidate(BadGradient(), [images()], expected_count=35)


def test_transform_and_label_free_signatures():
    module = api()
    image = Image.new("L", (300, 270), 255)
    value = module.image_transform(image)
    assert value.shape == (3, 224, 224)
    assert torch.equal(value, module.image_transform(image))
    assert value[:, 0, 0].tolist() == pytest.approx([2.2489082969432315, 2.428571428571429, 2.64])
    for name in ["create_candidate", "predict_pair", "extract_features"]:
        assert not ({"labels", "targets", "beta", "radius"} & set(inspect.signature(getattr(module, name)).parameters))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_optimizer_state_rejected(monkeypatch, bad):
    # Fault injection after a genuine Adam update; optimization itself remains real.
    original = torch.optim.Adam.step

    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        next(iter(self.state.values()))["exp_avg"].fill_(bad)
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", corrupt)
    with pytest.raises(ValueError, match="optimizer"):
        api().create_candidate(Tiny(), [images()], expected_count=35)


def test_candidate_receipt_detects_between_window_mutation():
    module = api()
    source = Tiny()
    candidate = module.create_candidate(source, [images()], expected_count=35)
    module.predict_pair(source, candidate, [images(3)], expected_count=3)
    candidate.model.bn.eps *= 2
    with pytest.raises(ValueError, match="mutat"):
        module.predict_pair(source, candidate, [images(3)], expected_count=3)


def test_feature_extraction_rejects_forged_diagnostics():
    from dataclasses import replace

    module = api()
    pair = module.predict_pair(Tiny(), Tiny(), [images(2)], expected_count=2)
    for changed in [
        replace(pair, count=3),
        replace(pair, frozen_predictions=(0,)),
        replace(pair, frozen_entropy_sum=float("nan")),
        replace(pair, candidate_entropy_sum=-1.0),
    ]:
        with pytest.raises(ValueError):
            module.extract_features(changed, 0.1)
    for update in [-1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            module.extract_features(pair, update)


def test_logit_class_count_must_match_pair_and_all_batches():
    class Changing(Tiny):
        def forward(self, x):
            out = super().forward(x)
            return torch.cat((out, out[:, :1]), 1) if len(x) < 32 else out

    module = api()
    with pytest.raises(ValueError, match="class"):
        module.predict_pair(Tiny(), Changing(), [images()], expected_count=35)
    with pytest.raises(ValueError, match="class"):
        module.create_candidate(Changing(), [images()], expected_count=35)


def test_receipt_pins_adaptation_class_count_for_deployment():
    class Changing(Tiny):
        def forward(self, x):
            out = super().forward(x)
            return torch.cat((out, out[:, :1]), 1) if not self.bn.training else out

    module = api()
    source = Changing()
    result = module.create_candidate(source, [images()], expected_count=35)
    with pytest.raises(ValueError, match="class"):
        module.predict_pair(source, result, [images(2)], expected_count=2)


@pytest.mark.parametrize("fault", ["parameter", "buffer", "nonpersistent_buffer"])
def test_adaptation_rejects_nonaffine_mutation(fault):
    class Mutating(Tiny):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.tensor(0), persistent=fault != "nonpersistent_buffer")

        def forward(self, x):
            out = super().forward(x)
            if self.bn.training:
                with torch.no_grad():
                    if fault == "parameter":
                        self.head.bias.add_(1)
                    else:
                        self.counter.add_(1)
            return out

    with pytest.raises(ValueError, match="mutat"):
        api().create_candidate(Mutating(), [images()], expected_count=35)


def test_deepcopy_cannot_mutate_source_silently():
    class Sharing(Tiny):
        def __deepcopy__(self, memo):
            return self

    source = Sharing()
    before = copy.deepcopy({k: v.clone() for k, v in source.state_dict().items()})
    with pytest.raises(ValueError, match="mutat|independent"):
        api().create_candidate(source, [images()], expected_count=35)
    assert not source.bn.training and not source.bn.weight.requires_grad
    assert all(torch.equal(source.state_dict()[k], v) for k, v in before.items())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_parameter_after_real_update_is_rejected(monkeypatch, bad):
    original = torch.optim.Adam.step

    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        with torch.no_grad():
            self.param_groups[0]["params"][0].fill_(bad)
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", corrupt)
    with pytest.raises(ValueError, match="nonfinite"):
        api().create_candidate(Tiny(), [images()], expected_count=35)
