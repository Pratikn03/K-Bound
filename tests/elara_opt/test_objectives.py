"""Unit tests for the ELARA-Opt unlabeled objectives."""
import math

import torch

from experiments.kbound.elara_opt import objectives as O


def test_entropy_bounds_and_uniform_max():
    C = 5
    uniform = torch.zeros(4, C)                      # logits 0 -> uniform softmax
    peaked = torch.full((4, C), -10.0)
    peaked[:, 0] = 10.0
    e_uniform = O.entropy_loss(uniform).item()
    e_peaked = O.entropy_loss(peaked).item()
    assert abs(e_uniform - math.log(C)) < 1e-4       # uniform entropy = ln C
    assert e_peaked < 1e-3                            # near-deterministic -> ~0
    assert e_peaked < e_uniform


def test_entropy_per_sample_shape():
    logits = torch.randn(7, 3)
    e = O.entropy_per_sample(logits)
    assert e.shape == (7,)
    assert torch.all(e >= -1e-6)


def test_filtered_entropy_keeps_confident_only():
    C = 4
    logits = torch.full((6, C), -10.0)
    logits[:3, 0] = 10.0                             # 3 confident (low entropy)
    logits[3:] = 0.0                                 # 3 uniform (high entropy)
    loss, kept = O.filtered_entropy_loss(logits, C, margin_frac=0.4)
    assert abs(kept - 0.5) < 1e-9                     # exactly half pass the margin
    assert loss.item() < 1e-3                         # kept set is near-zero entropy


def test_filtered_entropy_empty_is_graph_connected_zero():
    C = 4
    logits = torch.zeros(5, C, requires_grad=True)   # all uniform -> none reliable
    loss, kept = O.filtered_entropy_loss(logits, C, margin_frac=0.1)
    assert kept == 0.0
    loss.backward()                                  # must not raise; zero grad
    assert torch.allclose(logits.grad, torch.zeros_like(logits))


def test_symmetric_kl_zero_iff_equal_and_nonneg():
    a = torch.randn(5, 3)
    assert O.symmetric_kl(a, a.clone()).abs().max().item() < 1e-6
    b = torch.randn(5, 3)
    assert torch.all(O.symmetric_kl(a, b) >= -1e-6)


def test_aug_consistency_zero_when_identical():
    logits = torch.randn(8, 6)
    assert O.aug_consistency_loss(logits, logits.clone()).item() < 1e-6


def test_frozen_anchor_zero_iff_equal_and_nonneg():
    lf = torch.randn(5, 4)
    assert O.frozen_kl_anchor(lf, lf.clone()).item() < 1e-6
    lc = torch.randn(5, 4)
    assert O.frozen_kl_anchor(lf, lc).item() >= -1e-6


def test_augment_flip_is_deterministic_and_label_preserving_shape():
    x = torch.randn(3, 3, 8, 8)
    a1 = O.augment(x, seed=0)
    a2 = O.augment(x, seed=0)
    assert torch.equal(a1, a2)                        # deterministic
    assert a1.shape == x.shape
    assert torch.equal(O.augment(x), torch.flip(x, dims=[3]))  # NCHW -> hflip


def test_all_mixture_losses_keys():
    logits = torch.randn(6, 5)
    aug = torch.randn(6, 5)
    d, kept = O.all_mixture_losses(logits, aug, 5)
    assert set(d.keys()) == set(O.OBJECTIVE_NAMES)
    assert 0.0 <= kept <= 1.0
