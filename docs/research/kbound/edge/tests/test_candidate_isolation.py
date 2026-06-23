"""test_candidate_isolation -- the frozen model f0 is NEVER mutated by adaptation.

This is the central safety guarantee, checked with a real parameter+buffer hash:
the candidate is a deep copy; after one (or several) episodic Tent steps, f0's
content hash is bit-identical and only the candidate has changed.
"""

import pytest

torch = pytest.importorskip("torch")

from kbound_edge.model import build_model, state_dict_hash, bn_affine_param_names
from kbound_edge.tent_adapter import EpisodicTentAdapter


def _batch(n=4, size=48):
    torch.manual_seed(123)
    return torch.randn(n, 3, size, size)


def test_frozen_hash_unchanged_after_adapt():
    f0 = build_model(num_classes=4, pretrained=False, seed=0)
    h_before = state_dict_hash(f0)
    assert len(bn_affine_param_names(f0)) > 0

    adapter = EpisodicTentAdapter(f0, lr=1e-3, steps=1)
    res = adapter.adapt(_batch())

    assert state_dict_hash(f0) == h_before, "FROZEN MODEL WAS MUTATED"
    assert state_dict_hash(res.model) != h_before, "candidate did not change"
    assert res.upd_norm > 0.0


def test_repeated_episodic_adapts_do_not_drift_f0():
    f0 = build_model(num_classes=4, pretrained=False, seed=1)
    h_before = state_dict_hash(f0)
    adapter = EpisodicTentAdapter(f0, lr=1e-3, steps=1)
    x = _batch()
    for _ in range(3):
        adapter.adapt(x)
    assert state_dict_hash(f0) == h_before


def test_f0_requires_grad_flags_preserved():
    f0 = build_model(num_classes=4, pretrained=False, seed=2)
    before = [p.requires_grad for p in f0.parameters()]
    EpisodicTentAdapter(f0, lr=1e-3, steps=2).adapt(_batch())
    after = [p.requires_grad for p in f0.parameters()]
    assert before == after


def test_multi_step_still_isolated():
    f0 = build_model(num_classes=4, pretrained=False, seed=3)
    h_before = state_dict_hash(f0)
    res = EpisodicTentAdapter(f0, lr=5e-3, steps=3).adapt(_batch())
    assert state_dict_hash(f0) == h_before
    assert res.n_params > 0
