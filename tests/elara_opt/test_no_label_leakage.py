"""No-target-label-leakage tests for ELARA-Opt.

The strongest guarantee is behavioral: the adapted candidate must be a function of
the unlabeled batch ONLY, so permuting the (dev) labels cannot change the candidate
hash.  We also check signatures, the telemetry guard, and scan emitted telemetry.
"""
import inspect

import numpy as np
import pytest
import torch

from experiments.kbound.elara_opt import elara_opt_adapt, run_elara_candidate
from experiments.kbound.elara_opt import reliability as R
from experiments.kbound.elara_opt.telemetry import TelemetryCollector
from experiments.kbound.elara_opt.smoke_models import build_f0, synth_cell

_FORBIDDEN = ("label", "target", "y_true", "y_test", "ground_truth", "gt")


def _no_forbidden_params(fn):
    for name in inspect.signature(fn).parameters:
        low = name.lower()
        assert not any(b in low for b in _FORBIDDEN), f"{fn.__name__} has label-like param {name}"


def test_adapter_and_features_take_no_labels():
    _no_forbidden_params(elara_opt_adapt)
    _no_forbidden_params(R.compute_features)


def test_candidate_is_invariant_to_label_permutation():
    """If labels leaked into the update, permuting dev_y would move the candidate."""
    f0 = build_f0(10, 3, seed=0)
    stream, eval_x, dev_y = synth_cell(10, 16, 3, 32, seed=0)
    rng = np.random.default_rng(123)
    dev_y_perm = dev_y.copy()
    rng.shuffle(dev_y_perm)
    assert not np.array_equal(dev_y, dev_y_perm)

    r1 = run_elara_candidate(f0, stream, eval_x, dev_y, 10, "elara_rule", steps=1, lr=1e-3, seed=0)
    r2 = run_elara_candidate(f0, stream, eval_x, dev_y_perm, 10, "elara_rule", steps=1, lr=1e-3, seed=0)
    # the adapted candidate is identical; only the (legitimately dev-labeled) benefit may differ
    assert r1["candidate_hash"] == r2["candidate_hash"]


def test_telemetry_guard_rejects_label_key_and_raw_tensor():
    tc = TelemetryCollector(mode="elara_rule", seed=0)
    with pytest.raises(ValueError):
        tc.log_step({"target_label": 3})
    with pytest.raises(ValueError):
        tc.log_step({"loss_entropy": torch.tensor([1.0, 2.0])})  # raw tensor could smuggle labels


def test_emitted_telemetry_has_no_label_content():
    f0 = build_f0(10, 3, seed=0)
    stream, _, _ = synth_cell(10, 16, 3, 32, seed=0)
    _, _, tele = elara_opt_adapt(f0, stream, 1, 1e-3, 10, mode="elara_rule", seed=0)

    def scan(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert not any(b in k.lower() for b in _FORBIDDEN), f"forbidden key {k}"
                scan(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                scan(v)
        else:
            assert isinstance(obj, (str, int, float, bool)) or obj is None

    scan(tele)
