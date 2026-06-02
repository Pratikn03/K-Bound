"""Regression guard for the degenerate-channel filter (D18 lego_propeller audit).

The D18 Real-IAD-D3 held-out run was inflated by one degenerate point-cloud
channel: an inverted xyz (validation AUROC 0.000) and a saturated rgb
(validation std 0.000). These tests pin the validation-only detection of both
modes and confirm that genuinely informative channels are preserved.
"""

from __future__ import annotations

import numpy as np

from elara.evaluation.degenerate_channel_guard import (
    channel_diagnostic,
    diagnose_channels,
    guarded_channel_mask,
    guarded_reliability,
)


def _labels(n: int = 120, frac_pos: float = 0.4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < frac_pos).astype(int)
    # Guarantee both classes present.
    y[0], y[1] = 1, 0
    return y


def _informative(y: np.ndarray, sep: float = 0.3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.2, 0.6, size=len(y))
    s[y == 1] = np.clip(s[y == 1] + sep, 0, 1)
    return s


def _inverted(y: np.ndarray, sep: float = 0.3, seed: int = 2) -> np.ndarray:
    # Anomalies score LOWER than normals -> AUROC near 0.
    return 1.0 - _informative(y, sep=sep, seed=seed)


def _saturated(y: np.ndarray, value: float = 0.999, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.full(len(y), value) + rng.normal(0, 1e-4, size=len(y))


def test_informative_channel_is_kept():
    y = _labels()
    d = channel_diagnostic(_informative(y), y)
    assert not d.degenerate
    assert d.val_auc > 0.7
    assert d.reasons == ()


def test_inverted_channel_is_flagged():
    y = _labels()
    d = channel_diagnostic(_inverted(y), y)
    assert d.degenerate
    assert "inverted" in d.reasons
    assert d.val_auc < 0.45
    assert d.separation < 0  # anomalies below normals


def test_saturated_channel_is_flagged_even_with_high_auc():
    y = _labels()
    # A near-constant channel that happens to tie-break in label order: AUROC can
    # read high, but the channel carries no real discrimination.
    s = _saturated(y)
    s[y == 1] += 1e-5  # micro-ordering -> AUROC ~1.0 artifact
    d = channel_diagnostic(np.clip(s, 0, 1), y)
    assert d.degenerate
    assert any("saturated" in r for r in d.reasons)


def test_mask_drops_degenerate_keeps_good():
    y = _labels()
    s_val = np.column_stack([_saturated(y), _informative(y), _inverted(y)])
    mask = guarded_channel_mask(s_val, y)
    assert mask.tolist() == [False, True, False]


def test_reliability_zeroes_degenerate_channels():
    y = _labels()
    s_val = np.column_stack([_saturated(y), _informative(y), _inverted(y)])
    rel = guarded_reliability(s_val, y)
    assert rel[0] == 0.0 and rel[2] == 0.0
    assert rel[1] > 0.0


def test_never_zeroes_all_channels():
    # Even if every channel is weak, keep the single best so fusion can rank.
    y = _labels()
    weak = np.column_stack([_inverted(y), _saturated(y)])
    mask = guarded_channel_mask(weak, y)
    assert mask.sum() == 1


def test_single_class_validation_is_degenerate():
    y = np.ones(50, dtype=int)
    d = channel_diagnostic(np.linspace(0, 1, 50), y)
    assert d.degenerate
    assert "single_class_validation" in d.reasons


def test_diagnose_channels_shape():
    y = _labels()
    s_val = np.column_stack([_informative(y), _inverted(y)])
    diags = diagnose_channels(s_val, y)
    assert [d.index for d in diags] == [0, 1]
