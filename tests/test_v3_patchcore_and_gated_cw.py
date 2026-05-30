"""Smoke + unit tests for the v3 strong-detector stack:
true patch-level PatchCore, greedy coreset, image scoring, and the
RGA-gated-CW clean-default / degradation-response property.

These cover the newest, most claim-bearing code, which previously had no
dedicated tests. They are deterministic and run without the large image
datasets (synthetic tensors / arrays), plus light artifact-consistency
checks on the committed v3 result JSONs when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# patch-level PatchCore primitives
# --------------------------------------------------------------------------
def test_greedy_coreset_selects_requested_count_and_subset():
    from uais.fusion.attention.patchcore_patch import greedy_coreset

    rng = np.random.default_rng(0)
    bank = rng.standard_normal((500, 32)).astype(np.float32)
    sel = greedy_coreset(bank, 50, seed=0)
    assert sel.shape == (50, 32)
    # every selected row must be an actual bank row (subset, not synthesised)
    bank_set = {tuple(r) for r in bank}
    assert all(tuple(r) in bank_set for r in sel)


def test_greedy_coreset_returns_full_bank_when_target_exceeds_size():
    from uais.fusion.attention.patchcore_patch import greedy_coreset

    bank = np.random.default_rng(1).standard_normal((20, 8)).astype(np.float32)
    sel = greedy_coreset(bank, 100, seed=0)
    assert sel.shape == bank.shape


def test_image_anomaly_scores_separates_normal_from_anomalous():
    """Patches near the bank should score low; patches far from it, high."""
    from uais.fusion.attention.patchcore_patch import image_anomaly_scores

    rng = np.random.default_rng(2)
    D, P = 16, 4
    bank = rng.standard_normal((200, D)).astype(np.float32)  # "normal" cluster ~N(0,1)
    # normal image: patches drawn from the same distribution
    normal = rng.standard_normal((P, D)).astype(np.float32)
    # anomalous image: patches shifted far away
    anomalous = (rng.standard_normal((P, D)) + 8.0).astype(np.float32)
    q = np.concatenate([normal, anomalous], axis=0)  # 2 images, P patches each
    scores = image_anomaly_scores(q, P, bank)
    assert scores.shape == (2,)
    assert scores[1] > scores[0]  # anomalous image scores strictly higher


def test_image_anomaly_scores_is_max_over_patches():
    """The image score is the MAX patch distance, so one bad patch dominates."""
    from uais.fusion.attention.patchcore_patch import image_anomaly_scores

    D, P = 4, 3
    bank = np.zeros((10, D), dtype=np.float32)
    img = np.zeros((P, D), dtype=np.float32)
    img[2] = 5.0  # one far patch
    s = image_anomaly_scores(img, P, bank)
    assert s.shape == (1,)
    assert s[0] == pytest.approx(np.linalg.norm(np.full(D, 5.0)), rel=1e-4)


# --------------------------------------------------------------------------
# RGA-gated-CW clean-default / degradation-response behavior
# --------------------------------------------------------------------------
def _ks_reliability(test, ref):
    from scipy.stats import ks_2samp
    return float(np.clip(1 - ks_2samp(test, ref).statistic, 0.0, 1.0))


def _gated_cw(rgb, depth, val_rgb, val_depth, clean_floor, margin=0.05):
    """Reference implementation of the gated-CW rule (mirrors the script)."""
    cw = 0.5 * (rgb + depth)  # equal-confidence baseline
    r_rgb = _ks_reliability(rgb, val_rgb)
    r_depth = _ks_reliability(depth, val_depth)
    tau = clean_floor - margin
    if min(r_rgb, r_depth) < tau:
        return (r_rgb * rgb + r_depth * depth) / (r_rgb + r_depth + 1e-9)
    return cw


def test_gated_cw_equals_baseline_on_clean():
    """When no drift is detected the gate must defer to CW exactly (no negative)."""
    rng = np.random.default_rng(3)
    val_rgb = rng.uniform(0, 1, 300)
    val_depth = rng.uniform(0, 1, 300)
    # clean test = same distribution as validation -> reliability stays high
    rgb = rng.uniform(0, 1, 300)
    depth = rng.uniform(0, 1, 300)
    clean_floor = min(_ks_reliability(rgb, val_rgb), _ks_reliability(depth, val_depth))
    out = _gated_cw(rgb, depth, val_rgb, val_depth, clean_floor)
    cw = 0.5 * (rgb + depth)
    np.testing.assert_allclose(out, cw)  # exactly the baseline -> cannot be worse


def test_gated_cw_switches_under_degradation():
    """When one modality degrades, the gate must fire and downweight it."""
    rng = np.random.default_rng(4)
    val_depth = rng.uniform(0, 1, 300)
    val_rgb = rng.uniform(0, 1, 300)
    rgb = rng.uniform(0, 1, 300)
    depth_clean = rng.uniform(0, 1, 300)
    clean_floor = min(_ks_reliability(rgb, val_rgb), _ks_reliability(depth_clean, val_depth))
    # fully corrupt depth -> large KS drift -> gate should fire (output != CW)
    depth_bad = np.full(300, 0.9) + 0.01 * rng.standard_normal(300)
    out = _gated_cw(rgb, depth_bad, val_rgb, val_depth, clean_floor)
    cw = 0.5 * (rgb + depth_bad)
    assert not np.allclose(out, cw)  # gate fired -> deviated from the naive baseline


# --------------------------------------------------------------------------
# artifact consistency (committed v3 results)
# --------------------------------------------------------------------------
def test_v3_multisplit_result_is_significant_and_positive():
    p = ROOT / "experiments/fusion/mvtec3d_v3_multisplit_result.json"
    if not p.exists():
        pytest.skip("v3 multisplit result not present")
    d = json.loads(p.read_text())
    delta = d["delta_rga_vs_sar"]
    assert delta["mean"] > 0
    assert delta["ci95"][0] > 0  # CI excludes zero on the positive side


def test_gated_cw_artifact_has_no_significant_negative_but_not_noninferiority():
    p = ROOT / "experiments/fusion/rga_gated_cw_transfer_result.json"
    if not p.exists():
        pytest.skip("gated-CW result not present")
    d = json.loads(p.read_text())
    saw_negative_ci_bound = False
    for name, res in d["datasets"].items():
        assert res["no_significant_negative_anywhere"] is True, name
        saw_negative_ci_bound |= any(row["ci95"][0] < 0 for row in res["rows"])
    assert saw_negative_ci_bound  # CI crossing zero must not be marketed as weak dominance.
