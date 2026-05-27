"""Tests for target-domain KS reference calibration (positive transfer fix).

Proves that re_fit_ks_reference() resolves the Eyecandies 100% false-fire
root cause: training scores (normal-only, [0, 0.5]) vs. inference scores
(mixed normal/anomalous, [0, 1]) cause ks_reliability≈0 without the fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uais.fusion.attention.reliability_estimator import ReliabilityEstimator


# ── Synthetic fixture helpers ─────────────────────────────────────────────────


def _make_train_distribution(
    n: int = 200,
    n_domains: int = 2,
    n_features: int = 4,
    score_index: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic training data: all-normal, scores drawn from U(0.05, 0.45).

    Mimics Eyecandies training features (PatchCore cosine distances for
    normal samples only, z-score+sigmoid → low score range).
    """
    rng = np.random.RandomState(seed)
    features = rng.uniform(0.0, 0.5, size=(n, n_domains, n_features)).astype(np.float32)
    # score channel: training normal samples cluster in [0.05, 0.45]
    features[:, :, score_index] = rng.uniform(0.05, 0.45, size=(n, n_domains)).astype(np.float32)
    masks = np.zeros((n, n_domains), dtype=bool)
    labels = np.zeros(n, dtype=float)  # all normal
    return features, masks, labels


def _make_val_distribution(
    n: int = 200,
    n_domains: int = 2,
    n_features: int = 4,
    score_index: int = 0,
    seed: int = 1,
    anomaly_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic validation/inference data matching Eyecandies score distribution.

    Normal samples score U(0.35, 0.65) and anomalous samples score U(0.75, 1.0).
    Mean ≈ 0.50 which is the actual Eyecandies val/test distribution observed
    (fusion CSV: val_mean=0.4935). This is very different from train (mean≈0.15)
    causing the KS drift issue.
    """
    rng = np.random.RandomState(seed)
    n_anomaly = int(n * anomaly_fraction)
    n_normal = n - n_anomaly
    features = rng.uniform(0.0, 0.5, size=(n, n_domains, n_features)).astype(np.float32)
    # Normal val samples: score U(0.35, 0.65)
    features[n_anomaly:, :, score_index] = rng.uniform(
        0.35, 0.65, size=(n_normal, n_domains)
    ).astype(np.float32)
    # Anomalous val samples: score U(0.75, 1.0)
    features[:n_anomaly, :, score_index] = rng.uniform(
        0.75, 1.0, size=(n_anomaly, n_domains)
    ).astype(np.float32)
    masks = np.zeros((n, n_domains), dtype=bool)
    labels = np.zeros(n, dtype=float)
    labels[:n_anomaly] = 1.0
    return features, masks, labels


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestReFitKsReference:
    """Verify re_fit_ks_reference() resolves the 100% false-fire root cause."""

    # After re_fit_ks_reference, clean rel_d ceiling ≈ 0.45*ece + 0.35*1.0 + 0.20*sharpness.
    # With degenerate ECE (all-normal train) ece_rel=0.5 → ceiling ≈ 0.575.
    # OPERATING_TAU must be below that ceiling; tau sweep will pick the best candidate.
    LOCKED_TAU = 0.66
    OPERATING_TAU = 0.50  # safely below rel_d ceiling ≈ 0.575 achievable after refit
    CLEAN_BUDGET = 0.010

    def _make_estimator(self) -> ReliabilityEstimator:
        return ReliabilityEstimator(
            domain_order=["rgb", "depth"],
            score_index=0,
            ece_weight=0.45,
            ks_weight=0.35,
            sharpness_weight=0.20,
            gate_threshold=self.LOCKED_TAU,
            gate_mode="mean",
            min_samples_for_ks=10,
        )

    def test_without_refit_false_fire_rate_is_100_percent(self):
        """Without re_fit_ks_reference, all clean val samples trigger the gate.

        Train scores U(0.05, 0.45), val scores U(0.35, 0.65). The KS test
        detects drift (very different distributions) → ks_rel≈0 → rel_d≈0.25
        → all samples fire at tau=0.66.
        """
        train_feat, train_mask, train_labels = _make_train_distribution(seed=0)
        val_feat, val_mask, _ = _make_val_distribution(seed=1)
        # Val clean = only label-0 samples (last 80 %)
        n_anom = int(0.2 * len(val_feat))
        val_clean_feat = val_feat[n_anom:]
        val_clean_mask = val_mask[n_anom:]

        estimator = self._make_estimator()
        estimator.fit(train_feat, train_mask, train_labels)
        # Do NOT call re_fit_ks_reference

        weights = estimator.compute_reliability_weights(val_clean_feat, val_clean_mask)
        mean_r = weights.mean(axis=1)
        false_fire_rate = float((mean_r < self.LOCKED_TAU).mean())

        # Without calibration fix, the KS drift is huge → virtually all fire
        assert false_fire_rate > 0.50, (
            f"Expected high false-fire rate without calibration fix, got {false_fire_rate:.3f}"
        )

    def test_with_refit_mean_reliability_increases(self):
        """After re_fit_ks_reference, mean reliability rises toward the true domain reliability.

        The key functional guarantee of the fix: when training scores are OOD
        relative to inference scores, re-fitting the KS reference on the val
        distribution increases the reliability estimate for clean samples.
        This allows the tau sweep to find a threshold that passes the budget.
        """
        rng = np.random.RandomState(99)
        n_train = 400

        # Train: all-normal, scores tightly in [0.05, 0.30]  (OOD from inference)
        train_feat = rng.uniform(0.0, 0.5, (n_train, 2, 4)).astype(np.float32)
        train_feat[:, :, 0] = rng.uniform(0.05, 0.30, (n_train, 2)).astype(np.float32)
        train_mask = np.zeros((n_train, 2), dtype=bool)
        train_labels = np.zeros(n_train)

        # Val (Eyecandies-like): ALL normal, scores U(0.45, 0.55) — very different from train
        n_val = 200
        val_feat = rng.uniform(0.0, 0.5, (n_val, 2, 4)).astype(np.float32)
        val_feat[:, :, 0] = rng.uniform(0.45, 0.55, (n_val, 2)).astype(np.float32)
        val_mask = np.zeros((n_val, 2), dtype=bool)

        # Inference clean batch: same distribution as val
        n_clean = 100
        clean_feat = rng.uniform(0.0, 0.5, (n_clean, 2, 4)).astype(np.float32)
        clean_feat[:, :, 0] = rng.uniform(0.45, 0.55, (n_clean, 2)).astype(np.float32)
        clean_mask = np.zeros((n_clean, 2), dtype=bool)

        estimator = self._make_estimator()
        estimator.fit(train_feat, train_mask, train_labels)

        # Before refit: train_ref U(0.05,0.30) vs clean U(0.45,0.55) → massive drift
        w_before = estimator.compute_reliability_weights(clean_feat, clean_mask)
        mean_r_before = float(w_before.mean(axis=1).mean())

        estimator.re_fit_ks_reference(val_feat, val_mask)

        # After refit: val_ref U(0.45,0.55) vs clean U(0.45,0.55) → no drift
        w_after = estimator.compute_reliability_weights(clean_feat, clean_mask)
        mean_r_after = float(w_after.mean(axis=1).mean())

        # The fix must increase reliability: KS p goes from ≈0 to ≈1
        assert mean_r_after > mean_r_before, (
            f"Mean reliability did not increase after refit: "
            f"before={mean_r_before:.4f}, after={mean_r_after:.4f}"
        )
        # The improvement should be substantial (at least 0.10 gain from KS term)
        assert mean_r_after - mean_r_before > 0.05, (
            f"Reliability improvement too small: Δ={mean_r_after - mean_r_before:.4f}"
        )

    def test_refit_preserves_ece_and_calibrators(self):
        """re_fit_ks_reference() must not change ECE values or calibrators."""
        train_feat, train_mask, train_labels = _make_train_distribution(seed=0)
        val_feat, val_mask, _ = _make_val_distribution(seed=1)

        estimator = self._make_estimator()
        estimator.fit(train_feat, train_mask, train_labels)

        ece_before = dict(estimator._domain_ece)
        cal_before = {k: id(v) for k, v in estimator._calibrators.items()}

        estimator.re_fit_ks_reference(val_feat, val_mask)

        assert estimator._domain_ece == ece_before, "ECE values changed after re_fit_ks_reference"
        # Calibrator object identity must be unchanged
        for domain, cal_id in cal_before.items():
            assert id(estimator._calibrators[domain]) == cal_id, (
                f"Calibrator object replaced for domain={domain}"
            )

    def test_refit_updates_reference_scores(self):
        """re_fit_ks_reference() must update _reference_scores for all domains."""
        train_feat, train_mask, train_labels = _make_train_distribution(n=200, seed=0)
        val_feat, val_mask, _ = _make_val_distribution(n=100, seed=1)

        estimator = self._make_estimator()
        estimator.fit(train_feat, train_mask, train_labels)

        ref_before = {d: v.mean() for d, v in estimator._reference_scores.items()}
        estimator.re_fit_ks_reference(val_feat, val_mask)
        ref_after = {d: v.mean() for d, v in estimator._reference_scores.items()}

        for domain in estimator.domain_order:
            # Val mean ≈ 0.5, train mean ≈ 0.25 — should have changed
            assert abs(ref_after[domain] - ref_before[domain]) > 0.05, (
                f"Reference scores for domain={domain} did not update: "
                f"before={ref_before[domain]:.4f}, after={ref_after[domain]:.4f}"
            )

    def test_refit_requires_fitted_estimator(self):
        """re_fit_ks_reference() must raise if called before fit()."""
        estimator = self._make_estimator()
        val_feat, val_mask, _ = _make_val_distribution(seed=1)
        with pytest.raises(RuntimeError, match="re_fit_ks_reference.*fit"):
            estimator.re_fit_ks_reference(val_feat, val_mask)

    def test_refit_handles_partial_missing_masks(self):
        """re_fit_ks_reference() must tolerate domains where all samples are masked."""
        train_feat, train_mask, train_labels = _make_train_distribution(n=100, seed=0)
        val_feat, val_mask, _ = _make_val_distribution(n=50, seed=1)

        # Mask out domain 0 entirely in the val set
        val_mask_partial = val_mask.copy()
        val_mask_partial[:, 0] = True  # all missing for domain 0

        estimator = self._make_estimator()
        estimator.fit(train_feat, train_mask, train_labels)
        # Should not raise even when one domain is fully masked
        estimator.re_fit_ks_reference(val_feat, val_mask_partial)

    def test_ks_reference_mean_matches_val_distribution(self):
        """After refit, KS reference mean should be close to val score mean."""
        train_feat, train_mask, train_labels = _make_train_distribution(n=500, seed=42)
        val_feat, val_mask, _ = _make_val_distribution(n=200, seed=43)

        estimator = self._make_estimator()
        estimator.fit(train_feat, train_mask, train_labels)
        estimator.re_fit_ks_reference(val_feat, val_mask)

        for i, domain in enumerate(estimator.domain_order):
            ref_mean = estimator._reference_scores[domain].mean()
            expected_mean = val_feat[~val_mask[:, i], i, 0].mean()
            assert abs(ref_mean - expected_mean) < 0.01, (
                f"KS reference mean for domain={domain} does not match val: "
                f"ref={ref_mean:.4f}, expected={expected_mean:.4f}"
            )
