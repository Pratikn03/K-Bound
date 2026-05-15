"""Calibrated Reliability Scoring (CRS) for test-time adaptive fusion.

ReliabilityEstimator fits post-hoc isotonic calibrators on a validation split
and computes per-domain reliability weights at inference time using three signals:

  ECE (Expected Calibration Error) — how well-calibrated was the domain at training?
  KS drift                         — has the domain's score distribution shifted?
  Sharpness                        — does the domain produce confident (non-0.5) scores?

These weights replace the static distance-from-0.5 heuristic and are injected
directly into CrossModalAttentionFusion.forward(confidence_weights=...) without
retraining the fusion model. This is the Test-Time Reliability Adaptation (TTRA)
contribution of CRAF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression

from uais.utils.metrics import expected_calibration_error


class ReliabilityEstimator:
    """Post-hoc, calibration-aware domain reliability scorer.

    Fit once on the validation split after the fusion model is trained.
    Call compute_reliability_weights() at inference to get [N, D] weights
    suitable for injection into CrossModalAttentionFusion.forward().
    """

    def __init__(
        self,
        domain_order: List[str],
        score_index: int,
        ece_weight: float = 0.45,
        ks_weight: float = 0.35,
        sharpness_weight: float = 0.20,
        n_calibration_bins: int = 10,
        min_samples_for_ks: int = 30,
        gate_threshold: float = 0.66,
    ) -> None:
        if abs(ece_weight + ks_weight + sharpness_weight - 1.0) > 1e-6:
            raise ValueError("ece_weight + ks_weight + sharpness_weight must sum to 1.0")
        self.domain_order = list(domain_order)
        self.score_index = score_index
        self.ece_weight = ece_weight
        self.ks_weight = ks_weight
        self.sharpness_weight = sharpness_weight
        self.n_calibration_bins = n_calibration_bins
        self.min_samples_for_ks = min_samples_for_ks
        self.gate_threshold = gate_threshold

        self._calibrators: Dict[str, IsotonicRegression] = {}
        self._reference_scores: Dict[str, np.ndarray] = {}
        self._domain_ece: Dict[str, float] = {}
        self.fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
    ) -> "ReliabilityEstimator":
        """Fit isotonic calibrators and store reference distributions.

        Should be called on the same validation split used for early stopping,
        mirroring standard post-hoc calibration practice.

        Parameters
        ----------
        features : [N, D, F]
        masks    : [N, D] bool — True = domain missing for that sample
        labels   : [N] binary float
        """
        labels = np.asarray(labels, dtype=float)
        for i, domain in enumerate(self.domain_order):
            available = ~masks[:, i]
            scores = features[available, i, self.score_index].astype(float)
            y = labels[available]

            if len(scores) < 2 or len(np.unique(y)) < 2:
                # Insufficient data: treat as uncalibrated but present
                self._calibrators[domain] = None
                self._reference_scores[domain] = scores.copy()
                self._domain_ece[domain] = 0.5
                continue

            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(scores, y)
            calibrated = cal.predict(scores).astype(float)

            ece = expected_calibration_error(y, calibrated, n_bins=self.n_calibration_bins)

            self._calibrators[domain] = cal
            self._reference_scores[domain] = scores.copy()
            self._domain_ece[domain] = float(ece)

        self.fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute_reliability_weights(
        self,
        features: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Compute [N, D] reliability weights for the current inference batch.

        Weights are batch-level per domain (all samples in the batch receive
        the same domain reliability scalar). Missing domains receive 0.0.

        KS test compares the current batch's domain scores against the
        reference distribution stored at fit time — high p-value (no drift)
        means high reliability.
        """
        if not self.fitted:
            raise RuntimeError("ReliabilityEstimator must be fitted before computing weights.")

        n_samples = features.shape[0]
        n_domains = len(self.domain_order)
        weights = np.zeros((n_samples, n_domains), dtype=np.float32)

        for i, domain in enumerate(self.domain_order):
            available_mask = ~masks[:, i]
            if not available_mask.any():
                continue  # all missing → weights stay 0.0

            cur_scores = features[available_mask, i, self.score_index].astype(float)

            # --- KS drift ---
            ref = self._reference_scores.get(domain, np.array([]))
            if len(ref) >= self.min_samples_for_ks and len(cur_scores) >= self.min_samples_for_ks:
                _, ks_p = ks_2samp(ref, cur_scores)
                ks_reliability = float(np.clip(ks_p, 0.0, 1.0))
            else:
                ks_reliability = 1.0  # not enough data to detect drift → assume reliable

            # --- Sharpness: mean squared distance from 0.5, scaled to [0,1] ---
            sharpness = float(np.clip(np.mean((cur_scores - 0.5) ** 2) * 4.0, 0.0, 1.0))

            # --- ECE component: higher stored ECE at training → less reliable ---
            stored_ece = self._domain_ece.get(domain, 0.5)
            ece_reliability = float(max(0.0, 1.0 - stored_ece))

            rel_d = (
                self.ece_weight * ece_reliability
                + self.ks_weight * ks_reliability
                + self.sharpness_weight * sharpness
            )
            rel_d = float(np.clip(rel_d, 0.0, 1.0))

            # Broadcast scalar to all available rows
            weights[available_mask, i] = rel_d

        return weights

    def gate_decisions(self, weights: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Return [N] bool: True = use reliability path, False = use static path.

        For each sample, mean reliability is computed over present (non-masked)
        domains only. If that mean falls below gate_threshold, the reliability
        path is activated for that sample.  This is the conservative gate from
        the RGA paper: static attention is the default; reliability weights are
        only injected when domain quality evidence indicates degradation.
        """
        n_present = (~masks).sum(axis=1).astype(np.float32)
        # weights[masks] == 0.0 by contract of compute_reliability_weights
        mean_r = np.where(
            n_present > 0,
            weights.sum(axis=1) / np.maximum(n_present, 1.0),
            0.0,
        )
        return mean_r < self.gate_threshold  # True → use reliability path

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_domain_ece(self) -> Dict[str, float]:
        """Return stored per-domain ECE from the calibration fit."""
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        return dict(self._domain_ece)

    def get_domain_calibrated_scores(
        self, scores: np.ndarray, domain: str
    ) -> np.ndarray:
        """Apply the fitted isotonic calibrator for a given domain."""
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        cal = self._calibrators.get(domain)
        if cal is None:
            return scores.astype(float)
        return cal.predict(scores.astype(float)).astype(float)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise to disk using joblib (consistent with existing model artifacts)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "domain_order": self.domain_order,
            "score_index": self.score_index,
            "ece_weight": self.ece_weight,
            "ks_weight": self.ks_weight,
            "sharpness_weight": self.sharpness_weight,
            "n_calibration_bins": self.n_calibration_bins,
            "min_samples_for_ks": self.min_samples_for_ks,
            "gate_threshold": self.gate_threshold,
            "calibrators": self._calibrators,
            "reference_scores": self._reference_scores,
            "domain_ece": self._domain_ece,
            "fitted": self.fitted,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "ReliabilityEstimator":
        """Load a previously saved ReliabilityEstimator."""
        payload = joblib.load(path)
        obj = cls(
            domain_order=payload["domain_order"],
            score_index=payload["score_index"],
            ece_weight=payload["ece_weight"],
            ks_weight=payload["ks_weight"],
            sharpness_weight=payload["sharpness_weight"],
            n_calibration_bins=payload["n_calibration_bins"],
            min_samples_for_ks=payload["min_samples_for_ks"],
            gate_threshold=payload.get("gate_threshold", 0.66),
        )
        obj._calibrators = payload["calibrators"]
        obj._reference_scores = payload["reference_scores"]
        obj._domain_ece = payload["domain_ece"]
        obj.fitted = payload["fitted"]
        return obj


# ---------------------------------------------------------------------------
# Paper-name alias
# ---------------------------------------------------------------------------
# The paper (ELARA, 2026) calls this component "RGA" (Reliability-Gated Attention).
# The code uses the internal project name "CRAF" (Calibration-aware Reliability-
# Adaptive Fusion).  Both names refer to the same class.
RGAReliabilityEstimator = ReliabilityEstimator

__all__ = ["ReliabilityEstimator", "RGAReliabilityEstimator"]
