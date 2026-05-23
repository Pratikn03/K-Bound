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
        gate_mode: str = "mean",
        min_gate_threshold: Optional[float] = None,
    ) -> None:
        if abs(ece_weight + ks_weight + sharpness_weight - 1.0) > 1e-6:
            raise ValueError("ece_weight + ks_weight + sharpness_weight must sum to 1.0")
        if gate_mode not in {"mean", "minimum", "hybrid"}:
            raise ValueError("gate_mode must be one of 'mean', 'minimum', or 'hybrid'.")
        self.domain_order = list(domain_order)
        self.score_index = score_index
        self.ece_weight = ece_weight
        self.ks_weight = ks_weight
        self.sharpness_weight = sharpness_weight
        self.n_calibration_bins = n_calibration_bins
        self.min_samples_for_ks = min_samples_for_ks
        self.gate_threshold = gate_threshold
        self.gate_mode = gate_mode
        self.min_gate_threshold = 1.0 - gate_threshold if min_gate_threshold is None else float(min_gate_threshold)

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

    def gate_decisions(
        self,
        weights: np.ndarray,
        masks: np.ndarray,
        gate_threshold: Optional[float] = None,
        gate_mode: Optional[str] = None,
        min_gate_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Return [N] bool: True = use reliability path, False = use static path.

        For each sample, mean reliability is computed over present (non-masked)
        domains only. The default ``mean`` gate preserves the conservative RGA
        rule: if the mean falls below gate_threshold, the reliability path is
        activated for that sample. The optional RGA-v2 ``minimum`` and
        ``hybrid`` modes add sensitivity to isolated catastrophic domain
        failures without changing the default paper configuration.
        """
        threshold = self.gate_threshold if gate_threshold is None else float(gate_threshold)
        mode = self.gate_mode if gate_mode is None else gate_mode
        min_threshold = self.min_gate_threshold if min_gate_threshold is None else float(min_gate_threshold)
        if mode not in {"mean", "minimum", "hybrid"}:
            raise ValueError("gate_mode must be one of 'mean', 'minimum', or 'hybrid'.")
        n_present = (~masks).sum(axis=1).astype(np.float32)
        present_weights = np.where(masks, 1.0, weights)
        mean_r = np.where(
            n_present > 0,
            weights.sum(axis=1) / np.maximum(n_present, 1.0),
            0.0,
        )
        min_r = np.where(n_present > 0, present_weights.min(axis=1), 0.0)
        mean_fire = mean_r < threshold
        min_fire = min_r < min_threshold
        if mode == "mean":
            return mean_fire
        if mode == "minimum":
            return min_fire
        return mean_fire | min_fire

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
            "gate_mode": self.gate_mode,
            "min_gate_threshold": self.min_gate_threshold,
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
            gate_mode=payload.get("gate_mode", "mean"),
            min_gate_threshold=payload.get("min_gate_threshold"),
        )
        obj._calibrators = payload["calibrators"]
        obj._reference_scores = payload["reference_scores"]
        obj._domain_ece = payload["domain_ece"]
        obj.fitted = payload["fitted"]
        return obj


class CategoryAwareReliabilityEstimator(ReliabilityEstimator):
    """Reliability estimator with category-conditional KS drift references.

    The global estimator can fire on legitimate batch-composition changes when
    each category has a different score distribution. This variant compares the
    current scores against the matching validation category when categories are
    supplied, falling back to the global reference for unknown categories.
    """

    def __init__(self, *args, unknown_category_policy: str = "global", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if unknown_category_policy not in {"global", "assume_reliable"}:
            raise ValueError("unknown_category_policy must be 'global' or 'assume_reliable'.")
        self.unknown_category_policy = unknown_category_policy
        self._category_reference_scores: Dict[str, Dict[str, np.ndarray]] = {}

    def fit(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        categories: Optional[np.ndarray] = None,
    ) -> "CategoryAwareReliabilityEstimator":
        super().fit(features, masks, labels)
        self._category_reference_scores = {domain: {} for domain in self.domain_order}
        if categories is None:
            return self

        categories = np.asarray(categories).astype(str)
        if categories.shape[0] != features.shape[0]:
            raise ValueError("categories must have one value per sample.")

        for i, domain in enumerate(self.domain_order):
            available = ~masks[:, i]
            for category in sorted(set(categories[available])):
                cat_mask = available & (categories == category)
                scores = features[cat_mask, i, self.score_index].astype(float)
                self._category_reference_scores[domain][category] = scores.copy()
        return self

    def _reference_for_category(self, domain: str, category: str) -> np.ndarray:
        category_refs = self._category_reference_scores.get(domain, {})
        if category in category_refs:
            return category_refs[category]
        if self.unknown_category_policy == "assume_reliable":
            return np.array([], dtype=float)
        return self._reference_scores.get(domain, np.array([], dtype=float))

    def compute_reliability_weights(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        categories: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if categories is None or not self._category_reference_scores:
            return super().compute_reliability_weights(features, masks)
        if not self.fitted:
            raise RuntimeError("ReliabilityEstimator must be fitted before computing weights.")

        categories = np.asarray(categories).astype(str)
        if categories.shape[0] != features.shape[0]:
            raise ValueError("categories must have one value per sample.")

        n_samples = features.shape[0]
        n_domains = len(self.domain_order)
        weights = np.zeros((n_samples, n_domains), dtype=np.float32)

        for i, domain in enumerate(self.domain_order):
            available_mask = ~masks[:, i]
            if not available_mask.any():
                continue

            for category in sorted(set(categories[available_mask])):
                cat_mask = available_mask & (categories == category)
                cur_scores = features[cat_mask, i, self.score_index].astype(float)
                ref = self._reference_for_category(domain, category)
                if len(ref) >= self.min_samples_for_ks and len(cur_scores) >= self.min_samples_for_ks:
                    _, ks_p = ks_2samp(ref, cur_scores)
                    ks_reliability = float(np.clip(ks_p, 0.0, 1.0))
                else:
                    ks_reliability = 1.0

                sharpness = float(np.clip(np.mean((cur_scores - 0.5) ** 2) * 4.0, 0.0, 1.0))
                stored_ece = self._domain_ece.get(domain, 0.5)
                ece_reliability = float(max(0.0, 1.0 - stored_ece))
                rel_d = (
                    self.ece_weight * ece_reliability
                    + self.ks_weight * ks_reliability
                    + self.sharpness_weight * sharpness
                )
                weights[cat_mask, i] = float(np.clip(rel_d, 0.0, 1.0))

        return weights

    def save(self, path: str | Path) -> None:
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
            "gate_mode": self.gate_mode,
            "min_gate_threshold": self.min_gate_threshold,
            "calibrators": self._calibrators,
            "reference_scores": self._reference_scores,
            "domain_ece": self._domain_ece,
            "fitted": self.fitted,
            "unknown_category_policy": self.unknown_category_policy,
            "category_reference_scores": self._category_reference_scores,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "CategoryAwareReliabilityEstimator":
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
            gate_mode=payload.get("gate_mode", "mean"),
            min_gate_threshold=payload.get("min_gate_threshold"),
            unknown_category_policy=payload.get("unknown_category_policy", "global"),
        )
        obj._calibrators = payload["calibrators"]
        obj._reference_scores = payload["reference_scores"]
        obj._domain_ece = payload["domain_ece"]
        obj.fitted = payload["fitted"]
        obj._category_reference_scores = payload.get("category_reference_scores", {})
        return obj


class PerSampleReliabilityEstimator(ReliabilityEstimator):
    """Per-sample reliability variant for streaming / low-batch deployment.

    The base ``ReliabilityEstimator`` returns a *batch-level* reliability
    scalar per domain that is broadcast to every available row in the
    inference batch. Under streaming or single-sample inference the
    batch-level reliability degenerates to a constant, which is the
    deployment-grade objection: the gate does not actually adapt per
    sample.

    This estimator decomposes the reliability into three terms that are
    each computed per-sample, plus a single global drift correction:

      r_{i,d} = ece_weight * ece_reliability_d                    (per-domain)
              + ks_weight * ks_local_reliability_{i,d}            (per-sample)
              + sharpness_weight * sharpness_local_{i,d}          (per-sample)

    where ``ks_local_reliability_{i,d}`` is computed from the local
    neighbourhood of sample i in the validation reference distribution
    (rank-based: 1 - 2|F_d(s_{i,d}) - 0.5|, so higher when the sample's
    score sits near the validation distribution's median and lower at
    the tails). ``sharpness_local_{i,d}`` is the per-sample sharpness
    ``(s_{i,d} - 0.5)^2 * 4`` clipped to [0,1]. The ECE component is
    necessarily per-domain because ECE itself is a population-level
    summary; it acts as a uniform per-domain prior. The drift term thus
    varies sample-by-sample, which is the property that makes this
    estimator deployment-grade under streaming.

    This subclass is API-compatible with ``ReliabilityEstimator``: the
    same ``fit`` and ``compute_reliability_weights`` signatures hold.
    """

    def compute_reliability_weights(
        self,
        features: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ReliabilityEstimator must be fitted before computing weights.")
        n_samples = features.shape[0]
        n_domains = len(self.domain_order)
        weights = np.zeros((n_samples, n_domains), dtype=np.float32)

        for i, domain in enumerate(self.domain_order):
            available_mask = ~masks[:, i]
            if not available_mask.any():
                continue
            cur_scores = features[available_mask, i, self.score_index].astype(float)
            # --- Per-sample rank-based local KS reliability ---
            ref = self._reference_scores.get(domain, np.array([]))
            if len(ref) >= self.min_samples_for_ks:
                # Empirical CDF of the validation reference at each test score.
                sorted_ref = np.sort(ref)
                ranks = np.searchsorted(sorted_ref, cur_scores, side="right")
                cdf = ranks / max(len(sorted_ref), 1)
                # 1 - 2|F-0.5| equals 1 at the median, 0 at the tails.
                ks_local = 1.0 - 2.0 * np.abs(cdf - 0.5)
                ks_local = np.clip(ks_local, 0.0, 1.0)
            else:
                ks_local = np.ones_like(cur_scores)

            # --- Per-sample sharpness ---
            sharp_local = np.clip(((cur_scores - 0.5) ** 2) * 4.0, 0.0, 1.0)

            # --- Per-domain ECE prior ---
            stored_ece = self._domain_ece.get(domain, 0.5)
            ece_reliability = float(max(0.0, 1.0 - stored_ece))

            rel = (
                self.ece_weight * ece_reliability
                + self.ks_weight * ks_local
                + self.sharpness_weight * sharp_local
            )
            rel = np.clip(rel.astype(np.float32), 0.0, 1.0)
            weights[available_mask, i] = rel
        return weights


# ---------------------------------------------------------------------------
# Paper-name alias
# ---------------------------------------------------------------------------
# The paper (ELARA, 2026) calls this component "RGA" (Reliability-Gated Attention).
# The code uses the internal project name "CRAF" (Calibration-aware Reliability-
# Adaptive Fusion).  Both names refer to the same class.
RGAReliabilityEstimator = ReliabilityEstimator

__all__ = [
    "ReliabilityEstimator",
    "CategoryAwareReliabilityEstimator",
    "PerSampleReliabilityEstimator",
    "RGAReliabilityEstimator",
]
