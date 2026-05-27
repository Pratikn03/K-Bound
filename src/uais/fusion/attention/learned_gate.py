"""Learned reliability gate — alternative to the heuristic τ threshold.

The default RGA gate is a hard threshold on mean per-sample reliability:
``mean(r_i) < τ → use reliability path``.  A common reviewer question is
"why this τ and not a learned alternative?".  This module trains a small
classifier on per-sample reliability vectors using a self-supervised signal
generated from synthetic drift episodes on the validation split:

  - On each validation drift episode, for every sample compute static
    prediction p_s, reliability prediction p_r, and the true label y.
  - Sample-level training label is ``|p_r - y| < |p_s - y|`` (i.e. RGA is
    closer to ground truth than static for this sample).
  - Train a logistic regression on the per-sample reliability vector
    r ∈ R^D as features → binary "should fire" label.

At inference the learned gate replaces the heuristic comparison; downstream
code consumes a [N]-bool decision array in the same way as the heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class LearnedGateConfig:
    """Hyperparameters for the learned gate."""

    drift_levels: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3)
    feature_mode: str = "vector"  # "vector" → r ∈ R^D + mean; "scalar" → mean only
    use_mask_indicators: bool = True  # append [D] missing-domain flags
    class_weight: str = "balanced"
    C: float = 1.0
    random_state: int = 42


class LearnedReliabilityGate:
    """Classifier that decides per sample whether to use the reliability path.

    The decision interface mirrors ``ReliabilityEstimator.gate_decisions`` so
    callers can pass the returned boolean array straight into
    ``_predict_craf_with_stats(..., per_sample_gating=True)`` machinery.

    Parameters
    ----------
    config : LearnedGateConfig
        Hyperparameters; defaults are fine for first runs.
    """

    def __init__(self, config: LearnedGateConfig | None = None) -> None:
        self.config = config or LearnedGateConfig()
        self._clf: LogisticRegression | None = None
        self.fitted: bool = False
        self.train_label_balance_: float | None = None
        self.train_n_episodes_: int = 0

    # ------------------------------------------------------------------
    # Featurization
    # ------------------------------------------------------------------
    def _featurize(self, weights: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Build [N, F] feature matrix from per-sample reliability + masks.

        weights : [N, D] float — reliability vector r_i
        masks   : [N, D] bool — True = domain missing
        """
        weights = np.asarray(weights, dtype=np.float32)
        masks = np.asarray(masks, dtype=bool)
        n_present = (~masks).sum(axis=1).astype(np.float32)
        mean_r = np.where(
            n_present > 0,
            weights.sum(axis=1) / np.maximum(n_present, 1.0),
            0.0,
        ).astype(np.float32)
        if self.config.feature_mode == "scalar":
            features = mean_r.reshape(-1, 1)
        else:  # "vector"
            features = np.concatenate(
                [weights, mean_r.reshape(-1, 1)],
                axis=1,
            )
        if self.config.use_mask_indicators:
            mask_int = masks.astype(np.float32)
            features = np.concatenate([features, mask_int], axis=1)
        return features

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit_from_episodes(
        self,
        episodes: list[dict],
    ) -> LearnedReliabilityGate:
        """Fit from pre-computed episode dicts.

        Each episode dict must contain:
          - "weights": [N, D] reliability weights for this episode
          - "masks":   [N, D] mask
          - "labels":  [N] training label (1 = RGA should fire for this sample)

        Use this entry point when you've generated drift episodes manually;
        ``fit()`` below wraps the common case where episodes are produced from
        a perturbation engine.
        """
        if not episodes:
            raise ValueError("No drift episodes provided to fit the learned gate.")
        Xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for ep in episodes:
            X = self._featurize(ep["weights"], ep["masks"])
            Xs.append(X)
            ys.append(np.asarray(ep["labels"], dtype=int))
        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        # Need both classes present to fit a binary classifier
        n_pos = int(y_all.sum())
        n_neg = len(y_all) - n_pos
        if n_pos == 0 or n_neg == 0:
            # Degenerate: fall back to a constant decision matching the data
            self._clf = None
            self.train_label_balance_ = float(n_pos / max(len(y_all), 1))
            self.train_n_episodes_ = len(episodes)
            self.fitted = True
            return self
        self._clf = LogisticRegression(
            C=self.config.C,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            max_iter=500,
        )
        self._clf.fit(X_all, y_all)
        self.train_label_balance_ = float(n_pos / len(y_all))
        self.train_n_episodes_ = len(episodes)
        self.fitted = True
        return self

    def fit(
        self,
        val_features: np.ndarray,
        val_masks: np.ndarray,
        val_labels: np.ndarray,
        compute_reliability_weights: Callable[[np.ndarray, np.ndarray], np.ndarray],
        predict_static: Callable[[np.ndarray, np.ndarray], np.ndarray],
        predict_reliability: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        perturbation_fns: list[Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]],
    ) -> LearnedReliabilityGate:
        """Generate val drift episodes and fit the gate.

        ``perturbation_fns`` is a list of callables (features, masks) →
        (perturbed_features, perturbed_masks); each callable defines one
        drift episode (e.g. per-domain Gaussian noise at level σ).
        """
        episodes = []
        for fn in perturbation_fns:
            pert_features, pert_masks = fn(val_features, val_masks)
            weights = compute_reliability_weights(pert_features, pert_masks)
            p_static = predict_static(pert_features, pert_masks)
            p_rel = predict_reliability(pert_features, pert_masks, weights)
            y = np.asarray(val_labels, dtype=float)
            # Per-sample: RGA should fire if reliability prediction is closer
            # to ground truth than static for this sample.
            fires = (np.abs(p_rel - y) < np.abs(p_static - y)).astype(int)
            episodes.append({"weights": weights, "masks": pert_masks, "labels": fires})
        return self.fit_from_episodes(episodes)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def decide(self, weights: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Return [N] bool: True = use reliability path, False = use static path."""
        if not self.fitted:
            raise RuntimeError("Call fit() or fit_from_episodes() first.")
        if self._clf is None:
            # Degenerate fit — return majority decision constant
            n = weights.shape[0]
            constant = bool((self.train_label_balance_ or 0.0) > 0.5)
            return np.full(n, constant, dtype=bool)
        X = self._featurize(weights, masks)
        return self._clf.predict(X).astype(bool)

    def decision_probabilities(self, weights: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Return [N] float in [0,1]: probability that RGA should fire."""
        if not self.fitted:
            raise RuntimeError("Call fit() or fit_from_episodes() first.")
        if self._clf is None:
            n = weights.shape[0]
            return np.full(n, float(self.train_label_balance_ or 0.0), dtype=np.float32)
        X = self._featurize(weights, masks)
        return self._clf.predict_proba(X)[:, 1].astype(np.float32)


__all__ = ["LearnedReliabilityGate", "LearnedGateConfig"]
