"""Flagship RGA+ variants: validation-selected TTA + optional category features."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from uais.fusion.attention.reliability_boosted_fusion import (
    ReliabilityBoostedFusion,
    _probability,
    _safe_auc,
    _selection_score,
    _validation_metrics,
)

__all__ = [
    "ReliabilityBoostedFusionFlagship",
    "FLAGSHIP_RGA_VARIANTS",
]


def _probs_to_logits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _apply_entropy_tta(
    base_logits: np.ndarray,
    *,
    adaptation_steps: int,
    lr: float = 0.01,
    l2_anchor: float = 0.005,
) -> np.ndarray:
    if adaptation_steps <= 0:
        p = np.asarray(base_logits, dtype=np.float64)
        return (1.0 / (1.0 + np.exp(-p))).astype(np.float32)

    base_t = torch.tensor(base_logits, dtype=torch.float32)
    log_temperature = torch.zeros((), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature, bias], lr=lr)

    for _ in range(int(adaptation_steps)):
        optimizer.zero_grad()
        logits = base_t / torch.exp(log_temperature).clamp_min(1e-3) + bias
        probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs)).mean()
        anchor = l2_anchor * (log_temperature.pow(2) + bias.pow(2))
        (entropy + anchor).backward()
        optimizer.step()

    with torch.no_grad():
        logits = base_t / torch.exp(log_temperature).clamp_min(1e-3) + bias
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def _category_matrix(
    categories: np.ndarray | None,
    vocabulary: dict[str, int],
    n_samples: int,
) -> np.ndarray:
    if not vocabulary:
        return np.zeros((n_samples, 0), dtype=np.float32)
    out = np.zeros((n_samples, len(vocabulary)), dtype=np.float32)
    if categories is None:
        return out
    cats = np.asarray(categories).astype(str)
    for i, cat in enumerate(cats):
        idx = vocabulary.get(cat)
        if idx is not None:
            out[i, idx] = 1.0
    return out


class ReliabilityBoostedFusionFlagship(ReliabilityBoostedFusion):
    """RGA+ with optional category one-hot features and val-selected entropy TTA."""

    def __init__(
        self,
        score_index: int = 0,
        confidence_index: int | None = None,
        random_seed: int = 42,
        selection_metric: str = "roc_auc",
        *,
        use_category_features: bool = False,
        tta_candidates: Sequence[int] = (0, 10, 25),
        tta_lr: float = 0.01,
        tta_l2_anchor: float = 0.005,
    ) -> None:
        super().__init__(
            score_index=score_index,
            confidence_index=confidence_index,
            random_seed=random_seed,
            selection_metric=selection_metric,
        )
        self.use_category_features = bool(use_category_features)
        self.tta_candidates = tuple(int(v) for v in tta_candidates)
        self.tta_lr = float(tta_lr)
        self.tta_l2_anchor = float(tta_l2_anchor)
        self.selected_tta_steps: int = 0
        self._category_vocabulary: dict[str, int] = {}
        self._train_categories: np.ndarray | None = None

    def _features(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        categories: np.ndarray | None = None,
    ) -> np.ndarray:
        base = super()._features(features, masks)
        if not self.use_category_features or not self._category_vocabulary:
            return base
        cat_block = _category_matrix(categories, self._category_vocabulary, features.shape[0])
        if cat_block.shape[1] == 0:
            return base
        return np.concatenate([base, cat_block], axis=1)

    def fit(
        self,
        train_features: np.ndarray,
        train_masks: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_masks: np.ndarray,
        val_labels: np.ndarray,
        reliability_estimator=None,
        train_categories: np.ndarray | None = None,
        val_categories: np.ndarray | None = None,
    ) -> ReliabilityBoostedFusionFlagship:
        self._train_categories = train_categories
        if self.use_category_features and train_categories is not None:
            uniq = sorted(set(np.asarray(train_categories).astype(str).tolist()))
            self._category_vocabulary = {c: i for i, c in enumerate(uniq)}
        else:
            self._category_vocabulary = {}

        train_labels = np.asarray(train_labels, dtype=np.int64)
        val_labels = np.asarray(val_labels, dtype=np.int64)
        self._reliability_estimator = reliability_estimator

        if len(np.unique(train_labels)) < 2:
            self._constant_prediction = float(np.mean(train_labels)) if len(train_labels) else 0.5
            self.selected_candidate = "constant"
            self.candidate_validation_auc = {"constant": 0.5}
            self.candidate_validation_metrics = {
                "constant": {"roc_auc": 0.5, "pr_auc": 0.0, "f1": 0.0, "brier": 0.0}
            }
            self.selected_tta_steps = 0
            return self

        x_train = self._features(train_features, train_masks, train_categories)
        x_val = self._features(val_features, val_masks, val_categories)
        best_name = None
        best_score = -np.inf
        best_model = None
        self.candidate_validation_auc = {}
        self.candidate_validation_metrics = {}

        for candidate in self._candidate_grid():
            model = candidate.factory()
            model.fit(x_train, train_labels)
            val_probs = _probability(model, x_val)
            metrics = _validation_metrics(val_labels, val_probs)
            score = _selection_score(metrics, self.selection_metric)
            self.candidate_validation_auc[candidate.name] = metrics["roc_auc"]
            self.candidate_validation_metrics[candidate.name] = metrics
            if score > best_score or (np.isclose(score, best_score) and candidate.name < str(best_name)):
                best_score = score
                best_name = candidate.name
                best_model = model

        self._model = best_model
        self.selected_candidate = best_name
        self._constant_prediction = None

        val_base = self.predict_proba(val_features, val_masks, categories=val_categories, _skip_tta=True)
        best_tta = 0
        best_auc = _safe_auc(val_labels, val_base)
        for steps in self.tta_candidates:
            logits = _probs_to_logits(val_base)
            adapted = _apply_entropy_tta(
                logits,
                adaptation_steps=steps,
                lr=self.tta_lr,
                l2_anchor=self.tta_l2_anchor,
            )
            auc = _safe_auc(val_labels, adapted)
            if auc >= best_auc:
                best_auc = float(auc)
                best_tta = int(steps)
        self.selected_tta_steps = best_tta
        return self

    def predict_proba(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        categories: np.ndarray | None = None,
        _skip_tta: bool = False,
    ) -> np.ndarray:
        if self._constant_prediction is not None:
            return np.full(features.shape[0], self._constant_prediction, dtype=np.float32)
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        base = _probability(self._model, self._features(features, masks, categories))
        if _skip_tta or self.selected_tta_steps <= 0:
            return base
        return _apply_entropy_tta(
            _probs_to_logits(base),
            adaptation_steps=self.selected_tta_steps,
            lr=self.tta_lr,
            l2_anchor=self.tta_l2_anchor,
        )


FLAGSHIP_RGA_VARIANTS: dict[str, dict] = {
    "rga_plus_baseline": {
        "use_category_features": False,
        "tta_candidates": (0,),
    },
    "rga_plus_tta": {
        "use_category_features": False,
        "tta_candidates": (0, 10, 25),
    },
    "rga_plus_category": {
        "use_category_features": True,
        "tta_candidates": (0,),
    },
    "rga_plus_tta_category": {
        "use_category_features": True,
        "tta_candidates": (0, 10, 25),
    },
}
