from __future__ import annotations

import numpy as np

from uais.fusion.attention.reliability_boosted_fusion import ReliabilityBoostedFusion, _Candidate


class _FixedProbabilityModel:
    def __init__(self, probs: np.ndarray) -> None:
        self.probs = np.asarray(probs, dtype=np.float32)
        self.classes_ = np.array([0, 1])

    def fit(self, _features, _labels):
        return self

    def predict_proba(self, features):
        probs = self.probs[: len(features)]
        return np.column_stack([1.0 - probs, probs])


class _FixedGridReliabilityBoostedFusion(ReliabilityBoostedFusion):
    def __init__(self, candidate_probs: dict[str, np.ndarray], **kwargs) -> None:
        super().__init__(**kwargs)
        self.candidate_probs = candidate_probs

    def _candidate_grid(self) -> list[_Candidate]:
        return [
            _Candidate(name=name, factory=lambda probs=probs: _FixedProbabilityModel(probs))
            for name, probs in self.candidate_probs.items()
        ]


def test_reliability_boosted_fusion_selects_validation_candidate():
    rng = np.random.default_rng(7)
    train_features = rng.normal(size=(80, 2, 3)).astype(np.float32)
    train_masks = np.zeros((80, 2), dtype=bool)
    train_labels = (train_features[:, 0, 0] + 0.8 * train_features[:, 1, 0] > 0.15).astype(np.int64)
    val_features = rng.normal(size=(40, 2, 3)).astype(np.float32)
    val_masks = np.zeros((40, 2), dtype=bool)
    val_labels = (val_features[:, 0, 0] + 0.8 * val_features[:, 1, 0] > 0.15).astype(np.int64)

    model = ReliabilityBoostedFusion(random_seed=7).fit(
        train_features,
        train_masks,
        train_labels,
        val_features,
        val_masks,
        val_labels,
    )

    probs = model.predict_proba(val_features, val_masks)

    assert probs.shape == (40,)
    assert model.selected_candidate
    assert model.candidate_validation_auc[model.selected_candidate] >= 0.5


def test_reliability_boosted_fusion_handles_single_class_training_fold():
    features = np.full((12, 2, 3), 0.25, dtype=np.float32)
    masks = np.zeros((12, 2), dtype=bool)
    labels = np.zeros(12, dtype=np.int64)

    model = ReliabilityBoostedFusion(random_seed=1).fit(
        features,
        masks,
        labels,
        features,
        masks,
        labels,
    )

    assert np.allclose(model.predict_proba(features, masks), 0.0)


def test_reliability_boosted_fusion_honors_selection_metric():
    train_features = np.zeros((6, 2, 3), dtype=np.float32)
    train_masks = np.zeros((6, 2), dtype=bool)
    train_labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    val_features = np.zeros((6, 2, 3), dtype=np.float32)
    val_masks = np.zeros((6, 2), dtype=bool)
    val_labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    candidate_probs = {
        "weak": np.array([0.2, 0.3, 0.4, 0.9, 0.8, 0.7]),
        "strong_pr": np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05]),
    }

    model = _FixedGridReliabilityBoostedFusion(
        candidate_probs,
        selection_metric="pr_auc",
        random_seed=7,
    ).fit(
        train_features,
        train_masks,
        train_labels,
        val_features,
        val_masks,
        val_labels,
    )

    assert model.selected_candidate == "strong_pr"
    assert model.candidate_validation_metrics["strong_pr"]["pr_auc"] == 1.0
