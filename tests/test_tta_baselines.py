import numpy as np

from uais.fusion.attention.baselines import (
    TentScoreAdapter,
    TTTPseudoLabelAdapter,
    run_baseline_suite,
)


def _toy_fusion_data(n=96, d=3, f=5):
    rng = np.random.default_rng(7)
    labels = rng.binomial(1, 0.45, size=n)
    features = rng.normal(0.45, 0.12, size=(n, d, f)).astype(np.float32)
    features[:, :, 0] = np.clip(0.2 + labels[:, None] * 0.55 + rng.normal(0, 0.08, size=(n, d)), 0, 1)
    features[:, :, 1] = np.clip(2.0 * np.abs(features[:, :, 0] - 0.5), 0, 1)
    masks = rng.random((n, d)) < 0.08
    return features, masks, labels


def test_tent_and_ttt_adapters_return_finite_probabilities():
    features, masks, labels = _toy_fusion_data()
    train_feat, train_mask, train_y = features[:60], masks[:60], labels[:60]
    test_feat, test_mask = features[60:], masks[60:]

    for adapter_cls in (TentScoreAdapter, TTTPseudoLabelAdapter):
        adapter = adapter_cls(random_seed=11, adaptation_steps=2)
        adapter.fit(train_feat, train_mask, train_y)
        probs = adapter.predict_proba(test_feat, test_mask)
        assert probs.shape == (len(test_feat),)
        assert np.isfinite(probs).all()
        assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_baseline_suite_includes_tent_and_ttt_metrics():
    features, masks, labels = _toy_fusion_data()
    train_idx = np.arange(0, 50)
    val_idx = np.arange(50, 70)
    test_idx = np.arange(70, 96)

    metrics = run_baseline_suite(
        features,
        masks,
        labels,
        train_idx,
        val_idx,
        test_idx,
        random_seed=5,
        baseline_epochs=2,
        tta_steps=2,
    )

    assert "tent_score_adapter" in metrics
    assert "ttt_pseudo_label_adapter" in metrics
    assert np.isfinite(metrics["tent_score_adapter"]["roc_auc"])
    assert np.isfinite(metrics["ttt_pseudo_label_adapter"]["roc_auc"])
