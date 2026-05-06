"""Tests for CounterfactualDomainExplainer (CDA component of CRAF)."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest
import torch

from uais.fusion.attention.counterfactual_explainer import (
    CounterfactualDomainExplainer,
    CounterfactualResult,
)
from uais.fusion.attention.cross_modal_attention import AttentionFusionModel


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

N_DOMAINS = 3
INPUT_DIM = 5
EMBED_DIM = 16
DOMAIN_ORDER = ["fraud", "cyber", "behavior"]


def _make_model() -> AttentionFusionModel:
    """Build a minimal (untrained) AttentionFusionModel for deterministic testing."""
    model = AttentionFusionModel(
        num_domains=N_DOMAINS,
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        use_confidence=False,
        use_input_confidence=False,
        confidence_index=None,
        use_domain_embeddings=True,
        use_positional_embeddings=True,
        use_missing_embedding=True,
    )
    model.eval()
    return model


def _make_explainer(use_craf: bool = False):
    model = _make_model()
    return CounterfactualDomainExplainer(
        model=model,
        domain_order=DOMAIN_ORDER,
        device=torch.device("cpu"),
        reliability_estimator=None,
        use_craf_weights=use_craf,
    ), model


def _sample_features(n: int = 4, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    features = rng.random((n, N_DOMAINS, INPUT_DIM)).astype(np.float32)
    masks = np.zeros((n, N_DOMAINS), dtype=bool)
    return features, masks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_explain_single_sample_shapes():
    explainer, _ = _make_explainer()
    features, masks = _sample_features(n=1)
    results = explainer.explain(features, masks, sample_ids=[42])

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, CounterfactualResult)
    assert r.sample_id == 42
    assert len(r.cf_impacts) == N_DOMAINS, "Should have one impact per domain"
    assert len(r.cf_impacts_pct) == N_DOMAINS
    assert set(r.cf_impacts.keys()) == set(DOMAIN_ORDER)


def test_cf_impact_bounds():
    """All finite CF impacts must satisfy |delta| <= 1.0 (probabilities are in [0,1])."""
    explainer, _ = _make_explainer()
    features, masks = _sample_features(n=8)
    results = explainer.explain(features, masks)

    for r in results:
        for domain, delta in r.cf_impacts.items():
            if math.isfinite(delta):
                assert abs(delta) <= 1.0 + 1e-6, (
                    f"CF impact for {domain} out of bounds: {delta:.4f}"
                )


def test_fully_masked_domain_returns_nan():
    """A domain that is already missing should produce nan impact and percentage."""
    explainer, _ = _make_explainer()
    features, masks = _sample_features(n=1)
    masks[0, 1] = True  # mask out "cyber"

    results = explainer.explain(features, masks)
    r = results[0]

    assert math.isnan(r.cf_impacts["cyber"]), "Missing domain should have nan impact"
    assert math.isnan(r.cf_impacts_pct["cyber"]), "Missing domain should have nan pct"
    # Other domains should still have finite impacts
    for domain in ["fraud", "behavior"]:
        assert math.isfinite(r.cf_impacts[domain]), f"{domain} should have finite impact"


def test_narrative_contains_domain_name_and_percent():
    """Narrative must reference each domain name and include '%' symbol."""
    explainer, _ = _make_explainer()
    features, masks = _sample_features(n=1)
    results = explainer.explain(features, masks)
    r = results[0]

    for domain in DOMAIN_ORDER:
        assert domain in r.narrative, f"Domain '{domain}' not found in narrative"
    assert "%" in r.narrative, "Narrative must contain a percentage"


def test_explain_batch_length():
    """explain_batch must return exactly n_samples results."""
    explainer, _ = _make_explainer()
    n = 20
    features, masks = _sample_features(n=n)
    results = explainer.explain_batch(features, masks, batch_size=7)

    assert len(results) == n, f"Expected {n} results, got {len(results)}"


def test_correlation_with_shap_finite():
    """correlation_with_shap must return a finite float when enough domains overlap."""
    explainer, _ = _make_explainer()
    features, masks = _sample_features(n=10)
    results = explainer.explain(features, masks)

    # Fake SHAP importances for all three domains
    shap_importances = {"fraud": 0.5, "cyber": 0.3, "behavior": 0.2}
    corr = explainer.correlation_with_shap(results, shap_importances)

    # Should be a float (nan is allowed only if fewer than 3 domains overlap)
    assert isinstance(corr, float)
    # With 3 domains all present, result should be finite
    assert math.isfinite(corr), f"Expected finite correlation, got {corr}"
    assert -1.0 <= corr <= 1.0, f"Spearman correlation out of [-1, 1]: {corr}"


def test_single_sample_2d_input():
    """explain() must accept 2-D inputs [D, F] for a single sample."""
    explainer, _ = _make_explainer()
    rng = np.random.default_rng(7)
    features_2d = rng.random((N_DOMAINS, INPUT_DIM)).astype(np.float32)
    masks_1d = np.zeros(N_DOMAINS, dtype=bool)

    results = explainer.explain(features_2d, masks_1d)
    assert len(results) == 1
    r = results[0]
    assert len(r.cf_impacts) == N_DOMAINS
