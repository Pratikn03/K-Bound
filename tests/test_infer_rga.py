"""Smoke tests for the infer_rga runtime-only package.

These tests do *not* train a real model on a real benchmark — they
construct a small fusion model + a small fitted reliability estimator
in-memory, dump them to disk via the standard checkpoint/save APIs,
load them through ``InferRGA.from_checkpoint``, and verify that
inference shapes, types, and value ranges are sane.

This is the minimum honest test that the package's deployment-time
surface (load + predict_proba + predict_with_gate + reliability)
works end-to-end with the trained-model artefacts the research
codebase actually produces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from infer_rga import InferRGA
from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DOMAINS = 2
INPUT_DIM = 5
N_SAMPLES = 32


def _make_synthetic_fit_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.random((N_SAMPLES, N_DOMAINS, INPUT_DIM)).astype(np.float32)
    labels = (rng.random(N_SAMPLES) < 0.3).astype(np.float32)
    masks = np.zeros((N_SAMPLES, N_DOMAINS), dtype=bool)
    return features, masks, labels


@pytest.fixture
def trained_artifacts(tmp_path: Path):
    """Build a tiny fusion model + fitted reliability estimator on disk."""
    model = AttentionFusionModel(
        num_domains=N_DOMAINS,
        input_dim=INPUT_DIM,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        use_confidence=True,
        use_input_confidence=True,
        confidence_index=0,
    )

    # Tiny training loop so the weights aren't pathological.
    features, masks, labels = _make_synthetic_fit_data(seed=0)
    feat_t = torch.tensor(features)
    mask_t = torch.tensor(masks)
    lbl_t = torch.tensor(labels)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        logits, _, _ = model(feat_t, key_padding_mask=mask_t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(-1), lbl_t
        )
        opt.zero_grad(); loss.backward(); opt.step()

    model_path = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_args": {
                "num_domains": N_DOMAINS,
                "input_dim": INPUT_DIM,
                "embed_dim": 16,
                "num_heads": 2,
                "num_layers": 1,
                "use_confidence": True,
                "use_input_confidence": True,
                "confidence_index": 0,
            },
        },
        model_path,
    )

    estimator = ReliabilityEstimator(
        domain_order=["dom0", "dom1"],
        score_index=0,
        ece_weight=0.4,
        ks_weight=0.4,
        sharpness_weight=0.2,
        n_calibration_bins=5,
        min_samples_for_ks=10,
    )
    estimator.fit(features, masks, labels)
    estimator_path = tmp_path / "reliability.joblib"
    estimator.save(estimator_path)

    return model_path, estimator_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_from_checkpoint_loads(trained_artifacts):
    model_path, estimator_path = trained_artifacts
    rga = InferRGA.from_checkpoint(model_path=model_path, estimator_path=estimator_path)
    assert rga.metadata.num_domains == N_DOMAINS
    assert rga.metadata.input_dim == INPUT_DIM
    assert rga.metadata.estimator_kind in ("ReliabilityEstimator", "PerSampleReliabilityEstimator")


def test_predict_proba_shape_and_range(trained_artifacts):
    model_path, estimator_path = trained_artifacts
    rga = InferRGA.from_checkpoint(model_path=model_path, estimator_path=estimator_path)
    features = np.random.rand(8, N_DOMAINS, INPUT_DIM).astype(np.float32)
    masks = np.zeros((8, N_DOMAINS), dtype=bool)
    probs = rga.predict_proba(features, masks)
    assert probs.shape == (8,)
    assert probs.dtype == np.float32
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0


def test_predict_with_gate_returns_observe_only_diagnostics(trained_artifacts):
    model_path, estimator_path = trained_artifacts
    rga = InferRGA.from_checkpoint(model_path=model_path, estimator_path=estimator_path)
    features = np.random.rand(16, N_DOMAINS, INPUT_DIM).astype(np.float32)
    masks = np.zeros((16, N_DOMAINS), dtype=bool)
    out = rga.predict_with_gate(features, masks)
    assert set(out.keys()) == {"static_probs", "mean_reliability", "gate_fired"}
    assert out["static_probs"].shape == (16,)
    assert out["mean_reliability"].shape == (16,)
    assert out["gate_fired"].shape == (16,)
    assert out["gate_fired"].dtype == bool
    assert float(out["mean_reliability"].min()) >= 0.0
    assert float(out["mean_reliability"].max()) <= 1.0


def test_reliability_returns_per_domain_array(trained_artifacts):
    model_path, estimator_path = trained_artifacts
    rga = InferRGA.from_checkpoint(model_path=model_path, estimator_path=estimator_path)
    features = np.random.rand(4, N_DOMAINS, INPUT_DIM).astype(np.float32)
    masks = np.zeros((4, N_DOMAINS), dtype=bool)
    rel = rga.reliability(features, masks)
    assert rel.shape == (4, N_DOMAINS)
    assert float(rel.min()) >= 0.0
    assert float(rel.max()) <= 1.0


def test_predict_proba_no_estimator(trained_artifacts):
    """Inference should still work if the estimator is omitted."""
    model_path, _ = trained_artifacts
    rga = InferRGA.from_checkpoint(model_path=model_path, estimator_path=None)
    features = np.random.rand(4, N_DOMAINS, INPUT_DIM).astype(np.float32)
    masks = np.zeros((4, N_DOMAINS), dtype=bool)
    probs = rga.predict_proba(features, masks)
    assert probs.shape == (4,)
    # predict_with_gate must still work, gate-fired must all be False
    out = rga.predict_with_gate(features, masks)
    assert not out["gate_fired"].any()


def test_reliability_without_estimator_raises(trained_artifacts):
    model_path, _ = trained_artifacts
    rga = InferRGA.from_checkpoint(model_path=model_path, estimator_path=None)
    with pytest.raises(RuntimeError):
        rga.reliability(
            np.zeros((1, N_DOMAINS, INPUT_DIM), dtype=np.float32),
            np.zeros((1, N_DOMAINS), dtype=bool),
        )
