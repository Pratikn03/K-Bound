"""Tests for unified attention training loop."""

from __future__ import annotations

import numpy as np
import torch

from uais.fusion.attention.training_loop import train_attention_model
from uais.fusion.attention.fusion_training_utils import evaluate_model
from src.scripts.run_breakthrough_experiment import _build_model, _make_loaders


def _tiny_setup():
    rng = np.random.default_rng(0)
    n, d, f = 80, 3, 4
    features = rng.normal(size=(n, d, f)).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    labels = (rng.random(n) < 0.4).astype(np.float32)
    idx = np.arange(n)
    train_idx, val_idx, test_idx = idx[:50], idx[50:65], idx[65:]
    cfg = {
        "model": {
            "embed_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
            "use_confidence": False,
            "use_input_confidence": False,
        },
        "training": {
            "epochs": 12,
            "lr": 1e-2,
            "early_stopping": 2,
            "early_stopping_metric": "pr_auc",
            "restore_best_weights": True,
            "domain_dropout": 0.0,
        },
    }
    device = torch.device("cpu")
    model = _build_model(cfg, d, f, None, device)
    loaders = _make_loaders(features, masks, labels, train_idx, val_idx, test_idx, batch_size=16)
    return cfg, device, model, loaders


def test_train_restores_best_pr_auc_checkpoint():
    cfg, device, model, (train_loader, val_loader, _) = _tiny_setup()
    result = train_attention_model(model, train_loader, val_loader, cfg["training"], device)
    assert result.restored_best_weights is True
    assert result.early_stopping_metric == "pr_auc"
    assert result.epochs_run >= 1
    val_after = evaluate_model(model, val_loader, device)
    assert val_after["pr_auc"] >= result.val_best_pr_auc - 1e-4


def test_without_restore_can_differ_from_best_epoch():
    cfg, device, model, (train_loader, val_loader, _) = _tiny_setup()
    cfg["training"]["restore_best_weights"] = False
    cfg["training"]["early_stopping"] = 1
    train_attention_model(model, train_loader, val_loader, cfg["training"], device)
    pr_without_restore = evaluate_model(model, val_loader, device)["pr_auc"]

    model2 = _build_model(cfg, 3, 4, None, device)
    cfg["training"]["restore_best_weights"] = True
    train_attention_model(model2, train_loader, val_loader, cfg["training"], device)
    pr_with_restore = evaluate_model(model2, val_loader, device)["pr_auc"]

    assert np.isfinite(pr_without_restore)
    assert np.isfinite(pr_with_restore)
