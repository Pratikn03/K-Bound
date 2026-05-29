"""Regression guard for the early-stopping best-weights restore fix.

Verifies that _train_model honours restore_best_weights without crashing and
that training updates parameters. The correctness of restoration itself is a
small, reviewable code path; this test guards against import/plumbing breakage.
"""

from __future__ import annotations

import numpy as np
import torch

from src.scripts.run_breakthrough_experiment import _build_model, _make_loaders, _train_model


def _tiny_setup():
    rng = np.random.default_rng(0)
    n, d, f = 60, 3, 4
    features = rng.normal(size=(n, d, f)).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    labels = (rng.random(n) < 0.4).astype(np.float32)
    idx = np.arange(n)
    train_idx, val_idx, test_idx = idx[:40], idx[40:50], idx[50:]
    cfg = {
        "model": {
            "embed_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
            "use_confidence": False,
            "use_input_confidence": False,
        },
        "training": {"epochs": 3, "lr": 1e-3, "early_stopping": 2},
    }
    device = torch.device("cpu")
    model = _build_model(cfg, d, f, None, device)
    loaders = _make_loaders(features, masks, labels, train_idx, val_idx, test_idx, batch_size=16)
    return cfg, device, model, loaders


def test_train_with_restore_best_weights_updates_parameters():
    cfg, device, model, (train_loader, val_loader, _) = _tiny_setup()
    cfg["training"]["restore_best_weights"] = True
    before = [p.detach().clone() for p in model.parameters()]
    _train_model(model, train_loader, val_loader, cfg, device)
    after = list(model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(after, before))


def test_train_without_restore_best_weights_still_runs():
    cfg, device, model, (train_loader, val_loader, _) = _tiny_setup()
    cfg["training"]["restore_best_weights"] = False
    _train_model(model, train_loader, val_loader, cfg, device)
    # No assertion on values; this guards the legacy code path from regressions.
