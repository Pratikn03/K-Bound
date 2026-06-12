"""Unified training harness for the UAIS / K-Bound anomaly-intelligence system.

This package provides a thin, reproducible orchestration layer *around* the
project's existing ``train_*.py`` scripts. It standardises seeding, experiment
tracking, artifact persistence, and model-card emission, while delegating all
model logic to the original training entrypoints via registry adapters.

Public API
----------
* :class:`Trainer` -- abstract lifecycle base class (build/fit/evaluate/run).
* :class:`TrainConfig` -- YAML-loadable run configuration.
* :func:`set_seed` -- deterministic multi-RNG seeding.
* :func:`get_tracker` -- MLflow-or-JSON tracking backend factory.

The registry adapters (in :mod:`uais.training.registry`) are imported here so
that registration happens on package import and trainers are immediately
resolvable via :func:`uais.training.registry.get_trainer`.

Importing this package never requires PyTorch, TensorFlow, or MLflow: those
dependencies are imported lazily only when a trainer that needs them is run, or
fall back gracefully when absent.
"""

from __future__ import annotations

# Importing the registry triggers registration of all built-in adapters as a
# side effect. ``available_trainers``/``get_trainer``/``register`` are also
# re-exported for convenience.
from uais.training.registry import (  # noqa: E402  (intentional: registration side effect)
    TRAINERS,
    available_trainers,
    get_trainer,
    register,
)
from uais.training.tracking import (
    BaseTracker,
    JSONTracker,
    MLflowTracker,
    get_tracker,
    mlflow_available,
)
from uais.training.trainer import TrainConfig, Trainer, set_seed

__all__ = [
    # Core
    "Trainer",
    "TrainConfig",
    "set_seed",
    # Tracking
    "get_tracker",
    "BaseTracker",
    "JSONTracker",
    "MLflowTracker",
    "mlflow_available",
    # Registry
    "TRAINERS",
    "register",
    "get_trainer",
    "available_trainers",
]
