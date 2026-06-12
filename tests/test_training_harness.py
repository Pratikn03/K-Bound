"""Tests for the unified UAIS training harness (``uais.training``).

These tests require only NumPy and pytest -- no PyTorch, TensorFlow, or MLflow.
They exercise the dependency-light core of the harness:

* :func:`~uais.training.set_seed` determinism,
* :class:`~uais.training.TrainConfig` YAML round-trip and validation,
* :class:`~uais.training.tracking.JSONTracker` on-disk behaviour, and
* :meth:`~uais.training.trainer.Trainer.emit_model_card` required keys,

plus a full no-op lifecycle run that touches every orchestration step without
any heavy training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from uais.training import (
    TrainConfig,
    Trainer,
    get_tracker,
    set_seed,
)
from uais.training.tracking import JSONTracker
from uais.training.trainer import set_seed as trainer_set_seed

# ---------------------------------------------------------------------------
# set_seed determinism
# ---------------------------------------------------------------------------


def test_set_seed_makes_numpy_draws_reproducible() -> None:
    """Two identical seeds must produce identical NumPy draws."""
    set_seed(123)
    first = np.random.rand(10)
    set_seed(123)
    second = np.random.rand(10)
    assert np.array_equal(first, second)


def test_set_seed_differs_across_seeds() -> None:
    """Different seeds must (with overwhelming probability) differ."""
    set_seed(1)
    a = np.random.rand(10)
    set_seed(2)
    b = np.random.rand(10)
    assert not np.array_equal(a, b)


def test_set_seed_returns_seed_and_seeds_python_random() -> None:
    """`set_seed` echoes its seed and also seeds the stdlib RNG."""
    import random

    returned = set_seed(7)
    assert returned == 7
    set_seed(7)
    x = random.random()
    set_seed(7)
    y = random.random()
    assert x == y


def test_set_seed_reexport_identity() -> None:
    """The package-level and module-level `set_seed` are the same callable."""
    assert set_seed is trainer_set_seed


# ---------------------------------------------------------------------------
# TrainConfig: round-trip + validation
# ---------------------------------------------------------------------------


def test_trainconfig_from_yaml_round_trip(tmp_path: Path) -> None:
    """A config written to YAML reloads with identical field values."""
    cfg = TrainConfig(
        name="demo",
        seed=42,
        data_path="data/processed/demo.parquet",
        output_dir="experiments/demo",
        hyperparams={"contamination": 0.05, "target": "label"},
        tracker="json",
    )
    yaml_path = tmp_path / "demo.yaml"
    cfg.to_yaml(yaml_path)

    loaded = TrainConfig.from_yaml(yaml_path)
    assert loaded.name == cfg.name
    assert loaded.seed == cfg.seed
    assert loaded.data_path == cfg.data_path
    assert loaded.output_dir == cfg.output_dir
    assert loaded.hyperparams == cfg.hyperparams
    assert loaded.tracker == cfg.tracker


def test_trainconfig_from_dict_preserves_unknown_keys() -> None:
    """Unrecognised top-level keys land in `extra` (lossless round-trip)."""
    cfg = TrainConfig.from_dict({"name": "x", "notes": "hello", "owner": "team"})
    assert cfg.extra["notes"] == "hello"
    assert cfg.extra["owner"] == "team"


def test_trainconfig_trainer_key_defaults_to_name() -> None:
    """`trainer_key` falls back to `name` when `trainer` is unset."""
    assert TrainConfig(name="iforest").trainer_key == "iforest"
    assert TrainConfig(name="iforest", trainer="isolation_forest").trainer_key == "isolation_forest"


def test_trainconfig_rejects_empty_name() -> None:
    """An empty name fails validation."""
    with pytest.raises(ValueError):
        TrainConfig(name="")


def test_trainconfig_rejects_negative_seed() -> None:
    """A negative seed fails validation."""
    with pytest.raises(ValueError):
        TrainConfig(name="ok", seed=-1)


def test_trainconfig_rejects_unknown_tracker() -> None:
    """An unrecognised tracker value fails validation."""
    with pytest.raises(ValueError):
        TrainConfig(name="ok", tracker="wandb")


def test_trainconfig_from_yaml_missing_file(tmp_path: Path) -> None:
    """Loading a non-existent YAML path raises `FileNotFoundError`."""
    with pytest.raises(FileNotFoundError):
        TrainConfig.from_yaml(tmp_path / "does_not_exist.yaml")


def test_trainconfig_resolved_paths_absolute(tmp_path: Path) -> None:
    """Absolute data/output paths are returned unchanged."""
    abs_data = tmp_path / "d.parquet"
    abs_out = tmp_path / "out"
    cfg = TrainConfig(name="x", data_path=str(abs_data), output_dir=str(abs_out))
    assert cfg.resolved_data_path() == abs_data
    assert cfg.resolved_output_dir() == abs_out


# ---------------------------------------------------------------------------
# JSONTracker
# ---------------------------------------------------------------------------


def test_json_tracker_writes_metrics(tmp_path: Path) -> None:
    """JSONTracker persists params + metrics to runs/<name>/metrics.json."""
    tracker = JSONTracker(name="unit", runs_dir=tmp_path)
    tracker.start_run()
    tracker.log_params({"seed": 42, "lr": 0.01})
    tracker.log_metrics({"roc_auc": 0.91, "n_samples": 100})
    tracker.end_run()

    metrics_file = tmp_path / "unit" / "metrics.json"
    assert metrics_file.exists()
    records = json.loads(metrics_file.read_text())
    assert isinstance(records, list) and len(records) == 1
    record = records[0]
    assert record["name"] == "unit"
    assert record["params"]["seed"] == 42
    assert record["metrics"]["roc_auc"] == 0.91
    assert "timestamp" in record


def test_json_tracker_appends_history(tmp_path: Path) -> None:
    """Multiple runs accumulate as separate records in the same file."""
    for i in range(3):
        tracker = JSONTracker(name="hist", runs_dir=tmp_path)
        tracker.start_run()
        tracker.log_metrics({"run_index": i})
        tracker.end_run()
    records = json.loads((tmp_path / "hist" / "metrics.json").read_text())
    assert len(records) == 3
    assert [r["metrics"]["run_index"] for r in records] == [0, 1, 2]


def test_json_tracker_context_manager(tmp_path: Path) -> None:
    """JSONTracker works as a context manager (start/end bracketing)."""
    with JSONTracker(name="ctx", runs_dir=tmp_path) as tracker:
        tracker.log_metrics({"x": 1})
    records = json.loads((tmp_path / "ctx" / "metrics.json").read_text())
    assert records[0]["metrics"]["x"] == 1


def test_get_tracker_falls_back_to_json(tmp_path: Path) -> None:
    """Requesting 'mlflow' without mlflow installed yields a JSONTracker."""
    tracker = get_tracker("mlflow", name="fallback", runs_dir=tmp_path)
    # In a torch/mlflow-free CI environment this must be the JSON backend.
    from uais.training.tracking import mlflow_available

    if not mlflow_available():
        assert isinstance(tracker, JSONTracker)


# ---------------------------------------------------------------------------
# Trainer lifecycle + model card
# ---------------------------------------------------------------------------


class _DummyTrainer(Trainer):
    """Minimal concrete trainer used to exercise the lifecycle without ML deps."""

    def build(self) -> dict[str, Any]:
        """Return a trivial 'model' object."""
        return {"built": True}

    def fit(self) -> dict[str, Any]:
        """Return a trivial fitted 'model' (a dict, joblib-serialisable)."""
        return {"weights": [1, 2, 3], "seed": self.config.seed}

    def evaluate(self) -> dict[str, Any]:
        """Return deterministic dummy metrics."""
        return {"accuracy": 0.99, "n_samples": 42}


def _dummy_config(tmp_path: Path) -> TrainConfig:
    """Build a TrainConfig whose output_dir lives under tmp_path."""
    return TrainConfig(
        name="dummy",
        seed=42,
        data_path=None,
        output_dir=str(tmp_path / "out"),
        hyperparams={"alpha": 0.1},
        tracker="json",
    )


def test_emit_model_card_has_required_keys(tmp_path: Path) -> None:
    """The emitted model card contains all required provenance keys."""
    trainer = _DummyTrainer(_dummy_config(tmp_path))
    card = trainer.emit_model_card(metrics={"accuracy": 0.99})

    required = {
        "model_name",
        "trainer",
        "created_at",
        "seed",
        "hyperparams",
        "tracker",
        "metrics",
        "environment",
    }
    assert required.issubset(card.keys())
    assert card["model_name"] == "dummy"
    assert card["seed"] == 42
    assert card["metrics"]["accuracy"] == 0.99
    assert "python" in card["environment"]

    # Card is also written to disk.
    card_file = Path(card["card_path"])
    assert card_file.exists()
    on_disk = json.loads(card_file.read_text())
    assert on_disk["model_name"] == "dummy"


def test_trainer_run_full_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`Trainer.run()` seeds, fits, evaluates, logs, saves, and emits a card."""
    # Route the JSON tracker's runs/ dir under tmp_path to avoid touching repo.
    import uais.training.tracking as tracking_mod

    monkeypatch.setattr(tracking_mod, "PROJECT_ROOT", tmp_path, raising=True)

    trainer = _DummyTrainer(_dummy_config(tmp_path))
    summary = trainer.run()

    assert summary["name"] == "dummy"
    assert summary["seed"] == 42
    assert summary["metrics"]["accuracy"] == 0.99
    # Artifact persisted.
    assert summary["artifact_path"] is not None
    assert Path(summary["artifact_path"]).exists()
    # Model card present with metrics.
    assert summary["model_card"]["metrics"]["accuracy"] == 0.99
    # Tracker wrote its metrics file under the patched runs/ dir.
    metrics_file = tmp_path / "runs" / "dummy" / "metrics.json"
    assert metrics_file.exists()


def test_registry_has_expected_trainers() -> None:
    """The built-in adapters register under their documented keys."""
    from uais.training import available_trainers

    names = set(available_trainers())
    for expected in {"isolation_forest", "lof", "ocsvm", "autoencoder", "fusion_meta", "vae"}:
        assert expected in names


def test_get_trainer_unknown_raises() -> None:
    """Resolving an unregistered trainer name raises KeyError."""
    from uais.training import get_trainer

    with pytest.raises(KeyError):
        get_trainer("not_a_real_trainer", TrainConfig(name="x"))
