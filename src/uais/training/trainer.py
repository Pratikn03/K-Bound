"""Core abstractions for the unified UAIS training harness.

This module is intentionally free of any heavyweight ML dependency at import
time. It exposes three building blocks:

* :func:`set_seed` -- deterministically seed the standard library ``random``
  module, NumPy, and (when importable) PyTorch.
* :class:`TrainConfig` -- a serialisable training-run configuration with a
  :meth:`TrainConfig.from_yaml` loader.
* :class:`Trainer` -- an abstract base class whose :meth:`Trainer.run` method
  orchestrates a reproducible train/evaluate/log/persist lifecycle around the
  project's existing ``train_*.py`` entrypoints.

The harness is *additive*: concrete trainers (see :mod:`uais.training.registry`)
adapt the pre-existing training functions rather than re-implementing them.
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uais.training.tracking import BaseTracker, get_tracker, mlflow_available
from uais.utils.paths import PROJECT_ROOT

# NumPy is a core dependency of the project, but we still import defensively so
# that ``set_seed`` degrades gracefully in a stripped-down environment.
try:
    import numpy as _np

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy is a hard project dependency
    _np = None
    _NUMPY_AVAILABLE = False


def set_seed(seed: int = 42, *, deterministic: bool = True) -> int:
    """Seed all relevant RNGs for reproducible training runs.

    Seeds, in order: the ``PYTHONHASHSEED`` environment variable, the standard
    library :mod:`random`, NumPy's global RNG, and -- when PyTorch is importable
    -- the CPU and all CUDA RNGs. When ``deterministic`` is ``True`` and PyTorch
    is present, cuDNN is also placed in deterministic mode.

    Parameters
    ----------
    seed:
        The non-negative integer seed to apply across all RNGs.
    deterministic:
        When ``True`` (default), additionally request deterministic cuDNN
        behaviour from PyTorch if it is installed. Has no effect without torch.

    Returns
    -------
    int
        The ``seed`` that was applied (echoed back for convenience/logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if _NUMPY_AVAILABLE:
        _np.random.seed(seed)

    # PyTorch is optional; guard the import so this module works without it.
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # No torch installed -- random + numpy seeding is sufficient.
        pass

    return seed


@dataclass
class TrainConfig:
    """Declarative configuration for a single training run.

    Instances are typically constructed from YAML via
    :meth:`TrainConfig.from_yaml`, but may be built directly in code or tests.

    Attributes
    ----------
    name:
        Unique run/experiment name. Also selects the trainer in the registry
        unless overridden by ``trainer``.
    seed:
        RNG seed applied by :func:`set_seed` before any work begins.
    data_path:
        Path (absolute, or relative to the project root) to the input data.
        May be ``None`` for trainers that source data internally.
    output_dir:
        Directory into which artifacts and model cards are written.
    hyperparams:
        Free-form mapping of model/training hyper-parameters forwarded to the
        underlying training entrypoint.
    tracker:
        Tracking backend selector, ``"mlflow"`` or ``"json"``.
    trainer:
        Optional explicit registry key. Defaults to ``name`` when omitted.
    extra:
        Catch-all for additional YAML keys, preserved for round-tripping.
    """

    name: str
    seed: int = 42
    data_path: str | None = None
    output_dir: str = "experiments"
    hyperparams: dict[str, Any] = field(default_factory=dict)
    tracker: str = "json"
    trainer: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field types/values after dataclass initialisation.

        Raises
        ------
        ValueError
            If ``name`` is empty, ``seed`` is negative, ``tracker`` is not a
            recognised backend, or ``hyperparams``/``extra`` are not mappings.
        """
        if not self.name or not str(self.name).strip():
            raise ValueError("TrainConfig.name must be a non-empty string.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("TrainConfig.seed must be an int.")
        if self.seed < 0:
            raise ValueError("TrainConfig.seed must be non-negative.")
        if self.tracker not in {"mlflow", "json"}:
            raise ValueError(f"TrainConfig.tracker must be 'mlflow' or 'json', got {self.tracker!r}.")
        if not isinstance(self.hyperparams, dict):
            raise ValueError("TrainConfig.hyperparams must be a mapping.")
        if not isinstance(self.extra, dict):
            raise ValueError("TrainConfig.extra must be a mapping.")

    @property
    def trainer_key(self) -> str:
        """Registry key used to resolve the concrete trainer.

        Returns
        -------
        str
            ``trainer`` when set, otherwise ``name``.
        """
        return self.trainer or self.name

    def resolved_data_path(self) -> Path | None:
        """Return ``data_path`` resolved against the project root.

        Returns
        -------
        Optional[pathlib.Path]
            Absolute path to the data, or ``None`` when no ``data_path`` is
            configured. Relative paths are resolved against ``PROJECT_ROOT``.
        """
        if not self.data_path:
            return None
        path = Path(self.data_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolved_output_dir(self) -> Path:
        """Return ``output_dir`` resolved against the project root.

        Returns
        -------
        pathlib.Path
            Absolute output directory. Relative paths resolve against
            ``PROJECT_ROOT``.
        """
        path = Path(self.output_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-``dict`` representation suitable for serialisation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        """Build a :class:`TrainConfig` from a mapping.

        Recognised keys are mapped to dataclass fields; any unrecognised keys
        are collected into :attr:`extra` so that round-tripping is lossless.

        Parameters
        ----------
        data:
            Mapping of configuration values (e.g. parsed YAML).

        Returns
        -------
        TrainConfig
            The constructed, validated configuration.

        Raises
        ------
        ValueError
            If ``data`` is not a mapping or required fields are missing/invalid.
        """
        if not isinstance(data, dict):
            raise ValueError("TrainConfig.from_dict expects a mapping.")
        known = {"name", "seed", "data_path", "output_dir", "hyperparams", "tracker", "trainer", "extra"}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        extra = dict(kwargs.get("extra", {}) or {})
        for key, value in data.items():
            if key not in known:
                extra[key] = value
        if extra:
            kwargs["extra"] = extra
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load a :class:`TrainConfig` from a YAML file.

        Parameters
        ----------
        path:
            Filesystem path to a YAML document describing one training run.

        Returns
        -------
        TrainConfig
            The constructed, validated configuration.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ImportError
            If PyYAML is not installed.
        ValueError
            If the YAML root is not a mapping.
        """
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - PyYAML is a project dep
            raise ImportError("TrainConfig.from_yaml requires PyYAML ('pyyaml').") from exc

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Training config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data is None:
            raise ValueError(f"Training config is empty: {config_path}")
        if not isinstance(data, dict):
            raise ValueError(f"Training config root must be a mapping: {config_path}")
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> Path:
        """Serialise this config to a YAML file.

        Parameters
        ----------
        path:
            Destination path for the YAML document.

        Returns
        -------
        pathlib.Path
            The path that was written.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        """
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - PyYAML is a project dep
            raise ImportError("TrainConfig.to_yaml requires PyYAML ('pyyaml').") from exc

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        return out_path


class Trainer(ABC):
    """Abstract base class orchestrating a reproducible training lifecycle.

    Subclasses implement :meth:`build`, :meth:`fit`, and :meth:`evaluate`; the
    concrete :meth:`run` method wires them together with seeding, tracking,
    artifact persistence, and model-card emission. The standard flow is::

        set_seed -> build -> fit -> evaluate -> log_metrics
                 -> save_artifact -> emit_model_card

    Concrete trainers in this repository are thin adapters that delegate the
    heavy lifting to the existing ``train_*.py`` functions, so this base class
    never duplicates model logic.
    """

    def __init__(self, config: TrainConfig) -> None:
        """Initialise the trainer with its run configuration.

        Parameters
        ----------
        config:
            The :class:`TrainConfig` driving this run.
        """
        self.config = config
        self._tracker: BaseTracker | None = None
        self._model: Any = None
        self._metrics: dict[str, Any] = {}

    # --- Lifecycle hooks (implemented by subclasses) --------------------

    @abstractmethod
    def build(self) -> Any:
        """Construct and return the (untrained) model or pipeline.

        Returns
        -------
        Any
            The model/estimator object to be fitted in :meth:`fit`.
        """

    @abstractmethod
    def fit(self) -> Any:
        """Train the model. Implementations may use :attr:`config`.

        Returns
        -------
        Any
            The fitted model. Conventionally stored on the instance and used
            by :meth:`evaluate` and :meth:`save_artifact`.
        """

    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        """Evaluate the fitted model and return a metrics mapping.

        Returns
        -------
        Dict[str, Any]
            Flat mapping of metric name to value.
        """

    # --- Shared orchestration -------------------------------------------

    @property
    def tracker(self) -> BaseTracker:
        """Lazily-instantiated tracking backend for this run.

        Returns
        -------
        BaseTracker
            The tracker selected by :attr:`TrainConfig.tracker`.
        """
        if self._tracker is None:
            self._tracker = get_tracker(self.config.tracker, name=self.config.name)
        return self._tracker

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Record ``metrics`` to the active tracking backend.

        Parameters
        ----------
        metrics:
            Flat mapping of metric name to value.
        """
        self.tracker.log_metrics(metrics)

    def save_artifact(self, obj: Any, path: str | Path) -> Path:
        """Persist a Python object to ``path`` using a best-effort serialiser.

        Uses :mod:`joblib` when available (the project's standard for sklearn
        artifacts), falling back to :mod:`pickle`. Parent directories are
        created as needed.

        Parameters
        ----------
        obj:
            The object to persist (model, scaler, dict, etc.).
        path:
            Destination path. Relative paths resolve against the configured
            output directory.

        Returns
        -------
        pathlib.Path
            The path the artifact was written to.
        """
        out_path = Path(path)
        if not out_path.is_absolute():
            out_path = self.config.resolved_output_dir() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import joblib  # type: ignore

            joblib.dump(obj, out_path)
        except ImportError:  # pragma: no cover - joblib is a project dep
            import pickle

            with out_path.open("wb") as handle:
                pickle.dump(obj, handle)
        return out_path

    def emit_model_card(
        self,
        metrics: dict[str, Any] | None = None,
        artifact_path: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build and persist a MODEL_CARD-style metadata record.

        The card captures the run's identity, configuration, environment, and
        results so a trained artifact is self-describing and reproducible. It
        is written as ``model_card.json`` inside the run's output directory.

        Parameters
        ----------
        metrics:
            Metrics to embed. Defaults to the most recent :meth:`evaluate`
            results captured during :meth:`run`.
        artifact_path:
            Optional path to the saved model artifact, recorded in the card.
        extra:
            Optional additional fields merged into the card under their keys.

        Returns
        -------
        Dict[str, Any]
            The model-card dictionary that was written to disk.
        """
        metrics = metrics if metrics is not None else self._metrics
        card: dict[str, Any] = {
            "model_name": self.config.name,
            "trainer": self.config.trainer_key,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seed": self.config.seed,
            "data_path": str(self.config.resolved_data_path()) if self.config.data_path else None,
            "output_dir": str(self.config.resolved_output_dir()),
            "hyperparams": dict(self.config.hyperparams),
            "tracker": self.config.tracker,
            "metrics": dict(metrics),
            "artifact_path": str(artifact_path) if artifact_path else None,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "mlflow_available": mlflow_available(),
                "numpy_available": _NUMPY_AVAILABLE,
            },
        }
        if extra:
            card.update(extra)

        card_dir = self.config.resolved_output_dir()
        card_dir.mkdir(parents=True, exist_ok=True)
        card_path = card_dir / "model_card.json"
        with card_path.open("w", encoding="utf-8") as handle:
            json.dump(card, handle, indent=2, default=str)
        card["card_path"] = str(card_path)
        return card

    def run(self) -> dict[str, Any]:
        """Execute the full training lifecycle and return a run summary.

        Orchestrates, in order: :func:`set_seed`, :meth:`build`, :meth:`fit`,
        :meth:`evaluate`, :meth:`log_metrics`, :meth:`save_artifact` (when a
        fitted model is available), and :meth:`emit_model_card`. Parameters
        and metrics are logged to the tracking backend; the tracking run is
        always closed, even if a lifecycle step raises.

        Returns
        -------
        Dict[str, Any]
            Summary mapping with keys ``name``, ``seed``, ``metrics``,
            ``artifact_path``, ``model_card``, and ``duration_sec``.
        """
        started_at = time.time()
        set_seed(self.config.seed)

        tracker = self.tracker
        tracker.start_run()
        artifact_path: Path | None = None
        try:
            tracker.log_params(
                {
                    "seed": self.config.seed,
                    "trainer": self.config.trainer_key,
                    "data_path": self.config.data_path,
                    **{f"hp.{k}": v for k, v in self.config.hyperparams.items()},
                }
            )

            self._model = self.build()
            self._model = self.fit()
            self._metrics = dict(self.evaluate())
            self.log_metrics(self._metrics)

            if self._model is not None:
                artifact_path = self.save_artifact(self._model, "model.joblib")

            model_card = self.emit_model_card(metrics=self._metrics, artifact_path=artifact_path)
        finally:
            tracker.end_run()

        return {
            "name": self.config.name,
            "seed": self.config.seed,
            "metrics": self._metrics,
            "artifact_path": str(artifact_path) if artifact_path else None,
            "model_card": model_card,
            "duration_sec": round(time.time() - started_at, 4),
        }


__all__ = ["set_seed", "TrainConfig", "Trainer"]
