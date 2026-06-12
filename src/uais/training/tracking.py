"""Experiment-tracking backends for the unified UAIS training harness.

This module provides a tiny, dependency-light tracking abstraction with two
interchangeable backends:

* :class:`MLflowTracker` -- used automatically when the optional ``mlflow``
  package is importable. Runs, params, and metrics are logged to the active
  MLflow tracking store.
* :class:`JSONTracker` -- a zero-dependency fallback that appends run records
  to ``runs/<name>/metrics.json`` under the project root. This guarantees the
  harness is fully functional (and tested) without any heavyweight optional
  dependency installed.

Both backends implement the identical :class:`BaseTracker` interface so callers
never need to branch on which one is active. Use :func:`get_tracker` to obtain
the appropriate backend for a desired ``kind``.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from uais.utils.paths import PROJECT_ROOT

# ``mlflow`` is an *optional* dependency. We detect availability lazily and
# never hard-fail if it is missing; the JSON fallback is always available.
try:  # pragma: no cover - exercised only when mlflow is installed
    import mlflow as _mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - the common, dependency-free path
    _mlflow = None
    _MLFLOW_AVAILABLE = False


def mlflow_available() -> bool:
    """Return ``True`` when the optional ``mlflow`` dependency is importable.

    Returns
    -------
    bool
        Whether MLflow can be used as a tracking backend in this environment.
    """
    return _MLFLOW_AVAILABLE


class BaseTracker(ABC):
    """Abstract base class defining the experiment-tracking interface.

    Concrete trackers manage the lifecycle of a single *run*: a logical unit
    of work bracketed by :meth:`start_run` / :meth:`end_run`, during which
    parameters and metrics may be recorded.
    """

    def __init__(self, name: str) -> None:
        """Initialise the tracker for a named experiment.

        Parameters
        ----------
        name:
            Human-readable experiment/run name. Used by the JSON backend to
            choose the on-disk location and by MLflow as the run name.
        """
        self.name = name

    @abstractmethod
    def start_run(self) -> None:
        """Begin a new tracking run.

        Implementations should make any necessary preparations (creating
        directories, opening an MLflow run, etc.). Calling :meth:`log_params`
        or :meth:`log_metrics` before ``start_run`` is undefined behaviour.
        """

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Record a mapping of hyper-parameters / config values.

        Parameters
        ----------
        params:
            Flat mapping of parameter name to (JSON-serialisable) value.
        """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Record a mapping of evaluation metrics.

        Parameters
        ----------
        metrics:
            Flat mapping of metric name to value. Nested structures are
            permitted by the JSON backend but flattened/skipped by MLflow.
        """

    @abstractmethod
    def end_run(self) -> None:
        """Finalise the current run, flushing any buffered state."""

    def __enter__(self) -> BaseTracker:
        """Context-manager entry: starts the run and returns ``self``."""
        self.start_run()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Context-manager exit: ends the run (always, even on error)."""
        self.end_run()


class JSONTracker(BaseTracker):
    """File-based tracker that appends run records to a JSON document.

    Each run is serialised as a single JSON object containing the run name,
    a UTC timestamp, the logged params, and the logged metrics. Records for a
    given experiment ``name`` accumulate in ``runs/<name>/metrics.json`` as a
    JSON array, so the history of every run is preserved.
    """

    def __init__(self, name: str, runs_dir: Path | None = None) -> None:
        """Initialise a JSON tracker.

        Parameters
        ----------
        name:
            Experiment name; also the sub-directory under ``runs/``.
        runs_dir:
            Optional base directory for run artifacts. Defaults to
            ``<PROJECT_ROOT>/runs``. Provided primarily to make the backend
            easy to unit-test against a temporary directory.
        """
        super().__init__(name)
        base = Path(runs_dir) if runs_dir is not None else PROJECT_ROOT / "runs"
        self.run_dir: Path = base / name
        self.metrics_path: Path = self.run_dir / "metrics.json"
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._started_at: float | None = None

    def start_run(self) -> None:
        """Create the run directory and reset buffered params/metrics."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._params = {}
        self._metrics = {}
        self._started_at = time.time()

    def log_params(self, params: dict[str, Any]) -> None:
        """Merge ``params`` into the buffered parameter record."""
        self._params.update(dict(params))

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge ``metrics`` into the buffered metric record."""
        self._metrics.update(dict(metrics))

    def _read_existing(self) -> list[dict[str, Any]]:
        """Return the existing list of run records, or an empty list.

        Tolerant of a missing or corrupt file: any read/parse failure yields
        an empty history rather than raising, so a single bad write never
        blocks future logging.
        """
        if not self.metrics_path.exists():
            return []
        try:
            with self.metrics_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return []
        if isinstance(data, list):
            return data
        # Backwards-compatible: a single object becomes a one-element history.
        return [data]

    def end_run(self) -> None:
        """Append the buffered run record to ``metrics.json``."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "name": self.name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "params": self._params,
            "metrics": self._metrics,
        }
        if self._started_at is not None:
            record["duration_sec"] = round(time.time() - self._started_at, 4)
        history = self._read_existing()
        history.append(record)
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, default=str)
        self._started_at = None


class MLflowTracker(BaseTracker):
    """MLflow-backed tracker (active only when ``mlflow`` is installed).

    Thin adapter over the MLflow fluent API. Falls back gracefully is not the
    responsibility of this class: :func:`get_tracker` selects it only when
    :func:`mlflow_available` is ``True``.
    """

    def __init__(self, name: str) -> None:
        """Initialise an MLflow tracker for experiment ``name``.

        Raises
        ------
        RuntimeError
            If instantiated when ``mlflow`` is not importable.
        """
        super().__init__(name)
        if not _MLFLOW_AVAILABLE:  # pragma: no cover - guarded by get_tracker
            raise RuntimeError("MLflowTracker requires the optional 'mlflow' package.")
        self._active = False

    def start_run(self) -> None:  # pragma: no cover - requires mlflow
        """Set the MLflow experiment and start a run."""
        _mlflow.set_experiment(self.name)
        _mlflow.start_run(run_name=self.name)
        self._active = True

    def log_params(self, params: dict[str, Any]) -> None:  # pragma: no cover
        """Log scalar params; nested values are JSON-encoded as strings."""
        flat = {k: (v if _is_scalar(v) else json.dumps(v, default=str)) for k, v in params.items()}
        _mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, Any]) -> None:  # pragma: no cover
        """Log scalar/numeric metrics; non-numeric entries are skipped."""
        numeric = {k: float(v) for k, v in metrics.items() if _is_number(v)}
        if numeric:
            _mlflow.log_metrics(numeric)

    def end_run(self) -> None:  # pragma: no cover - requires mlflow
        """End the active MLflow run if one is open."""
        if self._active:
            _mlflow.end_run()
            self._active = False


def _is_number(value: Any) -> bool:
    """Return ``True`` for real (non-bool) numeric scalars."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_scalar(value: Any) -> bool:
    """Return ``True`` for values MLflow can store as a param directly."""
    return isinstance(value, (str, int, float, bool)) or value is None


def get_tracker(kind: str = "json", name: str = "default", **kwargs: Any) -> BaseTracker:
    """Return a tracker backend for the requested ``kind``.

    Selection rules:

    * ``kind == "mlflow"`` -> :class:`MLflowTracker` when MLflow is importable,
      otherwise transparently falls back to :class:`JSONTracker`.
    * ``kind == "json"`` (or anything else) -> :class:`JSONTracker`.

    Parameters
    ----------
    kind:
        Desired backend, ``"mlflow"`` or ``"json"``.
    name:
        Experiment/run name handed to the tracker.
    **kwargs:
        Backend-specific keyword arguments (e.g. ``runs_dir`` for the JSON
        backend). Ignored by backends that do not accept them.

    Returns
    -------
    BaseTracker
        An initialised (but not yet started) tracker instance.
    """
    normalized = (kind or "json").strip().lower()
    if normalized == "mlflow" and _MLFLOW_AVAILABLE:  # pragma: no cover
        return MLflowTracker(name=name)
    runs_dir = kwargs.get("runs_dir")
    return JSONTracker(name=name, runs_dir=runs_dir)


__all__ = [
    "BaseTracker",
    "JSONTracker",
    "MLflowTracker",
    "get_tracker",
    "mlflow_available",
]
