"""Production-grade vision orchestration built on Prefect.

This module wraps the small vision experiment
(:func:`uais.vision.train_vision_model.run_vision_experiment`) in a real Prefect
flow with a retried task. When Prefect is not installed, the :mod:`._compat`
shim degrades the decorators to no-ops so the flow still runs as a plain
function.

The underlying computation, configuration defaults, and default dataset
directory resolution are unchanged from the original wrapper. The TensorFlow
import is performed lazily inside the task so importing the orchestration
package never pulls in the heavy vision stack (preserving the original design
goal that optional domains do not break unrelated orchestration imports).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from uais.utils.logging_utils import setup_logging

from ._compat import PREFECT, flow, get_run_logger, task

logger = setup_logging(__name__)

#: Training defaults preserved from the legacy wrapper.
_DEFAULT_EPOCHS = 1
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_IMAGE_SIZE = 128


def _default_dataset_dir() -> Path:
    """Return the default vision dataset directory.

    Resolves to ``src/data/raw/vision/document_forgery`` (relative to the
    ``src`` package directory), matching the legacy wrapper.
    """

    project_root = Path(__file__).resolve().parents[1]
    return project_root / "data" / "raw" / "vision" / "document_forgery"


@task(name="vision-run-experiment", retries=2, retry_delay_seconds=5, log_prints=True)
def _run_vision_experiment_task(
    dataset_dir: str,
    epochs: int,
    batch_size: int,
    image_size: int,
) -> dict[str, Any] | None:
    """Train the small vision model and return its metrics.

    Parameters
    ----------
    dataset_dir:
        Directory of class-foldered images.
    epochs:
        Number of training epochs.
    batch_size:
        Training batch size.
    image_size:
        Square image side length in pixels.

    Returns
    -------
    dict or None
        The metrics dictionary returned by ``run_vision_experiment`` (keys
        ``val_loss``, ``val_accuracy``, ``history``), or ``None`` if the
        experiment failed.
    """

    run_logger = get_run_logger()
    try:
        from uais.vision.train_vision_model import VisionConfig, run_vision_experiment
    except Exception as exc:  # pragma: no cover - import guard
        run_logger.error("Vision pipeline import failed: %s", exc)
        return None

    cfg = VisionConfig(
        dataset_dir=Path(dataset_dir),
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
    )
    try:
        run_logger.info("Starting vision pipeline on %s", dataset_dir)
        metrics = run_vision_experiment(cfg)
        run_logger.info("Vision metrics: %s", metrics)
        return metrics
    except Exception as exc:  # pragma: no cover - best-effort
        run_logger.error("Vision pipeline failed: %s", exc)
        return None


@flow(name="vision-flow", log_prints=True)
def vision_pipeline(
    dataset_dir: str | None = None,
    epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> dict[str, Any]:
    """Run a small vision experiment (simple CNN by default) as a Prefect flow.

    Parameters
    ----------
    dataset_dir:
        Optional directory of class-foldered images. Defaults to
        ``src/data/raw/vision/document_forgery`` (the legacy default).
    epochs:
        Number of training epochs (default ``1``).
    batch_size:
        Training batch size (default ``16``).
    image_size:
        Square image side length in pixels (default ``128``).

    Returns
    -------
    dict
        Result payload with ``domain``, ``status``, the resolved
        ``dataset_dir``, the ``metrics`` dict (or ``None``), and ``prefect``.
    """

    resolved_dir = Path(dataset_dir) if dataset_dir else _default_dataset_dir()
    metrics = _run_vision_experiment_task(
        dataset_dir=str(resolved_dir),
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
    )
    return {
        "domain": "vision",
        "status": "completed" if metrics is not None else "failed",
        "dataset_dir": str(resolved_dir),
        "metrics": metrics,
        "prefect": PREFECT,
    }


__all__ = ["vision_pipeline"]
