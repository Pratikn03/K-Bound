"""Production-grade behavior anomaly orchestration built on Prefect.

This module wraps the existing end-to-end behavior experiment
(:func:`src.scripts.run_behavior_experiment.main`) in a real Prefect flow with a
retried task. When Prefect is not installed, the :mod:`._compat` shim degrades
the decorators to no-ops so the flow still runs as a plain function.

The underlying computation is unchanged: the same experiment script is invoked.
The flow only adds orchestration concerns (retries, structured logging, a
machine-readable result dict).
"""

from __future__ import annotations

from typing import Any

from uais.utils.logging_utils import setup_logging

from ._compat import PREFECT, flow, get_run_logger, task

logger = setup_logging(__name__)


@task(name="behavior-run-experiment", retries=2, retry_delay_seconds=5, log_prints=True)
def _run_behavior_experiment_task() -> bool:
    """Run the behavior experiment script end-to-end.

    The experiment trains an autoencoder and a Local Outlier Factor model on the
    behavior features, evaluates anomaly scores, and persists metrics/scores.
    The heavy import is performed lazily and guarded so an optional-dependency
    failure surfaces as a logged, recoverable ``False``.

    Returns
    -------
    bool
        ``True`` if the experiment ran to completion, ``False`` if it could not
        be imported/started.
    """

    run_logger = get_run_logger()
    try:
        from src.scripts.run_behavior_experiment import main as run_behavior
    except Exception as exc:  # pragma: no cover - import guard
        run_logger.error("Behavior pipeline import failed: %s", exc)
        return False
    run_logger.info("Starting behavior pipeline via run_behavior_experiment")
    run_behavior()
    run_logger.info("Behavior pipeline complete")
    return True


@flow(name="behavior-flow", log_prints=True)
def behavior_pipeline() -> dict[str, Any]:
    """Run the behavior experiment as a Prefect flow.

    Returns
    -------
    dict
        Result payload with keys ``domain``, ``status`` (``"completed"`` or
        ``"failed"``), and ``prefect`` (whether real Prefect is active).
    """

    completed = _run_behavior_experiment_task()
    return {
        "domain": "behavior",
        "status": "completed" if completed else "failed",
        "prefect": PREFECT,
    }


__all__ = ["behavior_pipeline"]
