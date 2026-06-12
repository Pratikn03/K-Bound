"""Production-grade fraud anomaly orchestration built on Prefect.

This module wraps the existing end-to-end fraud experiment
(:func:`src.scripts.run_fraud_experiment.main`) in a real Prefect flow with a
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


@task(name="fraud-run-experiment", retries=2, retry_delay_seconds=5, log_prints=True)
def _run_fraud_experiment_task() -> bool:
    """Run the fraud experiment script end-to-end.

    The experiment trains a supervised model plus an Isolation Forest, blends
    their scores, evaluates, and persists metrics/plots/scores artifacts. The
    heavy import is performed lazily and guarded so an optional-dependency
    failure surfaces as a logged, recoverable ``False`` rather than crashing the
    package import.

    Returns
    -------
    bool
        ``True`` if the experiment ran to completion, ``False`` if it could not
        be imported/started.
    """

    run_logger = get_run_logger()
    try:
        from src.scripts.run_fraud_experiment import main as run_fraud
    except Exception as exc:  # pragma: no cover - import guard
        run_logger.error("Fraud pipeline import failed: %s", exc)
        return False
    run_logger.info("Starting fraud pipeline via run_fraud_experiment")
    run_fraud()
    run_logger.info("Fraud pipeline complete")
    return True


@flow(name="fraud-flow", log_prints=True)
def fraud_pipeline() -> dict[str, Any]:
    """Run the fraud experiment as a Prefect flow.

    Returns
    -------
    dict
        Result payload with keys ``domain``, ``status`` (``"completed"`` or
        ``"failed"``), and ``prefect`` (whether real Prefect is active).
    """

    completed = _run_fraud_experiment_task()
    return {
        "domain": "fraud",
        "status": "completed" if completed else "failed",
        "prefect": PREFECT,
    }


__all__ = ["fraud_pipeline"]
