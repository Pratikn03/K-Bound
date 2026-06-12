"""Production-grade cyber-intrusion orchestration built on Prefect.

This module wraps the existing end-to-end cyber experiment
(:func:`src.scripts.run_cyber_experiment.main`) in a real Prefect flow with a
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


@task(name="cyber-run-experiment", retries=2, retry_delay_seconds=5, log_prints=True)
def _run_cyber_experiment_task() -> bool:
    """Run the cyber experiment script end-to-end.

    The experiment loads the cyber config, builds features, trains a supervised
    model and an Isolation Forest, evaluates, and persists metrics/scores. The
    heavy import is performed lazily and guarded so an optional-dependency
    failure surfaces as a logged, recoverable ``False``.

    Returns
    -------
    bool
        ``True`` if the experiment ran to completion, ``False`` if it could not
        be imported/started.
    """

    run_logger = get_run_logger()
    try:
        from src.scripts.run_cyber_experiment import main as run_cyber
    except Exception as exc:  # pragma: no cover - import guard
        run_logger.error("Cyber pipeline import failed: %s", exc)
        return False
    run_logger.info("Starting cyber pipeline via run_cyber_experiment")
    run_cyber()
    run_logger.info("Cyber pipeline complete")
    return True


@flow(name="cyber-flow", log_prints=True)
def cyber_pipeline() -> dict[str, Any]:
    """Run the cyber experiment as a Prefect flow.

    Returns
    -------
    dict
        Result payload with keys ``domain``, ``status`` (``"completed"`` or
        ``"failed"``), and ``prefect`` (whether real Prefect is active).
    """

    completed = _run_cyber_experiment_task()
    return {
        "domain": "cyber",
        "status": "completed" if completed else "failed",
        "prefect": PREFECT,
    }


__all__ = ["cyber_pipeline"]
