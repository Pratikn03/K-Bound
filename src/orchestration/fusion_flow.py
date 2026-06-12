"""Production-grade fusion orchestration built on Prefect.

This module wraps the existing fusion experiment
(:func:`src.scripts.run_fusion_experiment.main`) plus its two optional
post-steps (attention validation and the attention harness) in a real Prefect
flow composed of three retried tasks. When Prefect is not installed, the
:mod:`._compat` shim degrades the decorators to no-ops so the flow still runs as
a plain function.

The underlying computation is unchanged: the same scripts are invoked. The
optional attention steps preserve the legacy environment-variable gating
(``RUN_ATTENTION_VALIDATION`` / ``RUN_ATTENTION_HARNESS`` set to ``"1"``) while
also being controllable through typed flow parameters.
"""

from __future__ import annotations

import os
from typing import Any

from uais.utils.logging_utils import setup_logging

from ._compat import PREFECT, flow, get_run_logger, task

logger = setup_logging(__name__)


def _env_flag(name: str) -> bool:
    """Return ``True`` when environment variable ``name`` equals ``"1"``."""

    return os.getenv(name) == "1"


@task(name="fusion-run-experiment", retries=2, retry_delay_seconds=5, log_prints=True)
def _run_fusion_experiment_task() -> bool:
    """Run the (legacy dashboard) fusion experiment script end-to-end.

    Returns
    -------
    bool
        ``True`` if the experiment ran to completion, ``False`` if it could not
        be imported/started.
    """

    run_logger = get_run_logger()
    try:
        from src.scripts.run_fusion_experiment import main as run_fusion
    except Exception as exc:  # pragma: no cover - import guard
        run_logger.error("Fusion pipeline import failed: %s", exc)
        return False
    run_logger.info("Starting fusion pipeline via run_fusion_experiment")
    run_fusion()
    run_logger.info("Fusion pipeline complete")
    return True


@task(name="fusion-attention-validation", retries=2, retry_delay_seconds=5, log_prints=True)
def _run_attention_validation_task() -> bool:
    """Run the optional attention-input validation step.

    Returns
    -------
    bool
        ``True`` if the validation ran, ``False`` on import/run failure.
    """

    run_logger = get_run_logger()
    try:
        from src.scripts.run_attention_validation import main as run_validation

        run_validation()
        return True
    except Exception as exc:  # pragma: no cover - optional path
        run_logger.error("Attention validation failed: %s", exc)
        return False


@task(name="fusion-attention-harness", retries=2, retry_delay_seconds=5, log_prints=True)
def _run_attention_harness_task() -> bool:
    """Run the optional attention-evaluation harness step.

    Returns
    -------
    bool
        ``True`` if the harness ran, ``False`` on import/run failure.
    """

    run_logger = get_run_logger()
    try:
        from src.scripts.run_attention_harness import main as run_harness

        run_harness()
        return True
    except Exception as exc:  # pragma: no cover - optional path
        run_logger.error("Attention harness failed: %s", exc)
        return False


@flow(name="fusion-flow", log_prints=True)
def fusion_pipeline(
    run_attention_validation: bool | None = None,
    run_attention_harness: bool | None = None,
) -> dict[str, Any]:
    """Run the fusion experiment and optional attention steps as a Prefect flow.

    Parameters
    ----------
    run_attention_validation:
        Whether to run attention-input validation after the fusion experiment.
        ``None`` (default) preserves the legacy behavior of reading the
        ``RUN_ATTENTION_VALIDATION`` environment variable (active when ``"1"``).
    run_attention_harness:
        Whether to run the attention-evaluation harness after the fusion
        experiment. ``None`` (default) preserves the legacy behavior of reading
        the ``RUN_ATTENTION_HARNESS`` environment variable (active when ``"1"``).

    Returns
    -------
    dict
        Result payload with ``domain``, ``status``, the resolved
        ``attention_validation`` / ``attention_harness`` booleans, and
        ``prefect``.
    """

    do_validation = (
        _env_flag("RUN_ATTENTION_VALIDATION") if run_attention_validation is None else run_attention_validation
    )
    do_harness = _env_flag("RUN_ATTENTION_HARNESS") if run_attention_harness is None else run_attention_harness

    completed = _run_fusion_experiment_task()

    validation_ran = False
    if do_validation:
        validation_ran = _run_attention_validation_task()

    harness_ran = False
    if do_harness:
        harness_ran = _run_attention_harness_task()

    return {
        "domain": "fusion",
        "status": "completed" if completed else "failed",
        "attention_validation": validation_ran,
        "attention_harness": harness_ran,
        "prefect": PREFECT,
    }


__all__ = ["fusion_pipeline"]
