"""Production-grade NLP orchestration built on Prefect.

This module wraps the baseline TF-IDF + logistic-regression text experiment
(:func:`uais.nlp.train_text_classifier.run_text_experiment`) in a real Prefect
flow with a retried task. When Prefect is not installed, the :mod:`._compat`
shim degrades the decorators to no-ops so the flow still runs as a plain
function.

The underlying computation, configuration defaults, and default dataset path
resolution are unchanged from the original wrapper. Heavy imports (scikit-learn
via the training module) are performed lazily inside the task so importing the
orchestration package stays lightweight and dependency-tolerant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from uais.utils.logging_utils import setup_logging

from ._compat import PREFECT, flow, get_run_logger, task

logger = setup_logging(__name__)

#: Default columns / sampling caps preserved from the legacy wrapper.
_DEFAULT_TEXT_COLUMN = "content"
_DEFAULT_LABEL_COLUMN = "label"
_DEFAULT_MAX_SAMPLES = 5000
_DEFAULT_MAX_FEATURES = 5000


def _default_dataset_path() -> Path:
    """Return the default NLP dataset path (``src/data/raw/nlp/enron_emails.csv``).

    This mirrors the original wrapper's resolution, which is relative to the
    ``src`` package directory (the parent of ``orchestration``).
    """

    project_root = Path(__file__).resolve().parents[1]
    return project_root / "data" / "raw" / "nlp" / "enron_emails.csv"


@task(name="nlp-run-text-experiment", retries=2, retry_delay_seconds=5, log_prints=True)
def _run_text_experiment_task(
    dataset_path: str,
    text_column: str,
    label_column: str,
    max_samples: int,
    max_features: int,
) -> dict[str, float] | None:
    """Train the baseline TF-IDF + logistic-regression model and return metrics.

    Parameters
    ----------
    dataset_path:
        Path to the CSV dataset.
    text_column:
        Name of the text column.
    label_column:
        Name of the binary label column.
    max_samples:
        Maximum number of rows to sample for the experiment.
    max_features:
        Maximum TF-IDF vocabulary size.

    Returns
    -------
    dict or None
        The metrics dictionary returned by ``run_text_experiment`` (keys
        ``roc_auc`` and ``accuracy``), or ``None`` if the experiment failed.
    """

    run_logger = get_run_logger()
    try:
        from uais.nlp.train_text_classifier import NLPConfig, run_text_experiment
    except Exception as exc:  # pragma: no cover - import guard
        run_logger.error("NLP pipeline import failed: %s", exc)
        return None

    cfg = NLPConfig(
        dataset_path=Path(dataset_path),
        text_column=text_column,
        label_column=label_column,
        max_samples=max_samples,
        max_features=max_features,
    )
    try:
        run_logger.info("Starting NLP pipeline on %s", cfg.dataset_path)
        metrics = run_text_experiment(cfg)
        run_logger.info("NLP metrics: %s", metrics)
        return metrics
    except Exception as exc:  # pragma: no cover - best-effort
        run_logger.error("NLP pipeline failed: %s", exc)
        return None


@flow(name="nlp-flow", log_prints=True)
def nlp_pipeline(
    dataset_path: str | None = None,
    text_column: str = _DEFAULT_TEXT_COLUMN,
    label_column: str = _DEFAULT_LABEL_COLUMN,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    max_features: int = _DEFAULT_MAX_FEATURES,
) -> dict[str, Any]:
    """Run the baseline TF-IDF + logistic-regression NLP experiment as a flow.

    Parameters
    ----------
    dataset_path:
        Optional path to the CSV dataset. Defaults to
        ``src/data/raw/nlp/enron_emails.csv`` (the legacy default).
    text_column:
        Name of the text column (default ``"content"``).
    label_column:
        Name of the binary label column (default ``"label"``).
    max_samples:
        Maximum number of rows to sample (default ``5000``).
    max_features:
        Maximum TF-IDF vocabulary size (default ``5000``).

    Returns
    -------
    dict
        Result payload with ``domain``, ``status``, the resolved
        ``dataset_path``, the ``metrics`` dict (or ``None``), and ``prefect``.
    """

    resolved_path = Path(dataset_path) if dataset_path else _default_dataset_path()
    metrics = _run_text_experiment_task(
        dataset_path=str(resolved_path),
        text_column=text_column,
        label_column=label_column,
        max_samples=max_samples,
        max_features=max_features,
    )
    return {
        "domain": "nlp",
        "status": "completed" if metrics is not None else "failed",
        "dataset_path": str(resolved_path),
        "metrics": metrics,
        "prefect": PREFECT,
    }


__all__ = ["nlp_pipeline"]
