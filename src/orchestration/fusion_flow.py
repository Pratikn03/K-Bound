"""Best-effort fusion pipeline wrapper."""

from __future__ import annotations

from uais.utils.logging_utils import setup_logging

logger = setup_logging(__name__)


def fusion_pipeline():
    """Run the fusion experiment script."""
    try:
        from src.scripts.run_fusion_experiment import main as run_fusion
    except Exception as exc:  # pragma: no cover - import guard
        logger.error("Fusion pipeline import failed: %s", exc)
        return
    logger.info("Starting fusion pipeline via run_fusion_experiment")
    run_fusion()
    if __import__("os").getenv("RUN_ATTENTION_VALIDATION") == "1":
        try:
            from src.scripts.run_attention_validation import main as run_validation

            run_validation()
        except Exception as exc:  # pragma: no cover - optional path
            logger.error("Attention validation failed: %s", exc)
    if __import__("os").getenv("RUN_ATTENTION_HARNESS") == "1":
        try:
            from src.scripts.run_attention_harness import main as run_harness

            run_harness()
        except Exception as exc:  # pragma: no cover - optional path
            logger.error("Attention harness failed: %s", exc)
    logger.info("Fusion pipeline complete")


__all__ = ["fusion_pipeline"]
