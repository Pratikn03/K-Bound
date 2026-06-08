"""Stub for NLP drift detection.

Replace with real text drift checks (e.g., embedding shift, word frequency PSI).
"""

from __future__ import annotations


def drift_report_nlp(ref_texts, cur_texts) -> dict[str, float]:
    # TODO: implement real drift detection
    return {"js_divergence": 0.0}


__all__ = ["drift_report_nlp"]
