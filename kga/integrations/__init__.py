"""Optional integrations for the KGA decision layer."""

from kga.integrations.claims import assess_promotion
from kga.integrations.elara import (
    ELARAKGAGuard,
    ELARAKGAResult,
    EvaluationMode,
    FrozenLinearBenefitEstimator,
    evaluate_result,
)

__all__ = [
    "ELARAKGAGuard",
    "ELARAKGAResult",
    "EvaluationMode",
    "FrozenLinearBenefitEstimator",
    "assess_promotion",
    "evaluate_result",
]
