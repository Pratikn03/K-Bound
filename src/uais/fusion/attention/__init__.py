"""Attention-based fusion modules for Phase 2."""

from uais.fusion.attention.cross_modal_attention import (
    AttentionFusionModel,
    CrossModalAttentionBlock,
    CrossModalAttentionFusion,
)
from uais.fusion.attention.train_attention_fusion import train_attention_fusion
from uais.fusion.attention.evaluate_attention_fusion import evaluate_attention_fusion
from uais.fusion.attention.evaluate_attention_harness import evaluate_attention_harness
from uais.fusion.attention.validate_fusion_inputs import validate_attention_inputs
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator
from uais.fusion.attention.counterfactual_explainer import (
    CounterfactualDomainExplainer,
    CounterfactualResult,
)
from uais.fusion.attention.adversarial_robustness import (
    AdversarialAttackType,
    AdversarialPerturbationEngine,
)
from uais.fusion.attention.baselines import (
    EarlyFusionMLP,
    LateFusionEnsemble,
    RandomForestFusion,
    ConfidenceWeightedMean,
    run_baseline_suite,
)

__all__ = [
    "AttentionFusionModel",
    "CrossModalAttentionBlock",
    "CrossModalAttentionFusion",
    "train_attention_fusion",
    "evaluate_attention_fusion",
    "evaluate_attention_harness",
    "validate_attention_inputs",
    # CRAF: Calibration-aware Reliability-Adaptive Fusion
    "ReliabilityEstimator",
    "CounterfactualDomainExplainer",
    "CounterfactualResult",
    "AdversarialAttackType",
    "AdversarialPerturbationEngine",
    # Strong baselines for fair comparison
    "EarlyFusionMLP",
    "LateFusionEnsemble",
    "RandomForestFusion",
    "ConfidenceWeightedMean",
    "run_baseline_suite",
]
