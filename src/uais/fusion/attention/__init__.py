"""Attention-based fusion modules for Phase 2.

Paper names (VERA/RGA) vs. code names (UAIS-V/CRAF) cross-reference:
  VERA  = Verifiable Evidence Reliability for Anomalies  (paper system name)
  RGA   = Reliability-Gated Attention                    (paper method name)
  CRAF  = Calibration-aware Reliability-Adaptive Fusion  (code / internal name)
  TTRA  = Test-Time Reliability Adaptation               (CRAF injection step)
  CRS   = Calibrated Reliability Scoring                 (ReliabilityEstimator)
  CDA   = Counterfactual Domain Attribution              (CounterfactualDomainExplainer)
"""

from uais.fusion.attention.cross_modal_attention import (
    AttentionFusionModel,
    CrossModalAttentionBlock,
    CrossModalAttentionFusion,
)
from uais.fusion.attention.train_attention_fusion import train_attention_fusion
from uais.fusion.attention.evaluate_attention_fusion import evaluate_attention_fusion
from uais.fusion.attention.evaluate_attention_harness import evaluate_attention_harness
from uais.fusion.attention.validate_fusion_inputs import validate_attention_inputs
from uais.fusion.attention.reliability_estimator import (
    ReliabilityEstimator,
    RGAReliabilityEstimator,   # paper-name alias for ReliabilityEstimator
)
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
    TentAdapter,
    PseudoLabelTTTAdapter,
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
    # CRAF / RGA — Calibration-aware Reliability-Adaptive Fusion (paper: RGA)
    "ReliabilityEstimator",
    "RGAReliabilityEstimator",
    "CounterfactualDomainExplainer",
    "CounterfactualResult",
    "AdversarialAttackType",
    "AdversarialPerturbationEngine",
    # Baselines — static + test-time adaptive
    "EarlyFusionMLP",
    "LateFusionEnsemble",
    "RandomForestFusion",
    "ConfidenceWeightedMean",
    "TentAdapter",
    "PseudoLabelTTTAdapter",
    "run_baseline_suite",
]
