"""Attention-based fusion modules for Phase 2.

Paper names (ELARA/RGA) vs. code names (UAIS-V/CRAF) cross-reference:
  ELARA  = Verifiable Evidence Reliability for Anomalies  (paper system name)
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
    CategoryAwareReliabilityEstimator,
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
    TentScoreAdapter,
    TTTPseudoLabelAdapter,
    run_baseline_suite,
)
from uais.fusion.attention.unsupervised_baselines import (
    BGMMConfig, GMMConfig, KMeansConfig, IForestConfig,
    OCSVMConfig, LOFConfig, AEConfig,
    BGMMAnomalyDetector, GMMAnomalyDetector, KMeansAnomalyDetector,
    IsolationForestDetector, OneClassSVMDetector, LOFAnomalyDetector,
    AutoencoderAnomalyDetector,
    run_unsupervised_suite,
)
from uais.fusion.attention.dim_reduction import (
    DimReducer, NoOpReducer, PCAReducer, AutoencoderReducer,
    PCAReducerConfig, AEReducerConfig, make_reducer,
)
from uais.fusion.attention.leakage_guard import (
    check_train_test_contamination,
    check_label_overlap,
    assert_no_oversampling_in_test,
    assert_normal_only_training,
    flag_suspicious_metrics,
    PipelineGuard,
    assert_split_before_preprocess,
)
from uais.fusion.attention.cv_evaluator import (
    BaselineSpec, CVConfig,
    cross_validate_baselines,
    pairwise_delong_from_predictions,
)
from uais.fusion.attention.learned_gate import (
    LearnedReliabilityGate,
    LearnedGateConfig,
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
    "CategoryAwareReliabilityEstimator",
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
    "TentScoreAdapter",
    "TTTPseudoLabelAdapter",
    "run_baseline_suite",
    # Unsupervised baselines — normal-only training protocol
    "BGMMConfig", "GMMConfig", "KMeansConfig", "IForestConfig",
    "OCSVMConfig", "LOFConfig", "AEConfig",
    "BGMMAnomalyDetector", "GMMAnomalyDetector", "KMeansAnomalyDetector",
    "IsolationForestDetector", "OneClassSVMDetector", "LOFAnomalyDetector",
    "AutoencoderAnomalyDetector",
    "run_unsupervised_suite",
    # Consistent dimensionality reduction
    "DimReducer", "NoOpReducer", "PCAReducer", "AutoencoderReducer",
    "PCAReducerConfig", "AEReducerConfig", "make_reducer",
    # Leakage detection + protocol enforcement
    "check_train_test_contamination", "check_label_overlap",
    "assert_no_oversampling_in_test", "assert_normal_only_training",
    "flag_suspicious_metrics", "PipelineGuard", "assert_split_before_preprocess",
    # Unified CV evaluator
    "BaselineSpec", "CVConfig",
    "cross_validate_baselines",
    "pairwise_delong_from_predictions",
    # Learned gate (alternative to heuristic τ)
    "LearnedReliabilityGate", "LearnedGateConfig",
]
