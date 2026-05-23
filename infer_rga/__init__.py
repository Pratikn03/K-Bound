"""infer_rga: runtime-only inference for ELARA RGA reliability-gated fusion.

This package is the deployment-side surface of the research codebase.
It exposes a small ``InferRGA`` class that loads a trained fusion
checkpoint and a fitted reliability estimator, and produces fused
anomaly scores with optional per-domain reliability weights.

The training and benchmarking code lives in ``src/uais/fusion``; this
package deliberately ships *only* the inference path. The intent is
that a production user can install ``infer_rga`` (and its narrow
runtime dependencies: ``torch``, ``numpy``) without pulling the
research codebase's broader dependency tree (matplotlib, statsmodels,
scikit-learn extras, the MVTec / VisA / UNSW dataset loaders, the
adversarial sweep machinery, the LaTeX emit scripts, etc.).

Usage
-----

    from infer_rga import InferRGA

    rga = InferRGA.from_checkpoint(
        model_path="checkpoints/fusion_visa.pt",
        estimator_path="checkpoints/reliability_visa.json",
    )

    # features: (batch, num_domains, feature_dim) float32 in [0, 1]
    # masks:    (batch, num_domains) bool  (True = domain missing/dropped)
    scores = rga.predict_proba(features, masks)
"""

from .inference import InferRGA, RuntimeMetadata

__all__ = ["InferRGA", "RuntimeMetadata"]
__version__ = "0.1.0"
