"""Counterfactual Domain Attribution (CDA) for multimodal fusion explanations.

For each prediction, masks one domain at a time and measures the probability
change, producing human-readable statements like:
  "If fraud evidence were absent, anomaly risk would drop from 87% to 52% (Δ = -0.35, -40%)"

When used with a fitted ReliabilityEstimator (use_craf_weights=True), reliability
weights are injected directly into CrossModalAttentionFusion.forward() for the
baseline and all counterfactual predictions.

CRAF injection pattern
----------------------
AttentionFusionModel.forward() does not accept external confidence_weights, but
CrossModalAttentionFusion.forward() does. We bypass the outer model by running
domain_encoders manually then calling model.fusion directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator


@dataclass
class CounterfactualResult:
    sample_id: Any
    p_baseline: float
    cf_impacts: Dict[str, float]      # domain -> delta (positive = domain raised risk)
    cf_impacts_pct: Dict[str, float]  # domain -> (delta / p_baseline) * 100
    narrative: str                    # multi-line human-readable explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "p_baseline": self.p_baseline,
            "cf_impacts": self.cf_impacts,
            "cf_impacts_pct": self.cf_impacts_pct,
            "narrative": self.narrative,
        }


class CounterfactualDomainExplainer:
    """Explains individual fusion predictions via deterministic domain masking.

    For each domain, masks it out and computes the change in predicted
    anomaly probability. The resulting attribution tells analysts which
    domain is driving each specific decision.
    """

    def __init__(
        self,
        model: AttentionFusionModel,
        domain_order: List[str],
        device: Optional[torch.device] = None,
        reliability_estimator: Optional[ReliabilityEstimator] = None,
        use_craf_weights: bool = False,
    ) -> None:
        self.model = model
        self.model.eval()
        self.domain_order = list(domain_order)
        self.device = device or next(model.parameters()).device
        self.reliability_estimator = reliability_estimator
        self.use_craf_weights = use_craf_weights and (reliability_estimator is not None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        sample_ids: Optional[List] = None,
    ) -> List[CounterfactualResult]:
        """Explain all samples in features.

        features : [N, D, F] or [D, F] for a single sample
        masks    : [N, D] or [D] bool — True = missing
        """
        if features.ndim == 2:
            features = features[np.newaxis]
            masks = masks[np.newaxis]

        n = features.shape[0]
        ids = sample_ids if sample_ids is not None else list(range(n))
        return [self._explain_one(features[i], masks[i], ids[i]) for i in range(n)]

    def explain_batch(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        sample_ids: Optional[List] = None,
        batch_size: int = 64,
    ) -> List[CounterfactualResult]:
        """Memory-efficient batched explanation."""
        results: List[CounterfactualResult] = []
        n = features.shape[0]
        ids = sample_ids if sample_ids is not None else list(range(n))
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_f = features[start:end]
            batch_m = masks[start:end]
            batch_ids = ids[start:end]
            results.extend(self.explain(batch_f, batch_m, batch_ids))
        return results

    def correlation_with_shap(
        self,
        cf_results: List[CounterfactualResult],
        shap_domain_importance: Dict[str, float],
    ) -> float:
        """Spearman correlation between mean |CF_impact| and SHAP domain importance.

        Used as a sanity check: CDA and SHAP should agree on which domains matter.
        Returns nan if fewer than 3 domains overlap.
        """
        from scipy.stats import spearmanr

        domains = [d for d in self.domain_order if d in shap_domain_importance]
        if len(domains) < 3:
            return float("nan")

        mean_cf = np.array([
            np.mean([abs(r.cf_impacts.get(d, 0.0)) for r in cf_results if d in r.cf_impacts])
            for d in domains
        ])
        shap_vals = np.array([shap_domain_importance[d] for d in domains])
        if not np.isfinite(mean_cf).all() or not np.isfinite(shap_vals).all():
            return float("nan")
        corr, _ = spearmanr(mean_cf, shap_vals)
        return float(corr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _explain_one(
        self,
        features_1d: np.ndarray,   # [D, F]
        masks_1d: np.ndarray,       # [D] bool
        sample_id: Any,
    ) -> CounterfactualResult:
        # Compute CRAF weights if requested (using single-sample batch)
        craf_w: Optional[np.ndarray] = None
        if self.use_craf_weights and self.reliability_estimator is not None:
            craf_w = self.reliability_estimator.compute_reliability_weights(
                features_1d[np.newaxis], masks_1d[np.newaxis]
            )[0]  # shape [D]

        p_baseline = self._forward_single(features_1d, masks_1d, craf_w)

        cf_impacts: Dict[str, float] = {}
        p_masked_per_domain: Dict[str, float] = {}

        for di, domain in enumerate(self.domain_order):
            if masks_1d[di]:
                # Domain already missing — counterfactual is undefined
                cf_impacts[domain] = float("nan")
                p_masked_per_domain[domain] = float("nan")
                continue

            cf_masks = masks_1d.copy()
            cf_masks[di] = True  # mask out this domain

            # If masking this domain leaves nothing available, skip
            if cf_masks.all():
                cf_impacts[domain] = float("nan")
                p_masked_per_domain[domain] = float("nan")
                continue

            # Recompute CRAF weights with domain masked
            cf_craf_w: Optional[np.ndarray] = None
            if self.use_craf_weights and self.reliability_estimator is not None:
                cf_craf_w = self.reliability_estimator.compute_reliability_weights(
                    features_1d[np.newaxis], cf_masks[np.newaxis]
                )[0]

            p_masked = self._forward_single(features_1d, cf_masks, cf_craf_w)
            p_masked_per_domain[domain] = p_masked
            cf_impacts[domain] = p_baseline - p_masked  # positive = domain increased risk

        cf_impacts_pct = {
            d: (v / p_baseline * 100.0) if (math.isfinite(v) and p_baseline > 1e-9) else float("nan")
            for d, v in cf_impacts.items()
        }

        narrative = self._build_narrative(p_baseline, cf_impacts, p_masked_per_domain)
        return CounterfactualResult(
            sample_id=sample_id,
            p_baseline=p_baseline,
            cf_impacts=cf_impacts,
            cf_impacts_pct=cf_impacts_pct,
            narrative=narrative,
        )

    def _forward_single(
        self,
        features_1d: np.ndarray,             # [D, F]
        masks_1d: np.ndarray,                 # [D] bool
        reliability_weights: Optional[np.ndarray] = None,  # [D] float | None
    ) -> float:
        """Run one sample through the model, optionally injecting CRAF weights.

        When reliability_weights is not None, we bypass AttentionFusionModel.forward()
        and call model.fusion directly with the supplied weights.
        """
        feat_t = torch.tensor(
            features_1d[np.newaxis], dtype=torch.float32, device=self.device
        )
        mask_t = torch.tensor(
            masks_1d[np.newaxis], dtype=torch.bool, device=self.device
        )

        with torch.no_grad():
            if reliability_weights is not None:
                # CRAF injection: encode each domain, then call fusion with external weights
                embeds = [
                    enc(feat_t[:, i, :])
                    for i, enc in enumerate(self.model.domain_encoders)
                ]
                domain_embeds = torch.stack(embeds, dim=1)  # [1, D, embed_dim]
                craf_t = torch.tensor(
                    reliability_weights[np.newaxis], dtype=torch.float32, device=self.device
                )
                craf_t = craf_t.masked_fill(mask_t, 0.0)
                logits, _ = self.model.fusion(
                    domain_embeds,
                    key_padding_mask=mask_t,
                    confidence_weights=craf_t,
                )
            else:
                logits, _, _ = self.model(feat_t, key_padding_mask=mask_t)

        return float(torch.sigmoid(logits).item())

    @staticmethod
    def _build_narrative(
        p_baseline: float,
        cf_impacts: Dict[str, float],
        p_masked: Dict[str, float],
    ) -> str:
        lines = []
        for domain, delta in cf_impacts.items():
            if not math.isfinite(delta):
                lines.append(f"  [{domain}] already absent — counterfactual undefined")
                continue
            direction = "drop" if delta > 0 else "rise"
            pct = (delta / p_baseline * 100.0) if p_baseline > 1e-9 else 0.0
            pm = p_masked.get(domain, float("nan"))
            lines.append(
                f"  [{domain}] If absent, risk would {direction} "
                f"from {p_baseline:.0%} to {pm:.0%} "
                f"(Δ = {delta:+.2f}, {pct:+.0f}%)"
            )
        return "\n".join(lines)


__all__ = ["CounterfactualResult", "CounterfactualDomainExplainer"]
