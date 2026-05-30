"""ELARA-CHF: validation-locked certified heterogeneous fusion (T8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elara.theory.t8_certified_heterogeneous_fusion import (
    CHFCertificate,
    batch_coherence_scores,
    predict_chf,
    select_chf_route_on_validation,
)
from uais.fusion.attention.gate_decision_rule import per_sample_mean_reliability

__all__ = ["CertifiedHeterogeneousFusion", "fit_chf_on_validation"]


@dataclass
class CertifiedHeterogeneousFusion:
    certificate: CHFCertificate
    stack_model: Any | None = None
    method_name: str = "elara_chf_v1"

    def predict_proba(self, sar_probs: np.ndarray, rga_probs: np.ndarray, static_probs: np.ndarray, **kwargs) -> np.ndarray:
        fn = predict_chf
        if kwargs.get("shift_aware"):
            from elara.theory.t8_certified_heterogeneous_fusion import predict_chf_shift_aware

            fn = predict_chf_shift_aware
            return fn(
                self.certificate,
                sar_probs=sar_probs,
                rga_probs=rga_probs,
                static_probs=static_probs,
                coherence_per_sample=kwargs["coherence_per_sample"],
                reliability_mean=kwargs["reliability_mean"],
                stack_model=self.stack_model,
                eval_categories=kwargs.get("eval_categories"),
                train_categories=kwargs.get("train_categories"),
            )
        return fn(
            self.certificate,
            sar_probs=sar_probs,
            rga_probs=rga_probs,
            static_probs=static_probs,
            coherence_per_sample=kwargs["coherence_per_sample"],
            reliability_mean=kwargs["reliability_mean"],
            stack_model=self.stack_model,
        )


def fit_chf_on_validation(
    val_labels: np.ndarray,
    *,
    sar_val: np.ndarray,
    rga_val: np.ndarray,
    static_val: np.ndarray,
    reliability_weights_val: np.ndarray,
    masks_val: np.ndarray,
    switching_certified: bool = True,
    coherence_threshold: float = 0.5,
) -> CertifiedHeterogeneousFusion:
    rel_mean = per_sample_mean_reliability(reliability_weights_val, masks_val)
    coh = batch_coherence_scores(reliability_weights_val, masks_val)
    cert = select_chf_route_on_validation(
        val_labels,
        sar_probs=sar_val,
        rga_probs=rga_val,
        static_probs=static_val,
        coherence_per_sample=coh,
        reliability_mean=rel_mean,
        switching_certified=switching_certified,
        coherence_threshold=coherence_threshold,
    )
    stack_model = None
    if cert.route == "stack5":
        from sklearn.linear_model import LogisticRegression

        x = np.column_stack([sar_val, rga_val, static_val, rel_mean, coh]).astype(np.float64)
        stack_model = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=0)
        stack_model.fit(x, np.asarray(val_labels, dtype=int))
    return CertifiedHeterogeneousFusion(certificate=cert, stack_model=stack_model)
