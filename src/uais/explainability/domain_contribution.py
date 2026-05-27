"""Aggregate attention weights into per-domain contributions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def summarize_domain_contributions(attention_weights, domain_labels: Sequence[str] | None = None) -> dict[str, float]:
    weights = attention_weights
    if hasattr(weights, "detach"):
        weights = weights.detach().cpu().numpy()
    weights = np.asarray(weights)

    if weights.ndim == 4:
        contrib = weights.mean(axis=(0, 1, 2))
    elif weights.ndim == 3:
        contrib = weights.mean(axis=(0, 1))
    elif weights.ndim == 2:
        contrib = weights.mean(axis=0)
    else:
        raise ValueError("Unexpected attention weight shape.")

    if domain_labels is None:
        domain_labels = [f"domain_{i}" for i in range(len(contrib))]
    return {label: float(score) for label, score in zip(domain_labels, contrib)}


__all__ = ["summarize_domain_contributions"]
