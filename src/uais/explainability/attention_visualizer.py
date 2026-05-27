"""Attention visualization helpers for fusion models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_attention_heatmap(
    attention_weights,
    domain_labels: Sequence[str] | None = None,
    sample_idx: int = 0,
    head_idx: int | None = None,
    out_path: str | Path | None = None,
):
    weights = attention_weights
    if hasattr(weights, "detach"):
        weights = weights.detach().cpu().numpy()
    weights = np.asarray(weights)

    if weights.ndim == 4:
        weights = weights[sample_idx]
        if head_idx is None:
            weights = weights.mean(axis=0)
        else:
            weights = weights[head_idx]
    elif weights.ndim == 3:
        weights = weights[sample_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(weights, cmap="viridis")
    fig.colorbar(im, ax=ax)

    if domain_labels:
        ax.set_xticks(range(len(domain_labels)))
        ax.set_yticks(range(len(domain_labels)))
        ax.set_xticklabels(domain_labels, rotation=45, ha="right")
        ax.set_yticklabels(domain_labels)

    ax.set_title("Attention Heatmap")
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig


def summarize_attention_weights(
    attention_weights,
    domain_labels: Sequence[str] | None = None,
) -> dict:
    weights = attention_weights
    if hasattr(weights, "detach"):
        weights = weights.detach().cpu().numpy()
    weights = np.asarray(weights)

    if weights.ndim == 4:
        weights = weights.mean(axis=1).mean(axis=0)
    elif weights.ndim == 3:
        weights = weights.mean(axis=0)
    if weights.ndim != 2:
        raise ValueError("Attention weights must be 2D, 3D, or 4D.")

    if domain_labels is None:
        domain_labels = [f"domain_{i}" for i in range(weights.shape[0])]

    contributions = weights.mean(axis=0)
    return {label: float(contributions[idx]) for idx, label in enumerate(domain_labels)}


def plot_domain_contributions(
    contributions: dict,
    title: str = "Domain Contribution (Mean Attention)",
    out_path: str | Path | None = None,
):
    labels = list(contributions.keys())
    values = [contributions[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="teal")
    ax.set_ylabel("Mean attention")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig


def plot_confidence_map(
    confidence_scores: Sequence[float],
    domain_labels: Sequence[str] | None = None,
    title: str = "Domain Confidence",
    out_path: str | Path | None = None,
):
    scores = np.asarray(confidence_scores, dtype=float)
    if domain_labels is None:
        domain_labels = [f"domain_{i}" for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(domain_labels, scores, color="slateblue")
    ax.set_ylabel("Confidence")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(domain_labels)))
    ax.set_xticklabels(domain_labels, rotation=45, ha="right")
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig


__all__ = [
    "plot_attention_heatmap",
    "summarize_attention_weights",
    "plot_domain_contributions",
    "plot_confidence_map",
]
