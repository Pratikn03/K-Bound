"""Confidence estimation head for domain embeddings."""

from __future__ import annotations

from torch import nn


class DomainConfidenceEstimator(nn.Module):
    """Predicts per-domain confidence scores in [0, 1]."""

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings):
        confidences = self.confidence_net(embeddings)
        return confidences.squeeze(-1)


__all__ = ["DomainConfidenceEstimator"]
