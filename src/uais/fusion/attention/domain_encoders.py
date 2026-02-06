"""Domain encoder modules for attention fusion."""

from __future__ import annotations

from torch import nn


class DomainEncoder(nn.Module):
    """Encodes domain-specific inputs into a shared embedding space."""

    def __init__(
        self,
        input_dim: int = 1,
        embed_dim: int = 64,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.encoder(x)


__all__ = ["DomainEncoder"]
