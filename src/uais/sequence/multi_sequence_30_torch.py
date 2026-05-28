"""PyTorch multi-sequence classifier for the 30-sequence behavior task.

Input layout: ``(batch, 30, seq_len, n_features)``. Each of the 30 sub-sequences
is encoded with a shared 1-D causal-convolution stack, the 30 resulting latent
vectors are concatenated, and a small MLP produces the final logits.

The model is intentionally compact so the multi-sequence path stays
unit-testable on CPU; it is not intended to compete with specialized
behavior-anomaly architectures.
"""

from __future__ import annotations

import torch
from torch import nn


class _PerSequenceTCN(nn.Module):
    """Shared TCN that encodes a single ``(batch, seq_len, n_features)`` slice."""

    def __init__(self, n_features: int, latent_dim: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(latent_dim, 2 * n_features)
        padding = kernel_size - 1  # causal-style left padding via slicing in forward
        self.conv1 = nn.Conv1d(n_features, hidden, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(hidden, latent_dim, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        self._padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) → (batch, n_features, seq_len)
        h = x.transpose(1, 2)
        h = self.conv1(h)
        if self._padding:
            h = h[..., : -self._padding]
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self._padding:
            h = h[..., : -self._padding]
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = torch.relu(h)
        # Global pool across time → (batch, latent_dim)
        return h.mean(dim=-1)


class MultiSequenceTCNClassifier(nn.Module):
    """Multi-sequence TCN classifier.

    Parameters
    ----------
    seq_len : int
        Length of each individual sub-sequence.
    n_features : int
        Number of features per timestep within a sub-sequence.
    latent_dim : int
        Per-sub-sequence latent dimension produced by the shared TCN.
    num_outputs : int
        Output logit dimension. Set to 2 (or higher) for multi-class
        classification, or 1 for binary BCE-with-logits training.
    num_sequences : int, default 30
        Number of paired sub-sequences expected on the second input axis.
    dropout : float, default 0.1
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        latent_dim: int,
        num_outputs: int,
        num_sequences: int = 30,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.num_outputs = num_outputs
        self.num_sequences = num_sequences
        self.encoder = _PerSequenceTCN(n_features=n_features, latent_dim=latent_dim, dropout=dropout)
        hidden = max(latent_dim, 2 * num_outputs)
        self.classifier = nn.Sequential(
            nn.Linear(num_sequences * latent_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected input shape (batch, num_sequences, seq_len, n_features); got {tuple(x.shape)}"
            )
        batch, num_seq, seq_len, feat = x.shape
        if num_seq != self.num_sequences:
            raise ValueError(f"Expected {self.num_sequences} sub-sequences, got {num_seq}")
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        if feat != self.n_features:
            raise ValueError(f"Expected n_features={self.n_features}, got {feat}")
        # Encode each sub-sequence independently using the shared TCN.
        latents = self.encoder(x.reshape(batch * num_seq, seq_len, feat))
        latents = latents.reshape(batch, num_seq * self.latent_dim)
        return self.classifier(latents)


__all__ = ["MultiSequenceTCNClassifier"]
