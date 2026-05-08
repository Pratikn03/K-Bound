"""Cross-modal attention fusion modules."""

from __future__ import annotations

from typing import Tuple
import math

import torch
from torch import nn

from uais.fusion.attention.confidence_estimator import DomainConfidenceEstimator
from uais.fusion.attention.domain_encoders import DomainEncoder


class CrossModalAttentionBlock(nn.Module):
    """Single attention block with residual connection and layer norm."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        confidence_weights: torch.Tensor | None = None,
    ):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if confidence_weights is not None:
            conf = confidence_weights.clamp(min=0.0).unsqueeze(1).unsqueeze(-1)
            q = q * conf
            k = k * conf
            v = v * conf

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                attn_scores[all_masked] = 0.0

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        x = self.layer_norm(x + self.dropout(attn_output))
        return x, attn_weights


class CrossModalAttentionFusion(nn.Module):
    """Fuse domain embeddings via multi-head self-attention."""

    def __init__(
        self,
        num_domains: int,
        domain_embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_domain_embeddings: bool = True,
        use_positional_embeddings: bool = True,
        use_missing_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.use_attention = use_attention
        self.use_domain_embeddings = use_domain_embeddings
        self.use_positional_embeddings = use_positional_embeddings
        self.use_missing_embedding = use_missing_embedding

        self.domain_embeddings = nn.Embedding(num_domains, domain_embed_dim) if use_domain_embeddings else None
        self.position_embeddings = nn.Embedding(num_domains, domain_embed_dim) if use_positional_embeddings else None
        self.missing_embedding = nn.Parameter(torch.zeros(domain_embed_dim)) if use_missing_embedding else None

        self.blocks = nn.ModuleList(
            [CrossModalAttentionBlock(domain_embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        ) if use_attention else nn.ModuleList()
        self.ffn = nn.Sequential(
            nn.Linear(domain_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        domain_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        confidence_weights: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if domain_embeddings.shape[1] != self.num_domains:
            raise ValueError("domain_embeddings must have shape [batch, num_domains, embed_dim]")
        if domain_ids is None:
            domain_ids = torch.arange(self.num_domains, device=domain_embeddings.device)
            domain_ids = domain_ids.unsqueeze(0).expand(domain_embeddings.shape[0], -1)
        if position_ids is None:
            position_ids = domain_ids
        x = domain_embeddings
        if self.domain_embeddings is not None:
            x = x + self.domain_embeddings(domain_ids)
        if self.position_embeddings is not None:
            x = x + self.position_embeddings(position_ids)
        if self.missing_embedding is not None and key_padding_mask is not None:
            x = x.clone()
            x[key_padding_mask] = self.missing_embedding
        attn_weights = None
        if self.use_attention:
            for block in self.blocks:
                x, attn_weights = block(x, key_padding_mask=key_padding_mask, confidence_weights=confidence_weights)
        fused = self._pool(x, key_padding_mask, confidence_weights)
        logits = self.ffn(fused)
        return logits, attn_weights

    @staticmethod
    def _pool(
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        confidence_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if confidence_weights is not None:
            weights = confidence_weights
            if key_padding_mask is not None:
                weights = weights.masked_fill(key_padding_mask, 0.0)
            weights = weights.clamp(min=0.0)
            denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            weights = weights / denom
            return torch.sum(x * weights.unsqueeze(-1), dim=1)
        if key_padding_mask is not None:
            available = (~key_padding_mask).float()
            denom = available.sum(dim=1, keepdim=True).clamp(min=1.0)
            return torch.sum(x * available.unsqueeze(-1), dim=1) / denom
        return x.mean(dim=1)


class AttentionFusionModel(nn.Module):
    """End-to-end attention fusion with per-domain encoders."""

    def __init__(
        self,
        num_domains: int,
        input_dim: int,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_confidence: bool = True,
        use_input_confidence: bool = True,
        confidence_index: int | None = None,
        use_attention: bool = True,
        use_domain_embeddings: bool = True,
        use_positional_embeddings: bool = True,
        use_missing_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.domain_encoders = nn.ModuleList(
            [DomainEncoder(input_dim, embed_dim, hidden_dim=32, dropout=dropout) for _ in range(num_domains)]
        )
        self.fusion = CrossModalAttentionFusion(
            num_domains=num_domains,
            domain_embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention,
            use_domain_embeddings=use_domain_embeddings,
            use_positional_embeddings=use_positional_embeddings,
            use_missing_embedding=use_missing_embedding,
        )
        self.confidence_estimator = DomainConfidenceEstimator(embed_dim, hidden_dim=32, dropout=dropout) if use_confidence else None
        self.use_input_confidence = use_input_confidence
        self.confidence_index = confidence_index

    def forward(self, features: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        if features.shape[1] != self.num_domains:
            raise ValueError("features must have shape [batch, num_domains, input_dim]")
        embeddings = []
        for idx, encoder in enumerate(self.domain_encoders):
            embeddings.append(encoder(features[:, idx, :]))
        domain_embeddings = torch.stack(embeddings, dim=1)
        input_confidences = None
        if self.use_input_confidence and self.confidence_index is not None:
            input_confidences = features[:, :, self.confidence_index].clamp(0.0, 1.0)
        confidences = None
        if self.confidence_estimator is not None:
            confidences = self.confidence_estimator(domain_embeddings)
        confidence_weights = None
        if input_confidences is not None and confidences is not None:
            confidence_weights = input_confidences * confidences
        elif input_confidences is not None:
            confidence_weights = input_confidences
        elif confidences is not None:
            confidence_weights = confidences
        if confidence_weights is not None:
            if key_padding_mask is not None:
                confidence_weights = confidence_weights.masked_fill(key_padding_mask, 0.0)
        logits, attn_weights = self.fusion(
            domain_embeddings,
            key_padding_mask=key_padding_mask,
            confidence_weights=confidence_weights,
        )
        return logits, attn_weights, confidences


__all__ = ["CrossModalAttentionBlock", "CrossModalAttentionFusion", "AttentionFusionModel"]
