"""Sentiment-style binary text classification via DistilBERT fine-tuning.

Delegates to train_transformer_text so both entry points share the same
Trainer-based pipeline and return the same metrics contract.
"""

from __future__ import annotations

from typing import Dict, List

from .train_transformer_text import train_transformer_text


def train_sentiment_model(
    texts: List[str],
    labels: List[int],
    model_name: str = "distilbert-base-uncased",
    **kwargs: object,
) -> Dict[str, float]:
    """Fine-tune DistilBERT for sentiment-based anomaly detection.

    Any extra kwargs (batch_size, num_epochs, learning_rate, etc.) are
    forwarded directly to train_transformer_text.
    """
    return train_transformer_text(texts=texts, labels=labels, model_name=model_name, **kwargs)


__all__ = ["train_sentiment_model"]
