"""DistilBERT fine-tuning for binary text anomaly / phishing detection.

Uses the HuggingFace Trainer API with early stopping and full metrics
(accuracy, F1, ROC-AUC) computed on a held-out validation split.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


class _TextDataset(Dataset):
    def __init__(self, encodings: dict, labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


def train_transformer_text(
    texts: List[str],
    labels: List[int],
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    batch_size: int = 32,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    test_size: float = 0.2,
    output_dir: str = "/tmp/distilbert_finetuned",
    seed: int = 42,
    **_: object,
) -> Dict[str, float]:
    """Fine-tune a DistilBERT model for binary text classification.

    Splits texts/labels into train/val, tokenizes with the model's own
    tokenizer, trains with HuggingFace Trainer, and returns eval metrics.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
    val_enc = tokenizer(X_val, truncation=True, padding=True, max_length=max_length)

    train_ds = _TextDataset(train_enc, y_train)
    val_ds = _TextDataset(val_enc, y_val)

    num_labels = len(set(labels))
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    eval_results = trainer.evaluate()

    return {
        "model": model_name,
        "accuracy": float(eval_results.get("eval_accuracy", 0.0)),
        "f1": float(eval_results.get("eval_f1", 0.0)),
        "roc_auc": float(eval_results.get("eval_roc_auc", 0.0)),
        "eval_loss": float(eval_results.get("eval_loss", 0.0)),
    }


__all__ = ["train_transformer_text"]
