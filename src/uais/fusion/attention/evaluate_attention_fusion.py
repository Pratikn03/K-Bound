"""Evaluation helper for attention fusion checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from uais.fusion.attention.attention_utils import (
    FusionDataset,
    build_fusion_tensors,
    infer_feature_columns,
    load_fusion_dataframe,
)
from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.utils.config_loader import load_yaml
from uais.utils.metrics import classification_metrics
from uais.utils.paths import PROJECT_ROOT

DEFAULT_CONFIG = Path("src/uais/fusion/attention/attention_config.yaml")


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def evaluate_attention_fusion(cfg_path: Path = DEFAULT_CONFIG) -> Dict[str, float]:
    cfg = load_yaml(cfg_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    data_path = _resolve_path(data_cfg.get("path", ""))
    df = load_fusion_dataframe(data_path)

    feature_columns = infer_feature_columns(
        df,
        score_column=data_cfg.get("score_column", "score"),
        confidence_column=data_cfg.get("confidence_column", "confidence"),
        embedding_prefix=data_cfg.get("embedding_prefix", "embedding_"),
        feature_columns=data_cfg.get("feature_columns") or None,
        include_confidence=bool(model_cfg.get("use_input_confidence", True)),
    )
    confidence_column = data_cfg.get("confidence_column", "confidence")
    confidence_index = (
        feature_columns.index(confidence_column)
        if confidence_column in feature_columns and model_cfg.get("use_input_confidence", True)
        else None
    )

    output_dir = _resolve_path(output_cfg.get("model_dir", "models/fusion/attention"))
    ckpt_path = output_dir / output_cfg.get("checkpoint_name", "attention_fusion.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    state_cfg = state.get("config", {}) if isinstance(state, dict) else {}
    if state_cfg:
        data_cfg = state_cfg.get("data", data_cfg)
        model_cfg = state_cfg.get("model", model_cfg)
        train_cfg = state_cfg.get("training", train_cfg)
    domain_order = state.get("domain_order")
    feature_columns = state.get("feature_columns", feature_columns)

    domain_order_cfg = data_cfg.get("domain_order") or model_cfg.get("domain_order")
    features, masks, labels, _, domain_order = build_fusion_tensors(
        df,
        id_column=data_cfg.get("id_column", "sample_id"),
        domain_column=data_cfg.get("domain_column", "domain"),
        label_column=data_cfg.get("label_column", "label"),
        score_column=data_cfg.get("score_column", "score"),
        confidence_column=confidence_column,
        embedding_prefix=data_cfg.get("embedding_prefix", "embedding_"),
        timestamp_column=data_cfg.get("timestamp_column"),
        domain_order=domain_order_cfg or domain_order,
        feature_columns=feature_columns,
    )
    if labels is None:
        raise ValueError("Label column is required for evaluation.")

    indices = np.arange(features.shape[0])
    stratify = labels if len(np.unique(labels)) > 1 else None
    _, test_idx = train_test_split(
        indices,
        test_size=train_cfg.get("test_size", 0.2),
        random_state=train_cfg.get("seed", 42),
        stratify=stratify,
    )

    test_set = FusionDataset(features[test_idx], masks[test_idx], labels[test_idx])
    test_loader = DataLoader(test_set, batch_size=int(train_cfg.get("batch_size", 128)))

    model = AttentionFusionModel(
        num_domains=len(domain_order),
        input_dim=features.shape[-1],
        embed_dim=int(model_cfg.get("embed_dim", 64)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_confidence=bool(model_cfg.get("use_confidence", True)),
        use_input_confidence=bool(model_cfg.get("use_input_confidence", True)),
        confidence_index=confidence_index,
        use_attention=bool(model_cfg.get("use_attention", True)),
        use_domain_embeddings=bool(model_cfg.get("use_domain_embeddings", True)),
        use_positional_embeddings=bool(model_cfg.get("use_positional_embeddings", True)),
        use_missing_embedding=bool(model_cfg.get("use_missing_embedding", True)),
    )
    model.load_state_dict(state["model_state"])
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for features_batch, masks_batch, labels_batch in test_loader:
            logits, _, _ = model(features_batch, key_padding_mask=masks_batch)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.numpy())
            all_labels.append(labels_batch.numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    metrics = classification_metrics(y_true, y_prob, threshold=0.5)
    return metrics


if __name__ == "__main__":  # pragma: no cover
    metrics_out = evaluate_attention_fusion()
    for key, value in metrics_out.items():
        print(f"{key}: {value:.4f}")
