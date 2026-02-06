"""Training script for cross-modal attention fusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from uais.fusion.attention.attention_utils import (
    FusionDataset,
    apply_domain_dropout,
    build_fusion_tensors,
    infer_feature_columns,
    load_fusion_dataframe,
    prepare_fusion_dataframe,
    validate_fusion_schema,
    hash_file,
)
from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.utils.config_loader import load_yaml
from uais.utils.metrics import classification_metrics
from uais.utils.mlflow_utils import load_mlflow_settings, log_run, setup_mlflow
from uais.utils.paths import PROJECT_ROOT

DEFAULT_CONFIG = Path("src/uais/fusion/attention/attention_config.yaml")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attention_fusion_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_weights: torch.Tensor | None,
    confidences: torch.Tensor | None = None,
    lambda_reg: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    entropy = torch.tensor(0.0, device=logits.device)
    if attention_weights is not None:
        attn = attention_weights.mean(dim=1)
        attn = attn.mean(dim=1)
        entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean()
    conf_reg = torch.tensor(0.0, device=logits.device)
    if confidences is not None:
        conf_reg = torch.mean((confidences - 1.0) ** 2)
    total_loss = bce_loss - lambda_reg * entropy + lambda_reg * conf_reg
    return total_loss, {
        "bce": float(bce_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "conf_reg": float(conf_reg.detach().cpu()),
    }


def evaluate_model(model: AttentionFusionModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            features, masks, labels = batch
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            logits, _, _ = model(features, key_padding_mask=masks)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    return classification_metrics(y_true, y_prob, threshold=0.5)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def train_attention_fusion(cfg_path: Path = DEFAULT_CONFIG):
    cfg = load_yaml(cfg_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    set_seed(int(train_cfg.get("seed", 42)))

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

    domain_order_cfg = data_cfg.get("domain_order") or model_cfg.get("domain_order")
    schema_stats = validate_fusion_schema(
        df,
        id_column=data_cfg.get("id_column", "sample_id"),
        domain_column=data_cfg.get("domain_column", "domain"),
        score_column=data_cfg.get("score_column", "score"),
        label_column=data_cfg.get("label_column", "label"),
        confidence_column=confidence_column,
        embedding_prefix=data_cfg.get("embedding_prefix", "embedding_"),
        timestamp_column=data_cfg.get("timestamp_column"),
    )
    _, prep_stats = prepare_fusion_dataframe(
        df,
        id_column=data_cfg.get("id_column", "sample_id"),
        domain_column=data_cfg.get("domain_column", "domain"),
        feature_columns=feature_columns,
        label_column=data_cfg.get("label_column", "label"),
        timestamp_column=data_cfg.get("timestamp_column"),
    )

    features, masks, labels, sample_ids, domain_order = build_fusion_tensors(
        df,
        id_column=data_cfg.get("id_column", "sample_id"),
        domain_column=data_cfg.get("domain_column", "domain"),
        label_column=data_cfg.get("label_column", "label"),
        score_column=data_cfg.get("score_column", "score"),
        confidence_column=confidence_column,
        embedding_prefix=data_cfg.get("embedding_prefix", "embedding_"),
        timestamp_column=data_cfg.get("timestamp_column"),
        domain_order=domain_order_cfg,
        feature_columns=feature_columns,
    )

    if labels is None:
        raise ValueError("Label column is required for supervised attention fusion training.")

    num_domains = len(domain_order)
    input_dim = features.shape[-1]
    if model_cfg.get("num_domains") and int(model_cfg.get("num_domains")) != num_domains:
        raise ValueError("num_domains in config does not match domains found in data.")

    indices = np.arange(features.shape[0])
    stratify = labels if len(np.unique(labels)) > 1 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=train_cfg.get("test_size", 0.2),
        random_state=train_cfg.get("seed", 42),
        stratify=stratify,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=train_cfg.get("val_size", 0.1),
        random_state=train_cfg.get("seed", 42),
        stratify=labels[train_idx] if stratify is not None else None,
    )

    train_set = FusionDataset(features[train_idx], masks[train_idx], labels[train_idx])
    val_set = FusionDataset(features[val_idx], masks[val_idx], labels[val_idx])
    test_set = FusionDataset(features[test_idx], masks[test_idx], labels[test_idx])

    batch_size = int(train_cfg.get("batch_size", 128))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionFusionModel(
        num_domains=num_domains,
        input_dim=input_dim,
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
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )

    best_pr_auc = -1.0
    patience = 0
    max_patience = int(train_cfg.get("early_stopping", 5))
    domain_dropout = float(train_cfg.get("domain_dropout", 0.0))
    lambda_reg = float(train_cfg.get("lambda_reg", 0.01))

    for epoch in range(int(train_cfg.get("epochs", 10))):
        model.train()
        epoch_losses = []
        for features_batch, masks_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            masks_batch = masks_batch.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(-1)
            masks_batch = apply_domain_dropout(masks_batch, domain_dropout)

            optimizer.zero_grad()
            logits, attn_weights, confidences = model(features_batch, key_padding_mask=masks_batch)
            loss, _ = attention_fusion_loss(
                logits,
                labels_batch,
                attn_weights,
                confidences=confidences,
                lambda_reg=lambda_reg,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        val_metrics = evaluate_model(model, val_loader, device)
        pr_auc = val_metrics.get("pr_auc", float("nan"))
        print(f"Epoch {epoch + 1}: train_loss={np.mean(epoch_losses):.4f}, val_pr_auc={pr_auc:.4f}")

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            patience = 0
            output_dir = _resolve_path(output_cfg.get("model_dir", "models/fusion/attention"))
            output_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = output_dir / output_cfg.get("checkpoint_name", "attention_fusion.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "domain_order": domain_order,
                    "feature_columns": feature_columns,
                    "config": cfg,
                },
                ckpt_path,
            )
        else:
            patience += 1
            if patience >= max_patience:
                print("Early stopping triggered.")
                break

    test_metrics = evaluate_model(model, test_loader, device)
    metrics_out = {
        "val_best_pr_auc": best_pr_auc,
        "test": test_metrics,
        "data_stats": {**schema_stats, **prep_stats, "data_hash": hash_file(data_path)},
        "domain_order": domain_order,
        "feature_columns": feature_columns,
    }

    metrics_path = output_cfg.get("metrics_path")
    if metrics_path:
        out_path = _resolve_path(metrics_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics_out, indent=2))

    mlflow_cfg = cfg.get("mlflow", {})
    if mlflow_cfg.get("enabled"):
        try:
            settings = load_mlflow_settings()
            setup_mlflow(
                experiment_name=mlflow_cfg.get("experiment_name", settings.get("experiment_name", "UAISV_Experiments")),
                tracking_uri=mlflow_cfg.get("tracking_uri", settings.get("tracking_uri")),
            )
            params = {
                "num_domains": num_domains,
                "embed_dim": int(model_cfg.get("embed_dim", 64)),
                "num_heads": int(model_cfg.get("num_heads", 8)),
                "num_layers": int(model_cfg.get("num_layers", 1)),
                "use_attention": bool(model_cfg.get("use_attention", True)),
                "use_confidence": bool(model_cfg.get("use_confidence", True)),
                "use_input_confidence": bool(model_cfg.get("use_input_confidence", True)),
                "domain_dropout": float(train_cfg.get("domain_dropout", 0.0)),
            }
            metrics = {"val_best_pr_auc": best_pr_auc}
            metrics.update(test_metrics)
            log_run(params=params, metrics=metrics)
        except Exception as exc:  # pragma: no cover - optional logging
            print(f"[warn] MLflow logging failed: {exc}")

    return model, metrics_out


if __name__ == "__main__":  # pragma: no cover
    train_attention_fusion()
