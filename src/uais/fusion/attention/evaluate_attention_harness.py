"""Evaluation harness for attention fusion ablations and robustness checks."""

from __future__ import annotations

import json
import resource
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.optimize import nnls
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from uais.fusion.attention.attention_utils import (
    FusionDataset,
    apply_domain_dropout,
    build_fusion_tensors,
    compute_domain_reliability,
    infer_feature_columns,
    load_fusion_dataframe,
)
from uais.fusion.attention.counterfactual_explainer import CounterfactualDomainExplainer
from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.fusion.attention.training_loop import train_attention_model
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator
from uais.fusion.attention.fusion_training_utils import attention_fusion_loss
from uais.fusion.attention.train_attention_fusion import set_seed
from uais.utils.config_loader import load_yaml
from uais.utils.metrics import classification_metrics
from uais.utils.paths import PROJECT_ROOT
from uais.utils.stats import bootstrap_ci, delong_roc_test

DEFAULT_CONFIG = Path("src/uais/fusion/attention/attention_config.yaml")


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _prepare_data(cfg: dict, include_confidence: bool | None = None):
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    data_path = _resolve_path(data_cfg.get("path", ""))
    df = load_fusion_dataframe(data_path)

    include_conf = (
        bool(model_cfg.get("use_input_confidence", True)) if include_confidence is None else include_confidence
    )
    feature_columns = infer_feature_columns(
        df,
        score_column=data_cfg.get("score_column", "score"),
        confidence_column=data_cfg.get("confidence_column", "confidence"),
        embedding_prefix=data_cfg.get("embedding_prefix", "embedding_"),
        feature_columns=data_cfg.get("feature_columns") or None,
        include_confidence=include_conf,
    )
    confidence_column = data_cfg.get("confidence_column", "confidence")
    confidence_index = (
        feature_columns.index(confidence_column) if confidence_column in feature_columns and include_conf else None
    )

    domain_order_cfg = data_cfg.get("domain_order") or model_cfg.get("domain_order")
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
        raise ValueError("Label column is required for evaluation.")

    score_column = data_cfg.get("score_column", "score")
    score_index = feature_columns.index(score_column) if score_column in feature_columns else None
    return features, masks, labels, sample_ids, domain_order, feature_columns, confidence_index, score_index


def _split_indices(labels: np.ndarray, train_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(labels.shape[0])
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
    return train_idx, val_idx, test_idx


def _build_model(
    model_cfg: dict,
    num_domains: int,
    input_dim: int,
    confidence_index: int | None,
) -> AttentionFusionModel:
    return AttentionFusionModel(
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
    )


def _collect_predictions(
    model: AttentionFusionModel,
    loader: DataLoader,
    device: torch.device,
    return_attention: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    all_probs = []
    all_labels = []
    all_attn = []
    with torch.no_grad():
        for batch in loader:
            features, masks, labels = batch
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            logits, attn_weights, _ = model(features, key_padding_mask=masks)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if return_attention:
                if attn_weights is None:
                    all_attn.append(torch.zeros((features.shape[0], features.shape[1]), device=features.device))
                else:
                    attn_mean = attn_weights.mean(dim=1).mean(dim=1)
                    all_attn.append(attn_mean)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    attn = None
    if return_attention:
        attn = torch.cat(all_attn, dim=0).cpu().numpy()
    return y_true, y_prob, attn


def _evaluate_model(model: AttentionFusionModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    y_true, y_prob, _ = _collect_predictions(model, loader, device, return_attention=False)
    return classification_metrics(y_true, y_prob, threshold=0.5)


def _train_model(
    cfg: dict,
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    domain_order: Sequence[str],
    confidence_index: int | None,
) -> tuple[AttentionFusionModel, dict[str, float], dict[str, float]]:
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    train_idx, val_idx, test_idx = _split_indices(labels, train_cfg)

    train_set = FusionDataset(features[train_idx], masks[train_idx], labels[train_idx])
    val_set = FusionDataset(features[val_idx], masks[val_idx], labels[val_idx])
    test_set = FusionDataset(features[test_idx], masks[test_idx], labels[test_idx])

    batch_size = int(train_cfg.get("batch_size", 128))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_cfg, len(domain_order), features.shape[-1], confidence_index).to(device)

    train_result = train_attention_model(model, train_loader, val_loader, train_cfg, device)

    y_true, y_prob, _ = _collect_predictions(model, test_loader, device, return_attention=False)
    test_metrics = classification_metrics(y_true, y_prob, threshold=0.5)
    return (
        model,
        {
            "val_best_pr_auc": train_result.val_best_pr_auc,
            "val_best_loss": train_result.val_best_loss,
            "epochs_run": train_result.epochs_run,
            "early_stopping_metric": train_result.early_stopping_metric,
            "restored_best_weights": train_result.restored_best_weights,
            "train_loss_last_epoch": train_result.train_loss_last_epoch,
            "test": test_metrics,
        },
        {
            "batch_size": batch_size,
            "device": device.type,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "y_true": y_true,
            "y_prob": y_prob,
        },
    )


def _evaluate_domain_dropout(
    model: AttentionFusionModel,
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    domain_order: Sequence[str],
    batch_size: int,
    device: torch.device,
    drop_probs: Sequence[float],
    drop_domains: Sequence[str],
    drop_pairs: bool,
) -> dict[str, dict[str, dict[str, float]]]:
    results: dict[str, dict[str, dict[str, float]]] = {"random": {}, "drop_domain": {}, "drop_pair": {}}
    for prob in drop_probs:
        new_masks = apply_domain_dropout(torch.as_tensor(masks), prob).cpu().numpy()
        dataset = FusionDataset(features, new_masks, labels)
        loader = DataLoader(dataset, batch_size=batch_size)
        results["random"][str(prob)] = _evaluate_model(model, loader, device)

    domain_to_idx = {domain: idx for idx, domain in enumerate(domain_order)}
    for domain in drop_domains:
        if domain not in domain_to_idx:
            continue
        new_masks = masks.copy()
        new_masks[:, domain_to_idx[domain]] = True
        dataset = FusionDataset(features, new_masks, labels)
        loader = DataLoader(dataset, batch_size=batch_size)
        results["drop_domain"][domain] = _evaluate_model(model, loader, device)

    if drop_pairs and len(domain_order) >= 2:
        from itertools import combinations

        for left, right in combinations(domain_order, 2):
            new_masks = masks.copy()
            new_masks[:, domain_to_idx[left]] = True
            new_masks[:, domain_to_idx[right]] = True
            dataset = FusionDataset(features, new_masks, labels)
            loader = DataLoader(dataset, batch_size=batch_size)
            results["drop_pair"][f"{left}+{right}"] = _evaluate_model(model, loader, device)
    return results


def _measure_performance(
    model: AttentionFusionModel,
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    warmup_batches: int,
    runs: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    dataset = FusionDataset(features, masks, labels)
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    total_samples = len(dataset)

    for idx, (features_batch, masks_batch, _) in enumerate(loader):
        if idx >= warmup_batches:
            break
        with torch.no_grad():
            _ = model(features_batch.to(device), key_padding_mask=masks_batch.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            for batch_idx, (features_batch, masks_batch, _) in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                _ = model(features_batch.to(device), key_padding_mask=masks_batch.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    avg_time = float(np.mean(timings)) if timings else float("nan")
    effective_samples = total_samples
    if max_batches is not None:
        effective_samples = min(total_samples, max_batches * batch_size)
    latency_ms = (avg_time / effective_samples) * 1000 if effective_samples > 0 else float("nan")
    throughput = effective_samples / avg_time if avg_time and avg_time > 0 else float("nan")

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if rss > 10**7 else rss / 1024

    return {
        "avg_runtime_sec": avg_time,
        "latency_ms_per_sample": latency_ms,
        "throughput_samples_per_sec": throughput,
        "max_rss_mb": float(rss_mb),
    }


def _collect_predictions_craf(
    model: AttentionFusionModel,
    estimator: ReliabilityEstimator,
    features_np: np.ndarray,
    masks_np: np.ndarray,
    labels_np: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect predictions using CRAF reliability weights injected at fusion layer."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for start in range(0, len(labels_np), batch_size):
        end = min(start + batch_size, len(labels_np))
        feat_batch = features_np[start:end]
        mask_batch = masks_np[start:end]
        lbl_batch = labels_np[start:end]

        rel_weights = estimator.compute_reliability_weights(feat_batch, mask_batch)
        gate = estimator.gate_decisions(rel_weights, mask_batch)  # [B] bool
        feat_t = torch.as_tensor(feat_batch, dtype=torch.float32).to(device)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.bool).to(device)

        with torch.no_grad():
            # Static path — always computed as the fallback
            logits_static, _, _ = model(feat_t, key_padding_mask=mask_t)
            probs_static = torch.sigmoid(logits_static.squeeze(-1))

            if gate.any():
                # Reliability path — only for samples below gate_threshold
                craf_t = torch.as_tensor(rel_weights, dtype=torch.float32).to(device)
                craf_t = craf_t.masked_fill(mask_t, 0.0)
                embeds = [enc(feat_t[:, i, :]) for i, enc in enumerate(model.domain_encoders)]
                domain_embeds = torch.stack(embeds, dim=1)
                logits_craf, _ = model.fusion(domain_embeds, key_padding_mask=mask_t, confidence_weights=craf_t)
                probs_craf = torch.sigmoid(logits_craf.squeeze(-1))
                gate_t = torch.as_tensor(gate, dtype=torch.bool).to(device)
                probs = torch.where(gate_t, probs_craf, probs_static).cpu().numpy()
            else:
                probs = probs_static.cpu().numpy()

        all_probs.append(probs)
        all_labels.append(lbl_batch)

    return np.concatenate(all_labels), np.concatenate(all_probs)


def _save_attention_samples(
    model: AttentionFusionModel,
    features: np.ndarray,
    masks: np.ndarray,
    sample_ids: Sequence[str],
    domain_order: Sequence[str],
    device: torch.device,
    max_samples: int,
    out_path: Path,
) -> None:
    if max_samples <= 0:
        return
    sample_features = torch.as_tensor(features[:max_samples], dtype=torch.float32).to(device)
    sample_masks = torch.as_tensor(masks[:max_samples], dtype=torch.bool).to(device)
    with torch.no_grad():
        _, attn_weights, _ = model(sample_features, key_padding_mask=sample_masks)
    if attn_weights is None:
        return
    weights = attn_weights.detach().cpu().numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        attention_weights=weights,
        domain_order=np.asarray(list(domain_order)),
        sample_ids=np.asarray(list(sample_ids[:max_samples])),
    )


def _bootstrap_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    def _f1_metric(y_true_arr: np.ndarray, y_prob_arr: np.ndarray) -> float:
        return classification_metrics(y_true_arr, y_prob_arr, threshold=0.5)["f1"]

    def _pr_auc_metric(y_true_arr: np.ndarray, y_prob_arr: np.ndarray) -> float:
        try:
            return float(average_precision_score(y_true_arr, y_prob_arr))
        except ValueError:
            return float("nan")

    pr_lower, pr_upper = bootstrap_ci(
        y_true, y_prob, _pr_auc_metric, n_bootstrap=n_bootstrap, alpha=alpha, random_state=seed
    )
    f1_lower, f1_upper = bootstrap_ci(
        y_true, y_prob, _f1_metric, n_bootstrap=n_bootstrap, alpha=alpha, random_state=seed
    )
    return {
        "pr_auc": {"lower": pr_lower, "upper": pr_upper},
        "f1": {"lower": f1_lower, "upper": f1_upper},
    }


def _stack_scores(scores: np.ndarray, masks: np.ndarray) -> np.ndarray:
    present = ~masks
    filled_scores = scores.copy()
    filled_scores[~present] = 0.0
    return np.concatenate([filled_scores, present.astype(np.float32)], axis=1)


def _baseline_predictions(
    scores: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray | None,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    domain_order: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    baselines: dict[str, dict[str, np.ndarray]] = {}
    present = ~masks
    denom = present.sum(axis=1).clip(min=1)
    avg_scores = (scores * present).sum(axis=1) / denom
    baselines["average"] = {
        "y_true": labels[test_idx],
        "y_prob": avg_scores[test_idx],
    }

    if confidence is not None:
        weights = confidence * present
        weight_sum = weights.sum(axis=1).clip(min=1e-6)
        conf_avg = (scores * weights).sum(axis=1) / weight_sum
        baselines["confidence_weighted_average"] = {
            "y_true": labels[test_idx],
            "y_prob": conf_avg[test_idx],
        }

    train_scores = scores[train_idx].copy()
    train_masks = masks[train_idx]
    train_scores[train_masks] = np.nan
    domain_weights = []
    for domain_idx, _domain in enumerate(domain_order):
        present = ~train_masks[:, domain_idx]
        if present.sum() < 2:
            domain_weights.append(0.0)
            continue
        try:
            auc = float(metrics.roc_auc_score(labels[train_idx][present], train_scores[present, domain_idx]))
        except ValueError:
            auc = 0.0
        domain_weights.append(max(auc, 0.0))
    domain_weights = np.asarray(domain_weights, dtype=float)
    if domain_weights.sum() > 0:
        domain_weights = domain_weights / domain_weights.sum()
        weighted_scores = []
        for idx in test_idx:
            present = ~masks[idx]
            weights = domain_weights.copy()
            weights[~present] = 0.0
            if weights.sum() == 0:
                weighted_scores.append(float(avg_scores[idx]))
            else:
                weights = weights / weights.sum()
                weighted_scores.append(float(np.sum(scores[idx] * weights)))
        baselines["reliability_weighted_average"] = {
            "y_true": labels[test_idx],
            "y_prob": np.asarray(weighted_scores),
        }

    X = _stack_scores(scores, masks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    try:
        model.fit(X_train, labels[train_idx])
        y_prob = model.predict_proba(X_test)[:, 1]
    except ValueError:
        fallback = float(np.mean(labels[train_idx])) if len(train_idx) > 0 else 0.5
        y_prob = np.full(len(test_idx), fallback)
    baselines["stacking_logreg"] = {"y_true": labels[test_idx], "y_prob": y_prob}

    means = np.nanmean(train_scores, axis=0)
    means = np.where(np.isfinite(means), means, 0.5)
    train_filled = np.where(np.isnan(train_scores), means, train_scores)
    test_scores = scores[test_idx].copy()
    test_scores[masks[test_idx]] = np.nan
    test_filled = np.where(np.isnan(test_scores), means, test_scores)
    X_train_blend = np.column_stack([train_filled, np.ones(train_filled.shape[0])])
    X_test_blend = np.column_stack([test_filled, np.ones(test_filled.shape[0])])
    try:
        weights, _ = nnls(X_train_blend, labels[train_idx])
        blend_pred = X_test_blend @ weights
        blend_pred = np.clip(blend_pred, 0.0, 1.0)
    except Exception:
        blend_pred = np.full(len(test_idx), float(np.mean(labels[train_idx])) if len(train_idx) > 0 else 0.5)
    baselines["blending_nnls"] = {"y_true": labels[test_idx], "y_prob": blend_pred}
    return baselines


def _interpretability_summary(
    model: AttentionFusionModel,
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    domain_order: Sequence[str],
    score_index: int | None,
    confidence_index: int | None,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    if score_index is None:
        return {}
    dataset = FusionDataset(features, masks, labels)
    loader = DataLoader(dataset, batch_size=batch_size)
    _, _, attn = _collect_predictions(model, loader, device, return_attention=True)
    attention_means: dict[str, float] = {}
    for idx, domain in enumerate(domain_order):
        present = ~masks[:, idx]
        if present.sum() == 0:
            attention_means[domain] = float("nan")
            continue
        attention_means[domain] = float(attn[present, idx].mean()) if attn is not None else float("nan")

    reliabilities = compute_domain_reliability(features, masks, labels, domain_order, score_index)
    confidence_means: dict[str, float] = {}
    if confidence_index is not None:
        conf = features[:, :, confidence_index]
        for idx, domain in enumerate(domain_order):
            present = ~masks[:, idx]
            if present.sum() == 0:
                confidence_means[domain] = float("nan")
            else:
                confidence_means[domain] = float(conf[present, idx].mean())

    def _spearman(a: dict[str, float], b: dict[str, float]) -> float:
        pairs = [(a[key], b[key]) for key in a if key in b and np.isfinite(a[key]) and np.isfinite(b[key])]
        if len(pairs) < 2:
            return float("nan")
        left, right = zip(*pairs)
        return float(stats.spearmanr(left, right).correlation)

    return {
        "attention_mean": attention_means,
        "domain_reliability": reliabilities,
        "confidence_mean": confidence_means,
        "spearman_attention_vs_reliability": _spearman(attention_means, reliabilities),
        "spearman_attention_vs_confidence": (
            _spearman(attention_means, confidence_means) if confidence_means else float("nan")
        ),
    }


def _aggregate_metric_dict(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}

    def _is_finite(value: float) -> bool:
        try:
            return bool(np.isfinite(value))
        except TypeError:
            return False

    keys = {key for metric in metric_dicts for key in metric}
    for key in sorted(keys):
        values = [metric[key] for metric in metric_dicts if key in metric and _is_finite(metric[key])]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return summary


def _aggregate_domain_values(domain_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}

    def _is_finite(value: float) -> bool:
        try:
            return bool(np.isfinite(value))
        except TypeError:
            return False

    keys = {key for metric in domain_dicts for key in metric}
    for key in sorted(keys):
        values = [metric[key] for metric in domain_dicts if key in metric and _is_finite(metric[key])]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return summary


def _aggregate_nested_metrics(
    nested_results: list[dict[str, dict[str, dict[str, float]]]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for category in ["random", "drop_domain", "drop_pair"]:
        category_entries: dict[str, dict[str, dict[str, float]]] = {}
        keys = {key for result in nested_results for key in result.get(category, {})}
        for key in sorted(keys):
            metric_dicts = [result.get(category, {}).get(key, {}) for result in nested_results]
            category_entries[key] = _aggregate_metric_dict(metric_dicts)
        summary[category] = category_entries
    return summary


def evaluate_attention_harness(cfg_path: Path = DEFAULT_CONFIG) -> dict[str, dict]:
    cfg = load_yaml(cfg_path)
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    base_seed = int(train_cfg.get("seed", 42))
    seeds = eval_cfg.get("seeds")
    if not seeds:
        num_seeds = int(eval_cfg.get("num_seeds", 1))
        seeds = [base_seed + idx for idx in range(num_seeds)]
    set_seed(base_seed)

    default_ablations = [
        {"name": "baseline", "overrides": {}},
        {"name": "no_attention", "overrides": {"use_attention": False}},
        {"name": "single_head", "overrides": {"num_heads": 1}},
        {"name": "heads_4", "overrides": {"num_heads": 4}},
        {"name": "heads_16", "overrides": {"num_heads": 16}},
        {"name": "no_domain_embeddings", "overrides": {"use_domain_embeddings": False}},
        {"name": "no_positional_embeddings", "overrides": {"use_positional_embeddings": False}},
        {"name": "no_confidence_estimator", "overrides": {"use_confidence": False}},
        {"name": "no_input_confidence", "overrides": {"use_input_confidence": False}},
    ]
    ablations = eval_cfg.get("ablations", default_ablations)

    drop_probs = eval_cfg.get("domain_dropout_probs", [0.0, 0.1, 0.3])
    drop_pairs = bool(eval_cfg.get("drop_domain_pairs", False))
    performance_runs = int(eval_cfg.get("performance_runs", 3))
    warmup_batches = int(eval_cfg.get("performance_warmup_batches", 2))
    n_bootstrap = int(eval_cfg.get("n_bootstrap", 200))
    alpha = float(eval_cfg.get("bootstrap_alpha", 0.05))

    results: dict[str, dict] = {}

    for ablation in ablations:
        name = ablation.get("name", "variant")
        overrides = ablation.get("overrides", {})
        base_cfg = json.loads(json.dumps(cfg))
        base_cfg.setdefault("model", {}).update(overrides)

        include_confidence = overrides.get("use_input_confidence")
        data_pack = _prepare_data(base_cfg, include_confidence=include_confidence)
        features, masks, labels, sample_ids, domain_order, _, confidence_index, score_index = data_pack

        embed_dim = int(base_cfg.get("model", {}).get("embed_dim", 64))
        heads = int(base_cfg.get("model", {}).get("num_heads", 8))
        if base_cfg.get("model", {}).get("use_attention", True) and embed_dim % heads != 0:
            results[name] = {"error": f"embed_dim {embed_dim} not divisible by num_heads {heads}"}
            continue

        seed_outputs: dict[str, dict] = {}
        test_metric_list: list[dict[str, float]] = []
        baseline_metric_lists: dict[str, list[dict[str, float]]] = {}
        baseline_p_lists: dict[str, list[dict[str, float]]] = {}
        interpretability_list: list[dict[str, object]] = []
        dropout_list: list[dict[str, dict[str, dict[str, float]]]] = []
        performance_list: list[dict[str, float]] = []

        for seed in seeds:
            local_cfg = json.loads(json.dumps(base_cfg))
            local_cfg.setdefault("training", {})["seed"] = seed
            set_seed(seed)

            model, metrics, perf_meta = _train_model(
                local_cfg,
                features,
                masks,
                labels,
                domain_order,
                confidence_index,
            )
            device = torch.device(perf_meta["device"])
            batch_size = perf_meta["batch_size"]
            train_idx = perf_meta["train_idx"]
            test_idx = perf_meta["test_idx"]

            y_true = perf_meta["y_true"]
            y_prob = perf_meta["y_prob"]
            ci = _bootstrap_summary(y_true, y_prob, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)

            test_features = features[test_idx]
            test_masks = masks[test_idx]
            test_labels = labels[test_idx]
            test_sample_ids = [sample_ids[idx] for idx in test_idx]

            dropout_metrics = _evaluate_domain_dropout(
                model,
                test_features,
                test_masks,
                test_labels,
                domain_order,
                batch_size,
                device,
                drop_probs,
                eval_cfg.get("drop_domains", domain_order),
                drop_pairs,
            )
            perf_batch_size = int(eval_cfg.get("performance_batch_size", batch_size))
            performance = _measure_performance(
                model,
                test_features,
                test_masks,
                test_labels,
                perf_batch_size,
                device,
                warmup_batches,
                performance_runs,
                max_batches=int(eval_cfg.get("performance_iters", 0)) or None,
            )

            attention_samples = int(eval_cfg.get("save_attention_samples", 0))
            attention_dir = eval_cfg.get(
                "attention_weights_dir",
                "experiments/fusion/attention_fusion/attention_weights",
            )
            if attention_samples > 0:
                out_path = _resolve_path(attention_dir) / f"attention_weights_seed_{seed}.npz"
                _save_attention_samples(
                    model,
                    test_features,
                    test_masks,
                    test_sample_ids,
                    domain_order,
                    device,
                    attention_samples,
                    out_path,
                )

            baseline_results = {}
            baseline_cis = {}
            baseline_p_values = {}
            if score_index is not None:
                scores = features[:, :, score_index]
                confidence = features[:, :, confidence_index] if confidence_index is not None else None
                baseline_preds = _baseline_predictions(
                    scores,
                    masks,
                    labels,
                    confidence,
                    train_idx,
                    test_idx,
                    domain_order,
                )
                for baseline_name, preds in baseline_preds.items():
                    baseline_metrics = classification_metrics(preds["y_true"], preds["y_prob"], threshold=0.5)
                    baseline_results[baseline_name] = baseline_metrics
                    baseline_cis[baseline_name] = _bootstrap_summary(
                        preds["y_true"], preds["y_prob"], n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
                    )
                    baseline_p_values[baseline_name] = {
                        "delong_p": delong_roc_test(preds["y_true"], y_prob, preds["y_prob"])
                    }
                    baseline_metric_lists.setdefault(baseline_name, []).append(baseline_metrics)
                    baseline_p_lists.setdefault(baseline_name, []).append(baseline_p_values[baseline_name])

            interpretability = _interpretability_summary(
                model,
                test_features,
                test_masks,
                test_labels,
                domain_order,
                score_index,
                confidence_index,
                batch_size,
                device,
            )

            # --- CRAF evaluation (gated by enable_craf) ---
            craf_metrics: dict = {}
            if eval_cfg.get("enable_craf", False) and score_index is not None:
                try:
                    val_idx = perf_meta["val_idx"]
                    rel_cfg_h = cfg.get("reliability", {})
                    estimator = ReliabilityEstimator(
                        domain_order=list(domain_order),
                        score_index=score_index,
                        ece_weight=rel_cfg_h.get("ece_weight", 0.45),
                        ks_weight=rel_cfg_h.get("ks_weight", 0.35),
                        sharpness_weight=rel_cfg_h.get("sharpness_weight", 0.20),
                        gate_threshold=rel_cfg_h.get("gate_threshold", 0.66),
                    )
                    estimator.fit(features[val_idx], masks[val_idx], labels[val_idx])
                    craf_y_true, craf_y_prob = _collect_predictions_craf(
                        model, estimator, test_features, test_masks, test_labels, batch_size, device
                    )
                    craf_m = classification_metrics(craf_y_true, craf_y_prob, threshold=0.5)
                    craf_delong_p = delong_roc_test(y_true, craf_y_prob, y_prob)
                    craf_metrics = {
                        "metrics": craf_m,
                        "delong_vs_static_p": float(craf_delong_p),
                        "domain_ece": estimator.get_domain_ece(),
                    }
                except Exception as exc:
                    craf_metrics = {"error": str(exc)}

            # --- CDA evaluation (gated by enable_cda) ---
            cda_output: dict = {}
            if eval_cfg.get("enable_cda", False) and score_index is not None:
                try:
                    cda_n = int(eval_cfg.get("cda_samples", 100))
                    cda_features = test_features[:cda_n]
                    cda_masks = test_masks[:cda_n]
                    cda_ids = test_sample_ids[:cda_n]
                    explainer = CounterfactualDomainExplainer(model, list(domain_order), device=device)
                    cf_results = explainer.explain_batch(cda_features, cda_masks, cda_ids)
                    mean_cf_impacts = {
                        d: float(np.nanmean([abs(r.cf_impacts.get(d, float("nan"))) for r in cf_results]))
                        for d in domain_order
                    }
                    cda_output = {
                        "mean_cf_impacts": mean_cf_impacts,
                        "sample_narratives": [r.narrative for r in cf_results[:5]],
                    }
                except Exception as exc:
                    cda_output = {"error": str(exc)}

            seed_outputs[str(seed)] = {
                "metrics": metrics,
                "metrics_ci": ci,
                "baselines": baseline_results,
                "baselines_ci": baseline_cis,
                "baselines_p_values": baseline_p_values,
                "interpretability": interpretability,
                "domain_dropout": dropout_metrics,
                "performance": performance,
                "craf": craf_metrics,
                "cda": cda_output,
            }

            test_metric_list.append(metrics["test"])
            interpretability_list.append(interpretability)
            dropout_list.append(dropout_metrics)
            performance_list.append(performance)

        summary = {
            "metrics": _aggregate_metric_dict(test_metric_list),
            "baselines": {name: _aggregate_metric_dict(vals) for name, vals in baseline_metric_lists.items()},
            "baselines_p_values": {name: _aggregate_metric_dict(vals) for name, vals in baseline_p_lists.items()},
            "domain_dropout": _aggregate_nested_metrics(dropout_list),
            "performance": _aggregate_metric_dict(performance_list),
        }

        if interpretability_list:
            attention_means = [item.get("attention_mean", {}) for item in interpretability_list]
            reliabilities = [item.get("domain_reliability", {}) for item in interpretability_list]
            confidences = [item.get("confidence_mean", {}) for item in interpretability_list]
            corr_rel = [item.get("spearman_attention_vs_reliability", float("nan")) for item in interpretability_list]
            corr_conf = [item.get("spearman_attention_vs_confidence", float("nan")) for item in interpretability_list]
            summary["interpretability"] = {
                "attention_mean": _aggregate_domain_values(attention_means),
                "domain_reliability": _aggregate_domain_values(reliabilities),
                "confidence_mean": _aggregate_domain_values(confidences),
                "spearman_attention_vs_reliability": _aggregate_metric_dict([{"value": v} for v in corr_rel]).get(
                    "value", {}
                ),
                "spearman_attention_vs_confidence": _aggregate_metric_dict([{"value": v} for v in corr_conf]).get(
                    "value", {}
                ),
            }

        results[name] = {
            "seeds": seed_outputs,
            "summary": summary,
            "model_overrides": overrides,
        }

    output_cfg = cfg.get("output", {})
    output_path = eval_cfg.get("metrics_path") or output_cfg.get(
        "harness_metrics_path",
        "experiments/fusion/attention_fusion/harness_metrics.json",
    )
    out_path = _resolve_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":  # pragma: no cover
    evaluate_attention_harness()
