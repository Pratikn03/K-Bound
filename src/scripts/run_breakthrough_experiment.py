"""End-to-end reliability-gated multimodal fusion experiment.

Produces auditable benchmark evidence across 8 phases:

  Phase 0 — Setup: load config, split data, set seeds
  Phase 1 — Train + fit ReliabilityEstimator on val split
  Phase 2 — clean data: reliability-gated attention vs static attention + baselines
  Phase 3 — Table 2 / Figure 1: domain-shift drift curves; reliability_degradation_auc
  Phase 4 — Table 3: adversarial attack robustness
  Phase 5 — Table 4: missing-modality extended dropout sweep
  Phase 6 — Table 5: calibration (ECE, Brier, bin-level)
  Phase 7 — CDA validation: counterfactual impacts on 100 samples; Spearman vs SHAP
  Phase 8 — Statistical tests + save JSON

Usage:
  python src/scripts/run_breakthrough_experiment.py
  python src/scripts/run_breakthrough_experiment.py --config path/to/config.yaml
  python src/scripts/run_breakthrough_experiment.py --synthetic --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn import metrics as sk_metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from uais.fusion.attention.adversarial_robustness import (
    AdversarialAttackType,
    AdversarialPerturbationEngine,
)
from uais.fusion.attention.baselines import run_baseline_suite
from uais.fusion.attention.attention_utils import (
    FusionDataset,
    apply_domain_dropout,
    build_fusion_tensors,
    infer_feature_columns,
    load_fusion_dataframe,
)
from uais.fusion.attention.counterfactual_explainer import CounterfactualDomainExplainer
from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.fusion.attention.learned_gate import LearnedGateConfig, LearnedReliabilityGate
from uais.fusion.attention.meta_router import fit_rga_meta_router
from uais.fusion.attention.reliability_boosted_fusion import ReliabilityBoostedFusion
from uais.fusion.attention.causal_attribution import (
    estimate_all_domain_effects,
    estimate_all_interventional_ates,
)
from uais.fusion.attention.reliability_estimator import (
    CategoryAwareReliabilityEstimator,
    ReliabilityEstimator,
)
from uais.fusion.attention.train_attention_fusion import attention_fusion_loss, set_seed
from uais.utils.config_loader import load_yaml
from uais.utils.metrics import (
    brier_score,
    classification_metrics,
    expected_calibration_error,
    reliability_degradation_auc,
    select_decision_threshold,
)
from uais.utils.paths import PROJECT_ROOT
from uais.utils.result_aggregation import aggregate_stress_rows, summarize_seed_metric_rows, summarize_values
from uais.utils.stats import bootstrap_ci, delong_roc_test, paired_ttest

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("src/uais/fusion/attention/attention_config.yaml")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _resolve(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    repo_root = PROJECT_ROOT.parent if PROJECT_ROOT.name == "src" else PROJECT_ROOT
    candidate = repo_root / path
    if candidate.exists() or str(path).startswith(("src/", "configs/", "experiments/", "docs/", "data/")):
        return candidate
    return PROJECT_ROOT / path


def _load_data(cfg: Dict):
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    path = _resolve(data_cfg.get("path", ""))
    df = load_fusion_dataframe(path)

    include_conf = bool(model_cfg.get("use_input_confidence", True))
    feature_columns = infer_feature_columns(
        df,
        score_column=data_cfg.get("score_column", "score"),
        confidence_column=data_cfg.get("confidence_column", "confidence"),
        embedding_prefix=data_cfg.get("embedding_prefix", "embedding_"),
        feature_columns=data_cfg.get("feature_columns") or None,
        include_confidence=include_conf,
    )
    conf_col = data_cfg.get("confidence_column", "confidence")
    confidence_index = (
        feature_columns.index(conf_col) if (conf_col in feature_columns and include_conf) else None
    )
    score_col = data_cfg.get("score_column", "score")
    score_index = feature_columns.index(score_col) if score_col in feature_columns else 0

    features, masks, labels, sample_ids, domain_order = build_fusion_tensors(
        df,
        id_column=data_cfg.get("id_column", "sample_id"),
        domain_column=data_cfg.get("domain_column", "domain"),
        label_column=data_cfg.get("label_column", "label"),
        score_column=score_col,
        confidence_column=conf_col,
        embedding_prefix=data_cfg.get("embedding_prefix", "embedding_"),
        timestamp_column=data_cfg.get("timestamp_column"),
        domain_order=data_cfg.get("domain_order") or model_cfg.get("domain_order"),
        feature_columns=feature_columns,
    )
    if labels is None:
        raise ValueError("Label column required for experiment.")
    split_column = train_cfg.get("split_column") or data_cfg.get("split_column")
    sample_splits = None
    if split_column:
        sample_splits = _sample_column_values(
            df,
            sample_ids=sample_ids,
            value_column=str(split_column),
            id_column=data_cfg.get("id_column", "sample_id"),
            timestamp_column=data_cfg.get("timestamp_column"),
        )
    category_column = data_cfg.get("category_column")
    sample_categories: np.ndarray | None = None
    if category_column:
        sample_categories = _sample_column_values(
            df,
            sample_ids=sample_ids,
            value_column=str(category_column),
            id_column=data_cfg.get("id_column", "sample_id"),
            timestamp_column=data_cfg.get("timestamp_column"),
        ).astype(str)
    return (
        features,
        masks,
        labels,
        sample_ids,
        domain_order,
        feature_columns,
        confidence_index,
        score_index,
        sample_splits,
        sample_categories,
    )


def _sample_column_values(
    df,
    sample_ids: list,
    value_column: str,
    id_column: str = "sample_id",
    timestamp_column: str | None = None,
) -> np.ndarray:
    if value_column not in df.columns:
        raise ValueError(f"Configured split column '{value_column}' is missing from fusion data.")
    if id_column not in df.columns:
        raise ValueError(f"Missing id column: {id_column}")

    if timestamp_column and timestamp_column in df.columns:
        keys = df[id_column].astype(str).fillna("") + "::" + df[timestamp_column].astype(str).fillna("")
    else:
        keys = df[id_column]

    work = pd.DataFrame({"_sample_key": keys, value_column: df[value_column]})
    unique_counts = work.groupby("_sample_key")[value_column].nunique(dropna=True)
    conflicts = unique_counts[unique_counts > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting '{value_column}' values for {len(conflicts)} sample_ids.")

    mapping = work.dropna(subset=[value_column]).groupby("_sample_key")[value_column].first()
    missing = [sample_id for sample_id in sample_ids if sample_id not in mapping.index]
    if missing:
        preview = ", ".join(str(item) for item in missing[:5])
        raise ValueError(f"Missing '{value_column}' value for sample_ids: {preview}")
    return mapping.reindex(sample_ids).to_numpy()


def _configured_split_values(train_cfg: Dict, key: str, default: list[str]) -> set[str]:
    values = train_cfg.get(key, default)
    if isinstance(values, str):
        values = [values]
    return {str(value) for value in values}


def _split(labels: np.ndarray, train_cfg: Dict, split_values: np.ndarray | None = None):
    if split_values is not None:
        split_values = np.asarray(split_values)
        if len(split_values) != len(labels):
            raise ValueError("predefined split values must have one entry per label.")
        train_values = _configured_split_values(train_cfg, "train_split_values", ["train"])
        val_values = _configured_split_values(train_cfg, "val_split_values", ["validation", "val"])
        test_values = _configured_split_values(train_cfg, "test_split_values", ["test"])
        overlap = (train_values & val_values) | (train_values & test_values) | (val_values & test_values)
        if overlap:
            raise ValueError(f"Predefined split values overlap across roles: {sorted(overlap)}")

        split_str = split_values.astype(str)
        train_mask = np.isin(split_str, list(train_values))
        val_mask = np.isin(split_str, list(val_values))
        test_mask = np.isin(split_str, list(test_values))
        assigned = train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int)
        if np.any(assigned == 0):
            unknown = sorted(set(split_str[assigned == 0]))
            raise ValueError(f"Predefined split values not assigned to train/validation/test: {unknown}")
        if np.any(assigned > 1):
            raise ValueError("Predefined split assignment produced overlapping sample indices.")

        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)
        test_idx = np.flatnonzero(test_mask)
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError("Predefined splits must provide non-empty train, validation, and test sets.")
        return train_idx, val_idx, test_idx

    idx = np.arange(len(labels))
    stratify = labels if len(np.unique(labels)) > 1 else None
    train_idx, test_idx = train_test_split(
        idx,
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


def _make_loaders(features, masks, labels, train_idx, val_idx, test_idx, batch_size: int):
    def _loader(idx, shuffle):
        ds = FusionDataset(features[idx], masks[idx], labels[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)


# ---------------------------------------------------------------------------
# Model construction + inference helpers
# ---------------------------------------------------------------------------

def _build_model(cfg: Dict, num_domains: int, input_dim: int, confidence_index: int | None, device: torch.device) -> AttentionFusionModel:
    m = cfg.get("model", {})
    model = AttentionFusionModel(
        num_domains=num_domains,
        input_dim=input_dim,
        embed_dim=int(m.get("embed_dim", 64)),
        num_heads=int(m.get("num_heads", 8)),
        num_layers=int(m.get("num_layers", 1)),
        dropout=float(m.get("dropout", 0.1)),
        use_confidence=bool(m.get("use_confidence", True)),
        use_input_confidence=bool(m.get("use_input_confidence", True)),
        confidence_index=confidence_index,
        use_attention=bool(m.get("use_attention", True)),
        use_domain_embeddings=bool(m.get("use_domain_embeddings", True)),
        use_positional_embeddings=bool(m.get("use_positional_embeddings", True)),
        use_missing_embedding=bool(m.get("use_missing_embedding", True)),
    )
    return model.to(device)


def _component_weights(rel_cfg: Dict, disabled: Tuple[str, ...] = ()) -> Dict[str, float]:
    """Return normalized reliability-component weights with selected terms removed."""
    name_to_key = {
        "ece": "ece_weight",
        "ks": "ks_weight",
        "sharpness": "sharpness_weight",
    }
    disabled_set = set(disabled)
    unknown = disabled_set.difference(name_to_key)
    if unknown:
        raise ValueError(f"Unknown reliability components: {sorted(unknown)}")

    weights = {
        key: 0.0 if name in disabled_set else float(rel_cfg.get(key, default))
        for name, key, default in [
            ("ece", "ece_weight", 0.4),
            ("ks", "ks_weight", 0.4),
            ("sharpness", "sharpness_weight", 0.2),
        ]
    }
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("At least one reliability component must remain enabled.")
    return {key: value / total for key, value in weights.items()}


def _make_reliability_estimator(
    rel_cfg: Dict,
    domain_order: List[str],
    score_index: int,
    disabled_components: Tuple[str, ...] = (),
) -> ReliabilityEstimator:
    weights = _component_weights(rel_cfg, disabled=disabled_components)
    return ReliabilityEstimator(
        domain_order=domain_order,
        score_index=score_index,
        ece_weight=weights["ece_weight"],
        ks_weight=weights["ks_weight"],
        sharpness_weight=weights["sharpness_weight"],
        n_calibration_bins=int(rel_cfg.get("n_calibration_bins", 10)),
        min_samples_for_ks=int(rel_cfg.get("min_samples_for_ks", 30)),
    )


def _train_model(model: AttentionFusionModel, train_loader, val_loader, cfg: Dict, device: torch.device) -> None:
    t_cfg = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(t_cfg.get("lr", 1e-3)),
        weight_decay=float(t_cfg.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    patience = int(t_cfg.get("early_stopping", 5))
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(int(t_cfg.get("epochs", 20))):
        model.train()
        for batch in train_loader:
            feats, msks, lbls = [x.to(device) for x in batch]
            domain_dropout_p = float(t_cfg.get("domain_dropout", 0.1))
            if domain_dropout_p > 0.0:
                msks = apply_domain_dropout(msks, drop_prob=domain_dropout_p)
            optimizer.zero_grad()
            logits, attn_weights, confidences = model(feats, key_padding_mask=msks)
            loss, _ = attention_fusion_loss(
                logits.squeeze(-1),
                lbls,
                attn_weights,
                confidences,
                lambda_reg=float(t_cfg.get("lambda_reg", 0.01)),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                feats, msks, lbls = [x.to(device) for x in batch]
                logits, attn_weights, confidences = model(feats, key_padding_mask=msks)
                loss, _ = attention_fusion_loss(
                    logits.squeeze(-1),
                    lbls,
                    attn_weights,
                    confidences,
                    lambda_reg=0.0,
                )
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break


@torch.no_grad()
def _predict_static(model: AttentionFusionModel, features: np.ndarray, masks: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    probs = []
    n = features.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feat_t = torch.tensor(features[start:end], dtype=torch.float32, device=device)
        mask_t = torch.tensor(masks[start:end], dtype=torch.bool, device=device)
        logits, _, _ = model(feat_t, key_padding_mask=mask_t)
        probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
    return np.concatenate(probs)


def _delong_pairs_against_baselines(
    test_labels: np.ndarray,
    rga_router_probs: np.ndarray,
    rga_boosted_probs: np.ndarray,
    baseline_predictions: dict,
    *,
    static_probs: np.ndarray | None = None,
) -> dict:
    """Compute DeLong's paired ROC p-value for RGA+ (router/boosted)
    against every non-router baseline available in this run.

    For each baseline name we return both p-values (router vs baseline,
    boost vs baseline). The milestone-2 emitter combines the two RGA+
    variants downstream by picking the max(router, boost) AUROC and the
    corresponding p-value.
    """
    out: dict = {}
    pairs: dict[str, np.ndarray] = {}
    if static_probs is not None:
        pairs["static_attention"] = np.asarray(static_probs, dtype=np.float64)
    for name, payload in baseline_predictions.items():
        probs = payload.get("test_probs") if isinstance(payload, dict) else None
        if probs is None:
            continue
        pairs[name] = np.asarray(probs, dtype=np.float64)
    test_labels = np.asarray(test_labels, dtype=int)
    for variant_name, variant_probs in (
        ("rga_meta_router", np.asarray(rga_router_probs, dtype=np.float64)),
        ("rga_boosted_fusion", np.asarray(rga_boosted_probs, dtype=np.float64)),
    ):
        per_baseline: dict[str, float] = {}
        for baseline_name, baseline_probs in pairs.items():
            if len(baseline_probs) != len(variant_probs):
                continue
            try:
                p = float(delong_roc_test(test_labels, variant_probs, baseline_probs))
            except Exception:
                p = float("nan")
            per_baseline[baseline_name] = p
        out[variant_name] = per_baseline
    return out


def _calibrate_polarity(
    model: AttentionFusionModel,
    val_feat: np.ndarray,
    val_mask: np.ndarray,
    val_labels: np.ndarray,
    score_index: int,
    device: torch.device,
    *,
    perturb_multiplier: float = 1.4,
    synthetic_fraction: float = 0.5,
    random_seed: int = 0,
) -> dict:
    """Detect output-polarity inversion on a synthetic-anomaly-augmented val set.

    Under canonical one-class training (val and train both normal-only) the
    supervised fusion head receives no anomaly gradient and can settle into
    an inverse-polarity solution where higher input scores predict *lower*
    anomaly probabilities. This helper builds a synthetic-anomaly-augmented
    calibration set by perturbing the score column upward on a fraction of
    val samples, computes the model's AUROC on that set, and returns
    ``flip_required = True`` when the AUROC is below 0.5. The flip is
    applied at evaluation time only; the trained model is never modified.
    """
    rng = np.random.default_rng(int(random_seed))
    n_val = val_feat.shape[0]
    if n_val < 4:
        return {"flip_required": False, "calibration_auroc": float("nan"), "n_calibration": int(n_val), "n_synthetic": 0}
    n_synth = max(2, int(round(synthetic_fraction * n_val)))
    perturb_idx = rng.choice(n_val, size=n_synth, replace=False)
    synth_feat = val_feat[perturb_idx].copy()
    synth_mask = val_mask[perturb_idx].copy()
    synth_feat[:, :, score_index] = np.clip(
        synth_feat[:, :, score_index] * float(perturb_multiplier), 0.0, 1.0
    )
    cal_feat = np.concatenate([val_feat, synth_feat], axis=0)
    cal_mask = np.concatenate([val_mask, synth_mask], axis=0)
    cal_labels = np.concatenate(
        [np.asarray(val_labels, dtype=int), np.ones(n_synth, dtype=int)]
    )
    if len(np.unique(cal_labels)) < 2:
        return {"flip_required": False, "calibration_auroc": float("nan"), "n_calibration": int(len(cal_labels)), "n_synthetic": int(n_synth)}
    cal_probs = _predict_static(model, cal_feat, cal_mask, device)
    try:
        auc = float(roc_auc_score(cal_labels, cal_probs))
    except ValueError:
        return {"flip_required": False, "calibration_auroc": float("nan"), "n_calibration": int(len(cal_labels)), "n_synthetic": int(n_synth)}
    return {
        "flip_required": bool(auc < 0.5),
        "calibration_auroc": auc,
        "n_calibration": int(len(cal_labels)),
        "n_synthetic": int(n_synth),
    }


@torch.no_grad()
def _predict_craf(
    model: AttentionFusionModel,
    estimator: ReliabilityEstimator,
    features: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    clean_gate_threshold: float = 0.66,
    per_sample_gating: bool = False,
) -> np.ndarray:
    probs, _ = _predict_craf_with_stats(
        model,
        estimator,
        features,
        masks,
        device,
        batch_size=batch_size,
        clean_gate_threshold=clean_gate_threshold,
        per_sample_gating=per_sample_gating,
    )
    return probs


def _gate_decision_stats(reliability_weights: np.ndarray, masks: np.ndarray, threshold: float) -> Dict[str, float | bool | int]:
    """Summarize whether the reliability gate adapts for one inference batch."""
    present_weights = reliability_weights[~masks]
    mean_reliability = float(np.nanmean(present_weights)) if present_weights.size else float("nan")
    adapted = bool(present_weights.size and mean_reliability < threshold)
    return {
        "adapted": adapted,
        "mean_reliability": mean_reliability,
        "n_present": int(present_weights.size),
        "n_samples": int(masks.shape[0]),
    }


@torch.no_grad()
def _predict_craf_with_stats(
    model: AttentionFusionModel,
    estimator: ReliabilityEstimator,
    features: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    clean_gate_threshold: float = 0.66,
    per_sample_gating: bool = False,
    learned_gate: Optional[LearnedReliabilityGate] = None,
) -> Tuple[np.ndarray, Dict[str, float | int]]:
    """Predict with reliability-gated attention weights.

    Two gating modes:
      - Batch-level (``per_sample_gating=False``, default): if the batch mean
        reliability over present domains is below ``clean_gate_threshold``, the
        whole batch is routed through the reliability path. Preserves
        bit-exact reproducibility for paper numbers generated to date.
      - Per-sample (``per_sample_gating=True``): for each sample independently,
        the mean reliability over its present domains is compared to the gate
        threshold (via ``estimator.gate_decisions``); samples above threshold
        keep the static path, samples below get the reliability-injected path.
        Matches the per-sample r_{i,d} formalism stated in the ELARA paper.
    """
    model.eval()
    probs = []
    n = features.shape[0]
    adapted_samples = 0
    adapted_batches = 0
    reliability_numer = 0.0
    reliability_denom = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feat_np = features[start:end]
        mask_np = masks[start:end]
        craf_w = estimator.compute_reliability_weights(feat_np, mask_np)
        feat_t = torch.tensor(feat_np, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device)
        gate_stats = _gate_decision_stats(craf_w, mask_np, clean_gate_threshold)
        batch_n = end - start
        if np.isfinite(float(gate_stats["mean_reliability"])):
            reliability_numer += float(gate_stats["mean_reliability"]) * batch_n
            reliability_denom += batch_n

        if per_sample_gating or learned_gate is not None:
            if learned_gate is not None:
                # Learned gate overrides the heuristic threshold.
                gate_per_sample = learned_gate.decide(craf_w, mask_np)
            else:
                # Per-sample decisions using the per-call threshold (so τ-sweep works).
                # Mirror estimator.gate_decisions logic but use clean_gate_threshold,
                # not the estimator's construction-time gate_threshold.
                n_present = (~mask_np).sum(axis=1).astype(np.float32)
                mean_r_per_sample = np.where(
                    n_present > 0,
                    craf_w.sum(axis=1) / np.maximum(n_present, 1.0),
                    0.0,
                )
                gate_per_sample = mean_r_per_sample < clean_gate_threshold  # [B] bool
            n_adapted = int(gate_per_sample.sum())
            if n_adapted == 0:
                logits, _, _ = model(feat_t, key_padding_mask=mask_t)
                probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
                continue
            # Both paths needed when at least one sample is gated.
            logits_static, _, _ = model(feat_t, key_padding_mask=mask_t)
            probs_static = torch.sigmoid(logits_static.squeeze(-1))
            craf_t = torch.tensor(craf_w, dtype=torch.float32, device=device)
            craf_t = craf_t.masked_fill(mask_t, 0.0)
            embeds = [enc(feat_t[:, i, :]) for i, enc in enumerate(model.domain_encoders)]
            domain_embeds = torch.stack(embeds, dim=1)
            logits_craf, _ = model.fusion(
                domain_embeds, key_padding_mask=mask_t, confidence_weights=craf_t
            )
            probs_craf = torch.sigmoid(logits_craf.squeeze(-1))
            gate_t = torch.tensor(gate_per_sample, dtype=torch.bool, device=device)
            batch_probs = torch.where(gate_t, probs_craf, probs_static)
            probs.append(batch_probs.cpu().numpy())
            adapted_samples += n_adapted
            adapted_batches += 1
            continue

        # Batch-level mode (default — preserves existing paper numbers)
        if not gate_stats["adapted"]:
            logits, _, _ = model(feat_t, key_padding_mask=mask_t)
            probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
            continue
        adapted_samples += batch_n
        adapted_batches += 1
        craf_t = torch.tensor(craf_w, dtype=torch.float32, device=device)
        craf_t = craf_t.masked_fill(mask_t, 0.0)
        embeds = [enc(feat_t[:, i, :]) for i, enc in enumerate(model.domain_encoders)]
        domain_embeds = torch.stack(embeds, dim=1)
        logits, _ = model.fusion(domain_embeds, key_padding_mask=mask_t, confidence_weights=craf_t)
        probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
    stats = {
        "adaptation_rate": float(adapted_samples / n) if n else 0.0,
        "adapted_batches": int(adapted_batches),
        "n_batches": int(math.ceil(n / batch_size)) if batch_size > 0 else 0,
        "mean_reliability": float(reliability_numer / reliability_denom) if reliability_denom else float("nan"),
        "per_sample_gating": bool(per_sample_gating or learned_gate is not None),
        "learned_gate": bool(learned_gate is not None),
    }
    return np.concatenate(probs), stats


# Baseline models are in baselines.py — run_baseline_suite() is the entry point.


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    result = []
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        result.append({
            "bin_center": float((bins[i] + bins[i + 1]) / 2),
            "mean_confidence": float(np.mean(y_prob[mask])),
            "mean_accuracy": float(np.mean(y_true[mask])),
            "count": int(np.sum(mask)),
        })
    return result


# ---------------------------------------------------------------------------
# Synthetic data for smoke testing
# ---------------------------------------------------------------------------

def _make_synthetic(n_samples: int = 800, n_domains: int = 3, n_features: int = 5, seed: int = 42) -> Tuple:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_samples) < 0.15).astype(np.float32)
    features = rng.random((n_samples, n_domains, n_features)).astype(np.float32)
    # Make scores (index 0) predictive
    for d in range(n_domains):
        features[labels == 1, d, 0] = np.clip(features[labels == 1, d, 0] + 0.35, 0, 1)
    # 15% random missingness
    masks = rng.random((n_samples, n_domains)) < 0.15
    sample_ids = list(range(n_samples))
    domain_order = [f"domain_{i}" for i in range(n_domains)]
    return features, masks.astype(bool), labels, sample_ids, domain_order


def _metric_bootstrap_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    seed: int,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float | None]]:
    """Bootstrap CIs for core probability metrics on one evaluated split."""

    def _safe_roc(y, p):
        try:
            return float(sk_metrics.roc_auc_score(y, p))
        except ValueError:
            return float("nan")

    def _safe_pr(y, p):
        try:
            return float(sk_metrics.average_precision_score(y, p))
        except ValueError:
            return float("nan")

    def _safe_f1(y, p):
        return float(sk_metrics.f1_score(y, (p >= threshold).astype(int), zero_division=0))

    out: Dict[str, Dict[str, float | None]] = {}
    for offset, (name, fn) in enumerate({"roc_auc": _safe_roc, "pr_auc": _safe_pr, "f1": _safe_f1}.items()):
        low, high = bootstrap_ci(
            y_true,
            y_prob,
            fn,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            random_state=seed + offset,
        )
        out[name] = {
            "ci_low": float(low) if np.isfinite(low) else None,
            "ci_high": float(high) if np.isfinite(high) else None,
        }
    return out


def _metrics_from_validation_threshold(
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    *,
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    strategy: str | None,
) -> Dict[str, float | str]:
    threshold = select_decision_threshold(val_labels, val_probs, strategy=strategy)
    metrics = classification_metrics(test_labels, test_probs, threshold=threshold)
    metrics["threshold_strategy"] = (strategy or "fixed_0p5").strip().lower()
    return metrics


def _json_float(value: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _fit_rga_meta_router_metrics(
    *,
    val_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    random_seed: int,
    threshold_strategy: str | None,
    selection_metric: str = "roc_auc",
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Fit and score the validation-only ELARA/RGA router.

    This is intentionally kept test-label blind. The router may select the
    original RGA path, static attention, a strong adapter, or a validation-fit
    stack, and the selected candidate is recorded so the paper cannot silently
    relabel a baseline as an RGA gain.
    """
    router = fit_rga_meta_router(
        val_predictions,
        val_labels,
        random_seed=random_seed,
        selection_metric=selection_metric,
    )
    router_val_probs = router.predict_proba(val_predictions)
    router_test_probs = router.predict_proba(test_predictions)
    metrics = _metrics_from_validation_threshold(
        test_labels,
        router_test_probs,
        val_labels=val_labels,
        val_probs=router_val_probs,
        strategy=threshold_strategy,
    )
    metrics["selected_candidate"] = router.selected_candidate
    metrics["selection_metric"] = selection_metric
    metrics["candidate_validation_roc_auc"] = {
        name: _json_float(values.get("roc_auc"))
        for name, values in sorted((router.candidate_metric_scores or {}).items())
    }
    metrics["candidate_validation_scores"] = {
        name: {metric: _json_float(score) for metric, score in sorted(values.items())}
        for name, values in sorted((router.candidate_metric_scores or {}).items())
    }
    return metrics, router_val_probs, router_test_probs


def _evaluate_drift(
    model: AttentionFusionModel,
    estimator: ReliabilityEstimator,
    test_feat: np.ndarray,
    test_mask: np.ndarray,
    test_labels: np.ndarray,
    domain_order: List[str],
    score_index: int,
    device: torch.device,
    noise_levels: List[float],
    clean_gate_threshold: float,
    seed: int,
    per_sample_gating: bool = False,
    static_decision_threshold: float = 0.5,
    craf_decision_threshold: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    engine = AdversarialPerturbationEngine(domain_order, score_index, random_seed=seed)
    curve_rows: list[dict] = []
    degradation_rows: list[dict] = []
    for domain in domain_order:
        drift_feats_map = engine.simulate_domain_drift(test_feat, test_mask, domain, noise_levels)
        auc_static_curve = []
        auc_craf_curve = []
        for level in noise_levels:
            pert_feat = drift_feats_map[level]
            s_probs = _predict_static(model, pert_feat, test_mask, device)
            c_probs = _predict_craf(
                model,
                estimator,
                pert_feat,
                test_mask,
                device,
                clean_gate_threshold=clean_gate_threshold,
                per_sample_gating=per_sample_gating,
            )
            s_m = classification_metrics(test_labels, s_probs, threshold=static_decision_threshold)
            c_m = classification_metrics(test_labels, c_probs, threshold=craf_decision_threshold)
            auc_static_curve.append(s_m.get("roc_auc", float("nan")))
            auc_craf_curve.append(c_m.get("roc_auc", float("nan")))
            curve_rows.append(
                {
                    "seed": int(seed),
                    "domain": domain,
                    "noise_level": float(level),
                    "static_auc": s_m.get("roc_auc"),
                    "craf_auc": c_m.get("roc_auc"),
                    "static_pr_auc": s_m.get("pr_auc"),
                    "craf_pr_auc": c_m.get("pr_auc"),
                    "static_f1": s_m.get("f1"),
                    "craf_f1": c_m.get("f1"),
                    "static_decision_threshold": float(static_decision_threshold),
                    "craf_decision_threshold": float(craf_decision_threshold),
                }
            )
        noise_arr = np.array(noise_levels, dtype=float)
        deg_auc_static = reliability_degradation_auc(noise_arr, np.array(auc_static_curve))
        deg_auc_craf = reliability_degradation_auc(noise_arr, np.array(auc_craf_curve))
        degradation_rows.append(
            {
                "seed": int(seed),
                "domain": domain,
                "static": float(deg_auc_static),
                "craf": float(deg_auc_craf),
                "delta": float(deg_auc_craf - deg_auc_static),
                "craf_better": bool(deg_auc_craf > deg_auc_static),
            }
        )
    return curve_rows, degradation_rows


def _evaluate_adversarial(
    model: AttentionFusionModel,
    estimator: ReliabilityEstimator,
    test_feat: np.ndarray,
    test_mask: np.ndarray,
    test_labels: np.ndarray,
    domain_order: List[str],
    score_index: int,
    device: torch.device,
    attack_names: List[str],
    sigma: float,
    clean_gate_threshold: float,
    seed: int,
    per_sample_gating: bool = False,
    static_decision_threshold: float = 0.5,
    craf_decision_threshold: float = 0.5,
) -> list[dict]:
    engine = AdversarialPerturbationEngine(domain_order, score_index, random_seed=seed)
    rows: list[dict] = []
    for attack_name in attack_names:
        try:
            attack_type = AdversarialAttackType(attack_name)
        except ValueError:
            continue
        for target in domain_order + [None]:
            pert_feat, _ = engine.apply_attack(
                test_feat,
                test_mask,
                attack_type,
                target_domain=target,
                sigma=sigma,
            )
            s_probs = _predict_static(model, pert_feat, test_mask, device)
            c_probs = _predict_craf(
                model,
                estimator,
                pert_feat,
                test_mask,
                device,
                clean_gate_threshold=clean_gate_threshold,
                per_sample_gating=per_sample_gating,
            )
            s_m = classification_metrics(test_labels, s_probs, threshold=static_decision_threshold)
            c_m = classification_metrics(test_labels, c_probs, threshold=craf_decision_threshold)
            rows.append(
                {
                    "seed": int(seed),
                    "attack": attack_name,
                    "target_domain": target if target is not None else "all",
                    "static_auc": s_m.get("roc_auc"),
                    "craf_auc": c_m.get("roc_auc"),
                    "static_pr_auc": s_m.get("pr_auc"),
                    "craf_pr_auc": c_m.get("pr_auc"),
                    "static_f1": s_m.get("f1"),
                    "craf_f1": c_m.get("f1"),
                    "static_decision_threshold": float(static_decision_threshold),
                    "craf_decision_threshold": float(craf_decision_threshold),
                    "delta_auc": (c_m.get("roc_auc", 0.0) or 0.0) - (s_m.get("roc_auc", 0.0) or 0.0),
                }
            )
    return rows


def _evaluate_missing(
    model: AttentionFusionModel,
    estimator: ReliabilityEstimator,
    test_feat: np.ndarray,
    test_mask: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    dropout_probs: List[float],
    clean_gate_threshold: float,
    seed: int,
    per_sample_gating: bool = False,
    static_decision_threshold: float = 0.5,
    craf_decision_threshold: float = 0.5,
) -> list[dict]:
    rows: list[dict] = []
    for p_drop in dropout_probs:
        rng_seed = np.random.default_rng(seed + int(float(p_drop) * 10_000))
        drop_mask = test_mask.copy()
        if p_drop > 0.0:
            drop_mask = drop_mask | (rng_seed.random(drop_mask.shape) < p_drop)
        s_probs = _predict_static(model, test_feat, drop_mask, device)
        c_probs = _predict_craf(
            model,
            estimator,
            test_feat,
            drop_mask,
            device,
            clean_gate_threshold=clean_gate_threshold,
            per_sample_gating=per_sample_gating,
        )
        s_m = classification_metrics(test_labels, s_probs, threshold=static_decision_threshold)
        c_m = classification_metrics(test_labels, c_probs, threshold=craf_decision_threshold)
        rows.append(
            {
                "seed": int(seed),
                "dropout_prob": float(p_drop),
                "static_auc": s_m.get("roc_auc"),
                "craf_auc": c_m.get("roc_auc"),
                "static_pr_auc": s_m.get("pr_auc"),
                "craf_pr_auc": c_m.get("pr_auc"),
                "static_f1": s_m.get("f1"),
                "craf_f1": c_m.get("f1"),
                "static_decision_threshold": float(static_decision_threshold),
                "craf_decision_threshold": float(craf_decision_threshold),
            }
        )
    return rows


def _all_domain_conditions(
    test_feat: np.ndarray,
    test_mask: np.ndarray,
    domain_order: List[str],
    score_index: int,
    attack_names: List[str],
    sigma: float,
    seed: int,
    include_clean: bool = True,
) -> list[dict]:
    conditions: list[dict] = []
    if include_clean:
        conditions.append(
            {
                "condition": "clean",
                "attack": "none",
                "target_domain": "none",
                "features": test_feat,
                "masks": test_mask,
            }
        )
    engine = AdversarialPerturbationEngine(domain_order, score_index, random_seed=seed)
    for attack_name in attack_names:
        try:
            attack_type = AdversarialAttackType(attack_name)
        except ValueError:
            continue
        pert_feat, pert_mask = engine.apply_attack(
            test_feat,
            test_mask,
            attack_type,
            target_domain=None,
            sigma=sigma,
        )
        conditions.append(
            {
                "condition": f"{attack_name}:all",
                "attack": attack_name,
                "target_domain": "all",
                "features": pert_feat,
                "masks": pert_mask,
            }
        )
    return conditions


def _evaluate_tau_sweep(
    model: AttentionFusionModel,
    estimator: ReliabilityEstimator,
    test_feat: np.ndarray,
    test_mask: np.ndarray,
    test_labels: np.ndarray,
    domain_order: List[str],
    score_index: int,
    device: torch.device,
    thresholds: List[float],
    attack_names: List[str],
    sigma: float,
    seed: int,
    per_sample_gating: bool = False,
    static_decision_threshold: float = 0.5,
    craf_decision_threshold: float = 0.5,
    learned_gate: Optional[LearnedReliabilityGate] = None,
) -> list[dict]:
    rows: list[dict] = []
    if not thresholds and learned_gate is None:
        return rows
    conditions = _all_domain_conditions(
        test_feat,
        test_mask,
        domain_order,
        score_index,
        attack_names,
        sigma,
        seed=seed + 17_000,
        include_clean=True,
    )
    for condition in conditions:
        condition_feat = condition["features"]
        condition_mask = condition["masks"]
        static_probs = _predict_static(model, condition_feat, condition_mask, device)
        static_m = classification_metrics(test_labels, static_probs, threshold=static_decision_threshold)
        for threshold in thresholds:
            craf_probs, gate_stats = _predict_craf_with_stats(
                model,
                estimator,
                condition_feat,
                condition_mask,
                device,
                clean_gate_threshold=float(threshold),
                per_sample_gating=per_sample_gating,
            )
            craf_m = classification_metrics(test_labels, craf_probs, threshold=craf_decision_threshold)
            rows.append(
                {
                    "seed": int(seed),
                    "condition": condition["condition"],
                    "attack": condition["attack"],
                    "target_domain": condition["target_domain"],
                    "tau": float(threshold),
                    "static_auc": static_m.get("roc_auc"),
                    "craf_auc": craf_m.get("roc_auc"),
                    "static_pr_auc": static_m.get("pr_auc"),
                    "craf_pr_auc": craf_m.get("pr_auc"),
                    "static_f1": static_m.get("f1"),
                    "craf_f1": craf_m.get("f1"),
                    "static_decision_threshold": float(static_decision_threshold),
                    "craf_decision_threshold": float(craf_decision_threshold),
                    "adaptation_rate": gate_stats["adaptation_rate"],
                    "mean_reliability": gate_stats["mean_reliability"],
                }
            )
        if learned_gate is not None:
            craf_probs, gate_stats = _predict_craf_with_stats(
                model,
                estimator,
                condition_feat,
                condition_mask,
                device,
                clean_gate_threshold=0.0,  # ignored when learned_gate is set
                per_sample_gating=True,
                learned_gate=learned_gate,
            )
            craf_m = classification_metrics(test_labels, craf_probs, threshold=craf_decision_threshold)
            rows.append(
                {
                    "seed": int(seed),
                    "condition": condition["condition"],
                    "attack": condition["attack"],
                    "target_domain": condition["target_domain"],
                    "tau": "learned",
                    "static_auc": static_m.get("roc_auc"),
                    "craf_auc": craf_m.get("roc_auc"),
                    "static_pr_auc": static_m.get("pr_auc"),
                    "craf_pr_auc": craf_m.get("pr_auc"),
                    "static_f1": static_m.get("f1"),
                    "craf_f1": craf_m.get("f1"),
                    "static_decision_threshold": float(static_decision_threshold),
                    "craf_decision_threshold": float(craf_decision_threshold),
                    "adaptation_rate": gate_stats["adaptation_rate"],
                    "mean_reliability": gate_stats["mean_reliability"],
                }
            )
    return rows


def _component_ablation_specs(names: List[str]) -> list[dict]:
    if not names:
        return []
    spec_by_name = {
        "full": {"variant": "full", "disabled_components": (), "gate_threshold": None},
        "no_ece": {"variant": "no_ece", "disabled_components": ("ece",), "gate_threshold": None},
        "no_ks": {"variant": "no_ks", "disabled_components": ("ks",), "gate_threshold": None},
        "no_sharpness": {
            "variant": "no_sharpness",
            "disabled_components": ("sharpness",),
            "gate_threshold": None,
        },
        "no_gate": {"variant": "no_gate", "disabled_components": (), "gate_threshold": 1.01},
    }
    unknown = [name for name in names if name not in spec_by_name]
    if unknown:
        raise ValueError(f"Unknown reliability ablation variants: {unknown}")
    return [spec_by_name[name] for name in names]


def _evaluate_component_ablation(
    model: AttentionFusionModel,
    rel_cfg: Dict,
    val_feat: np.ndarray,
    val_mask: np.ndarray,
    val_labels: np.ndarray,
    test_feat: np.ndarray,
    test_mask: np.ndarray,
    test_labels: np.ndarray,
    domain_order: List[str],
    score_index: int,
    device: torch.device,
    variant_names: List[str],
    attack_names: List[str],
    sigma: float,
    clean_gate_threshold: float,
    seed: int,
    per_sample_gating: bool = False,
    threshold_strategy: str | None = "fixed_0p5",
    static_decision_threshold: float = 0.5,
) -> list[dict]:
    rows: list[dict] = []
    specs = _component_ablation_specs(variant_names)
    if not specs:
        return rows

    variant_estimators: dict[str, tuple[ReliabilityEstimator, float, dict[str, float], float]] = {}
    for spec in specs:
        weights = _component_weights(rel_cfg, disabled=spec["disabled_components"])
        estimator = _make_reliability_estimator(
            rel_cfg,
            domain_order,
            score_index,
            disabled_components=spec["disabled_components"],
        )
        estimator.fit(val_feat, val_mask, val_labels)
        gate_threshold = clean_gate_threshold if spec["gate_threshold"] is None else float(spec["gate_threshold"])
        val_probs = _predict_craf(
            model,
            estimator,
            val_feat,
            val_mask,
            device,
            clean_gate_threshold=gate_threshold,
            per_sample_gating=per_sample_gating,
        )
        decision_threshold = select_decision_threshold(val_labels, val_probs, strategy=threshold_strategy)
        variant_estimators[spec["variant"]] = (estimator, gate_threshold, weights, decision_threshold)

    conditions = _all_domain_conditions(
        test_feat,
        test_mask,
        domain_order,
        score_index,
        attack_names,
        sigma,
        seed=seed + 29_000,
        include_clean=False,
    )
    for condition in conditions:
        condition_feat = condition["features"]
        condition_mask = condition["masks"]
        static_probs = _predict_static(model, condition_feat, condition_mask, device)
        static_m = classification_metrics(test_labels, static_probs, threshold=static_decision_threshold)
        for variant, (variant_estimator, threshold, weights, decision_threshold) in variant_estimators.items():
            craf_probs, gate_stats = _predict_craf_with_stats(
                model,
                variant_estimator,
                condition_feat,
                condition_mask,
                device,
                clean_gate_threshold=threshold,
                per_sample_gating=per_sample_gating,
            )
            craf_m = classification_metrics(test_labels, craf_probs, threshold=decision_threshold)
            rows.append(
                {
                    "seed": int(seed),
                    "variant": variant,
                    "attack": condition["attack"],
                    "target_domain": condition["target_domain"],
                    "gate_threshold": threshold,
                    "threshold_strategy": (threshold_strategy or "fixed_0p5").strip().lower(),
                    "static_decision_threshold": float(static_decision_threshold),
                    "craf_decision_threshold": float(decision_threshold),
                    "ece_weight": weights["ece_weight"],
                    "ks_weight": weights["ks_weight"],
                    "sharpness_weight": weights["sharpness_weight"],
                    "static_auc": static_m.get("roc_auc"),
                    "craf_auc": craf_m.get("roc_auc"),
                    "static_pr_auc": static_m.get("pr_auc"),
                    "craf_pr_auc": craf_m.get("pr_auc"),
                    "static_f1": static_m.get("f1"),
                    "craf_f1": craf_m.get("f1"),
                    "adaptation_rate": gate_stats["adaptation_rate"],
                    "mean_reliability": gate_stats["mean_reliability"],
                    "delta_auc": (craf_m.get("roc_auc", 0.0) or 0.0) - (static_m.get("roc_auc", 0.0) or 0.0),
                }
            )
    return rows


def _evaluate_calibration(
    estimator: ReliabilityEstimator,
    test_labels: np.ndarray,
    static_probs: np.ndarray,
    craf_probs: np.ndarray,
    seed: int,
) -> dict:
    return {
        "seed": int(seed),
        "static_ece": float(expected_calibration_error(test_labels, static_probs)),
        "craf_ece": float(expected_calibration_error(test_labels, craf_probs)),
        "static_brier": float(brier_score(test_labels, static_probs)),
        "craf_brier": float(brier_score(test_labels, craf_probs)),
        "static_bins": _calibration_bins(test_labels, static_probs),
        "craf_bins": _calibration_bins(test_labels, craf_probs),
        "domain_ece_at_fit": estimator.get_domain_ece(),
    }


def _is_finite_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _cda_spearman_status(value: float, domain_order: List[str]) -> str:
    if _is_finite_number(value):
        return "computed"
    if len(domain_order) < 3:
        return "undefined: fewer than three finite domains"
    return "undefined: insufficient finite CDA impacts or constant comparison values"


def _evaluate_cda(
    model: AttentionFusionModel,
    estimator: ReliabilityEstimator,
    test_feat: np.ndarray,
    test_mask: np.ndarray,
    sample_ids: list,
    test_idx: np.ndarray,
    domain_order: List[str],
    device: torch.device,
    n_cda: int,
    seed: int,
) -> dict:
    cda_idx = np.arange(min(n_cda, len(test_idx)))
    cda_feat = test_feat[cda_idx]
    cda_mask = test_mask[cda_idx]
    cda_ids = [sample_ids[test_idx[i]] for i in cda_idx] if sample_ids else list(range(len(cda_idx)))
    explainer = CounterfactualDomainExplainer(
        model=model,
        domain_order=domain_order,
        device=device,
        reliability_estimator=estimator,
        use_craf_weights=True,
    )
    cf_results = explainer.explain_batch(cda_feat, cda_mask, cda_ids, batch_size=32)
    mean_cf_impacts = {
        d: float(np.nanmean([abs(r.cf_impacts.get(d, float("nan"))) for r in cf_results]))
        for d in domain_order
    }
    domain_ece = estimator.get_domain_ece()
    spearman_vs_ece = explainer.correlation_with_shap(cf_results, {d: 1.0 - v for d, v in domain_ece.items()})
    return {
        "seed": int(seed),
        "n_samples": len(cf_results),
        "mean_cf_impacts_abs": mean_cf_impacts,
        "spearman_cda_vs_ece_reliability": float(spearman_vs_ece),
        "spearman_cda_vs_ece_reliability_status": _cda_spearman_status(spearman_vs_ece, domain_order),
        "sample_narratives": [r.narrative for r in cf_results[:5]],
    }


def _failure_case(
    case_type: str,
    idx: int,
    labels: np.ndarray,
    static_probs: np.ndarray,
    craf_probs: np.ndarray,
    sample_ids: list,
    test_idx: np.ndarray,
    features: np.ndarray,
    masks: np.ndarray,
    reliability_weights: np.ndarray,
    domain_order: List[str],
    score_index: int,
    static_decision_threshold: float = 0.5,
    craf_decision_threshold: float = 0.5,
) -> dict:
    domain_scores = {
        domain: None if bool(masks[idx, d]) else float(features[idx, d, score_index])
        for d, domain in enumerate(domain_order)
    }
    domain_reliability = {
        domain: float(reliability_weights[idx, d])
        for d, domain in enumerate(domain_order)
    }
    return {
        "case_type": case_type,
        "sample_id": sample_ids[test_idx[idx]] if sample_ids else int(idx),
        "label": int(labels[idx]),
        "static_prob": float(static_probs[idx]),
        "craf_prob": float(craf_probs[idx]),
        "static_pred": int(static_probs[idx] >= static_decision_threshold),
        "craf_pred": int(craf_probs[idx] >= craf_decision_threshold),
        "static_decision_threshold": float(static_decision_threshold),
        "craf_decision_threshold": float(craf_decision_threshold),
        "static_abs_error": float(abs(static_probs[idx] - labels[idx])),
        "craf_abs_error": float(abs(craf_probs[idx] - labels[idx])),
        "domain_scores": domain_scores,
        "domain_reliability": domain_reliability,
    }


def _extract_failure_cases(
    estimator: ReliabilityEstimator,
    test_feat: np.ndarray,
    test_mask: np.ndarray,
    test_labels: np.ndarray,
    static_probs: np.ndarray,
    craf_probs: np.ndarray,
    sample_ids: list,
    test_idx: np.ndarray,
    domain_order: List[str],
    score_index: int,
    static_decision_threshold: float = 0.5,
    craf_decision_threshold: float = 0.5,
) -> list[dict]:
    reliability_weights = estimator.compute_reliability_weights(test_feat, test_mask)
    static_pred = (static_probs >= static_decision_threshold).astype(int)
    craf_pred = (craf_probs >= craf_decision_threshold).astype(int)
    improvement = np.abs(static_probs - test_labels) - np.abs(craf_probs - test_labels)
    candidates: list[tuple[str, np.ndarray]] = [
        ("biggest_elara_win", np.argsort(-improvement)),
        ("biggest_elara_loss", np.argsort(improvement)),
        ("elara_correct_static_wrong", np.flatnonzero((craf_pred == test_labels) & (static_pred != test_labels))),
        ("static_correct_elara_wrong", np.flatnonzero((static_pred == test_labels) & (craf_pred != test_labels))),
        (
            "high_confidence_elara_failure",
            np.argsort(-np.where(craf_pred != test_labels, np.abs(craf_probs - craf_decision_threshold), -1)),
        ),
    ]
    cases = []
    used: set[int] = set()
    for case_type, idxs in candidates:
        for raw_idx in idxs:
            idx = int(raw_idx)
            if idx < 0 or idx >= len(test_labels) or idx in used:
                continue
            if case_type in {"elara_correct_static_wrong", "static_correct_elara_wrong"} and len(idxs) == 0:
                continue
            if case_type == "high_confidence_elara_failure" and craf_pred[idx] == test_labels[idx]:
                continue
            cases.append(
                _failure_case(
                    case_type,
                    idx,
                    test_labels,
                    static_probs,
                    craf_probs,
                    sample_ids,
                    test_idx,
                    test_feat,
                    test_mask,
                    reliability_weights,
                    domain_order,
                    score_index,
                    static_decision_threshold=static_decision_threshold,
                    craf_decision_threshold=craf_decision_threshold,
                )
            )
            used.add(idx)
            break
    return cases


def _aggregate_drift_curves(rows: list[dict], alpha: float) -> dict[str, list[dict]]:
    aggregated = aggregate_stress_rows(
        rows,
        group_keys=("domain", "noise_level"),
        metric_keys=("static_auc", "craf_auc", "static_pr_auc", "craf_pr_auc", "static_f1", "craf_f1"),
        alpha=alpha,
    )
    curves: dict[str, list[dict]] = {}
    for row in aggregated:
        curves.setdefault(row["domain"], []).append(row)
    for domain in curves:
        curves[domain] = sorted(curves[domain], key=lambda item: item["noise_level"])
    return curves


def _aggregate_degradation(rows: list[dict], alpha: float) -> dict[str, dict]:
    aggregated = aggregate_stress_rows(rows, group_keys=("domain",), metric_keys=("static", "craf"), alpha=alpha)
    output = {}
    for row in aggregated:
        static = row.get("static")
        craf = row.get("craf")
        output[row["domain"]] = {
            "static": static,
            "static_std": row.get("static_std"),
            "static_ci_low": row.get("static_ci_low"),
            "static_ci_high": row.get("static_ci_high"),
            "craf": craf,
            "craf_std": row.get("craf_std"),
            "craf_ci_low": row.get("craf_ci_low"),
            "craf_ci_high": row.get("craf_ci_high"),
            "delta": None if static is None or craf is None else float(craf - static),
            "craf_better": bool(craf is not None and static is not None and craf > static),
            "n_seeds": row.get("n_seeds", 0),
        }
    return output


def _aggregate_calibration(rows: list[dict]) -> dict:
    if not rows:
        return {}
    latest = rows[-1]
    out = {
        "static_bins": latest.get("static_bins", []),
        "craf_bins": latest.get("craf_bins", []),
        "domain_ece_at_fit": latest.get("domain_ece_at_fit", {}),
    }
    for metric in ("static_ece", "craf_ece", "static_brier", "craf_brier"):
        summary = summarize_values(row.get(metric) for row in rows)
        out[metric] = summary["mean"]
        out[f"{metric}_std"] = summary["std"]
        out[f"{metric}_ci_low"] = summary["ci_low"]
        out[f"{metric}_ci_high"] = summary["ci_high"]
    return out


def _evaluate_causal_attribution(
    model: AttentionFusionModel,
    estimator: ReliabilityEstimator,
    test_feat: np.ndarray,
    test_mask: np.ndarray,
    device: torch.device,
    domain_order: List[str],
    seed: int,
    *,
    n_bootstrap: int = 200,
    intervention: str = "population_mean",
    batch_size: int = 256,
) -> dict:
    """Interventional ATE of per-domain reliability on predictions.

    For each domain d we set every sample's domain-d reliability to its
    population mean (a do-calculus intervention on r_d), re-run the
    reliability-injected fusion forward pass, and compute the mean
    prediction shift versus the baseline reliability vector. A
    sample-level bootstrap gives the 95% CI.
    """
    reliability_weights = estimator.compute_reliability_weights(test_feat, test_mask)

    def predict_with_reliability(r_vec: np.ndarray) -> np.ndarray:
        model.eval()
        probs: list[np.ndarray] = []
        n = test_feat.shape[0]
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                feat_t = torch.tensor(test_feat[start:end], dtype=torch.float32, device=device)
                mask_t = torch.tensor(test_mask[start:end], dtype=torch.bool, device=device)
                r_t = torch.tensor(r_vec[start:end], dtype=torch.float32, device=device)
                r_t = r_t.masked_fill(mask_t, 0.0)
                embeds = [enc(feat_t[:, i, :]) for i, enc in enumerate(model.domain_encoders)]
                domain_embeds = torch.stack(embeds, dim=1)
                logits, _ = model.fusion(
                    domain_embeds, key_padding_mask=mask_t, confidence_weights=r_t
                )
                probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
        return np.concatenate(probs)

    effects = estimate_all_interventional_ates(
        predict_with_reliability,
        reliability_weights,
        domain_order=domain_order,
        n_bootstrap=n_bootstrap,
        random_state=seed,
        intervention=intervention,
    )
    return {
        "seed": int(seed),
        "n_test_samples": int(test_feat.shape[0]),
        "intervention": str(intervention),
        "per_domain": [
            {
                "domain": e.domain,
                "ate": float(e.ate),
                "ate_std_error": float(e.ate_std_error),
                "ate_ci_low": float(e.ate_ci_low),
                "ate_ci_high": float(e.ate_ci_high),
                "n_samples": int(e.n_samples),
            }
            for e in effects
        ],
    }


def _aggregate_causal_attribution(rows: list[dict], domain_order: List[str]) -> dict:
    if not rows:
        return {}
    out: dict = {"n_seeds": len(rows), "per_domain": []}
    for domain in domain_order:
        ates = [
            entry["ate"]
            for row in rows
            for entry in row.get("per_domain", [])
            if entry.get("domain") == domain and np.isfinite(entry.get("ate", float("nan")))
        ]
        if not ates:
            continue
        ates_arr = np.asarray(ates, dtype=np.float64)
        ate_mean = float(ates_arr.mean())
        ate_std = float(ates_arr.std(ddof=0))
        ate_se = float(ates_arr.std(ddof=0) / max(np.sqrt(len(ates)), 1.0))
        z = 1.959963984540054
        out["per_domain"].append(
            {
                "domain": domain,
                "ate_mean": ate_mean,
                "ate_std_across_seeds": ate_std,
                "ate_ci_low": ate_mean - z * ate_se,
                "ate_ci_high": ate_mean + z * ate_se,
                "n_seeds_with_finite_ate": int(len(ates)),
            }
        )
    return out


def _aggregate_category_aware(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out: dict = {"n_seeds": len(rows)}
    for metric in (
        "global_adapt_rate",
        "category_aware_adapt_rate",
        "global_mean_reliability",
        "category_aware_mean_reliability",
    ):
        summary = summarize_values(row.get(metric) for row in rows)
        out[metric] = summary["mean"]
        out[f"{metric}_std"] = summary["std"]
        out[f"{metric}_ci_low"] = summary["ci_low"]
        out[f"{metric}_ci_high"] = summary["ci_high"]
    out["misfire_reduction_absolute"] = (
        out["global_adapt_rate"] - out["category_aware_adapt_rate"]
    )
    return out


def _aggregate_cda(rows: list[dict], domain_order: List[str]) -> dict:
    if not rows:
        return {}
    latest = rows[-1]
    impacts = {
        domain: summarize_values(row.get("mean_cf_impacts_abs", {}).get(domain) for row in rows)["mean"]
        for domain in domain_order
    }
    spearman = summarize_values(row.get("spearman_cda_vs_ece_reliability") for row in rows)
    statuses = [row.get("spearman_cda_vs_ece_reliability_status") for row in rows if row.get("spearman_cda_vs_ece_reliability_status")]
    status = "computed" if _is_finite_number(spearman["mean"]) else (statuses[-1] if statuses else "undefined")
    return {
        "n_samples": int(sum(row.get("n_samples", 0) for row in rows)),
        "mean_cf_impacts_abs": impacts,
        "spearman_cda_vs_ece_reliability": spearman["mean"],
        "spearman_cda_vs_ece_reliability_std": spearman["std"],
        "spearman_cda_vs_ece_reliability_status": status,
        "sample_narratives": latest.get("sample_narratives", []),
        "per_seed": rows,
    }


def _run_experiment_arrays(
    cfg: Dict,
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    sample_ids: list,
    domain_order: List[str],
    confidence_index: int | None,
    score_index: int,
    sample_splits: np.ndarray | None = None,
    sample_categories: np.ndarray | None = None,
    seed_override: Optional[int] = None,
    device: torch.device | None = None,
) -> Dict:
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    rel_cfg = cfg.get("reliability", {})
    craf_cfg = cfg.get("craf", {})
    rga_plus_cfg = cfg.get("rga_plus", {})
    mechanism_cfg = craf_cfg.get("mechanism_isolation", {})

    seeds = [seed_override] if seed_override is not None else eval_cfg.get("seeds", [42])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_gate_threshold = float(rel_cfg.get("clean_gate_threshold", 0.66))
    # Per-sample RGA gating (paper formalism r_{i,d}). Default False preserves
    # batch-level gating used in current paper results.
    per_sample_gating = bool(rel_cfg.get("per_sample_gating", False))
    n_bootstrap = int(eval_cfg.get("n_bootstrap", 200))
    bootstrap_alpha = float(eval_cfg.get("bootstrap_alpha", 0.05))
    threshold_strategy = eval_cfg.get("decision_threshold_strategy", eval_cfg.get("decision_threshold", "fixed_0p5"))
    rga_plus_selection_metric = str(rga_plus_cfg.get("selection_metric", "roc_auc"))
    attack_names = craf_cfg.get("adversarial_attacks", ["zero_attack", "max_attack", "gaussian_noise"])
    adversarial_sigma = float(craf_cfg.get("adversarial_sigma", 0.1))
    tau_sweep_thresholds = [float(v) for v in mechanism_cfg.get("tau_sweep_thresholds", [])]
    component_ablation_variants = list(mechanism_cfg.get("component_ablation_variants", []))
    enable_learned_gate = bool(mechanism_cfg.get("learned_gate", False))

    per_seed_table1 = []
    per_seed_drift_rows: list[dict] = []
    per_seed_degradation_rows: list[dict] = []
    per_seed_adversarial_rows: list[dict] = []
    per_seed_missing_rows: list[dict] = []
    per_seed_calibration_rows: list[dict] = []
    per_seed_cda_rows: list[dict] = []
    per_seed_failure_cases: list[dict] = []
    per_seed_tau_sweep_rows: list[dict] = []
    per_seed_component_ablation_rows: list[dict] = []
    per_seed_category_aware_rows: list[dict] = []
    per_seed_causal_attribution_rows: list[dict] = []
    category_aware_enabled = bool(rel_cfg.get("category_aware", False)) and sample_categories is not None

    for seed in seeds:
        actual_seed = int(seed)
        set_seed(actual_seed)
        logger.info("Training and evaluating reliability-gated fusion (seed=%s)", actual_seed)

        train_idx, val_idx, test_idx = _split(labels, {**train_cfg, "seed": actual_seed}, split_values=sample_splits)
        train_loader, val_loader, _ = _make_loaders(
            features,
            masks,
            labels,
            train_idx,
            val_idx,
            test_idx,
            batch_size=int(train_cfg.get("batch_size", 128)),
        )

        model = _build_model(cfg, features.shape[1], features.shape[2], confidence_index, device)
        _train_model(model, train_loader, val_loader, cfg, device)
        model.eval()

        estimator = _make_reliability_estimator(rel_cfg, domain_order, score_index)
        estimator.fit(features[val_idx], masks[val_idx], labels[val_idx])

        val_feat = features[val_idx]
        val_mask = masks[val_idx]
        val_labels = labels[val_idx]
        test_feat = features[test_idx]
        test_mask = masks[test_idx]
        test_labels = labels[test_idx]

        if category_aware_enabled:
            cat_weights = _component_weights(rel_cfg, disabled=())
            category_estimator = CategoryAwareReliabilityEstimator(
                domain_order=domain_order,
                score_index=score_index,
                ece_weight=cat_weights["ece_weight"],
                ks_weight=cat_weights["ks_weight"],
                sharpness_weight=cat_weights["sharpness_weight"],
                n_calibration_bins=int(rel_cfg.get("n_calibration_bins", 10)),
                min_samples_for_ks=int(rel_cfg.get("min_samples_for_ks", 30)),
                gate_threshold=clean_gate_threshold,
                unknown_category_policy=str(rel_cfg.get("unknown_category_policy", "global")),
            )
            category_estimator.fit(
                val_feat,
                val_mask,
                val_labels,
                categories=sample_categories[val_idx],
            )
            global_test_weights = estimator.compute_reliability_weights(test_feat, test_mask)
            cat_test_weights = category_estimator.compute_reliability_weights(
                test_feat,
                test_mask,
                categories=sample_categories[test_idx],
            )
            n_present_per_sample = (~test_mask).sum(axis=1).astype(np.float32)
            global_mean_r = np.where(
                n_present_per_sample > 0,
                global_test_weights.sum(axis=1) / np.maximum(n_present_per_sample, 1.0),
                0.0,
            )
            cat_mean_r = np.where(
                n_present_per_sample > 0,
                cat_test_weights.sum(axis=1) / np.maximum(n_present_per_sample, 1.0),
                0.0,
            )
            per_seed_category_aware_rows.append(
                {
                    "seed": actual_seed,
                    "n_test_samples": int(test_feat.shape[0]),
                    "global_adapt_rate": float(np.mean(global_mean_r < clean_gate_threshold)),
                    "category_aware_adapt_rate": float(np.mean(cat_mean_r < clean_gate_threshold)),
                    "global_mean_reliability": float(np.mean(global_mean_r)),
                    "category_aware_mean_reliability": float(np.mean(cat_mean_r)),
                    "gate_threshold": float(clean_gate_threshold),
                    "n_categories_in_test": int(len(set(sample_categories[test_idx].tolist()))),
                }
            )
        static_val_probs = _predict_static(model, val_feat, val_mask, device)
        static_probs = _predict_static(model, test_feat, test_mask, device)
        craf_val_probs = _predict_craf(
            model,
            estimator,
            val_feat,
            val_mask,
            device,
            clean_gate_threshold=clean_gate_threshold,
            per_sample_gating=per_sample_gating,
        )
        craf_probs = _predict_craf(
            model,
            estimator,
            test_feat,
            test_mask,
            device,
            clean_gate_threshold=clean_gate_threshold,
            per_sample_gating=per_sample_gating,
        )

        # Post-hoc polarity calibration. When the supervised fusion head is
        # trained under a canonical one-class regime (or any regime that
        # leaves no anomaly gradient on val), the model can settle into an
        # inverse-polarity solution. We detect this by computing AUROC on a
        # synthetic-anomaly-augmented val set; if AUROC<0.5 we flip the
        # subsequent static/RGA predictions globally for this seed. The
        # trained weights are never modified.
        polarity_info = _calibrate_polarity(
            model,
            val_feat,
            val_mask,
            val_labels,
            score_index=score_index,
            device=device,
            random_seed=actual_seed,
        )
        if polarity_info["flip_required"]:
            static_val_probs = 1.0 - static_val_probs
            static_probs = 1.0 - static_probs
            craf_val_probs = 1.0 - craf_val_probs
            craf_probs = 1.0 - craf_probs

        static_metrics = _metrics_from_validation_threshold(
            test_labels,
            static_probs,
            val_labels=val_labels,
            val_probs=static_val_probs,
            strategy=threshold_strategy,
        )
        craf_metrics = _metrics_from_validation_threshold(
            test_labels,
            craf_probs,
            val_labels=val_labels,
            val_probs=craf_val_probs,
            strategy=threshold_strategy,
        )
        static_decision_threshold = float(static_metrics["decision_threshold"])
        craf_decision_threshold = float(craf_metrics["decision_threshold"])
        rga_boosted = ReliabilityBoostedFusion(
            score_index=score_index,
            confidence_index=confidence_index,
            random_seed=actual_seed,
            selection_metric=rga_plus_selection_metric,
        ).fit(
            features[train_idx],
            masks[train_idx],
            labels[train_idx],
            val_feat,
            val_mask,
            val_labels,
            reliability_estimator=estimator,
        )
        rga_boosted_val_probs = rga_boosted.predict_proba(val_feat, val_mask)
        rga_boosted_probs = rga_boosted.predict_proba(test_feat, test_mask)
        rga_boosted_metrics = _metrics_from_validation_threshold(
            test_labels,
            rga_boosted_probs,
            val_labels=val_labels,
            val_probs=rga_boosted_val_probs,
            strategy=threshold_strategy,
        )
        rga_boosted_metrics["selected_candidate"] = rga_boosted.selected_candidate
        rga_boosted_metrics["selection_metric"] = rga_plus_selection_metric
        rga_boosted_metrics["candidate_validation_roc_auc"] = {
            name: _json_float(score)
            for name, score in sorted(rga_boosted.candidate_validation_auc.items())
        }
        rga_boosted_metrics["candidate_validation_scores"] = {
            name: {metric: _json_float(score) for metric, score in sorted(values.items())}
            for name, values in sorted(rga_boosted.candidate_validation_metrics.items())
        }
        rga_boosted_decision_threshold = float(rga_boosted_metrics["decision_threshold"])
        baseline_metrics, baseline_predictions = run_baseline_suite(
            features,
            masks,
            labels,
            train_idx,
            val_idx,
            test_idx,
            score_index=score_index,
            device=device,
            random_seed=actual_seed,
            decision_threshold_strategy=threshold_strategy,
            return_predictions=True,
        )
        router_val_predictions = {
            "static_attention": static_val_probs,
            "craf_attention": craf_val_probs,
            "rga_boosted_fusion": rga_boosted_val_probs,
            **{
                name: prediction_payload["val_probs"]
                for name, prediction_payload in baseline_predictions.items()
            },
        }
        router_test_predictions = {
            "static_attention": static_probs,
            "craf_attention": craf_probs,
            "rga_boosted_fusion": rga_boosted_probs,
            **{
                name: prediction_payload["test_probs"]
                for name, prediction_payload in baseline_predictions.items()
            },
        }
        rga_router_metrics, _rga_router_val_probs, rga_router_probs = _fit_rga_meta_router_metrics(
            val_predictions=router_val_predictions,
            test_predictions=router_test_predictions,
            val_labels=val_labels,
            test_labels=test_labels,
            random_seed=actual_seed,
            threshold_strategy=threshold_strategy,
            selection_metric=rga_plus_selection_metric,
        )
        rga_router_decision_threshold = float(rga_router_metrics["decision_threshold"])
        per_seed_table1.append(
            {
                "seed": actual_seed,
                "static_attention": static_metrics,
                "craf_attention": craf_metrics,
                "rga_boosted_fusion": rga_boosted_metrics,
                "rga_meta_router": rga_router_metrics,
                "bootstrap_ci": {
                    "static_attention": _metric_bootstrap_intervals(
                        test_labels,
                        static_probs,
                        n_bootstrap,
                        bootstrap_alpha,
                        actual_seed,
                        threshold=static_decision_threshold,
                    ),
                    "craf_attention": _metric_bootstrap_intervals(
                        test_labels,
                        craf_probs,
                        n_bootstrap,
                        bootstrap_alpha,
                        actual_seed + 1000,
                        threshold=craf_decision_threshold,
                    ),
                    "rga_meta_router": _metric_bootstrap_intervals(
                        test_labels,
                        rga_router_probs,
                        n_bootstrap,
                        bootstrap_alpha,
                        actual_seed + 2000,
                        threshold=rga_router_decision_threshold,
                    ),
                    "rga_boosted_fusion": _metric_bootstrap_intervals(
                        test_labels,
                        rga_boosted_probs,
                        n_bootstrap,
                        bootstrap_alpha,
                        actual_seed + 3000,
                        threshold=rga_boosted_decision_threshold,
                    ),
                },
                "delong_p_craf_vs_static": float(delong_roc_test(test_labels, craf_probs, static_probs)),
                "polarity_calibration": polarity_info,
                "delong_p_rga_plus_vs_baseline": _delong_pairs_against_baselines(
                    test_labels,
                    rga_router_probs,
                    rga_boosted_probs,
                    baseline_predictions,
                    static_probs=static_probs,
                ),
                **baseline_metrics,
            }
        )

        drift_rows, degradation_rows = _evaluate_drift(
            model,
            estimator,
            test_feat,
            test_mask,
            test_labels,
            domain_order,
            score_index,
            device,
            craf_cfg.get("drift_noise_levels", [0.0, 0.05, 0.1, 0.2, 0.3]),
            clean_gate_threshold,
            actual_seed,
            per_sample_gating=per_sample_gating,
            static_decision_threshold=static_decision_threshold,
            craf_decision_threshold=craf_decision_threshold,
        )
        per_seed_drift_rows.extend(drift_rows)
        per_seed_degradation_rows.extend(degradation_rows)
        per_seed_adversarial_rows.extend(
            _evaluate_adversarial(
                model,
                estimator,
                test_feat,
                test_mask,
                test_labels,
                domain_order,
                score_index,
                device,
                attack_names,
                adversarial_sigma,
                clean_gate_threshold,
                actual_seed,
                per_sample_gating=per_sample_gating,
                static_decision_threshold=static_decision_threshold,
                craf_decision_threshold=craf_decision_threshold,
            )
        )
        learned_gate_for_seed: Optional[LearnedReliabilityGate] = None
        if enable_learned_gate:
            try:
                gate_engine = AdversarialPerturbationEngine(domain_order, score_index, random_seed=actual_seed + 33_000)
                perturbation_fns: list = [lambda f, m: (f.copy(), m.copy())]  # clean
                for attack_name in attack_names:
                    try:
                        attack_type = AdversarialAttackType(attack_name)
                    except ValueError:
                        continue
                    def _make_fn(at=attack_type, eng=gate_engine, sg=adversarial_sigma):
                        return lambda f, m: eng.apply_attack(f, m, at, target_domain=None, sigma=sg)
                    perturbation_fns.append(_make_fn())

                def _gate_static(f, m):
                    return _predict_static(model, f, m, device)

                def _gate_weights(f, m):
                    return estimator.compute_reliability_weights(f, m)

                def _gate_reliability(f, m, w):
                    # Force the reliability path on for every sample so we can
                    # measure where it would have helped vs hurt. clean_gate_threshold
                    # > 1.0 ensures the mean-reliability comparison is always
                    # satisfied, equivalent to "always fire".
                    probs, _ = _predict_craf_with_stats(
                        model, estimator, f, m, device,
                        clean_gate_threshold=2.0,
                        per_sample_gating=False,
                    )
                    return probs

                learned_gate_for_seed = LearnedReliabilityGate(LearnedGateConfig(random_state=actual_seed))
                learned_gate_for_seed.fit(
                    val_feat,
                    val_mask,
                    val_labels,
                    compute_reliability_weights=_gate_weights,
                    predict_static=_gate_static,
                    predict_reliability=_gate_reliability,
                    perturbation_fns=perturbation_fns,
                )
            except Exception as exc:
                logger.warning("Learned-gate fit failed for seed %d: %s", actual_seed, exc)
                learned_gate_for_seed = None

        per_seed_tau_sweep_rows.extend(
            _evaluate_tau_sweep(
                model,
                estimator,
                test_feat,
                test_mask,
                test_labels,
                domain_order,
                score_index,
                device,
                tau_sweep_thresholds,
                attack_names,
                adversarial_sigma,
                actual_seed,
                per_sample_gating=per_sample_gating,
                static_decision_threshold=static_decision_threshold,
                craf_decision_threshold=craf_decision_threshold,
                learned_gate=learned_gate_for_seed,
            )
        )
        per_seed_component_ablation_rows.extend(
            _evaluate_component_ablation(
                model,
                rel_cfg,
                features[val_idx],
                masks[val_idx],
                labels[val_idx],
                test_feat,
                test_mask,
                test_labels,
                domain_order,
                score_index,
                device,
                component_ablation_variants,
                attack_names,
                adversarial_sigma,
                clean_gate_threshold,
                actual_seed,
                per_sample_gating=per_sample_gating,
                threshold_strategy=threshold_strategy,
                static_decision_threshold=static_decision_threshold,
            )
        )
        per_seed_missing_rows.extend(
            _evaluate_missing(
                model,
                estimator,
                test_feat,
                test_mask,
                test_labels,
                device,
                eval_cfg.get("domain_dropout_probs_extended", [0.0, 0.1, 0.2, 0.3, 0.5]),
                clean_gate_threshold,
                actual_seed,
                per_sample_gating=per_sample_gating,
                static_decision_threshold=static_decision_threshold,
                craf_decision_threshold=craf_decision_threshold,
            )
        )
        per_seed_calibration_rows.append(
            _evaluate_calibration(estimator, test_labels, static_probs, craf_probs, actual_seed)
        )
        per_seed_cda_rows.append(
            _evaluate_cda(
                model,
                estimator,
                test_feat,
                test_mask,
                sample_ids,
                test_idx,
                domain_order,
                device,
                int(eval_cfg.get("cda_samples", 100)),
                actual_seed,
            )
        )
        per_seed_causal_attribution_rows.append(
            _evaluate_causal_attribution(
                model,
                estimator,
                test_feat,
                test_mask,
                device,
                domain_order,
                actual_seed,
            )
        )
        per_seed_failure_cases.append(
            {
                "seed": actual_seed,
                "cases": _extract_failure_cases(
                    estimator,
                    test_feat,
                    test_mask,
                    test_labels,
                    static_probs,
                    craf_probs,
                    sample_ids,
                    test_idx,
                    domain_order,
                    score_index,
                    static_decision_threshold=static_decision_threshold,
                    craf_decision_threshold=craf_decision_threshold,
                ),
            }
        )

    clean_methods = [
        "random_forest",
        "confidence_weighted_mean",
        "tent_score_adapter",
        "ttt_pseudo_label_adapter",
        "early_fusion_mlp",
        "late_fusion_ensemble",
        "static_attention",
        "craf_attention",
        "rga_boosted_fusion",
        "rga_meta_router",
    ]
    clean_summary = summarize_seed_metric_rows(per_seed_table1, methods=clean_methods)
    degradation_aucs = _aggregate_degradation(per_seed_degradation_rows, alpha=bootstrap_alpha)
    cda = _aggregate_cda(per_seed_cda_rows, domain_order)
    adversarial_summary = aggregate_stress_rows(
        per_seed_adversarial_rows,
        group_keys=("attack", "target_domain"),
        metric_keys=("static_auc", "craf_auc", "static_pr_auc", "craf_pr_auc", "static_f1", "craf_f1"),
        alpha=bootstrap_alpha,
    )
    missing_summary = aggregate_stress_rows(
        per_seed_missing_rows,
        group_keys=("dropout_prob",),
        metric_keys=("static_auc", "craf_auc", "static_pr_auc", "craf_pr_auc", "static_f1", "craf_f1"),
        alpha=bootstrap_alpha,
    )
    tau_sweep_summary = (
        aggregate_stress_rows(
            per_seed_tau_sweep_rows,
            group_keys=("condition", "tau"),
            metric_keys=(
                "static_auc",
                "craf_auc",
                "static_pr_auc",
                "craf_pr_auc",
                "static_f1",
                "craf_f1",
                "adaptation_rate",
                "mean_reliability",
            ),
            alpha=bootstrap_alpha,
        )
        if per_seed_tau_sweep_rows
        else []
    )
    component_ablation_summary = (
        aggregate_stress_rows(
            per_seed_component_ablation_rows,
            group_keys=("variant", "attack", "target_domain"),
            metric_keys=(
                "static_auc",
                "craf_auc",
                "static_pr_auc",
                "craf_pr_auc",
                "static_f1",
                "craf_f1",
                "adaptation_rate",
                "mean_reliability",
            ),
            alpha=bootstrap_alpha,
        )
        if per_seed_component_ablation_rows
        else []
    )

    static_aucs = [row["static_attention"].get("roc_auc", float("nan")) for row in per_seed_table1]
    craf_aucs = [row["craf_attention"].get("roc_auc", float("nan")) for row in per_seed_table1]
    paired_p = paired_ttest(np.array(craf_aucs), np.array(static_aucs)) if len(per_seed_table1) > 1 else float("nan")
    latest_cal = per_seed_calibration_rows[-1] if per_seed_calibration_rows else {}
    spearman_value = cda.get("spearman_cda_vs_ece_reliability")
    spearman_finite = spearman_value is not None and np.isfinite(float(spearman_value))
    claim_checks = {
        "delong_p_lt_0p05_on_last_seed": bool(per_seed_table1[-1]["delong_p_craf_vs_static"] < 0.05),
        "craf_better_drift_auc_n_domains": sum(v["craf_better"] for v in degradation_aucs.values()),
        "craf_better_drift_auc_all_domains": bool(
            sum(v["craf_better"] for v in degradation_aucs.values()) == len(domain_order)
        ),
        "spearman_cda_vs_ece_gt_0p6": bool(spearman_finite and float(spearman_value) > 0.6),
        "craf_ece_lt_static_ece": bool(latest_cal.get("craf_ece", float("inf")) < latest_cal.get("static_ece", 0.0)),
        "paired_ttest_p": float(paired_p) if np.isfinite(paired_p) else None,
    }

    return {
        "decision_thresholding": {
            "strategy": (threshold_strategy or "fixed_0p5").strip().lower(),
            "selection_split": "validation",
        },
        "table_1_clean_performance": per_seed_table1,
        "clean_metric_summary": clean_summary,
        "table_2_drift_robustness_per_seed": per_seed_drift_rows,
        "table_2_drift_robustness": _aggregate_drift_curves(per_seed_drift_rows, alpha=bootstrap_alpha),
        "figure_1_drift_curves": {
            "noise_levels": craf_cfg.get("drift_noise_levels", [0.0, 0.05, 0.1, 0.2, 0.3]),
            "degradation_aucs": degradation_aucs,
            "per_seed_degradation_aucs": per_seed_degradation_rows,
        },
        "table_3_adversarial_per_seed": per_seed_adversarial_rows,
        "table_3_adversarial": adversarial_summary,
        "table_4_missing_modality_per_seed": per_seed_missing_rows,
        "table_4_missing_modality": missing_summary,
        "table_5_calibration_per_seed": per_seed_calibration_rows,
        "table_5_calibration": _aggregate_calibration(per_seed_calibration_rows),
        "table_6_tau_sweep_per_seed": per_seed_tau_sweep_rows,
        "table_6_tau_sweep": tau_sweep_summary,
        "table_7_component_ablation_per_seed": per_seed_component_ablation_rows,
        "table_7_component_ablation": component_ablation_summary,
        "table_8_category_aware_per_seed": per_seed_category_aware_rows,
        "table_8_category_aware": _aggregate_category_aware(per_seed_category_aware_rows),
        "table_9_causal_attribution_per_seed": per_seed_causal_attribution_rows,
        "table_9_causal_attribution": _aggregate_causal_attribution(per_seed_causal_attribution_rows, domain_order),
        "cda_validation": cda,
        "failure_case_analysis": {
            "per_seed": per_seed_failure_cases,
            "representative_cases": per_seed_failure_cases[-1]["cases"] if per_seed_failure_cases else [],
        },
        "statistical_summary": {
            "per_seed_static_auc": static_aucs,
            "per_seed_craf_auc": craf_aucs,
            "paired_ttest_p_craf_vs_static": float(paired_p) if np.isfinite(paired_p) else None,
            "clean_metric_summary": clean_summary,
            "stress_metric_summary": {
                "drift_degradation": degradation_aucs,
                "adversarial": adversarial_summary,
                "missing_modality": missing_summary,
                "tau_sweep": tau_sweep_summary,
                "component_ablation": component_ablation_summary,
            },
            "claim_checks": claim_checks,
        },
    }


def run_experiment(cfg: Dict, seed_override: Optional[int] = None) -> Dict:
    logger.info("Phase 0: Loading data")
    (
        features,
        masks,
        labels,
        sample_ids,
        domain_order,
        _,
        confidence_index,
        score_index,
        sample_splits,
        sample_categories,
    ) = _load_data(cfg)
    return _run_experiment_arrays(
        cfg,
        features,
        masks,
        labels,
        sample_ids,
        domain_order,
        confidence_index,
        score_index or 0,
        sample_splits=sample_splits,
        sample_categories=sample_categories,
        seed_override=seed_override,
    )


def _run_synthetic_experiment(cfg, features, masks, labels, sample_ids, domain_order, seed_override=None) -> Dict:
    """Run experiment using pre-loaded synthetic arrays."""
    return _run_experiment_arrays(
        cfg,
        features,
        masks,
        labels,
        sample_ids,
        domain_order,
        confidence_index=None,
        score_index=0,
        seed_override=seed_override,
        device=torch.device("cpu"),
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Reliability-gated multimodal fusion experiment")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to attention_config.yaml")
    parser.add_argument("--output", default="experiments/fusion/attention_fusion/reliability_gated_results.json")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for an explicit smoke test")
    parser.add_argument("--seed", type=int, default=None, help="Override single seed")
    args = parser.parse_args()

    if args.synthetic:
        logger.warning("Running explicit synthetic smoke test; do not use this output as paper evidence.")
        features, masks, labels, sample_ids, domain_order = _make_synthetic()
        cfg = {
            "data": {"path": "/dev/null", "score_column": "score", "confidence_column": "confidence",
                     "embedding_prefix": "embedding_", "id_column": "sample_id",
                     "domain_column": "domain", "label_column": "label"},
            "model": {"num_domains": 3, "embed_dim": 32, "num_heads": 4, "num_layers": 1,
                      "dropout": 0.1, "use_confidence": False, "use_input_confidence": False,
                      "use_attention": True, "use_domain_embeddings": True,
                      "use_positional_embeddings": True, "use_missing_embedding": True},
            "training": {"seed": 42, "batch_size": 64, "epochs": 5, "lr": 1e-3,
                         "weight_decay": 0.01, "domain_dropout": 0.1,
                         "test_size": 0.2, "val_size": 0.1, "early_stopping": 3, "lambda_reg": 0.01},
            "evaluation": {"seeds": [42], "cda_samples": 20, "n_bootstrap": 50,
                           "domain_dropout_probs_extended": [0.0, 0.1, 0.3]},
            "reliability": {"ece_weight": 0.4, "ks_weight": 0.4, "sharpness_weight": 0.2,
                            "n_calibration_bins": 5, "min_samples_for_ks": 10},
            "craf": {"drift_noise_levels": [0.0, 0.1, 0.3],
                     "adversarial_attacks": ["zero_attack", "gaussian_noise"],
                     "adversarial_sigma": 0.1},
        }
        results = _run_synthetic_experiment(cfg, features, masks, labels, sample_ids, domain_order, seed_override=args.seed)
    else:
        cfg = load_yaml(str(_resolve(args.config)))
        results = run_experiment(cfg, seed_override=args.seed)

    out_path = _resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            value = float(obj)
            return None if value != value else value
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_to_serializable(results), f, indent=2)

    logger.info("Results saved to %s", out_path)

if __name__ == "__main__":
    main()
