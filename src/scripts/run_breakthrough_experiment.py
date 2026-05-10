"""End-to-end CRAF breakthrough experiment.

Produces publication-grade evidence across 8 phases:

  Phase 0 — Setup: load config, split data, set seeds
  Phase 1 — Train + fit ReliabilityEstimator on val split
  Phase 2 — Table 1: clean-data CRAF vs static attention + baselines; DeLong p-values
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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
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
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator
from uais.fusion.attention.train_attention_fusion import attention_fusion_loss, set_seed
from uais.utils.config_loader import load_yaml
from uais.utils.metrics import (
    brier_score,
    classification_metrics,
    expected_calibration_error,
    reliability_degradation_auc,
)
from uais.utils.paths import PROJECT_ROOT
from uais.utils.stats import bootstrap_ci, delong_roc_test, paired_ttest

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("src/uais/fusion/attention/attention_config.yaml")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_data(cfg: Dict):
    data_cfg = cfg.get("data", {})
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
    return features, masks, labels, sample_ids, domain_order, feature_columns, confidence_index, score_index


def _split(labels: np.ndarray, train_cfg: Dict):
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
                msks = apply_domain_dropout(msks, p=domain_dropout_p)
            optimizer.zero_grad()
            logits, _, confidence_w = model(feats, key_padding_mask=msks)
            loss = attention_fusion_loss(logits.squeeze(-1), lbls, confidence_w, lambda_reg=float(t_cfg.get("lambda_reg", 0.01)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                feats, msks, lbls = [x.to(device) for x in batch]
                logits, _, confidence_w = model(feats, key_padding_mask=msks)
                loss = attention_fusion_loss(logits.squeeze(-1), lbls, confidence_w, lambda_reg=0.0)
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


@torch.no_grad()
@torch.no_grad()
def _predict_craf(
    model: AttentionFusionModel,
    estimator: ReliabilityEstimator,
    features: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict using RGA: static path when reliable, reliability-injected path when degraded.

    Per the RGA paper, for each sample the mean reliability over present domains
    is compared against estimator.gate_threshold (default 0.66). Samples above
    the threshold use the static attention path; only samples below it receive
    reliability-weight injection. This keeps the method conservative: clean
    predictions are undisturbed, and adaptation only activates under degradation.
    """
    model.eval()
    probs = []
    n = features.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feat_np = features[start:end]
        mask_np = masks[start:end]

        craf_w = estimator.compute_reliability_weights(feat_np, mask_np)
        gate = estimator.gate_decisions(craf_w, mask_np)  # [B] bool

        feat_t = torch.tensor(feat_np, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device)

        # Static path — always run as fallback
        logits_static, _, _ = model(feat_t, key_padding_mask=mask_t)
        probs_static = torch.sigmoid(logits_static.squeeze(-1))

        if gate.any():
            # Reliability path — only for samples whose mean domain reliability < threshold
            craf_t = torch.tensor(craf_w, dtype=torch.float32, device=device)
            craf_t = craf_t.masked_fill(mask_t, 0.0)
            embeds = [enc(feat_t[:, i, :]) for i, enc in enumerate(model.domain_encoders)]
            domain_embeds = torch.stack(embeds, dim=1)
            logits_craf, _ = model.fusion(domain_embeds, key_padding_mask=mask_t, confidence_weights=craf_t)
            probs_craf = torch.sigmoid(logits_craf.squeeze(-1))
            gate_t = torch.tensor(gate, dtype=torch.bool, device=device)
            batch_probs = torch.where(gate_t, probs_craf, probs_static)
        else:
            batch_probs = probs_static

        probs.append(batch_probs.cpu().numpy())
    return np.concatenate(probs)


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


# ---------------------------------------------------------------------------
# Main experiment phases
# ---------------------------------------------------------------------------

def run_experiment(cfg: Dict, seed_override: Optional[int] = None) -> Dict:
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    rel_cfg = cfg.get("reliability", {})
    craf_cfg = cfg.get("craf", {})

    seeds = eval_cfg.get("seeds", [42])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: Dict = {}

    # ------------------------------------------------------------------
    # Phase 0 — Load data
    # ------------------------------------------------------------------
    logger.info("Phase 0: Loading data")
    features, masks, labels, sample_ids, domain_order, feature_columns, confidence_index, score_index = _load_data(cfg)
    if score_index is None:
        score_index = 0

    # ------------------------------------------------------------------
    # Phase 1 — Train model + fit ReliabilityEstimator on first seed
    # ------------------------------------------------------------------
    per_seed_table1 = []
    per_seed_static_probs = []
    per_seed_craf_probs = []

    for seed in seeds:
        actual_seed = seed_override if seed_override is not None else seed
        set_seed(actual_seed)
        logger.info(f"Phase 1: Training (seed={actual_seed})")

        train_idx, val_idx, test_idx = _split(labels, {**train_cfg, "seed": actual_seed})
        train_loader, val_loader, test_loader = _make_loaders(
            features, masks, labels, train_idx, val_idx, test_idx,
            batch_size=int(train_cfg.get("batch_size", 128)),
        )
        input_dim = features.shape[2]
        num_domains = features.shape[1]

        model = _build_model(cfg, num_domains, input_dim, confidence_index, device)
        _train_model(model, train_loader, val_loader, cfg, device)
        model.eval()

        # Fit ReliabilityEstimator on validation split
        estimator = ReliabilityEstimator(
            domain_order=domain_order,
            score_index=score_index,
            ece_weight=float(rel_cfg.get("ece_weight", 0.45)),
            ks_weight=float(rel_cfg.get("ks_weight", 0.35)),
            sharpness_weight=float(rel_cfg.get("sharpness_weight", 0.20)),
            n_calibration_bins=int(rel_cfg.get("n_calibration_bins", 10)),
            min_samples_for_ks=int(rel_cfg.get("min_samples_for_ks", 30)),
            gate_threshold=float(rel_cfg.get("gate_threshold", 0.66)),
        )
        estimator.fit(features[val_idx], masks[val_idx], labels[val_idx])

        test_feat = features[test_idx]
        test_mask = masks[test_idx]
        test_labels = labels[test_idx]

        # ------------------------------------------------------------------
        # Phase 2 — Table 1: static vs CRAF vs baselines
        # ------------------------------------------------------------------
        logger.info("Phase 2: Table 1 — clean performance")
        static_probs = _predict_static(model, test_feat, test_mask, device)
        craf_probs = _predict_craf(model, estimator, test_feat, test_mask, device)

        static_metrics = classification_metrics(test_labels, static_probs)
        craf_metrics = classification_metrics(test_labels, craf_probs)
        delong_p = delong_roc_test(test_labels, craf_probs, static_probs)

        logger.info("  Running baseline suite (MLP, ensemble, RF, CWM)...")
        baseline_metrics = run_baseline_suite(
            features, masks, labels,
            train_idx, val_idx, test_idx,
            score_index=score_index,
            device=device,
            random_seed=actual_seed,
        )
        seed_row = {
            "seed": actual_seed,
            "static_attention": static_metrics,
            "craf_attention": craf_metrics,
            "delong_p_craf_vs_static": float(delong_p),
            **baseline_metrics,
        }
        per_seed_table1.append(seed_row)
        per_seed_static_probs.append(static_probs)
        per_seed_craf_probs.append(craf_probs)

    results["table_1_clean_performance"] = per_seed_table1
    logger.info(f"Table 1 CRAF AUC (last seed): {per_seed_table1[-1]['craf_attention'].get('roc_auc', 'N/A'):.4f} "
                f"  DeLong p={per_seed_table1[-1]['delong_p_craf_vs_static']:.4f}")

    # Use last seed's model + estimator for remaining phases
    final_test_feat = features[test_idx]
    final_test_mask = masks[test_idx]
    final_test_labels = labels[test_idx]
    static_final = per_seed_static_probs[-1]
    craf_final = per_seed_craf_probs[-1]

    # ------------------------------------------------------------------
    # Phase 3 — Table 2 / Figure 1: domain-shift drift curves
    # ------------------------------------------------------------------
    logger.info("Phase 3: Table 2 — domain drift robustness")
    noise_levels = craf_cfg.get("drift_noise_levels", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    engine = AdversarialPerturbationEngine(domain_order, score_index)
    drift_results = {}
    degradation_aucs = {}

    for domain in domain_order:
        drift_feats_map = engine.simulate_domain_drift(
            final_test_feat, final_test_mask, domain, noise_levels
        )
        auc_static_curve = []
        auc_craf_curve = []
        domain_drift_rows = []
        for level in noise_levels:
            pert_feat = drift_feats_map[level]
            s_probs = _predict_static(model, pert_feat, final_test_mask, device)
            c_probs = _predict_craf(model, estimator, pert_feat, final_test_mask, device)
            s_m = classification_metrics(final_test_labels, s_probs)
            c_m = classification_metrics(final_test_labels, c_probs)
            auc_static_curve.append(s_m.get("roc_auc", float("nan")))
            auc_craf_curve.append(c_m.get("roc_auc", float("nan")))
            domain_drift_rows.append({
                "noise_level": float(level),
                "static_auc": s_m.get("roc_auc"),
                "craf_auc": c_m.get("roc_auc"),
            })

        noise_arr = np.array(noise_levels, dtype=float)
        deg_auc_static = reliability_degradation_auc(noise_arr, np.array(auc_static_curve))
        deg_auc_craf = reliability_degradation_auc(noise_arr, np.array(auc_craf_curve))
        degradation_aucs[domain] = {
            "static": float(deg_auc_static),
            "craf": float(deg_auc_craf),
            "craf_better": bool(deg_auc_craf > deg_auc_static),
        }
        drift_results[domain] = domain_drift_rows

    results["table_2_drift_robustness"] = drift_results
    results["figure_1_drift_curves"] = {
        "noise_levels": noise_levels,
        "degradation_aucs": degradation_aucs,
    }
    logger.info(f"Phase 3 done. CRAF better on {sum(v['craf_better'] for v in degradation_aucs.values())}/{len(domain_order)} domains")

    # ------------------------------------------------------------------
    # Phase 4 — Table 3: adversarial attacks
    # ------------------------------------------------------------------
    logger.info("Phase 4: Table 3 — adversarial attacks")
    attack_names = craf_cfg.get("adversarial_attacks", ["zero_attack", "max_attack", "gaussian_noise"])
    sigma = float(craf_cfg.get("adversarial_sigma", 0.1))
    adversarial_results = []

    for attack_name in attack_names:
        try:
            attack_type = AdversarialAttackType(attack_name)
        except ValueError:
            continue
        # Per-domain and all-domain attacks
        for target in domain_order + [None]:
            pert_feat, _ = engine.apply_attack(
                final_test_feat, final_test_mask, attack_type,
                target_domain=target, sigma=sigma,
            )
            s_probs = _predict_static(model, pert_feat, final_test_mask, device)
            c_probs = _predict_craf(model, estimator, pert_feat, final_test_mask, device)
            s_m = classification_metrics(final_test_labels, s_probs)
            c_m = classification_metrics(final_test_labels, c_probs)
            adversarial_results.append({
                "attack": attack_name,
                "target_domain": target if target is not None else "all",
                "static_auc": s_m.get("roc_auc"),
                "craf_auc": c_m.get("roc_auc"),
                "delta_auc": (c_m.get("roc_auc", 0.0) or 0.0) - (s_m.get("roc_auc", 0.0) or 0.0),
            })

    results["table_3_adversarial"] = adversarial_results
    logger.info(f"Phase 4 done. {len(adversarial_results)} attack scenarios evaluated.")

    # ------------------------------------------------------------------
    # Phase 5 — Table 4: missing modality extended dropout sweep
    # ------------------------------------------------------------------
    logger.info("Phase 5: Table 4 — missing modality robustness")
    dropout_probs = eval_cfg.get("domain_dropout_probs_extended", [0.0, 0.1, 0.2, 0.3, 0.5])
    missing_results = []

    for p_drop in dropout_probs:
        rng_seed = np.random.default_rng(99)
        drop_mask = final_test_mask.copy()
        if p_drop > 0.0:
            noise = rng_seed.random(drop_mask.shape) < p_drop
            drop_mask = drop_mask | noise

        s_probs = _predict_static(model, final_test_feat, drop_mask, device)
        c_probs = _predict_craf(model, estimator, final_test_feat, drop_mask, device)
        s_m = classification_metrics(final_test_labels, s_probs)
        c_m = classification_metrics(final_test_labels, c_probs)
        missing_results.append({
            "dropout_prob": float(p_drop),
            "static_auc": s_m.get("roc_auc"),
            "craf_auc": c_m.get("roc_auc"),
            "static_f1": s_m.get("f1"),
            "craf_f1": c_m.get("f1"),
        })

    results["table_4_missing_modality"] = missing_results
    logger.info("Phase 5 done.")

    # ------------------------------------------------------------------
    # Phase 6 — Table 5: calibration (ECE, Brier, bin-level)
    # ------------------------------------------------------------------
    logger.info("Phase 6: Table 5 — calibration quality")
    calibration_result = {
        "static_ece": float(expected_calibration_error(final_test_labels, static_final)),
        "craf_ece": float(expected_calibration_error(final_test_labels, craf_final)),
        "static_brier": float(brier_score(final_test_labels, static_final)),
        "craf_brier": float(brier_score(final_test_labels, craf_final)),
        "static_bins": _calibration_bins(final_test_labels, static_final),
        "craf_bins": _calibration_bins(final_test_labels, craf_final),
        "domain_ece_at_fit": estimator.get_domain_ece(),
    }
    results["table_5_calibration"] = calibration_result
    logger.info(
        f"Phase 6 done. ECE: static={calibration_result['static_ece']:.4f}, "
        f"CRAF={calibration_result['craf_ece']:.4f}"
    )

    # ------------------------------------------------------------------
    # Phase 6b — τ sweep + component ablation (paper Section 4.4)
    # ------------------------------------------------------------------
    logger.info("Phase 6b: τ sweep and component ablation")

    # τ sweep: AUC as a function of gate threshold
    TAU_SWEEP = [0.4, 0.5, 0.6, 0.66, 0.7, 0.8, 0.9]
    tau_sweep_results = []
    _saved_gate = estimator.gate_threshold
    for tau in TAU_SWEEP:
        estimator.gate_threshold = tau
        tau_probs = _predict_craf(model, estimator, final_test_feat, final_test_mask, device)
        tau_m = classification_metrics(final_test_labels, tau_probs)
        tau_sweep_results.append({
            "gate_threshold": tau,
            "auc": tau_m.get("roc_auc"),
            "f1": tau_m.get("f1"),
        })
    estimator.gate_threshold = _saved_gate  # restore

    results["tau_sweep"] = tau_sweep_results
    logger.info(f"τ sweep done over {len(TAU_SWEEP)} thresholds.")

    # Component ablation: zero-out each weight component to isolate its contribution
    # Also test always-gate (τ=0) to show effect of gate conservatism
    ABLATIONS: Dict[str, Dict] = {
        "full_rga":     {"ece_weight": 0.45, "ks_weight": 0.35, "sharpness_weight": 0.20, "gate_threshold": 0.66},
        "no_ece":       {"ece_weight": 0.0,  "ks_weight": 0.35, "sharpness_weight": 0.20, "gate_threshold": 0.66},
        "no_ks":        {"ece_weight": 0.45, "ks_weight": 0.0,  "sharpness_weight": 0.20, "gate_threshold": 0.66},
        "no_sharpness": {"ece_weight": 0.45, "ks_weight": 0.35, "sharpness_weight": 0.0,  "gate_threshold": 0.66},
        "always_gate":  {"ece_weight": 0.45, "ks_weight": 0.35, "sharpness_weight": 0.20, "gate_threshold": 0.0},
    }

    # Pre-compute zero-attack perturbed features for ablation stress test
    _zero_pert: Optional[np.ndarray] = None
    try:
        _zero_pert, _ = engine.apply_attack(
            final_test_feat, final_test_mask,
            AdversarialAttackType.ZERO_ATTACK, target_domain=None,
        )
    except Exception as _exc:
        logger.warning(f"Could not pre-compute zero_attack for ablation: {_exc}")

    _orig_ece_w = estimator.ece_weight
    _orig_ks_w  = estimator.ks_weight
    _orig_shr_w = estimator.sharpness_weight
    _orig_gate  = estimator.gate_threshold
    ablation_results: Dict[str, Dict] = {}
    for variant_name, params in ABLATIONS.items():
        estimator.ece_weight      = params["ece_weight"]
        estimator.ks_weight       = params["ks_weight"]
        estimator.sharpness_weight = params["sharpness_weight"]
        estimator.gate_threshold  = params["gate_threshold"]

        clean_probs = _predict_craf(model, estimator, final_test_feat, final_test_mask, device)
        clean_m = classification_metrics(final_test_labels, clean_probs)
        row: Dict[str, Any] = {"clean_auc": clean_m.get("roc_auc"), "clean_f1": clean_m.get("f1")}

        if _zero_pert is not None:
            adv_probs = _predict_craf(model, estimator, _zero_pert, final_test_mask, device)
            adv_m = classification_metrics(final_test_labels, adv_probs)
            row["zero_attack_auc"] = adv_m.get("roc_auc")
            row["zero_attack_delta"] = (
                (adv_m.get("roc_auc") or 0.0) - (clean_m.get("roc_auc") or 0.0)
            )
        ablation_results[variant_name] = row

    # Restore original weights
    estimator.ece_weight      = _orig_ece_w
    estimator.ks_weight       = _orig_ks_w
    estimator.sharpness_weight = _orig_shr_w
    estimator.gate_threshold  = _orig_gate

    results["component_ablation"] = ablation_results
    logger.info(f"Component ablation done: {list(ablation_results.keys())}")

    # ------------------------------------------------------------------
    # Phase 7 — CDA validation
    # ------------------------------------------------------------------
    logger.info("Phase 7: CDA validation")
    n_cda = int(eval_cfg.get("cda_samples", 100))
    cda_idx = np.arange(min(n_cda, len(test_idx)))
    cda_feat = final_test_feat[cda_idx]
    cda_mask = final_test_mask[cda_idx]
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
    # Spearman vs domain ECE (as a proxy for SHAP importance when SHAP isn't available)
    domain_ece = estimator.get_domain_ece()
    spearman_vs_ece = explainer.correlation_with_shap(cf_results, {d: 1.0 - v for d, v in domain_ece.items()})

    sample_narratives = [r.narrative for r in cf_results[:5]]

    cda_validation = {
        "n_samples": len(cf_results),
        "mean_cf_impacts_abs": mean_cf_impacts,
        "spearman_cda_vs_ece_reliability": float(spearman_vs_ece),
        "sample_narratives": sample_narratives,
    }
    results["cda_validation"] = cda_validation
    logger.info(f"Phase 7 done. Spearman CDA/ECE={spearman_vs_ece:.3f}")

    # ------------------------------------------------------------------
    # Phase 8 — Statistical tests + aggregate summary
    # ------------------------------------------------------------------
    logger.info("Phase 8: Statistical tests + summary")
    if len(seeds) > 1:
        static_aucs = [row["static_attention"].get("roc_auc", float("nan")) for row in per_seed_table1]
        craf_aucs = [row["craf_attention"].get("roc_auc", float("nan")) for row in per_seed_table1]
        paired_p = paired_ttest(np.array(craf_aucs), np.array(static_aucs))
    else:
        static_aucs = [per_seed_table1[0]["static_attention"].get("roc_auc", float("nan"))]
        craf_aucs = [per_seed_table1[0]["craf_attention"].get("roc_auc", float("nan"))]
        paired_p = float("nan")

    n_domains_craf_better_drift = sum(v["craf_better"] for v in degradation_aucs.values())
    breakthrough_checks = {
        "delong_p_lt_0p05_on_last_seed": bool(per_seed_table1[-1]["delong_p_craf_vs_static"] < 0.05),
        "craf_better_drift_auc_n_domains": n_domains_craf_better_drift,
        "craf_better_drift_auc_all_domains": bool(n_domains_craf_better_drift == len(domain_order)),
        "spearman_cda_vs_ece_gt_0p6": bool(np.isfinite(spearman_vs_ece) and spearman_vs_ece > 0.6),
        "craf_ece_lt_static_ece": bool(calibration_result["craf_ece"] < calibration_result["static_ece"]),
        "paired_ttest_p": float(paired_p) if np.isfinite(paired_p) else None,
    }

    results["statistical_summary"] = {
        "per_seed_static_auc": static_aucs,
        "per_seed_craf_auc": craf_aucs,
        "paired_ttest_p_craf_vs_static": float(paired_p) if np.isfinite(paired_p) else None,
        "breakthrough_checks": breakthrough_checks,
    }

    passed = sum([
        breakthrough_checks["delong_p_lt_0p05_on_last_seed"],
        breakthrough_checks["craf_better_drift_auc_all_domains"],
        breakthrough_checks["craf_ece_lt_static_ece"],
    ])
    logger.info(f"Breakthrough checks passed: {passed}/3")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="CRAF breakthrough experiment")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to attention_config.yaml")
    parser.add_argument("--output", default="experiments/fusion/attention_fusion/breakthrough_results.json")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke test)")
    parser.add_argument("--seed", type=int, default=None, help="Override single seed")
    args = parser.parse_args()

    if args.synthetic:
        logger.info("Running with synthetic data (smoke test mode)")
        features, masks, labels, sample_ids, domain_order = _make_synthetic()
        # Build a minimal cfg dict
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
            "evaluation": {"seeds": [42], "cda_samples": 20,
                           "domain_dropout_probs_extended": [0.0, 0.1, 0.3]},
            "reliability": {"ece_weight": 0.45, "ks_weight": 0.35, "sharpness_weight": 0.20,
                            "n_calibration_bins": 5, "min_samples_for_ks": 10,
                            "gate_threshold": 0.66},
            "craf": {"drift_noise_levels": [0.0, 0.1, 0.3],
                     "adversarial_attacks": ["zero_attack", "gaussian_noise"],
                     "adversarial_sigma": 0.1},
        }

        # Patch _load_data to return synthetic data
        def _load_data_synthetic(_cfg):
            n_features = features.shape[2]
            feature_columns = [f"feat_{i}" for i in range(n_features)]
            return features, masks, labels, sample_ids, domain_order, feature_columns, None, 0

        import uais.fusion.attention.run_breakthrough_experiment as this_module
        import types
        # Monkey-patch for synthetic path
        results = _run_synthetic_experiment(cfg, features, masks, labels, sample_ids, domain_order, seed_override=args.seed)
    else:
        cfg_path = _resolve(args.config)
        cfg = load_yaml(str(cfg_path))
        results = run_experiment(cfg, seed_override=args.seed)

    out_path = _resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if (v != v) else v  # nan → None
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

    logger.info(f"Results saved to {out_path}")
    checks = results.get("statistical_summary", {}).get("breakthrough_checks", {})
    logger.info("=== Breakthrough Summary ===")
    for k, v in checks.items():
        logger.info(f"  {k}: {v}")


def _run_synthetic_experiment(cfg, features, masks, labels, sample_ids, domain_order, seed_override=None) -> Dict:
    """Run experiment using pre-loaded synthetic arrays (bypasses _load_data)."""
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    rel_cfg = cfg.get("reliability", {})
    craf_cfg = cfg.get("craf", {})

    seeds = eval_cfg.get("seeds", [42])
    device = torch.device("cpu")
    results: Dict = {}

    score_index = 0
    n_features = features.shape[2]
    input_dim = n_features
    num_domains = features.shape[1]
    confidence_index = None

    per_seed_table1 = []
    per_seed_static_probs = []
    per_seed_craf_probs = []
    train_idx_last = val_idx_last = test_idx_last = None
    model_last = estimator_last = None

    for seed in seeds:
        actual_seed = seed_override if seed_override is not None else seed
        set_seed(actual_seed)

        train_idx, val_idx, test_idx = _split(labels, {**train_cfg, "seed": actual_seed})
        train_loader, val_loader, test_loader = _make_loaders(
            features, masks, labels, train_idx, val_idx, test_idx,
            batch_size=int(train_cfg.get("batch_size", 64)),
        )

        model = _build_model(cfg, num_domains, input_dim, confidence_index, device)
        _train_model(model, train_loader, val_loader, cfg, device)
        model.eval()

        estimator = ReliabilityEstimator(
            domain_order=domain_order,
            score_index=score_index,
            ece_weight=float(rel_cfg.get("ece_weight", 0.45)),
            ks_weight=float(rel_cfg.get("ks_weight", 0.35)),
            sharpness_weight=float(rel_cfg.get("sharpness_weight", 0.20)),
            n_calibration_bins=int(rel_cfg.get("n_calibration_bins", 5)),
            min_samples_for_ks=int(rel_cfg.get("min_samples_for_ks", 10)),
            gate_threshold=float(rel_cfg.get("gate_threshold", 0.66)),
        )
        estimator.fit(features[val_idx], masks[val_idx], labels[val_idx])

        test_feat = features[test_idx]
        test_mask = masks[test_idx]
        test_labels = labels[test_idx]

        static_probs = _predict_static(model, test_feat, test_mask, device)
        craf_probs = _predict_craf(model, estimator, test_feat, test_mask, device)
        static_metrics = classification_metrics(test_labels, static_probs)
        craf_metrics = classification_metrics(test_labels, craf_probs)
        delong_p = delong_roc_test(test_labels, craf_probs, static_probs)

        baseline_metrics = run_baseline_suite(
            features, masks, labels,
            train_idx, val_idx, test_idx,
            score_index=score_index,
            device=device,
            random_seed=actual_seed,
        )

        per_seed_table1.append({
            "seed": actual_seed,
            "static_attention": static_metrics,
            "craf_attention": craf_metrics,
            "delong_p_craf_vs_static": float(delong_p),
            **baseline_metrics,
        })
        per_seed_static_probs.append(static_probs)
        per_seed_craf_probs.append(craf_probs)
        train_idx_last, val_idx_last, test_idx_last = train_idx, val_idx, test_idx
        model_last, estimator_last = model, estimator

    results["table_1_clean_performance"] = per_seed_table1

    final_test_feat = features[test_idx_last]
    final_test_mask = masks[test_idx_last]
    final_test_labels = labels[test_idx_last]
    static_final = per_seed_static_probs[-1]
    craf_final = per_seed_craf_probs[-1]
    model = model_last
    estimator = estimator_last

    # Drift
    noise_levels = craf_cfg.get("drift_noise_levels", [0.0, 0.1, 0.3])
    engine = AdversarialPerturbationEngine(domain_order, score_index)
    drift_results = {}
    degradation_aucs = {}
    for domain in domain_order:
        drift_feats_map = engine.simulate_domain_drift(final_test_feat, final_test_mask, domain, noise_levels)
        auc_static_curve, auc_craf_curve, domain_rows = [], [], []
        for level in noise_levels:
            pf = drift_feats_map[level]
            sp = _predict_static(model, pf, final_test_mask, device)
            cp = _predict_craf(model, estimator, pf, final_test_mask, device)
            sm = classification_metrics(final_test_labels, sp)
            cm = classification_metrics(final_test_labels, cp)
            auc_static_curve.append(sm.get("roc_auc", float("nan")))
            auc_craf_curve.append(cm.get("roc_auc", float("nan")))
            domain_rows.append({"noise_level": float(level), "static_auc": sm.get("roc_auc"), "craf_auc": cm.get("roc_auc")})
        noise_arr = np.array(noise_levels, dtype=float)
        deg_s = reliability_degradation_auc(noise_arr, np.array(auc_static_curve))
        deg_c = reliability_degradation_auc(noise_arr, np.array(auc_craf_curve))
        degradation_aucs[domain] = {"static": float(deg_s), "craf": float(deg_c), "craf_better": bool(deg_c > deg_s)}
        drift_results[domain] = domain_rows
    results["table_2_drift_robustness"] = drift_results
    results["figure_1_drift_curves"] = {"noise_levels": noise_levels, "degradation_aucs": degradation_aucs}

    # Adversarial
    attack_names = craf_cfg.get("adversarial_attacks", ["zero_attack", "gaussian_noise"])
    sigma = float(craf_cfg.get("adversarial_sigma", 0.1))
    adv_results = []
    for attack_name in attack_names:
        try:
            at = AdversarialAttackType(attack_name)
        except ValueError:
            continue
        for target in domain_order + [None]:
            pf, _ = engine.apply_attack(final_test_feat, final_test_mask, at, target_domain=target, sigma=sigma)
            sp = _predict_static(model, pf, final_test_mask, device)
            cp = _predict_craf(model, estimator, pf, final_test_mask, device)
            sm = classification_metrics(final_test_labels, sp)
            cm = classification_metrics(final_test_labels, cp)
            adv_results.append({"attack": attack_name, "target_domain": target if target else "all",
                                 "static_auc": sm.get("roc_auc"), "craf_auc": cm.get("roc_auc"),
                                 "delta_auc": (cm.get("roc_auc") or 0) - (sm.get("roc_auc") or 0)})
    results["table_3_adversarial"] = adv_results

    # Missing modality
    dropout_probs = eval_cfg.get("domain_dropout_probs_extended", [0.0, 0.1, 0.3])
    missing_results = []
    for p_drop in dropout_probs:
        rng_seed = np.random.default_rng(99)
        drop_mask = final_test_mask.copy()
        if p_drop > 0.0:
            drop_mask = drop_mask | (rng_seed.random(drop_mask.shape) < p_drop)
        sp = _predict_static(model, final_test_feat, drop_mask, device)
        cp = _predict_craf(model, estimator, final_test_feat, drop_mask, device)
        sm = classification_metrics(final_test_labels, sp)
        cm = classification_metrics(final_test_labels, cp)
        missing_results.append({"dropout_prob": float(p_drop), "static_auc": sm.get("roc_auc"),
                                 "craf_auc": cm.get("roc_auc")})
    results["table_4_missing_modality"] = missing_results

    # Calibration
    cal = {
        "static_ece": float(expected_calibration_error(final_test_labels, static_final)),
        "craf_ece": float(expected_calibration_error(final_test_labels, craf_final)),
        "static_brier": float(brier_score(final_test_labels, static_final)),
        "craf_brier": float(brier_score(final_test_labels, craf_final)),
        "static_bins": _calibration_bins(final_test_labels, static_final),
        "craf_bins": _calibration_bins(final_test_labels, craf_final),
        "domain_ece_at_fit": estimator.get_domain_ece(),
    }
    results["table_5_calibration"] = cal

    # τ sweep + component ablation (synthetic path — condensed)
    _saved_gate_s = estimator.gate_threshold
    tau_sweep_s = []
    for tau in [0.4, 0.5, 0.6, 0.66, 0.8, 0.9]:
        estimator.gate_threshold = tau
        tp = _predict_craf(model, estimator, final_test_feat, final_test_mask, device)
        tm = classification_metrics(final_test_labels, tp)
        tau_sweep_s.append({"gate_threshold": tau, "auc": tm.get("roc_auc")})
    estimator.gate_threshold = _saved_gate_s
    results["tau_sweep"] = tau_sweep_s

    _orig_ece_ws = estimator.ece_weight
    _orig_ks_ws  = estimator.ks_weight
    _orig_shr_ws = estimator.sharpness_weight
    _orig_gate_s = estimator.gate_threshold
    ablation_s: Dict[str, Any] = {}
    for _vname, _params in [
        ("full_rga",     {"ece_weight": 0.45, "ks_weight": 0.35, "sharpness_weight": 0.20, "gate_threshold": 0.66}),
        ("no_ece",       {"ece_weight": 0.0,  "ks_weight": 0.35, "sharpness_weight": 0.20, "gate_threshold": 0.66}),
        ("no_ks",        {"ece_weight": 0.45, "ks_weight": 0.0,  "sharpness_weight": 0.20, "gate_threshold": 0.66}),
        ("no_sharpness", {"ece_weight": 0.45, "ks_weight": 0.35, "sharpness_weight": 0.0,  "gate_threshold": 0.66}),
        ("always_gate",  {"ece_weight": 0.45, "ks_weight": 0.35, "sharpness_weight": 0.20, "gate_threshold": 0.0}),
    ]:
        estimator.ece_weight      = _params["ece_weight"]
        estimator.ks_weight       = _params["ks_weight"]
        estimator.sharpness_weight = _params["sharpness_weight"]
        estimator.gate_threshold  = _params["gate_threshold"]
        _cp = _predict_craf(model, estimator, final_test_feat, final_test_mask, device)
        _cm = classification_metrics(final_test_labels, _cp)
        ablation_s[_vname] = {"clean_auc": _cm.get("roc_auc")}
    estimator.ece_weight      = _orig_ece_ws
    estimator.ks_weight       = _orig_ks_ws
    estimator.sharpness_weight = _orig_shr_ws
    estimator.gate_threshold  = _orig_gate_s
    results["component_ablation"] = ablation_s

    # CDA
    n_cda = int(eval_cfg.get("cda_samples", 20))
    cda_idx = np.arange(min(n_cda, len(test_idx_last)))
    cda_feat = final_test_feat[cda_idx]
    cda_mask = final_test_mask[cda_idx]
    cda_ids = list(range(len(cda_idx)))
    explainer = CounterfactualDomainExplainer(model=model, domain_order=domain_order, device=device,
                                              reliability_estimator=estimator, use_craf_weights=True)
    cf_results = explainer.explain_batch(cda_feat, cda_mask, cda_ids, batch_size=16)
    mean_cf_impacts = {
        d: float(np.nanmean([abs(r.cf_impacts.get(d, float("nan"))) for r in cf_results]))
        for d in domain_order
    }
    spearman_vs_ece = explainer.correlation_with_shap(cf_results, {d: 1.0 - v for d, v in estimator.get_domain_ece().items()})
    results["cda_validation"] = {
        "n_samples": len(cf_results),
        "mean_cf_impacts_abs": mean_cf_impacts,
        "spearman_cda_vs_ece_reliability": float(spearman_vs_ece),
        "sample_narratives": [r.narrative for r in cf_results[:3]],
    }

    # Stats
    if len(seeds) > 1:
        static_aucs = [row["static_attention"].get("roc_auc", float("nan")) for row in per_seed_table1]
        craf_aucs = [row["craf_attention"].get("roc_auc", float("nan")) for row in per_seed_table1]
        paired_p = paired_ttest(np.array(craf_aucs), np.array(static_aucs))
    else:
        static_aucs = [per_seed_table1[0]["static_attention"].get("roc_auc", float("nan"))]
        craf_aucs = [per_seed_table1[0]["craf_attention"].get("roc_auc", float("nan"))]
        paired_p = float("nan")

    n_better = sum(v["craf_better"] for v in degradation_aucs.values())
    results["statistical_summary"] = {
        "per_seed_static_auc": static_aucs,
        "per_seed_craf_auc": craf_aucs,
        "paired_ttest_p_craf_vs_static": float(paired_p) if np.isfinite(paired_p) else None,
        "breakthrough_checks": {
            "delong_p_lt_0p05_on_last_seed": bool(per_seed_table1[-1]["delong_p_craf_vs_static"] < 0.05),
            "craf_better_drift_auc_n_domains": n_better,
            "craf_better_drift_auc_all_domains": bool(n_better == len(domain_order)),
            "spearman_cda_vs_ece_gt_0p6": bool(np.isfinite(spearman_vs_ece) and spearman_vs_ece > 0.6),
            "craf_ece_lt_static_ece": bool(cal["craf_ece"] < cal["static_ece"]),
        },
    }
    return results


if __name__ == "__main__":
    main()
