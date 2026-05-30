"""Train-once, score flagship RGA+ variants + deploy v3 (validation / test)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from uais.fusion.attention.baselines import SARScoreAdapter
from uais.fusion.attention.elara_deploy_policy import (
    DeployPolicySpec,
    ElaraDeployArtifacts,
    predict_elara_deploy,
    select_validation_fallback,
)
from uais.fusion.attention.fusion_inference import build_gate_decision_calibration
from uais.fusion.attention.certified_heterogeneous_fusion import fit_chf_on_validation
from uais.fusion.attention.reliability_boosted_fusion_flagship import (
    FLAGSHIP_RGA_VARIANTS,
    ReliabilityBoostedFusionFlagship,
)
from uais.fusion.attention.train_attention_fusion import set_seed
from src.scripts.scenario_c.win_vs_sar_harness import (
    _metrics_with_threshold,
    _safe_auc,
)

logger = logging.getLogger(__name__)

DEFAULT_DEPLOY_V3 = DeployPolicySpec(
    policy_id="ELARA_DEPLOY_v3",
    routing_mode="gdr_shift_sar_fallback",
    fallback_method="sar_score_adapter",
    gate_decision_rule={
        "enabled": True,
        "coherence_min": 0.35,
        "tau": 0.55,
        "margin_epsilon": 0.0,
    },
)


def _unknown_category_mask(
    categories: np.ndarray | None,
    train_categories: np.ndarray | None,
) -> np.ndarray | None:
    if categories is None or train_categories is None:
        return None
    known = set(np.asarray(train_categories).astype(str).tolist())
    cats = np.asarray(categories).astype(str)
    return np.array([c not in known for c in cats], dtype=bool)


def predict_deploy_v3_shift_sar(
    artifacts: ElaraDeployArtifacts,
    features: np.ndarray,
    masks: np.ndarray,
    *,
    policy: DeployPolicySpec,
    primary_probs: np.ndarray,
    categories: np.ndarray | None = None,
    train_categories: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """GDR path when allowed; SAR for unknown held-out categories; else val-selected fallback."""
    unknown = _unknown_category_mask(categories, train_categories)
    sar_full = artifacts.sar.predict_proba(features, masks)
    if unknown is None or not unknown.any():
        deploy_probs, stats = predict_elara_deploy(
            artifacts,
            features,
            masks,
            policy=policy,
            fallback_probs=primary_probs,
        )
        stats["unknown_category_rate"] = 0.0
        return deploy_probs, stats

    known_mask = ~unknown
    out = np.asarray(primary_probs, dtype=np.float32).copy()
    if known_mask.any():
        known_probs, stats = predict_elara_deploy(
            artifacts,
            features[known_mask],
            masks[known_mask],
            policy=policy,
            fallback_probs=primary_probs[known_mask],
        )
        out[known_mask] = known_probs
    else:
        stats = {"policy_id": policy.policy_id, "routing_mode": policy.routing_mode}
    out[unknown] = sar_full[unknown]
    stats["unknown_category_rate"] = float(unknown.mean())
    stats["n_unknown"] = int(unknown.sum())
    stats["policy_id"] = policy.policy_id
    return out, stats


def evaluate_flagship_seed(
    cfg: dict[str, Any],
    *,
    seed: int,
    eval_split: str = "validation",
    rga_variants: list[str] | None = None,
    include_deploy_v3: bool = True,
) -> dict[str, Any]:
    from src.scripts.run_breakthrough_experiment import (
        _apply_calibration_transfer,
        _build_model,
        _load_data,
        _make_loaders,
        _make_reliability_estimator,
        _predict_static,
        _split,
        _train_model,
    )

    rga_variants = rga_variants or list(FLAGSHIP_RGA_VARIANTS.keys())
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    rel_cfg = cfg.get("reliability", {})
    rga_plus_cfg = cfg.get("rga_plus", {})
    threshold_strategy = eval_cfg.get("decision_threshold_strategy", eval_cfg.get("decision_threshold", "val_f1"))
    selection_metric = str(rga_plus_cfg.get("selection_metric", "roc_auc"))
    tta_steps = int(eval_cfg.get("tta_steps", 25))
    clean_gate_threshold = float(rel_cfg.get("clean_gate_threshold", rel_cfg.get("gate_threshold", 0.66)))
    gate_decision_cfg = DEFAULT_DEPLOY_V3.gate_decision_rule

    set_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        features,
        masks,
        labels,
        _sample_ids,
        domain_order,
        _fc,
        confidence_index,
        score_index,
        sample_splits,
        sample_categories,
    ) = _load_data(cfg)
    if (cfg.get("calibration_transfer") or {}).get("enabled"):
        features, _ = _apply_calibration_transfer(cfg, features, masks, domain_order, score_index or 0)

    train_idx, val_idx, test_idx = _split(labels, {**train_cfg, "seed": seed}, split_values=sample_splits)
    split_map = {"train": train_idx, "validation": val_idx, "val": val_idx, "test": test_idx}
    eval_key = eval_split.strip().lower()
    eval_idx = split_map[eval_key]
    if eval_key == "test" and not cfg.get("elara_deploy", {}).get("allow_test_eval", False):
        raise RuntimeError("Test eval blocked until elara_deploy.allow_test_eval: true")

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
    _train_model(model, train_loader, val_loader, cfg, device, score_index=score_index)
    model.eval()

    estimator = _make_reliability_estimator(rel_cfg, domain_order, score_index)
    estimator.fit(features[val_idx], masks[val_idx], labels[val_idx])

    train_feat = features[train_idx]
    train_mask = masks[train_idx]
    train_labels = labels[train_idx]
    val_feat = features[val_idx]
    val_mask = masks[val_idx]
    val_labels = labels[val_idx]
    eval_feat = features[eval_idx]
    eval_mask = masks[eval_idx]
    eval_labels = labels[eval_idx]
    train_cats = sample_categories[train_idx] if sample_categories is not None else None
    val_cats = sample_categories[val_idx] if sample_categories is not None else None
    eval_cats = sample_categories[eval_idx] if sample_categories is not None else None

    sar = SARScoreAdapter(random_seed=int(seed), adaptation_steps=tta_steps).fit(
        train_feat, train_mask, train_labels
    )
    sar_val = sar.predict_proba(val_feat, val_mask)
    sar_eval = sar.predict_proba(eval_feat, eval_mask)

    variant_rows: dict[str, dict[str, Any]] = {}
    best_val_probs = sar_val
    best_val_auc = -1.0
    for variant_name in rga_variants:
        opts = FLAGSHIP_RGA_VARIANTS[variant_name]
        rga = ReliabilityBoostedFusionFlagship(
            score_index=score_index,
            confidence_index=confidence_index,
            random_seed=int(seed),
            selection_metric=selection_metric,
            use_category_features=bool(opts["use_category_features"]),
            tta_candidates=tuple(opts["tta_candidates"]),
        ).fit(
            train_feat,
            train_mask,
            train_labels,
            val_feat,
            val_mask,
            val_labels,
            reliability_estimator=estimator,
            train_categories=train_cats,
            val_categories=val_cats,
        )
        val_probs = rga.predict_proba(val_feat, val_mask, categories=val_cats)
        eval_probs = rga.predict_proba(eval_feat, eval_mask, categories=eval_cats)
        metrics = _metrics_with_threshold(
            eval_labels,
            eval_probs,
            val_labels=val_labels,
            val_probs=val_probs,
            strategy=threshold_strategy,
        )
        variant_rows[variant_name] = {
            "roc_auc": float(metrics["roc_auc"]),
            "delta_vs_sar": float(metrics["roc_auc"]) - _safe_auc(eval_labels, sar_eval),
            "selected_candidate": rga.selected_candidate,
            "selected_tta_steps": int(rga.selected_tta_steps),
            "use_category_features": bool(opts["use_category_features"]),
        }
        if float(metrics["roc_auc"]) > best_val_auc:
            best_val_auc = float(metrics["roc_auc"])
            best_val_probs = val_probs

    deploy_row = None
    if include_deploy_v3:
        best_variant = max(variant_rows.items(), key=lambda kv: kv[1]["roc_auc"])[0]
        best_opts = FLAGSHIP_RGA_VARIANTS[best_variant]
        rga_best = ReliabilityBoostedFusionFlagship(
            score_index=score_index,
            confidence_index=confidence_index,
            random_seed=int(seed),
            selection_metric=selection_metric,
            use_category_features=bool(best_opts["use_category_features"]),
            tta_candidates=tuple(best_opts["tta_candidates"]),
        ).fit(
            train_feat,
            train_mask,
            train_labels,
            val_feat,
            val_mask,
            val_labels,
            reliability_estimator=estimator,
            train_categories=train_cats,
            val_categories=val_cats,
        )
        primary_eval = rga_best.predict_proba(eval_feat, eval_mask, categories=eval_cats)
        primary_val = rga_best.predict_proba(val_feat, val_mask, categories=val_cats)
        cal = build_gate_decision_calibration(
            model,
            estimator,
            val_feat,
            val_mask,
            val_labels,
            device,
            tau=float(gate_decision_cfg["tau"]),
        )
        artifacts = ElaraDeployArtifacts(
            model=model,
            estimator=estimator,
            sar=sar,
            device=device,
            gate_decision_calibration=cal,
            gate_decision_rule_cfg=gate_decision_cfg,
            clean_gate_threshold=clean_gate_threshold,
        )
        deploy_eval, deploy_stats = predict_deploy_v3_shift_sar(
            artifacts,
            eval_feat,
            eval_mask,
            policy=DEFAULT_DEPLOY_V3,
            primary_probs=primary_eval,
            categories=eval_cats,
            train_categories=train_cats,
        )
        deploy_val, _ = predict_deploy_v3_shift_sar(
            artifacts,
            val_feat,
            val_mask,
            policy=DEFAULT_DEPLOY_V3,
            primary_probs=primary_val,
            categories=val_cats,
            train_categories=train_cats,
        )
        d_metrics = _metrics_with_threshold(
            eval_labels,
            deploy_eval,
            val_labels=val_labels,
            val_probs=deploy_val,
            strategy=threshold_strategy,
        )
        deploy_row = {
            "roc_auc": float(d_metrics["roc_auc"]),
            "delta_vs_sar": float(d_metrics["roc_auc"]) - _safe_auc(eval_labels, sar_eval),
            "best_rga_variant": best_variant,
            "deploy_stats": deploy_stats,
        }

    static_val = _predict_static(model, val_feat, val_mask, device)
    static_eval = _predict_static(model, eval_feat, eval_mask, device)
    rel_val = estimator.compute_reliability_weights(val_feat, val_mask)
    rel_eval = estimator.compute_reliability_weights(eval_feat, eval_mask)

    best_name = max(variant_rows.items(), key=lambda kv: kv[1]["roc_auc"])[0]
    best_opts = FLAGSHIP_RGA_VARIANTS[best_name]
    rga_for_chf = ReliabilityBoostedFusionFlagship(
        score_index=score_index,
        confidence_index=confidence_index,
        random_seed=int(seed),
        selection_metric=selection_metric,
        use_category_features=bool(best_opts["use_category_features"]),
        tta_candidates=tuple(best_opts["tta_candidates"]),
    ).fit(
        train_feat,
        train_mask,
        train_labels,
        val_feat,
        val_mask,
        val_labels,
        reliability_estimator=estimator,
        train_categories=train_cats,
        val_categories=val_cats,
    )
    best_rga_val = rga_for_chf.predict_proba(val_feat, val_mask, categories=val_cats)
    best_rga_eval = rga_for_chf.predict_proba(eval_feat, eval_mask, categories=eval_cats)

    from uais.fusion.attention.gate_decision_rule import per_sample_mean_reliability
    from elara.theory.t8_certified_heterogeneous_fusion import batch_coherence_scores

    chf = fit_chf_on_validation(
        val_labels,
        sar_val=sar_val,
        rga_val=best_rga_val,
        static_val=static_val,
        reliability_weights_val=rel_val,
        masks_val=val_mask,
        switching_certified=True,
    )
    chf_kwargs = dict(
        coherence_per_sample=batch_coherence_scores(rel_eval, eval_mask),
        reliability_mean=per_sample_mean_reliability(rel_eval, eval_mask),
        shift_aware=True,
        eval_categories=eval_cats,
        train_categories=train_cats,
    )
    chf_eval = chf.predict_proba(sar_eval, best_rga_eval, static_eval, **chf_kwargs)
    chf_val = chf.predict_proba(
        sar_val,
        best_rga_val,
        static_val,
        coherence_per_sample=batch_coherence_scores(rel_val, val_mask),
        reliability_mean=per_sample_mean_reliability(rel_val, val_mask),
    )
    chf_metrics = _metrics_with_threshold(
        eval_labels,
        chf_eval,
        val_labels=val_labels,
        val_probs=chf_val,
        strategy=threshold_strategy,
    )
    chf_row = {
        "roc_auc": float(chf_metrics["roc_auc"]),
        "delta_vs_sar": float(chf_metrics["roc_auc"]) - _safe_auc(eval_labels, sar_eval),
        "certificate": chf.certificate.as_dict(),
    }

    fallback_choice, _, _ = select_validation_fallback(val_labels, sar_val, best_val_probs)

    return {
        "seed": int(seed),
        "eval_split": eval_key,
        "n_eval": int(len(eval_idx)),
        "sar_roc_auc": _safe_auc(eval_labels, sar_eval),
        "variants": variant_rows,
        "elara_deploy_v3": deploy_row,
        "elara_chf_v1": chf_row,
        "val_fallback_choice": fallback_choice,
    }
