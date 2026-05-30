"""Shared train/eval harness for WIN vs SAR validation (validation split only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from uais.fusion.attention.baselines import SARScoreAdapter
from uais.fusion.attention.elara_deploy_policy import (
    DeployPolicySpec,
    ElaraDeployArtifacts,
    deploy_policy_from_cfg,
    predict_elara_deploy,
    select_validation_fallback,
)
from uais.fusion.attention.fusion_inference import GateDecisionCalibration
from uais.fusion.attention.fusion_inference import build_gate_decision_calibration
from uais.fusion.attention.reliability_boosted_fusion import ReliabilityBoostedFusion
from uais.utils.metrics import classification_metrics, select_decision_threshold
from uais.fusion.attention.train_attention_fusion import set_seed

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return 0.5


def _metrics_with_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    strategy: str | None,
) -> dict[str, Any]:
    threshold = select_decision_threshold(val_labels, val_probs, strategy=strategy)
    metrics = classification_metrics(labels, probs, threshold=threshold)
    metrics["roc_auc"] = _safe_auc(labels, probs)
    metrics["threshold_strategy"] = (strategy or "fixed_0p5").strip().lower()
    return metrics


@dataclass
class SeedInferenceBundle:
    """Trained fusion + baseline probs for fast Phase-2 policy sweeps."""

    seed: int
    eval_labels: np.ndarray
    val_labels: np.ndarray
    eval_feat: np.ndarray
    eval_mask: np.ndarray
    val_feat: np.ndarray
    val_mask: np.ndarray
    sar_eval: np.ndarray
    sar_val: np.ndarray
    rga_eval: np.ndarray
    rga_val: np.ndarray
    val_fallback_choice: str
    val_fallback_sar_auc: float
    val_fallback_rga_auc: float
    model: Any
    estimator: Any
    sar: SARScoreAdapter
    device: torch.device
    clean_gate_threshold: float
    gate_calibrations: dict[float, GateDecisionCalibration] = field(default_factory=dict)

    def make_deploy_artifacts(
        self,
        gate_decision_rule_cfg: dict[str, Any],
        gate_calibration: GateDecisionCalibration | None,
    ) -> ElaraDeployArtifacts:
        return ElaraDeployArtifacts(
            model=self.model,
            estimator=self.estimator,
            sar=self.sar,
            device=self.device,
            gate_decision_calibration=gate_calibration,
            gate_decision_rule_cfg=gate_decision_rule_cfg,
            clean_gate_threshold=self.clean_gate_threshold,
            rga_probs=self.rga_eval,
        )


def train_seed_bundle(cfg: dict[str, Any], *, seed: int) -> SeedInferenceBundle:
    """Train once per seed; cache probabilities and per-tau GDR calibrations."""
    from src.scripts.run_breakthrough_experiment import (
        _apply_calibration_transfer,
        _build_model,
        _load_data,
        _make_loaders,
        _make_reliability_estimator,
        _split,
        _train_model,
    )

    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    rel_cfg = cfg.get("reliability", {})
    rga_plus_cfg = cfg.get("rga_plus", {})
    rga_plus_selection_metric = str(rga_plus_cfg.get("selection_metric", "roc_auc"))
    tta_steps = int(eval_cfg.get("tta_steps", 25))
    clean_gate_threshold = float(rel_cfg.get("clean_gate_threshold", rel_cfg.get("gate_threshold", 0.66)))
    tau_values = [float(v) for v in (cfg.get("elara_deploy") or {}).get("sweep_tau_values", [0.55, 0.66, 0.75])]

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
        _sample_categories,
    ) = _load_data(cfg)
    if (cfg.get("calibration_transfer") or {}).get("enabled"):
        features, _ = _apply_calibration_transfer(cfg, features, masks, domain_order, score_index or 0)

    train_idx, val_idx, test_idx = _split(labels, {**train_cfg, "seed": seed}, split_values=sample_splits)
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

    train_feat, train_mask, train_labels = features[train_idx], masks[train_idx], labels[train_idx]
    val_feat, val_mask, val_labels = features[val_idx], masks[val_idx], labels[val_idx]

    sar = SARScoreAdapter(random_seed=int(seed), adaptation_steps=tta_steps).fit(
        train_feat, train_mask, train_labels
    )
    sar_val = sar.predict_proba(val_feat, val_mask)

    rga_boosted = ReliabilityBoostedFusion(
        score_index=score_index,
        confidence_index=confidence_index,
        random_seed=int(seed),
        selection_metric=rga_plus_selection_metric,
    ).fit(
        train_feat,
        train_mask,
        train_labels,
        val_feat,
        val_mask,
        val_labels,
        reliability_estimator=estimator,
    )
    rga_val = rga_boosted.predict_proba(val_feat, val_mask)
    rga_eval = rga_boosted.predict_proba(val_feat, val_mask)

    # Bundle is validation-only for Phase 2 sweeps.
    eval_feat, eval_mask, eval_labels = val_feat, val_mask, val_labels
    sar_eval = sar.predict_proba(eval_feat, eval_mask)

    fallback_choice, _, _ = select_validation_fallback(val_labels, sar_val, rga_val)

    gate_calibrations: dict[float, GateDecisionCalibration] = {}
    for tau in tau_values:
        gate_calibrations[tau] = build_gate_decision_calibration(
            model,
            estimator,
            val_feat,
            val_mask,
            val_labels,
            device,
            tau=float(tau),
            margin_epsilon=0.0,
        )

    return SeedInferenceBundle(
        seed=int(seed),
        eval_labels=eval_labels,
        val_labels=val_labels,
        eval_feat=eval_feat,
        eval_mask=eval_mask,
        val_feat=val_feat,
        val_mask=val_mask,
        sar_eval=sar_eval,
        sar_val=sar_val,
        rga_eval=rga_eval,
        rga_val=rga_val,
        val_fallback_choice=fallback_choice,
        val_fallback_sar_auc=_safe_auc(val_labels, sar_val),
        val_fallback_rga_auc=_safe_auc(val_labels, rga_val),
        model=model,
        estimator=estimator,
        sar=sar,
        device=device,
        clean_gate_threshold=clean_gate_threshold,
        gate_calibrations=gate_calibrations,
    )


def evaluate_seed(
    cfg: dict[str, Any],
    *,
    seed: int,
    eval_split: str = "validation",
    deploy_policy: DeployPolicySpec | None = None,
    sar_adaptation_steps: int | None = None,
) -> dict[str, Any]:
    """Train fusion + baselines; score only ``eval_split`` (default validation)."""
    from src.scripts.run_breakthrough_experiment import (
        _apply_calibration_transfer,
        _build_model,
        _load_data,
        _make_loaders,
        _make_reliability_estimator,
        _per_category_auroc,
        _predict_craf,
        _predict_static,
        _split,
        _train_model,
    )

    deploy_policy = deploy_policy or deploy_policy_from_cfg(cfg)
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    rel_cfg = cfg.get("reliability", {})
    gate_decision_cfg = deploy_policy.gate_decision_rule
    rga_plus_cfg = cfg.get("rga_plus", {})
    threshold_strategy = eval_cfg.get("decision_threshold_strategy", eval_cfg.get("decision_threshold", "val_f1"))
    rga_plus_selection_metric = str(rga_plus_cfg.get("selection_metric", "roc_auc"))
    tta_steps = int(sar_adaptation_steps if sar_adaptation_steps is not None else eval_cfg.get("tta_steps", 25))
    clean_gate_threshold = float(rel_cfg.get("clean_gate_threshold", rel_cfg.get("gate_threshold", 0.66)))

    set_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        features,
        masks,
        labels,
        sample_ids,
        domain_order,
        _feature_columns,
        confidence_index,
        score_index,
        sample_splits,
        sample_categories,
    ) = _load_data(cfg)
    cal_bundle = None
    if (cfg.get("calibration_transfer") or {}).get("enabled"):
        features, cal_bundle = _apply_calibration_transfer(cfg, features, masks, domain_order, score_index or 0)

    train_idx, val_idx, test_idx = _split(labels, {**train_cfg, "seed": seed}, split_values=sample_splits)
    split_map = {
        "train": train_idx,
        "validation": val_idx,
        "val": val_idx,
        "test": test_idx,
    }
    eval_key = eval_split.strip().lower()
    if eval_key not in split_map:
        raise ValueError(f"Unknown eval_split={eval_split!r}; use train, validation, or test.")
    eval_idx = split_map[eval_key]
    if eval_key == "test" and not cfg.get("elara_deploy", {}).get("allow_test_eval", False):
        raise RuntimeError(
            "Refusing to score the locked test split. Set elara_deploy.allow_test_eval: true "
            "only for one-shot confirmatory runs after validation WIN vs SAR."
        )

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

    gate_calibration = None
    if gate_decision_cfg.get("enabled"):
        gate_calibration = build_gate_decision_calibration(
            model,
            estimator,
            features[val_idx],
            masks[val_idx],
            labels[val_idx],
            device,
            tau=float(gate_decision_cfg.get("tau", clean_gate_threshold)),
            margin_epsilon=float(gate_decision_cfg.get("margin_epsilon", 0.0)),
        )

    train_feat, train_mask, train_labels = features[train_idx], masks[train_idx], labels[train_idx]
    val_feat, val_mask, val_labels = features[val_idx], masks[val_idx], labels[val_idx]
    eval_feat, eval_mask, eval_labels = features[eval_idx], masks[eval_idx], labels[eval_idx]
    eval_categories = sample_categories[eval_idx] if sample_categories is not None else None

    static_val = _predict_static(model, val_feat, val_mask, device)
    static_eval = _predict_static(model, eval_feat, eval_mask, device)
    craf_gdr_val = _predict_craf(
        model,
        estimator,
        val_feat,
        val_mask,
        device,
        clean_gate_threshold=clean_gate_threshold,
        per_sample_gating=True,
        gate_decision_rule_cfg=gate_decision_cfg,
        gate_decision_calibration=gate_calibration,
    )
    craf_gdr_eval = _predict_craf(
        model,
        estimator,
        eval_feat,
        eval_mask,
        device,
        clean_gate_threshold=clean_gate_threshold,
        per_sample_gating=True,
        gate_decision_rule_cfg=gate_decision_cfg,
        gate_decision_calibration=gate_calibration,
    )

    sar = SARScoreAdapter(random_seed=int(seed), adaptation_steps=tta_steps).fit(
        train_feat, train_mask, train_labels
    )
    sar_val = sar.predict_proba(val_feat, val_mask)
    sar_eval = sar.predict_proba(eval_feat, eval_mask)

    rga_boosted = ReliabilityBoostedFusion(
        score_index=score_index,
        confidence_index=confidence_index,
        random_seed=int(seed),
        selection_metric=rga_plus_selection_metric,
    ).fit(
        train_feat,
        train_mask,
        train_labels,
        val_feat,
        val_mask,
        val_labels,
        reliability_estimator=estimator,
    )
    rga_val = rga_boosted.predict_proba(val_feat, val_mask)
    rga_eval = rga_boosted.predict_proba(eval_feat, eval_mask)

    deploy_artifacts = ElaraDeployArtifacts(
        model=model,
        estimator=estimator,
        sar=sar,
        device=device,
        gate_decision_calibration=gate_calibration,
        gate_decision_rule_cfg=gate_decision_cfg,
        clean_gate_threshold=clean_gate_threshold,
        rga_probs=rga_eval,
    )
    fallback_choice, _, _ = select_validation_fallback(val_labels, sar_val, rga_val)
    eval_fallback = rga_eval if (
        deploy_policy.routing_mode == "gdr_val_router_fallback"
        and fallback_choice == "rga_boosted_fusion"
    ) else None
    deploy_policy_eval = deploy_policy
    if deploy_policy.routing_mode == "gdr_val_router_fallback":
        deploy_policy_eval = DeployPolicySpec(
            policy_id=deploy_policy.policy_id,
            routing_mode=deploy_policy.routing_mode,
            fallback_method=fallback_choice,
            gate_decision_rule=deploy_policy.gate_decision_rule,
        )
    deploy_eval, deploy_stats = predict_elara_deploy(
        deploy_artifacts,
        eval_feat,
        eval_mask,
        policy=deploy_policy_eval,
        fallback_probs=eval_fallback,
    )
    val_fallback = rga_val if fallback_choice == "rga_boosted_fusion" else None
    deploy_val, _ = predict_elara_deploy(
        deploy_artifacts,
        val_feat,
        val_mask,
        policy=deploy_policy_eval,
        fallback_probs=val_fallback,
    )
    deploy_stats["val_fallback_choice"] = fallback_choice

    deploy_method_key = deploy_policy.policy_id.lower().replace("-", "_")
    method_probs = {
        "static_attention": static_eval,
        "craf_gdr": craf_gdr_eval,
        "rga_boosted_fusion": rga_eval,
        "sar_score_adapter": sar_eval,
        deploy_method_key: deploy_eval,
    }
    val_probs = {
        "static_attention": static_val,
        "craf_gdr": craf_gdr_val,
        "rga_boosted_fusion": rga_val,
        "sar_score_adapter": sar_val,
        deploy_method_key: deploy_val,
    }

    methods: dict[str, dict[str, Any]] = {}
    for name, eval_p in method_probs.items():
        methods[name] = _metrics_with_threshold(
            eval_labels,
            eval_p,
            val_labels=val_labels,
            val_probs=val_probs[name],
            strategy=threshold_strategy,
        )

    sar_auc = float(methods["sar_score_adapter"]["roc_auc"])
    deltas = {name: float(methods[name]["roc_auc"]) - sar_auc for name in methods}
    per_category = _per_category_auroc(eval_labels, method_probs, eval_categories)

    return {
        "seed": int(seed),
        "eval_split": eval_key,
        "n_eval": int(len(eval_idx)),
        "calibration_transfer": bool(cal_bundle is not None),
        "methods": methods,
        "delta_roc_auc_vs_sar": deltas,
        "deploy_stats": deploy_stats,
        "sar_adaptation_steps": tta_steps,
        "per_category": per_category,
        "deploy_method": deploy_method_key,
        "win_vs_sar": bool(deltas.get(deploy_method_key, -1.0) > 0.0),
    }


def aggregate_seed_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    method_names = sorted(rows[0]["methods"].keys())
    summary: dict[str, Any] = {"n_seeds": len(rows), "methods": {}, "delta_roc_auc_vs_sar": {}}
    for name in method_names:
        aucs = [float(r["methods"][name]["roc_auc"]) for r in rows]
        summary["methods"][name] = {
            "roc_auc_mean": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs)),
        }
        deltas = [float(r["delta_roc_auc_vs_sar"][name]) for r in rows]
        summary["delta_roc_auc_vs_sar"][name] = {
            "mean": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
        }
    deploy_key = rows[0].get("deploy_method") or "elara_deploy_v2"
    deploy_wins = sum(1 for r in rows if r.get("win_vs_sar"))
    summary["deploy_method"] = deploy_key
    summary["deploy_win_rate"] = float(deploy_wins / len(rows))
    summary["recommend_confirmatory_test"] = bool(
        summary["delta_roc_auc_vs_sar"].get(deploy_key, {}).get("mean", -1.0) > 0.0
    )
    return summary
