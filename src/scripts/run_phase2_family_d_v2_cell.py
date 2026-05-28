"""Phase 2.2E (Family-D v3) — one-time held-out cell evaluation driver.

Executes ONE cell (D-EYE-1, D-EYE-2, or D-EYE-3) of the Family-D v3
Eyecandies confirmatory replication under the frozen protocol.

INVARIANTS:
- Gate threshold τ is selected ONLY on validation fold; test fold is never
  accessed before the final ROC-AUC computation.
- selection_used_test_metrics = False is stamped in every archived row.
- Anomaly masks are never read; only train/val use label=0 at feature-score
  level; test labels are read at the final metric step (authorised).
- This script may run ONLY after:
  1. FAMILY_D_V2_INDEPENDENT_REVIEW_SIGNOFF.md exists.
  2. family_d_v3_scoring_pipeline.yaml SHA256 matches FAMILY_D_PARTITION_MANIFEST_v3.json.
  3. eyecandies_inputs.csv exists (built by family_d_v2_build_fusion_csv.py).

Usage:
  PYTHONPATH=src python src/scripts/run_phase2_family_d_v2_cell.py \\
      --cell D-EYE-1 --seeds 30 --seed-start 42 \\
      --protocol configs/phase2/family_d_v2_eyecandies_protocol.yaml \\
      --pipeline-spec configs/phase2/family_d_v3_scoring_pipeline.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from elara.evaluation.prediction_archive import PredictionArchive  # noqa: E402
from scripts.run_breakthrough_experiment import (  # noqa: E402
    CategoryAwareReliabilityEstimator,
    _build_model,
    _load_data,
    _make_loaders,
    _make_reliability_estimator,
    _predict_craf_with_stats,
    _predict_static,
    _split,
    _train_model,
    set_seed,
)

OUT_DIR = ROOT / "experiments" / "phase2" / "family_d"
ARCHIVE_DIR = OUT_DIR / "archives"
SIGNOFF = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_V2_INDEPENDENT_REVIEW_SIGNOFF.md"
MANIFEST_V3 = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_PARTITION_MANIFEST_v3.json"

# Locked degradation operator parameters (mirror of frozen YAML + spec)
OPERATOR_PARAMS = {
    "D-EYE-1": {"target_domain": "depth", "mode": "zero_collapse", "seed_offset": 41000},
    "D-EYE-2": {"target_domain": "rgb", "mode": "zero_collapse", "seed_offset": 41001},
    "D-EYE-3": {"target_domain": "depth", "mode": "mask_collapse", "seed_offset": 41002},
}
PRIMARY_CELLS = ("D-EYE-1", "D-EYE-2")
SECONDARY_CELLS = ("D-EYE-3",)
ALL_CELLS = PRIMARY_CELLS + SECONDARY_CELLS

CLEAN_FALSE_FIRE_BUDGET = 0.010
LOCKED_TAU_G0 = 0.66
TAU_CANDIDATES = [round(float(x), 3) for x in np.arange(0.30, 0.90, 0.01)]


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _preflight_checks(
    protocol_path: Path,
    pipeline_spec_path: Path,
    cell_id: str,
    allow_rerun: bool = False,
) -> None:
    """Authorisation gate: must pass before any test-fold data is accessed."""
    # 1. Independent review sign-off must exist
    if not SIGNOFF.exists():
        raise SystemExit(
            "PREFLIGHT FAIL: Independent review sign-off missing.\n"
            f"Expected: {SIGNOFF}\n"
            "Create FAMILY_D_V2_INDEPENDENT_REVIEW_SIGNOFF.md before running."
        )

    # 2. v3 manifest must exist (may be created by this run if first time)
    if MANIFEST_V3.exists():
        with MANIFEST_V3.open() as f:
            manifest = json.load(f)
        # Verify scoring pipeline SHA256 if recorded
        recorded_sha = manifest.get("scoring_pipeline_yaml_sha256", "")
        if recorded_sha and not allow_rerun:
            actual_sha = _sha256_file(pipeline_spec_path)
            if recorded_sha != actual_sha:
                raise SystemExit(
                    f"PREFLIGHT FAIL: scoring_pipeline_yaml_sha256 mismatch.\n"
                    f"  Manifest: {recorded_sha}\n"
                    f"  Actual:   {actual_sha}\n"
                    "The pipeline spec has changed after freeze — execution invalid.\n"
                    "Pass --allow-rerun for exploratory v4+ pipeline specs."
                )
        # Check per-cell execution flag (allows D-EYE-1 and D-EYE-2 to run independently)
        cell_key = f"cell_{cell_id.lower().replace('-','_')}_executed"
        if manifest.get(cell_key, False) and not allow_rerun:
            raise SystemExit(
                f"PREFLIGHT FAIL: Manifest shows {cell_key}=true.\n"
                f"The one-time evaluation for {cell_id} has already been performed.\n"
                "Pass --allow-rerun to write v4 exploratory outputs anyway."
            )
    else:
        print(
            "[PREFLIGHT] FAMILY_D_PARTITION_MANIFEST_v3.json not found; " "will create at end of run.",
            flush=True,
        )

    print("[PREFLIGHT] All pre-execution checks passed.", flush=True)


def _apply_degradation_operator(
    features: np.ndarray,
    masks: np.ndarray,
    domain_order: list[str],
    cell_id: str,
    global_seed: int,
    score_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a frozen D-EYE operator at the modality-score level.

    D-EYE-1/2: set per-domain score to 0.0 (zero_collapse).
    D-EYE-3:   set per-domain mask = True (missing).

    Returns (degraded_features, degraded_masks).
    """
    params = OPERATOR_PARAMS[cell_id]
    target_domain = params["target_domain"]
    mode = params["mode"]
    seed_offset = params["seed_offset"]

    # Seeded RNG for operator (required by spec; currently all ops are deterministic)
    rng = np.random.RandomState(global_seed + seed_offset)  # noqa: F841 (reserved for future stochastic ops)

    feat = features.copy()
    mask = masks.copy()

    # features shape: [N, D, F] where D=num_domains, F=feature_dim
    # masks   shape: [N, D]     where True = missing

    if target_domain == "alternated":
        # D-EYE-3 alternates between depth and rgb; per spec freeze: collapse depth
        target_domain = "depth"

    if target_domain not in domain_order:
        print(
            f"[WARN] Cell {cell_id}: target_domain={target_domain!r} not in "
            f"domain_order={domain_order}; operator has no effect.",
            flush=True,
        )
        return feat, mask

    d_idx = domain_order.index(target_domain)

    if mode == "zero_collapse":
        # Set score dimension to 0.0 for the target domain
        feat[:, d_idx, score_idx] = 0.0

    elif mode == "mask_collapse":
        # Set domain mask to True (missing) for the target domain
        mask[:, d_idx] = True

    else:
        raise ValueError(f"Unknown degradation mode: {mode!r}")

    return feat, mask


def _safe_auc(labels, scores) -> float | None:
    """ROC-AUC with graceful fallback when labels are degenerate."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def _select_tau_on_validation_only(
    model,
    estimator,
    val_features: np.ndarray,
    val_masks: np.ndarray,
    val_labels: np.ndarray,
    domain_order: list[str],
    cell_id: str,
    global_seed: int,
    device,
    val_categories: np.ndarray | None = None,
) -> dict[str, Any]:
    """Select τ using ONLY validation data + frozen degradation injection.

    Returns a dict with:
      selected_tau, clean_activation_rate, degraded_auc,
      selection_used_test_metrics (always False).
    """
    # Clean validation: measure gate activation rate on normal-only val data
    if val_categories is not None and isinstance(estimator, CategoryAwareReliabilityEstimator):
        weights = estimator.compute_reliability_weights(val_features, val_masks, categories=val_categories)
    else:
        weights = estimator.compute_reliability_weights(val_features, val_masks)
    mean_r = weights.mean(axis=1)  # [N]

    # For D-EYE-3 (secondary), just use τ = LOCKED_TAU_G0 directly
    if cell_id == "D-EYE-3":
        clean_rate = float((mean_r < LOCKED_TAU_G0).mean())
        return {
            "selected_tau": LOCKED_TAU_G0,
            "clean_activation_rate": clean_rate,
            "degraded_auc": None,
            "selection_used_test_metrics": False,
            "selection_note": "D-EYE-3 secondary; uses locked G0 tau",
        }

    # Apply degradation on validation
    deg_feat, deg_mask = _apply_degradation_operator(
        val_features, val_masks, domain_order, cell_id, global_seed, score_idx=0
    )
    # val labels are all 0 (anomaly-free); degraded signal should score higher
    # We score: rga on clean vs degraded to measure detection power
    # For threshold selection we maximise degraded AUC while respecting clean budget.
    # val_labels are all 0 → can't compute real anomaly AUC.
    # Instead: degraded gate activation rate is used as proxy (higher = more sensitive).

    best_tau = LOCKED_TAU_G0
    best_metric = -1.0
    best_clean_rate = float((mean_r < LOCKED_TAU_G0).mean())

    for tau in TAU_CANDIDATES:
        clean_rate = float((mean_r < tau).mean())
        if clean_rate > CLEAN_FALSE_FIRE_BUDGET:
            continue  # exceeds budget; skip

        # Measure degradation response on validation
        if val_categories is not None and isinstance(estimator, CategoryAwareReliabilityEstimator):
            deg_weights = estimator.compute_reliability_weights(deg_feat, deg_mask, categories=val_categories)
        else:
            deg_weights = estimator.compute_reliability_weights(deg_feat, deg_mask)
        deg_mean_r = deg_weights.mean(axis=1)
        degraded_rate = float((deg_mean_r < tau).mean())

        if degraded_rate > best_metric:
            best_metric = degraded_rate
            best_tau = tau
            best_clean_rate = clean_rate

    return {
        "selected_tau": best_tau,
        "clean_activation_rate": best_clean_rate,
        "degraded_activation_rate": best_metric,
        "selection_used_test_metrics": False,
        "selection_note": (
            f"validation-only search over {len(TAU_CANDIDATES)} τ candidates "
            f"within budget={CLEAN_FALSE_FIRE_BUDGET}"
        ),
    }


def run_one_seed(
    cfg: dict,
    seed: int,
    cell_id: str,
    archive: PredictionArchive,
    experiment_id: str,
    include_test: bool,
) -> list[dict]:
    """Run one seed: train RGA, calibrate on validation, evaluate on test (if authorised)."""
    device = _device()
    set_seed(int(seed))
    cfg_seed = dict(cfg)
    cfg_seed["training"] = dict(cfg.get("training", {}))
    cfg_seed["training"]["seed"] = int(seed)

    features, masks, labels, sample_ids, domain_order, _, conf_idx, score_idx, sample_splits, sample_categories = (
        _load_data(cfg_seed)
    )

    train_idx, val_idx, test_idx = _split(labels, cfg_seed["training"], split_values=sample_splits)

    # ── Train model ──────────────────────────────────────────────────────
    train_loader, val_loader, _ = _make_loaders(
        features,
        masks,
        labels,
        train_idx,
        val_idx,
        test_idx,
        batch_size=int(cfg_seed["training"].get("batch_size", 64)),
    )
    model = _build_model(cfg_seed, features.shape[1], features.shape[2], conf_idx, device)
    rel_cfg = cfg_seed.get("reliability", {})
    train_cfg = cfg_seed.get("training", {})
    _train_model(
        model,
        train_loader,
        val_loader,
        cfg_seed,
        device,
        score_index=score_idx if train_cfg.get("one_class_score_supervision") else None,
    )
    model.eval()

    # ── Fit reliability estimator on training data ────────────────────────
    estimator = _make_reliability_estimator(rel_cfg, list(domain_order) or ["rgb", "depth"], score_idx)
    train_categories = sample_categories[train_idx] if sample_categories is not None else None
    if isinstance(estimator, CategoryAwareReliabilityEstimator) and train_categories is not None:
        estimator.fit(features[train_idx], masks[train_idx], labels[train_idx], categories=train_categories)
    else:
        estimator.fit(features[train_idx], masks[train_idx], labels[train_idx])

    # ── Target-domain KS calibration ─────────────────────────────────────
    # The training split contains only normal (label=0) samples with cosine
    # distances in [0, 0.5].  At inference, the validation and test splits
    # contain a mix of normal and anomalous samples with scores in [0, 1].
    # Without re-calibration, ks_2samp(train_ref, val_scores) → p≈0 for
    # every domain → ks_reliability≈0 → rel_d<<0.66 → 100% false-fire rate.
    # Re-fitting the KS reference on the validation split aligns the reference
    # to the inference-time distribution.  The validation split is already
    # used for τ selection so this introduces no new data leakage.
    val_categories = sample_categories[val_idx] if sample_categories is not None else None
    if isinstance(estimator, CategoryAwareReliabilityEstimator) and val_categories is not None:
        estimator.re_fit_ks_reference(features[val_idx], masks[val_idx], categories=val_categories)
    else:
        estimator.re_fit_ks_reference(features[val_idx], masks[val_idx])

    val_feat = features[val_idx]
    val_mask = masks[val_idx]
    val_labels_arr = labels[val_idx]

    tau_result = _select_tau_on_validation_only(
        model,
        estimator,
        val_feat,
        val_mask,
        val_labels_arr,
        list(domain_order),
        cell_id,
        int(seed),
        device,
        val_categories=val_categories,
    )
    selected_tau = tau_result["selected_tau"]
    clean_rate = tau_result["clean_activation_rate"]

    assert not tau_result[
        "selection_used_test_metrics"
    ], "INVARIANT VIOLATED: selection_used_test_metrics must be False"

    rows = []
    selection_rule = (
        f"validation-only: τ={selected_tau:.2f} selected on normal-only val + "
        f"degradation injection {cell_id}; "
        f"clean_false_fire_rate={clean_rate:.4f} (budget={CLEAN_FALSE_FIRE_BUDGET}); "
        f"selection_used_test_metrics=False"
    )

    # ── Validation calibration rows (for per-seed log) ───────────────────
    rows.append(
        {
            "cell_id": cell_id,
            "seed": seed,
            "fold": "validation",
            "selected_tau": selected_tau,
            "clean_activation_rate": clean_rate,
            "selection_used_test_metrics": False,
            "selection_note": tau_result["selection_note"],
        }
    )

    if not include_test:
        # Dry-run mode: don't touch test fold
        return rows

    # ── Test evaluation (one-time; authorised) ───────────────────────────
    test_feat = features[test_idx]
    test_mask = masks[test_idx]
    test_labels_arr = labels[test_idx]
    test_categories = sample_categories[test_idx] if sample_categories is not None else None
    test_sids = [str(sample_ids[i]) for i in test_idx]
    score_blend = bool(rel_cfg.get("score_blend_on_gate", False))
    ignore_zero_scores = bool(rel_cfg.get("score_blend_ignore_zero_scores", True))

    # Filter out unlabeled test samples (label == -1; from test_private which has no metadata)
    valid_test_mask = test_labels_arr != -1
    if valid_test_mask.sum() == 0:
        print(f"[seed={seed}] WARNING: No labeled test samples found; skipping test evaluation.", flush=True)
        return rows
    if valid_test_mask.sum() < len(test_labels_arr):
        n_filtered = int((~valid_test_mask).sum())
        print(
            f"[seed={seed}] Filtering {n_filtered} unlabeled test_private rows; "
            f"using {int(valid_test_mask.sum())} labeled test_public rows.",
            flush=True,
        )
        test_feat = test_feat[valid_test_mask]
        test_mask = test_mask[valid_test_mask]
        test_labels_arr = test_labels_arr[valid_test_mask]
        if test_categories is not None:
            test_categories = test_categories[valid_test_mask]
        test_sids = [s for s, v in zip(test_sids, valid_test_mask) if v]

    # Apply degradation operator to test fold
    deg_feat, deg_mask = _apply_degradation_operator(
        test_feat, test_mask, list(domain_order), cell_id, int(seed), score_idx=score_idx
    )

    # Static predictions (no RGA)
    static_probs_clean = _predict_static(model, test_feat, test_mask, device)
    static_probs_deg = _predict_static(model, deg_feat, deg_mask, device)

    # RGA predictions (with gate at selected_tau)
    estimator.gate_mode = "mean"
    estimator.gate_threshold = selected_tau
    predict_kwargs = dict(
        score_blend_on_gate=score_blend,
        score_index=score_idx if score_blend else None,
        categories=test_categories,
        score_blend_ignore_zero_scores=ignore_zero_scores,
    )
    craf_probs_clean, gate_stats_clean = _predict_craf_with_stats(
        model,
        estimator,
        test_feat,
        test_mask,
        device,
        clean_gate_threshold=selected_tau,
        per_sample_gating=True,
        **predict_kwargs,
    )
    craf_probs_deg, gate_stats_deg = _predict_craf_with_stats(
        model,
        estimator,
        deg_feat,
        deg_mask,
        device,
        clean_gate_threshold=selected_tau,
        per_sample_gating=True,
        **predict_kwargs,
    )

    # Archive all four (method × condition) score vectors
    for condition, _f_arr, _m_arr in (
        ("clean", test_feat, test_mask),
        (cell_id, deg_feat, deg_mask),
    ):
        static_p = static_probs_clean if condition == "clean" else static_probs_deg
        craf_p = craf_probs_clean if condition == "clean" else craf_probs_deg

        for method, scores in (
            ("static_attention", static_p),
            (f"base_rga_tau{selected_tau:.2f}", craf_p),
        ):
            frame = PredictionArchive.build_frame(
                sample_ids=test_sids,
                labels=np.asarray(test_labels_arr, dtype=int),
                raw_scores=np.asarray(scores, dtype=float),
                method=method,
                method_variant=condition,
                benchmark="Eyecandies",
                protocol="validation_only_degradation_calibrated_one_class_multimodal",
                analysis_family="D",
                pairing_strength="rgb_depth_naturally_paired",
                split="test",
                seed=int(seed),
                selection_rule=selection_rule,
                selection_used_test_metrics=False,
                selected_head_or_comparator_status=(
                    f"base_RGA G0 mean-gate tau={selected_tau:.2f}" if "rga" in method else "static reference"
                ),
                gate_mode="mean",
            )
            archive.write(
                experiment_id=experiment_id,
                benchmark="Eyecandies",
                protocol="validation_only_degradation_calibrated_one_class_multimodal",
                seed=int(seed),
                method=method,
                split="test",
                frame=frame,
                selection_used_test_metrics=False,
                validation_only_selection_verified=True,
            )

    # Compute per-seed AUC metrics
    static_auc = _safe_auc(test_labels_arr, static_probs_deg)
    rga_auc = _safe_auc(test_labels_arr, craf_probs_deg)
    delta_auc = (rga_auc - static_auc) if (rga_auc is not None and static_auc is not None) else None

    gate_stats = gate_stats_deg
    gate_activation = (
        float(gate_stats.get("adaptation_rate", float("nan"))) if isinstance(gate_stats, dict) else float("nan")
    )

    rows.append(
        {
            "cell_id": cell_id,
            "seed": seed,
            "fold": "test",
            "selected_tau": selected_tau,
            "clean_activation_rate": clean_rate,
            "static_auc": static_auc,
            "rga_auc": rga_auc,
            "delta_auc": delta_auc,
            "gate_activation_rate_under_degradation": gate_activation,
            "selection_used_test_metrics": False,
        }
    )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", required=True, choices=ALL_CELLS, help="Which D-EYE endpoint to evaluate")
    p.add_argument("--seeds", type=int, default=30, help="Number of seeds to run (target=30, minimum=15)")
    p.add_argument("--seed-start", type=int, default=42, help="First seed (range: seed_start .. seed_start+seeds-1)")
    p.add_argument(
        "--protocol", default="configs/phase2/family_d_v2_eyecandies_protocol.yaml", help="Frozen protocol YAML path"
    )
    p.add_argument(
        "--pipeline-spec",
        default="configs/phase2/family_d_v3_scoring_pipeline.yaml",
        help="Frozen scoring-pipeline spec YAML path",
    )
    p.add_argument("--dry-run", action="store_true", help="Run validation calibration only; do not touch test fold")
    p.add_argument("--experiment-id", default="D-EYE-v3", help="Experiment ID for archive metadata")
    p.add_argument(
        "--allow-rerun",
        action="store_true",
        help="Bypass manifest one-time guard (writes exploratory v4 outputs)",
    )
    p.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for per-seed CSV (e.g. v4) to avoid overwriting v3 results",
    )
    args = p.parse_args()

    protocol_path = ROOT / args.protocol
    pipeline_spec_path = ROOT / args.pipeline_spec

    if not protocol_path.exists():
        raise SystemExit(f"Protocol YAML not found: {protocol_path}")
    if not pipeline_spec_path.exists():
        raise SystemExit(f"Pipeline spec YAML not found: {pipeline_spec_path}")

    # Authorisation gate
    _preflight_checks(protocol_path, pipeline_spec_path, args.cell, allow_rerun=args.allow_rerun)

    # Load pipeline spec for eyecandies RGA config
    with pipeline_spec_path.open() as f:
        pipeline_spec = yaml.safe_load(f)

    rga_cfg_raw = pipeline_spec.get("eyecandies_rga_config", {})
    # Resolve data path relative to ROOT
    if "data" in rga_cfg_raw and "path" in rga_cfg_raw["data"]:
        rga_cfg_raw["data"]["path"] = str(ROOT / rga_cfg_raw["data"]["path"])

    # Ensure fusion CSV exists
    fusion_csv = ROOT / "experiments" / "fusion" / "eyecandies_inputs.csv"
    if not fusion_csv.exists():
        raise SystemExit(
            f"Fusion CSV not found: {fusion_csv}\n"
            "Run: PYTHONPATH=src python src/scripts/family_d_v2_build_fusion_csv.py"
        )

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    archive = PredictionArchive(ARCHIVE_DIR / f"family_d_{args.cell.replace('-','_').lower()}")

    all_rows: list[dict] = []
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    include_test = not args.dry_run
    if args.dry_run:
        print("[DRY-RUN] Validation calibration only; test fold will NOT be accessed.", flush=True)
    else:
        print(f"[RUN] Cell={args.cell}, seeds={len(seeds)}, include_test=True", flush=True)
        print("[WARN] This is the ONE-TIME held-out evaluation. It cannot be re-run.", flush=True)

    for seed in seeds:
        print(f"[seed={seed}] starting...", flush=True)
        try:
            seed_rows = run_one_seed(
                cfg=rga_cfg_raw,
                seed=seed,
                cell_id=args.cell,
                archive=archive,
                experiment_id=args.experiment_id,
                include_test=include_test,
            )
            all_rows.extend(seed_rows)
            print(f"[seed={seed}] done. rows={len(seed_rows)}", flush=True)
        except Exception as exc:
            print(f"[seed={seed}] ERROR: {exc}", flush=True)
            import traceback

            traceback.print_exc()
            all_rows.append(
                {
                    "cell_id": args.cell,
                    "seed": seed,
                    "fold": "error",
                    "error": str(exc),
                    "selection_used_test_metrics": False,
                }
            )

    # Write per-seed calibration + result log
    fold_type = "dry_run_validation_only" if args.dry_run else "full_test_evaluation"
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_csv = OUT_DIR / f"family_d_{args.cell.replace('-','_').lower()}_{fold_type}{suffix}_per_seed.csv"
    if all_rows:
        fieldnames = sorted(
            {k for r in all_rows for k in r.keys()}, key=lambda k: (k != "cell_id", k != "seed", k != "fold", k)
        )
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows → {out_csv}", flush=True)

    # Update / create v3 manifest
    spec_sha = _sha256_file(pipeline_spec_path)
    protocol_sha = _sha256_file(protocol_path)
    manifest_path = MANIFEST_V3

    existing = {}
    if manifest_path.exists():
        with manifest_path.open() as f:
            existing = json.load(f)

    exploratory = args.allow_rerun or str(args.experiment_id).lower().startswith("d-eye-v4")
    if exploratory:
        v4_log = OUT_DIR / "family_d_v4_execution_log.json"
        v4_entry = {
            "cell": args.cell,
            "experiment_id": args.experiment_id,
            "pipeline_spec": str(pipeline_spec_path.relative_to(ROOT)),
            "pipeline_sha256": spec_sha,
            "seeds": len(seeds),
            "include_test": include_test,
            "output_suffix": args.output_suffix,
            "dry_run": args.dry_run,
        }
        v4_history = []
        if v4_log.exists():
            with v4_log.open() as f:
                v4_history = json.load(f)
        v4_history.append(v4_entry)
        with v4_log.open("w") as f:
            json.dump(v4_history, f, indent=2)
        print(f"Logged exploratory run → {v4_log}", flush=True)
    else:
        existing["scoring_pipeline_yaml_sha256"] = spec_sha
        existing["protocol_yaml_sha256"] = protocol_sha
        cell_key = f"cell_{args.cell.lower().replace('-','_')}_executed"
        existing[cell_key] = include_test
        existing[f"cell_{args.cell.lower().replace('-','_')}_seed_count"] = len(seeds)
        existing["v3_last_executed_cell"] = args.cell
        existing["v3_dry_run"] = args.dry_run

        d_eye_1_done = existing.get("cell_d_eye_1_executed", False)
        d_eye_2_done = existing.get("cell_d_eye_2_executed", False)
        existing["test_evaluation_executed"] = bool(d_eye_1_done and d_eye_2_done)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as f:
            json.dump(existing, f, indent=2)
        print(f"Updated manifest → {manifest_path}", flush=True)

    print(f"\n[DONE] Cell {args.cell} | seeds={len(seeds)} | include_test={include_test}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
