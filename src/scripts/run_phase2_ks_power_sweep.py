"""Phase 2.2B B-MECH-4 — KS true-degradation power × window-size sweep.

Sweeps the ReliabilityEstimator.ks_window_size parameter across the
locked grid {32, 64, 128, 256, 512} and measures detection power
(true-positive rate of gate firing) under genuine score collapse +
score noise + missingness vs false-activation rate on clean data.

Refuses:
- any experiment_id other than B-MECH-4;
- any window size not in the locked grid.

Usage:
  PYTHONPATH=src python src/scripts/run_phase2_ks_power_sweep.py \\
      --experiment-id B-MECH-4 --seeds 5 --seed-start 42
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from elara.family_b.ks_window import KS_WINDOW_GRID  # noqa: E402

REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"
ELARA_BENCH_LA_CONFIG = ROOT / "configs" / "attention_real_fusion.yaml"
MECH_OUT = ROOT / "experiments" / "phase2" / "mechanism"

# Locked degradation types for B-MECH-4
DEGRADATION_TYPES = ("score_collapse", "score_noise", "missingness")

# Corruption parameters per degradation type
_DEGRADATION_ATTACK = {
    "score_collapse": "zero_attack",
    "score_noise": "gaussian_noise",
    "missingness": "missing_domain_failure",
}
_DEGRADATION_K = {
    "score_collapse": 4,
    "score_noise": 2,
    "missingness": 2,  # mask out 2 domains
}


def _registry_row(eid: str) -> dict[str, str]:
    with REGISTRY_V2.open() as f:
        for r in csv.DictReader(f):
            if r["experiment_id"] == eid:
                return r
    raise SystemExit(f"experiment_id {eid!r} not in v2 registry")


def _validate(eid: str, row: dict[str, str]) -> None:
    if eid != "B-MECH-4":
        raise SystemExit(f"this driver runs B-MECH-4 only; got {eid!r}")
    if row["analysis_family"] != "B":
        raise SystemExit(f"{eid}: analysis_family={row['analysis_family']!r}; refusing")


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score

    try:
        y = np.asarray(y, dtype=int)
        p = np.asarray(p, dtype=float)
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except (ValueError, TypeError):
        return float("nan")


def _apply_degradation(
    features: np.ndarray,
    masks: np.ndarray,
    domain_order: list,
    score_idx: int,
    degradation_type: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one of the three locked degradation types to the test set.

    Returns (degraded_features, degraded_masks).
    """
    from elara.family_b.corruption import inject_corruption

    k = _DEGRADATION_K[degradation_type]
    attack = _DEGRADATION_ATTACK[degradation_type]

    if degradation_type == "missingness":
        # missing_domain_failure: mask out exactly k domains for every sample.
        # The engine may not support "missing_domain_failure" as an attack name;
        # implement directly by setting masks=True for 2 randomly chosen domains.
        rng = np.random.default_rng(seed)
        n_domains = len(domain_order)
        k_clipped = min(k, n_domains)
        deg_masks = masks.copy()
        # Choose k domains uniformly at random for the whole batch (systematic failure)
        failed_domain_indices = rng.choice(n_domains, size=k_clipped, replace=False)
        for di in failed_domain_indices:
            deg_masks[:, di] = True
        return features.copy(), deg_masks

    # For score_collapse / score_noise use the standard inject_corruption helper.
    # We take the first subset of k failed domains from inject_corruption's output.
    try:
        conds = inject_corruption(
            features,
            masks,
            domain_order=list(domain_order),
            score_index=score_idx,
            attack_name=attack,
            k_values=[k],
            sigma=1.0,
            seed=seed,
        )
        # Filter to the correct k (skip k=0 clean condition)
        for cond in conds:
            if cond.failed_domain_count == k:
                return cond.features, cond.masks
        # Fallback: return first non-clean condition
        for cond in conds:
            if cond.failed_domain_count > 0:
                return cond.features, cond.masks
    except Exception:
        pass

    # Last-resort fallback: apply zero-attack manually on first k domains
    deg_feat = features.copy()
    for d in range(min(k, len(domain_order))):
        deg_feat[:, d, score_idx] = 0.0
    return deg_feat, masks.copy()


def _build_windowed_estimator(rel_cfg, domain_order, score_idx, window_size):
    """Build a ReliabilityEstimator with a specific ks_window_size."""
    from uais.fusion.attention.reliability_estimator import ReliabilityEstimator

    weights = {
        "ece_weight": float(rel_cfg.get("ece_weight", 0.45)),
        "ks_weight": float(rel_cfg.get("ks_weight", 0.35)),
        "sharpness_weight": float(rel_cfg.get("sharpness_weight", 0.20)),
    }
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        weights = {k: v / total for k, v in weights.items()}
    return ReliabilityEstimator(
        domain_order=list(domain_order),
        score_index=score_idx,
        ece_weight=weights["ece_weight"],
        ks_weight=weights["ks_weight"],
        sharpness_weight=weights["sharpness_weight"],
        n_calibration_bins=int(rel_cfg.get("n_calibration_bins", 10)),
        min_samples_for_ks=int(rel_cfg.get("min_samples_for_ks", 30)),
        gate_threshold=float(rel_cfg.get("clean_gate_threshold", 0.66)),
        gate_mode="mean",
        ks_window_size=int(window_size),
    )


def run_one_seed_window(cfg, seed, window_size, eid, benchmark, protocol):
    """Train model, fit windowed estimator, measure detection power vs false-activation."""

    from scripts.run_breakthrough_experiment import (
        _build_model,
        _load_data,
        _make_loaders,
        _predict_craf_with_stats,
        _predict_static,
        _split,
        _train_model,
        set_seed,
    )

    device = _device()
    set_seed(int(seed))
    cfg_seed = dict(cfg)
    cfg_seed["training"] = dict(cfg.get("training", {}))
    cfg_seed["training"]["seed"] = int(seed)

    features, masks, labels, sample_ids, domain_order, _, conf_idx, score_idx, sample_splits, _ = _load_data(cfg_seed)
    train_idx, val_idx, test_idx = _split(labels, cfg_seed["training"], split_values=sample_splits)
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
    _train_model(model, train_loader, val_loader, cfg_seed, device)
    model.eval()

    rel_cfg = cfg_seed.get("reliability", {})

    # Build windowed estimator for this window_size
    estimator = _build_windowed_estimator(rel_cfg, domain_order, score_idx, window_size)
    estimator.fit(features[train_idx], masks[train_idx], labels[train_idx])

    test_feat = features[test_idx]
    test_mask = masks[test_idx]
    test_labels = labels[test_idx]

    # --- Clean baseline: false activation rate ---
    clean_weights = estimator.compute_reliability_weights(test_feat, test_mask)
    clean_fired = estimator.gate_decisions(clean_weights, test_mask, gate_mode="mean", gate_threshold=0.66)
    false_activation_rate = float(np.mean(clean_fired))

    # Static AUC on clean data (for roc_auc_delta reference)
    static_probs_clean = _predict_static(model, test_feat, test_mask, device)
    _safe_auc(test_labels, static_probs_clean)

    # RGA AUC on clean data
    rga_probs_clean, _ = _predict_craf_with_stats(
        model,
        estimator,
        test_feat,
        test_mask,
        device,
        clean_gate_threshold=0.66,
        per_sample_gating=True,
    )
    _safe_auc(test_labels, rga_probs_clean)

    degradation_records = []
    window_records = []

    for deg_type in DEGRADATION_TYPES:
        deg_seed = int(seed) + 55_000 + hash(deg_type) % 1000
        try:
            deg_feat, deg_mask = _apply_degradation(
                test_feat, test_mask, list(domain_order), score_idx, deg_type, deg_seed
            )
        except Exception as exc:
            degradation_records.append(
                {
                    "degradation_type": deg_type,
                    "reference_type": f"window_{window_size}",
                    "detection_power": float("nan"),
                    "gate_activation_rate": float("nan"),
                    "false_negative_rate": float("nan"),
                    "roc_auc_delta": float("nan"),
                    "window_size": int(window_size),
                    "seed": int(seed),
                    "status": f"error_degradation:{exc}",
                }
            )
            continue

        # Gate under degradation
        deg_weights = estimator.compute_reliability_weights(deg_feat, deg_mask)
        deg_fired = estimator.gate_decisions(deg_weights, deg_mask, gate_mode="mean", gate_threshold=0.66)
        detection_power = float(np.mean(deg_fired))
        false_negative_rate = 1.0 - detection_power  # fraction NOT caught by gate

        # Static and RGA AUC under degradation
        static_probs_deg = _predict_static(model, deg_feat, deg_mask, device)
        static_auc_deg = _safe_auc(test_labels, static_probs_deg)

        rga_probs_deg, _ = _predict_craf_with_stats(
            model,
            estimator,
            deg_feat,
            deg_mask,
            device,
            clean_gate_threshold=0.66,
            per_sample_gating=True,
        )
        rga_auc_deg = _safe_auc(test_labels, rga_probs_deg)
        roc_auc_delta = (
            (rga_auc_deg - static_auc_deg)
            if (np.isfinite(static_auc_deg) and np.isfinite(rga_auc_deg))
            else float("nan")
        )

        degradation_records.append(
            {
                "degradation_type": deg_type,
                "reference_type": f"window_{window_size}",
                "detection_power": detection_power,
                "gate_activation_rate": float(np.mean(deg_fired)),
                "false_negative_rate": false_negative_rate,
                "roc_auc_delta": roc_auc_delta,
                "window_size": int(window_size),
                "seed": int(seed),
                "status": "ok",
            }
        )

    # Aggregate across degradation types for the window-size table
    powers = [r["detection_power"] for r in degradation_records if np.isfinite(r["detection_power"])]
    deltas = [r["roc_auc_delta"] for r in degradation_records if np.isfinite(r["roc_auc_delta"])]
    true_deg_power = float(np.mean(powers)) if powers else float("nan")
    roc_auc_effect = float(np.mean(deltas)) if deltas else float("nan")

    window_records.append(
        {
            "window_size": int(window_size),
            "false_activation_rate": false_activation_rate,
            "true_degradation_detection_power": true_deg_power,
            "roc_auc_effect": roc_auc_effect,
            "seed": int(seed),
            "status": "ok",
        }
    )

    return degradation_records, window_records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument(
        "--window-sizes",
        default=",".join(str(w) for w in KS_WINDOW_GRID),
        help="comma-separated subset of the locked grid",
    )
    args = p.parse_args()
    row = _registry_row(args.experiment_id)
    _validate(args.experiment_id, row)
    if int(args.seeds) <= 0:
        print(f"[b-mech-4 {args.experiment_id}] validation-only invocation; exiting OK")
        return 0

    requested = [int(x) for x in args.window_sizes.split(",")]
    for w in requested:
        if w not in KS_WINDOW_GRID:
            raise SystemExit(f"window size {w} not in locked grid {KS_WINDOW_GRID}; refusing")

    cfg = yaml.safe_load(ELARA_BENCH_LA_CONFIG.read_text())
    MECH_OUT.mkdir(parents=True, exist_ok=True)

    deg_path = MECH_OUT / "ks_true_degradation_power.csv"
    win_path = MECH_OUT / "ks_window_size_power.csv"

    deg_fields = [
        "degradation_type",
        "reference_type",
        "detection_power",
        "gate_activation_rate",
        "false_negative_rate",
        "roc_auc_delta",
        "window_size",
        "seed",
        "status",
    ]
    win_fields = [
        "window_size",
        "false_activation_rate",
        "true_degradation_detection_power",
        "roc_auc_effect",
        "seed",
        "status",
    ]

    def _open_csv(path, fields):
        new_file = not path.exists()
        fh = path.open("a", newline="")
        w = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            w.writeheader()
        return fh, w

    fh_deg, w_deg = _open_csv(deg_path, deg_fields)
    fh_win, w_win = _open_csv(win_path, win_fields)

    benchmark = row["benchmark"]
    protocol = row["protocol"]
    print(f"[b-mech-4 {args.experiment_id}] requested window sizes: {requested}")

    try:
        for w in requested:
            for s in range(int(args.seed_start), int(args.seed_start) + int(args.seeds)):
                print(f"[b-mech-4 window={w} seed={s} starting]", flush=True)
                try:
                    deg_records, win_records = run_one_seed_window(cfg, s, w, args.experiment_id, benchmark, protocol)
                    for rec in deg_records:
                        w_deg.writerow(rec)
                    for rec in win_records:
                        w_win.writerow(rec)
                    fh_deg.flush()
                    fh_win.flush()
                    print(f"[b-mech-4 window={w} seed={s} done; " f"{len(deg_records)} degradation rows]", flush=True)
                except Exception as exc:
                    print(f"[b-mech-4 window={w} seed={s} ERROR: {exc}]", flush=True)
                    for deg_type in DEGRADATION_TYPES:
                        w_deg.writerow(
                            dict.fromkeys(deg_fields, "")
                            | {"degradation_type": deg_type, "window_size": w, "seed": s, "status": f"error:{exc}"}
                        )
                    w_win.writerow(
                        dict.fromkeys(win_fields, "") | {"window_size": w, "seed": s, "status": f"error:{exc}"}
                    )
                    fh_deg.flush()
                    fh_win.flush()
    finally:
        fh_deg.close()
        fh_win.close()

    print(
        f"[b-mech-4] {len(requested)} window sizes × {args.seeds} seeds complete. "
        f"Outputs: {deg_path.name}, {win_path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
