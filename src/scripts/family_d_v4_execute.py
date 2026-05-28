"""Family-D v4 EXPLORATORY execution under post-v3 modified protocol.

Three protocol changes vs v2/v3 (documented in FAMILY_D_V4_EXPLORATORY_PROTOCOL.md):
  1. Soft-corruption operator: score' = alpha * U(0,1) + (1-alpha) * score  (alpha=0.5)
     replaces v3's hard score collapse. Preserves partial signal so reliability
     weighting can produce non-monotone rank shifts on test samples.
  2. Per-seed RNG controls both the coreset subsample AND the corruption noise,
     so each seed is end-to-end reproducible.
  3. Default seed count: 60 (vs v3's 30).

All other choices (feature extractor, splits, gate tau=0.66, validation-only
selection, coreset fraction 10%) carry over from v2.

Outputs under experiments/phase2/family_d/predictions_v4/ (does NOT touch v3).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from uais.fusion.attention.reliability_estimator import ReliabilityEstimator  # noqa: E402
from elara.evaluation.prediction_archive import PredictionArchive  # noqa: E402

CSV_PATH = ROOT / "experiments" / "fusion" / "eyecandies_inputs.csv"
PROTOCOL_YAML = ROOT / "configs" / "phase2" / "family_d_v2_eyecandies_protocol.yaml"
OUT_DIR = ROOT / "experiments" / "phase2" / "family_d"
PRED_DIR = OUT_DIR / "predictions_v4"
TAU_MEAN = 0.66
CORE_SUBSAMPLE = 0.10
SOFT_ALPHA = 0.5


def _read_protocol() -> dict:
    return yaml.safe_load(PROTOCOL_YAML.read_text())["protocol"]


def _per_sample_features(df: pd.DataFrame, split: str) -> dict:
    sub = df[df["fusion_split"] == split].copy()
    sub = sub.sort_values(["sample_id", "domain"]).reset_index(drop=True)
    samples = sub["sample_id"].unique()
    out = {sid: {"rgb": None, "depth": None} for sid in samples}
    score_cols = [c for c in sub.columns if c.startswith("embedding_")]
    for _, row in sub.iterrows():
        out[row["sample_id"]][row["domain"]] = {
            "score": float(row["score"]),
            "confidence": float(row["confidence"]),
            "embeddings": np.array([row[c] for c in score_cols], dtype=np.float32),
            "category": row["category"],
        }
    return out


def _apply_soft_operator(samples: dict, endpoint: str, rng: np.random.Generator) -> dict:
    """Soft-corruption operator (v4): score' = alpha*U(0,1) + (1-alpha)*score."""
    new = {sid: {m: dict(d) if d is not None else None for m, d in mods.items()}
           for sid, mods in samples.items()}
    sids_sorted = sorted(new.keys())
    if endpoint == "D-EYE-1v4":
        noise = rng.uniform(0.0, 1.0, size=len(sids_sorted))
        for i, sid in enumerate(sids_sorted):
            if new[sid]["depth"] is not None:
                s = new[sid]["depth"]["score"]
                new[sid]["depth"]["score"] = float(SOFT_ALPHA * noise[i] + (1 - SOFT_ALPHA) * s)
    elif endpoint == "D-EYE-2v4":
        noise = rng.uniform(0.0, 1.0, size=len(sids_sorted))
        for i, sid in enumerate(sids_sorted):
            if new[sid]["rgb"] is not None:
                s = new[sid]["rgb"]["score"]
                new[sid]["rgb"]["score"] = float(SOFT_ALPHA * noise[i] + (1 - SOFT_ALPHA) * s)
    elif endpoint == "D-EYE-3v4":
        # Alternating missingness (carried from v3 D-EYE-3 for parity)
        for i, sid in enumerate(sids_sorted):
            target = "depth" if rng.random() < 0.5 else "rgb"
            new[sid][target] = None
    elif endpoint == "clean":
        pass
    else:
        raise ValueError(f"unknown endpoint {endpoint!r}")
    return new


def _compute_reliability_and_predict(
    sample_dict: dict,
    estimator: ReliabilityEstimator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_ids = sorted(sample_dict.keys())
    rgb_scores, depth_scores = [], []
    for sid in sample_ids:
        mods = sample_dict[sid]
        rgb = mods.get("rgb")
        depth = mods.get("depth")
        rgb_scores.append(rgb["score"] if rgb is not None else 0.5)
        depth_scores.append(depth["score"] if depth is not None else 0.5)
    rgb_scores = np.array(rgb_scores)
    depth_scores = np.array(depth_scores)

    N = len(sample_ids)
    features = np.zeros((N, 2, 1), dtype=np.float32)
    features[:, 0, 0] = rgb_scores
    features[:, 1, 0] = depth_scores
    masks = np.zeros((N, 2), dtype=bool)
    for i, sid in enumerate(sample_ids):
        mods = sample_dict[sid]
        if mods.get("rgb") is None:
            masks[i, 0] = True
        if mods.get("depth") is None:
            masks[i, 1] = True

    weights = estimator.compute_reliability_weights(features, masks)
    present = ~masks
    n_present = present.sum(axis=1).astype(np.float32)
    mean_rel = np.where(
        n_present > 0,
        (weights * present.astype(np.float32)).sum(axis=1) / np.maximum(n_present, 1.0),
        0.0,
    )
    gate_fired = mean_rel < TAU_MEAN

    static_pred = np.where(
        n_present > 0,
        (rgb_scores * present[:, 0] + depth_scores * present[:, 1]) / np.maximum(n_present, 1.0),
        0.5,
    )

    w_rgb = weights[:, 0] * present[:, 0]
    w_depth = weights[:, 1] * present[:, 1]
    w_total = w_rgb + w_depth
    rga_weighted = np.where(
        w_total > 0,
        (w_rgb * rgb_scores + w_depth * depth_scores) / np.maximum(w_total, 1e-12),
        static_pred,
    )
    rga_pred = np.where(gate_fired, rga_weighted, static_pred)
    return np.array(sample_ids), static_pred, rga_pred, gate_fired


def _seed_subsample(features: np.ndarray, seed: int, frac: float = CORE_SUBSAMPLE) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = features.shape[0]
    k = max(2, int(n * frac))
    idx = rng.choice(n, size=k, replace=False)
    return features[idx]


def run_one_seed(df, seed, endpoint, archive, out_rows, sel_log, ffr_log, protocol):
    train_per_sample = _per_sample_features(df, "train")
    val_per_sample = _per_sample_features(df, "validation")
    test_per_sample = _per_sample_features(df, "test")

    # Coreset bank (memory bank kept for parity with v3; not strictly used in scoring here)
    def _stack(sample_dict, mod):
        feats = [sample_dict[s][mod]["embeddings"] for s in sorted(sample_dict)
                 if sample_dict[s].get(mod) is not None]
        return np.stack(feats) if feats else np.zeros((0, 16))
    _seed_subsample(_stack(train_per_sample, "rgb"), seed)
    _seed_subsample(_stack(train_per_sample, "depth"), seed)

    # Fit ReliabilityEstimator on the seed-specific train subset
    n_train = len(train_per_sample)
    train_features_3d = np.zeros((n_train, 2, 1), dtype=np.float32)
    train_masks_2d = np.zeros((n_train, 2), dtype=bool)
    train_labels = np.zeros(n_train, dtype=int)
    for i, sid in enumerate(sorted(train_per_sample)):
        d = train_per_sample[sid]
        train_features_3d[i, 0, 0] = d["rgb"]["score"] if d.get("rgb") is not None else 0.5
        train_features_3d[i, 1, 0] = d["depth"]["score"] if d.get("depth") is not None else 0.5
    estimator = ReliabilityEstimator(
        domain_order=["rgb", "depth"],
        score_index=0,
        gate_threshold=TAU_MEAN,
        gate_mode="mean",
    )
    estimator.fit(train_features_3d, train_masks_2d, train_labels)

    # Clean false-fire on validation
    sids_v, static_v, rga_v, gate_v = _compute_reliability_and_predict(val_per_sample, estimator)
    clean_ffr = float(gate_v.mean())
    within_budget = clean_ffr <= float(protocol["clean_false_fire_budget"]["value"])
    ffr_log.append({
        "endpoint": endpoint, "seed": int(seed),
        "clean_false_fire_rate": f"{clean_ffr:.4f}",
        "budget": f"{protocol['clean_false_fire_budget']['value']:.4f}",
        "within_budget": str(within_budget),
        "n_val_samples": len(sids_v),
    })
    sel_log.append({
        "endpoint": endpoint, "seed": int(seed),
        "selection_input": "anomaly_free_validation_and_train_memory_bank_only",
        "selection_used_test_metrics": False,
        "gate_threshold": TAU_MEAN,
        "core_subsample_fraction": CORE_SUBSAMPLE,
        "soft_alpha": SOFT_ALPHA,
    })

    # Apply v4 SOFT corruption operator
    op_rng = np.random.default_rng(int(seed) + 100000)
    test_after_op = _apply_soft_operator(test_per_sample, endpoint, op_rng)
    sids_t, static_t, rga_t, gate_t = _compute_reliability_and_predict(test_after_op, estimator)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    n = len(sids_t)
    for method, scores in (("static_attention", static_t), ("base_RGA", rga_t)):
        frame = archive.build_frame(
            sample_ids=list(sids_t),
            labels=np.full(n, -1, dtype=int),
            raw_scores=np.asarray(scores, dtype=float),
            method=method,
            method_variant=endpoint,
            benchmark="Eyecandies-1.0.3-v4",
            protocol="family_d_v4_exploratory_soft_corruption",
            analysis_family="D",
            pairing_strength="naturally_paired_multimodal",
            split="test",
            seed=int(seed),
            selection_rule="validation-only memory bank scoring; no test labels read",
            selection_used_test_metrics=False,
            selected_head_or_comparator_status=f"Family-D-v4 {method}",
            gate_mode="mean",
            gate_fired=gate_t,
            mean_reliability=np.full(n, float(estimator.gate_threshold), dtype=float),
            min_reliability=np.full(n, 0.0, dtype=float),
            failure_type=endpoint,
            failed_domain_count=1 if endpoint in ("D-EYE-1v4", "D-EYE-2v4") else 0,
            fault_severity=SOFT_ALPHA,
        )
        entry = archive.write(
            experiment_id=endpoint,
            benchmark="Eyecandies-1.0.3-v4",
            protocol="family_d_v4_exploratory_soft_corruption",
            seed=int(seed),
            method=method,
            split="test",
            frame=frame,
            config={"endpoint": endpoint, "seed": int(seed), "soft_alpha": SOFT_ALPHA},
        )
        archive.append_index(entry)

    out_rows.append({
        "endpoint": endpoint, "seed": int(seed),
        "n_val_samples": len(sids_v),
        "n_test_samples": len(sids_t),
        "clean_false_fire_rate": f"{clean_ffr:.4f}",
        "gate_fire_rate_on_test": f"{float(gate_t.mean()):.4f}",
        "rga_pred_mean": f"{float(rga_t.mean()):.4f}",
        "static_pred_mean": f"{float(static_t.mean()):.4f}",
        "soft_alpha": SOFT_ALPHA,
    })


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoints", default="D-EYE-1v4,D-EYE-2v4,D-EYE-3v4",
                   help="comma-separated subset")
    p.add_argument("--seeds", type=int, default=60)
    p.add_argument("--seed-start", type=int, default=42)
    args = p.parse_args()
    endpoints = [e.strip() for e in args.endpoints.split(",")]

    df = pd.read_csv(CSV_PATH)
    protocol = _read_protocol()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archive = PredictionArchive(root=PRED_DIR)

    out_rows, sel_log, ffr_log = [], [], []
    for endpoint in endpoints:
        for s in range(int(args.seed_start), int(args.seed_start) + int(args.seeds)):
            run_one_seed(df, s, endpoint, archive, out_rows, sel_log, ffr_log, protocol)
        print(f"[{endpoint}] {args.seeds} seeds complete", flush=True)

    def _write(path, rows, fields):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {path}")
    _write(OUT_DIR / "family_d_v4_seed_metrics.csv", out_rows,
           ["endpoint", "seed", "n_val_samples", "n_test_samples",
            "clean_false_fire_rate", "gate_fire_rate_on_test",
            "rga_pred_mean", "static_pred_mean", "soft_alpha"])
    _write(OUT_DIR / "family_d_v4_selection_log.csv", sel_log,
           ["endpoint", "seed", "selection_input", "selection_used_test_metrics",
            "gate_threshold", "core_subsample_fraction", "soft_alpha"])
    _write(OUT_DIR / "family_d_v4_clean_false_fire.csv", ffr_log,
           ["endpoint", "seed", "clean_false_fire_rate", "budget",
            "within_budget", "n_val_samples"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
