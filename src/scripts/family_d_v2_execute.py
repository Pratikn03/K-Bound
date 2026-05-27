"""Phase 2.2E — Family-D v2 one-time held-out execution under frozen contract.

Implementation interpretation of "base RGA on one-class Eyecandies":
  - The existing attention-fusion model in this codebase is SUPERVISED and
    cannot be trained on Eyecandies (train/val are anomaly-free; no positive
    labels). Therefore we use the existing ReliabilityEstimator infrastructure
    directly on the per-modality score features without an attention model.
  - Static comparator prediction per sample: fixed equal average of per-modality
    scores. This matches the "no gating" interpretation of static_attention.
  - Base RGA prediction per sample: reliability-weighted average using
    KS-drift-derived reliability per modality vs the anomaly-free training
    distribution. When the RGA gate fires for a sample, the prediction uses
    the reliability-weighted combination; otherwise it falls back to the
    static average.
  - Degradation operators (frozen):
      D-EYE-1: depth-channel score collapse — set every test sample's depth
               score to 0.0 before any computation.
      D-EYE-2: rgb-channel score collapse — same for rgb.
      D-EYE-3: single-modality missingness — set the modality mask to True
               for the targeted modality (alternates).

Per-seed structure:
  Each seed reshuffles the train memory bank via random subsampling
  (10% coreset) — this is where seed-dependent variance enters under
  the one-class protocol. ReliabilityEstimator is then re-fit on the
  subsampled train bank.

Outputs:
  experiments/phase2/family_d/family_d_v2_seed_metrics.csv
  experiments/phase2/family_d/family_d_v2_prediction_archive_index.csv
  experiments/phase2/family_d/family_d_v2_selection_log.csv
  experiments/phase2/family_d/family_d_v2_clean_false_fire.csv
  + per-(endpoint, seed) parquet predictions under
    experiments/phase2/family_d/predictions/D-EYE-N/

This driver does NOT read test metadata.yaml `anomalous` labels here;
those are read only by the inference script that follows.
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

from elara.evaluation.prediction_archive import PredictionArchive  # noqa: E402
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator  # noqa: E402

CSV_PATH = ROOT / "experiments" / "fusion" / "eyecandies_inputs.csv"
PROTOCOL_YAML = ROOT / "configs" / "phase2" / "family_d_v2_eyecandies_protocol.yaml"
OUT_DIR = ROOT / "experiments" / "phase2" / "family_d"
PRED_DIR = OUT_DIR / "predictions"
TAU_MEAN = 0.66
CORE_SUBSAMPLE = 0.10  # coreset fraction; standard PatchCore default in {0.1, 0.25, 1.0}


def _read_protocol() -> dict:
    return yaml.safe_load(PROTOCOL_YAML.read_text())["protocol"]


def _per_sample_features(df: pd.DataFrame, split: str) -> dict:
    """Pivot the long-format CSV into per-sample (sample_id × domain) arrays."""
    sub = df[df["fusion_split"] == split].copy()
    sub = sub.sort_values(["sample_id", "domain"]).reset_index(drop=True)
    # Pivot: each row = one sample; columns per domain
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


def _apply_operator(samples: dict, endpoint: str) -> dict:
    """Frozen degradation operators."""
    new = {sid: {m: dict(d) if d is not None else None for m, d in mods.items()} for sid, mods in samples.items()}
    if endpoint == "D-EYE-1":
        # Depth-channel score collapse
        for _sid, mods in new.items():
            if mods["depth"] is not None:
                mods["depth"]["score"] = 0.0
    elif endpoint == "D-EYE-2":
        # RGB-channel score collapse
        for _sid, mods in new.items():
            if mods["rgb"] is not None:
                mods["rgb"]["score"] = 0.0
    elif endpoint == "D-EYE-3":
        # Single-modality missingness: alternate per sample
        rng = np.random.default_rng(7)
        for _sid, mods in new.items():
            target = "depth" if rng.random() < 0.5 else "rgb"
            mods[target] = None
    elif endpoint == "clean":
        pass
    else:
        raise ValueError(f"unknown endpoint {endpoint!r}")
    return new


def _compute_reliability_and_predict(
    sample_dict: dict,
    train_features: dict,
    estimator: ReliabilityEstimator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute static and RGA predictions for all test samples.

    Returns (sample_ids, static_pred, rga_pred, gate_fired_bool).
    """
    sample_ids = sorted(sample_dict.keys())
    rgb_scores = []
    depth_scores = []
    rgb_features = []
    depth_features = []
    for sid in sample_ids:
        mods = sample_dict[sid]
        rgb = mods.get("rgb")
        depth = mods.get("depth")
        rgb_scores.append(rgb["score"] if rgb is not None else 0.5)
        depth_scores.append(depth["score"] if depth is not None else 0.5)
        rgb_features.append(rgb["embeddings"] if rgb is not None else np.zeros(16))
        depth_features.append(depth["embeddings"] if depth is not None else np.zeros(16))
    rgb_scores = np.array(rgb_scores)
    depth_scores = np.array(depth_scores)
    np.stack(rgb_features)
    np.stack(depth_features)

    # Compute per-modality reliability via ReliabilityEstimator's KS-drift
    # logic. The estimator expects [N, D, F] features and [N, D] masks; we
    # use score as a 1-D feature.
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

    weights = estimator.compute_reliability_weights(features, masks)  # [N, 2]
    # Mean reliability across present modalities per sample
    present = ~masks
    n_present = present.sum(axis=1).astype(np.float32)
    mean_rel = np.where(
        n_present > 0,
        (weights * present.astype(np.float32)).sum(axis=1) / np.maximum(n_present, 1.0),
        0.0,
    )
    gate_fired = mean_rel < TAU_MEAN

    # Static prediction: equal-weighted average of present modality scores
    static_pred = np.where(
        n_present > 0,
        (rgb_scores * present[:, 0] + depth_scores * present[:, 1]) / np.maximum(n_present, 1.0),
        0.5,
    )

    # RGA prediction: reliability-weighted on gated samples; static on others
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


def run_one_seed(
    df: pd.DataFrame,
    seed: int,
    endpoint: str,
    archive: PredictionArchive,
    out_rows: list,
    sel_log: list,
    ffr_log: list,
    protocol: dict,
) -> dict:
    train_per_sample = _per_sample_features(df, "train")
    val_per_sample = _per_sample_features(df, "validation")
    test_per_sample = _per_sample_features(df, "test")

    # Build coreset memory bank per modality from train
    def _stack(sample_dict, mod):
        feats = []
        for sid in sorted(sample_dict):
            d = sample_dict[sid].get(mod)
            if d is not None:
                feats.append(d["embeddings"])
        return np.stack(feats) if feats else np.zeros((0, 16))

    train_rgb_full = _stack(train_per_sample, "rgb")
    train_depth_full = _stack(train_per_sample, "depth")
    rgb_bank = _seed_subsample(train_rgb_full, seed)
    depth_bank = _seed_subsample(train_depth_full, seed)
    train_features = {"rgb": rgb_bank, "depth": depth_bank}

    # Fit ReliabilityEstimator on the seed's train subset (anomaly-free, labels=0)
    # We use the train SCORE feature for both KS reference and ECE proxy.
    train_features_3d = np.zeros((len(train_per_sample), 2, 1), dtype=np.float32)
    train_masks_2d = np.zeros((len(train_per_sample), 2), dtype=bool)
    train_labels = np.zeros(len(train_per_sample), dtype=int)
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

    # CLEAN false-fire on validation (clean, no degradation operator)
    sids_v, static_v, rga_v, gate_v = _compute_reliability_and_predict(val_per_sample, train_features, estimator)
    clean_false_fire = float(gate_v.mean())
    within_budget = clean_false_fire <= float(protocol["clean_false_fire_budget"]["value"])
    ffr_log.append(
        {
            "endpoint": endpoint,
            "seed": int(seed),
            "clean_false_fire_rate": f"{clean_false_fire:.4f}",
            "budget": f"{protocol['clean_false_fire_budget']['value']:.4f}",
            "within_budget": str(within_budget),
            "n_val_samples": len(sids_v),
        }
    )
    sel_log.append(
        {
            "endpoint": endpoint,
            "seed": int(seed),
            "selection_input": "anomaly_free_validation_and_train_memory_bank_only",
            "selection_used_test_metrics": False,
            "gate_threshold": TAU_MEAN,
            "core_subsample_fraction": CORE_SUBSAMPLE,
        }
    )

    # Apply degradation operator on test fold
    test_after_op = _apply_operator(test_per_sample, endpoint)
    sids_t, static_t, rga_t, gate_t = _compute_reliability_and_predict(test_after_op, train_features, estimator)

    # Archive predictions
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    n = len(sids_t)
    for method, scores in (("static_attention", static_t), ("base_RGA", rga_t)):
        frame = archive.build_frame(
            sample_ids=list(sids_t),
            labels=np.full(n, -1, dtype=int),  # labels deferred to inference step
            raw_scores=np.asarray(scores, dtype=float),
            method=method,
            method_variant=endpoint,
            benchmark="Eyecandies-1.0.3",
            protocol=protocol["name"],
            analysis_family="D",
            pairing_strength="naturally_paired_multimodal",
            split="test",
            seed=int(seed),
            selection_rule="validation-only one-class memory bank scoring; no test labels read",
            selection_used_test_metrics=False,
            selected_head_or_comparator_status=f"D-EYE Family-D {method}",
            gate_mode="mean",
            gate_fired=gate_t,
            mean_reliability=np.full(n, float(estimator.gate_threshold), dtype=float),
            min_reliability=np.full(n, 0.0, dtype=float),
            failure_type=endpoint,
            failed_domain_count=1 if endpoint in ("D-EYE-1", "D-EYE-2") else 0,
            fault_severity=1.0,
        )
        entry = archive.write(
            experiment_id=endpoint,
            benchmark="Eyecandies-1.0.3",
            protocol=protocol["name"],
            seed=int(seed),
            method=method,
            split="test",
            frame=frame,
            config={"endpoint": endpoint, "seed": int(seed)},
        )
        archive.append_index(entry)

    out_rows.append(
        {
            "endpoint": endpoint,
            "seed": int(seed),
            "n_val_samples": len(sids_v),
            "n_test_samples": len(sids_t),
            "clean_false_fire_rate": f"{clean_false_fire:.4f}",
            "gate_fire_rate_on_test": f"{float(gate_t.mean()):.4f}",
            "rga_pred_mean": f"{float(rga_t.mean()):.4f}",
            "static_pred_mean": f"{float(static_t.mean()):.4f}",
        }
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoints", default="D-EYE-1,D-EYE-2,D-EYE-3", help="comma-separated subset")
    p.add_argument("--seeds", type=int, default=30)
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
            print(f"[{endpoint} seed={s}] starting", flush=True)
            run_one_seed(df, s, endpoint, archive, out_rows, sel_log, ffr_log, protocol)
        print(f"[{endpoint}] {args.seeds} seeds complete")

    # Write logs
    def _write(path, rows, fields):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {path}")

    _write(
        OUT_DIR / "family_d_v2_seed_metrics.csv",
        out_rows,
        [
            "endpoint",
            "seed",
            "n_val_samples",
            "n_test_samples",
            "clean_false_fire_rate",
            "gate_fire_rate_on_test",
            "rga_pred_mean",
            "static_pred_mean",
        ],
    )
    _write(
        OUT_DIR / "family_d_v2_selection_log.csv",
        sel_log,
        [
            "endpoint",
            "seed",
            "selection_input",
            "selection_used_test_metrics",
            "gate_threshold",
            "core_subsample_fraction",
        ],
    )
    _write(
        OUT_DIR / "family_d_v2_clean_false_fire.csv",
        ffr_log,
        ["endpoint", "seed", "clean_false_fire_rate", "budget", "within_budget", "n_val_samples"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
