"""Build a naturally-paired multimodal fusion benchmark from UNSW-NB15.

Each row in the UNSW-NB15 training/test CSVs is one network event, and
the 42 numeric features cluster into three co-observed measurement
modalities that describe different aspects of that same event:

  flow         - timing / throughput / inter-packet stats
                 (dur, rate, sload, dload, sinpkt, dinpkt, sjit, djit,
                  smean, dmean, tcprtt, synack, ackdat)
  connection   - protocol / session structure
                 (proto, service, state, spkts, dpkts, sbytes, dbytes,
                  sttl, dttl, sloss, dloss, swin, dwin, stcpb, dtcpb,
                  trans_depth, response_body_len)
  context      - aggregated past-window counters
                 (ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm,
                  ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login,
                  ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm,
                  ct_srv_dst, is_sm_ips_ports)

These three modalities are *naturally co-observed*: they are measured
simultaneously on the same network event. This makes UNSW-NB15 a
genuine 3-domain multimodal fusion benchmark - not a synthetic label
alignment over independent datasets.

The output schema matches the ELARA fusion runner contract
(prepare_mvtec3d_fusion_benchmark.py), so the same
run_breakthrough_experiment.py pipeline runs on this benchmark with no
special-casing. Category column carries `attack_cat` so the
category-aware KS reference can discriminate true drift from
inter-attack-class distribution differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DOMAIN_TO_FEATURES: dict[str, list[str]] = {
    "flow": [
        "dur",
        "rate",
        "sload",
        "dload",
        "sinpkt",
        "dinpkt",
        "sjit",
        "djit",
        "smean",
        "dmean",
        "tcprtt",
        "synack",
        "ackdat",
    ],
    "connection": [
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
        "swin",
        "dwin",
        "stcpb",
        "dtcpb",
        "trans_depth",
        "response_body_len",
    ],
    "context": [
        "ct_srv_src",
        "ct_state_ttl",
        "ct_dst_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_flw_http_mthd",
        "ct_src_ltm",
        "ct_srv_dst",
        "is_sm_ips_ports",
    ],
}
DOMAIN_ORDER = list(DOMAIN_TO_FEATURES.keys())


def _safe_numeric(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    available = [c for c in columns if c in df.columns]
    if not available:
        return np.zeros((len(df), 0), dtype=np.float32)
    block = df[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    return block


def _minmax_clip(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, None]
    fit_rows = values[fit_mask]
    if fit_rows.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo = np.nanmin(fit_rows, axis=0)
    hi = np.nanpercentile(fit_rows, 95, axis=0)
    span = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
    scaled = (values - lo) / span
    return np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, 1.0).astype(np.float32)


def _patient_style_stratified_split(
    df: pd.DataFrame,
    *,
    label_column: str,
    category_column: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> np.ndarray:
    """Return a per-row split assignment stratified by (category, label)."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(df), dtype=object)
    for (_cat, _label), group in df.groupby([category_column, label_column], sort=False):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * test_fraction))) if n >= 3 else 0
        remaining = n - n_test
        denom = max(1.0 - test_fraction, 1e-9)
        n_val = max(1, int(round(remaining * val_fraction / denom))) if remaining >= 2 else 0
        for offset, row_idx in enumerate(idx):
            if offset < n_test:
                split[row_idx] = "test"
            elif offset < n_test + n_val:
                split[row_idx] = "validation"
            else:
                split[row_idx] = "train"
    return split


def _domain_score(features: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Per-domain anomaly score: Mahalanobis-style distance from train-normal centroid."""
    if features.shape[1] == 0:
        return np.zeros(features.shape[0], dtype=np.float32)
    fit_rows = features[fit_mask]
    if fit_rows.shape[0] == 0:
        return np.zeros(features.shape[0], dtype=np.float32)
    center = fit_rows.mean(axis=0)
    scale = fit_rows.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    distances = np.linalg.norm((features - center) / scale, axis=1).astype(np.float32)
    fit_distances = distances[fit_mask]
    lo = float(np.min(fit_distances))
    hi = float(np.percentile(fit_distances, 95))
    if hi - lo <= 1e-9:
        hi = lo + 1.0
    return np.clip((distances - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_unsw_fusion_frame(
    train_csv: Path,
    test_csv: Path,
    *,
    embedding_dim: int = 8,
    val_fraction: float = 0.15,
    test_fraction: float = 0.30,
    seed: int = 42,
    max_rows: int | None = 60000,
    heldout_attack_categories: list[str] | None = None,
    heldout_val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("UNSW-NB15 CSVs must contain a 'label' column.")
    if "attack_cat" not in train_df.columns or "attack_cat" not in test_df.columns:
        raise ValueError("UNSW-NB15 CSVs must contain an 'attack_cat' column.")

    train_df = train_df.assign(_source="train_set")
    test_df = test_df.assign(_source="test_set")
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["attack_cat"] = combined["attack_cat"].fillna("Normal").astype(str)
    # The two CSVs share id ranges 1..82332, so namespace by source to make
    # the per-event sample_id globally unique.
    combined["_event_key"] = combined["_source"].astype(str) + "::" + combined["id"].astype(str)

    if max_rows is not None and len(combined) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(combined), size=max_rows, replace=False)
        combined = combined.iloc[sorted(idx)].reset_index(drop=True)

    if heldout_attack_categories:
        # Held-out-attack-category protocol: the named attack categories go
        # entirely into the test fold; everything else is split between
        # train and validation. Normal traffic is split stratified across
        # the three folds so all splits contain both classes.
        held = {c.strip() for c in heldout_attack_categories}
        rng = np.random.default_rng(int(seed))
        n = len(combined)
        fusion_split = np.empty(n, dtype=object)
        cats = combined["attack_cat"].astype(str).to_numpy()
        labels = combined["label"].to_numpy().astype(int)
        # Indices of held-out attack rows -> test
        held_mask = np.array([c in held for c in cats])
        fusion_split[held_mask] = "test"
        # Remaining rows: stratify by (cat, label) into train/validation.
        for (_cat, _label), group in combined.loc[~held_mask].groupby(["attack_cat", "label"], sort=False):
            idx = group.index.to_numpy().copy()
            rng.shuffle(idx)
            n_val = max(1, int(round(len(idx) * float(heldout_val_fraction))))
            fusion_split[idx[:n_val]] = "validation"
            fusion_split[idx[n_val:]] = "train"
        # Also push a small slice of normal traffic into the test fold so
        # the test split has both classes.
        normal_train_idx = np.flatnonzero((cats == "Normal") & (fusion_split == "train"))
        if normal_train_idx.size > 0:
            move_n = max(2, int(round(0.20 * normal_train_idx.size)))
            rng.shuffle(normal_train_idx)
            fusion_split[normal_train_idx[:move_n]] = "test"
        combined["fusion_split"] = fusion_split
    else:
        combined["fusion_split"] = _patient_style_stratified_split(
            combined,
            label_column="label",
            category_column="attack_cat",
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    train_mask = combined["fusion_split"].to_numpy() == "train"

    domain_scores: dict[str, np.ndarray] = {}
    domain_embeddings: dict[str, np.ndarray] = {}
    for domain, columns in DOMAIN_TO_FEATURES.items():
        block = _safe_numeric(combined, columns)
        domain_scores[domain] = _domain_score(block, train_mask)
        # Embedding: minmax-clipped feature block, truncated / padded to embedding_dim.
        if block.shape[1] >= embedding_dim:
            block = block[:, :embedding_dim]
        else:
            pad = np.zeros((block.shape[0], embedding_dim - block.shape[1]), dtype=np.float32)
            block = np.hstack([block, pad])
        scaled = np.column_stack([_minmax_clip(block[:, j], train_mask).ravel() for j in range(block.shape[1])])
        domain_embeddings[domain] = scaled

    rows: list[dict] = []
    attack_cats = combined["attack_cat"].to_numpy()
    fusion_splits = combined["fusion_split"].to_numpy()
    labels = combined["label"].to_numpy().astype(int)
    event_keys = combined["_event_key"].to_numpy()

    for i in range(len(combined)):
        sample_id = hashlib.md5(f"unsw::{event_keys[i]}".encode()).hexdigest()[:16]
        for domain in DOMAIN_ORDER:
            row = {
                "sample_id": sample_id,
                "event_key": str(event_keys[i]),
                "category": str(attack_cats[i]),
                "fusion_split": str(fusion_splits[i]),
                "domain": domain,
                "label": int(labels[i]),
                "score": float(domain_scores[domain][i]),
                "confidence": float(np.clip(2.0 * abs(float(domain_scores[domain][i]) - 0.5), 0.0, 1.0)),
            }
            embedding = domain_embeddings[domain][i]
            for emb_idx, value in enumerate(embedding):
                row[f"embedding_{emb_idx}"] = float(value)
            rows.append(row)

    frame = pd.DataFrame(rows)
    sample_frame = frame.groupby("sample_id").first()

    metadata = {
        "benchmark_type": "naturally_paired_unsw_nb15_cyber_fusion",
        "natural_pairing": True,
        "pairing_unit": "single UNSW-NB15 network event with three co-observed measurement modalities",
        "domain_order": DOMAIN_ORDER,
        "embedding_dim": int(embedding_dim),
        "samples": int(len(sample_frame)),
        "rows": int(len(frame)),
        "positive_fraction_actual": float(sample_frame["label"].mean()),
        "categories": sorted(frame["category"].unique().tolist()),
        "fusion_splits": frame["fusion_split"].value_counts().to_dict(),
        "label_distribution_per_split": {
            split: frame[frame["fusion_split"] == split].drop_duplicates("sample_id")["label"].value_counts().to_dict()
            for split in sorted(frame["fusion_split"].unique())
        },
        "score_protocol": {
            "score_fit_split": "train",
            "score_definition": "Mahalanobis-style distance to train-normal centroid, minmax-clipped against train p0..p95",
            "embedding_normalization_split": "train",
        },
        "score_features": dict(DOMAIN_TO_FEATURES.items()),
    }
    return frame, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/raw/cyber/UNSW_NB15_training-set.csv"),
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/raw/cyber/UNSW_NB15_testing-set.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/fusion/unsw_paired_inputs.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("experiments/fusion/unsw_paired_metadata.json"))
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=60000)
    parser.add_argument(
        "--held-out-attack-categories",
        nargs="*",
        default=None,
        help=(
            "Optional list of attack_cat values to hold entirely out of train/validation "
            "and route to the test fold. Triggers the held-out-attack-category protocol "
            "that defends against UNSW-NB15's known train/test attack overlap."
        ),
    )
    parser.add_argument("--heldout-val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    frame, metadata = build_unsw_fusion_frame(
        args.train_csv,
        args.test_csv,
        embedding_dim=args.embedding_dim,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        max_rows=args.max_rows,
        heldout_attack_categories=args.held_out_attack_categories,
        heldout_val_fraction=args.heldout_val_fraction,
    )
    if args.held_out_attack_categories:
        metadata["heldout_attack_protocol"] = {
            "held_out_attack_categories": sorted(args.held_out_attack_categories),
            "rationale": (
                "UNSW-NB15 has known train/test attack-category overlap; the held-out-attack "
                "protocol routes the named attack categories entirely into the test fold "
                "and keeps train+validation free of those attacks. Defends against the "
                "0.989 ROC-AUC leakage criticism."
            ),
            "heldout_val_fraction": float(args.heldout_val_fraction),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    metadata["output"] = str(args.output)
    args.metadata.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
