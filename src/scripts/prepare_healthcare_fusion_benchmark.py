"""Build a naturally-paired multimodal fusion benchmark from GridPulse vitals.

Each (patient_id, timestamp) row in the GridPulse features parquet carries
co-observed vital signs across four physiological channels:

  cardiac      (hr_bpm, hr_bpm_lag*, hr_bpm_roll*_mean, hr_bpm_roll*_std)
  oxygenation  (spo2_pct, spo2_pct_lag*, spo2_pct_roll*_mean, spo2_pct_roll*_std)
  respiratory  (respiratory_rate, respiratory_rate_lag*, respiratory_rate_roll*)
  hemodynamic  (shock_index, shock_index_lag*, shock_index_roll*)

These four channels are *naturally co-observed* on the same patient at the
same timestamp, which makes this a genuine multimodal-paired benchmark - not
a synthetic label alignment. The label (`is_critical`) is grounded in the
clinical source data (BIDMC, MIMIC-III).

The output schema matches the ELARA fusion runner contract used by MVTec
and ELARA-Bench-LA, so the same `run_breakthrough_experiment.py` pipeline
re-runs on this benchmark with no special-casing. The `category` column
carries the `source_dataset` so the category-aware KS reference can
discriminate true drift from inter-dataset distribution differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DOMAIN_TO_BASE_FEATURE: dict[str, str] = {
    "cardiac": "hr_bpm",
    "oxygenation": "spo2_pct",
    "respiratory": "respiratory_rate",
    "hemodynamic": "shock_index",
}
DOMAIN_ORDER = list(DOMAIN_TO_BASE_FEATURE.keys())


def _domain_feature_columns(df: pd.DataFrame, base: str) -> list[str]:
    candidates = [base]
    for col in df.columns:
        if col.startswith(f"{base}_lag") or col.startswith(f"{base}_roll"):
            candidates.append(col)
    return [c for c in candidates if c in df.columns]


def _minmax_clip(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    fit_rows = values[fit_mask]
    if fit_rows.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo = float(np.nanmin(fit_rows))
    hi = float(np.nanpercentile(fit_rows, 95))
    if hi - lo <= 1e-9:
        hi = lo + 1.0
    scaled = (values - lo) / (hi - lo)
    return np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, 1.0).astype(np.float32)


def _patient_stratified_assignment(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> dict[str, str]:
    """Return {patient_id: split} with each stratum (source_dataset + label) split independently."""
    rng = np.random.default_rng(seed)
    patient_table = (
        df.groupby("patient_id")
        .agg(source_dataset=("source_dataset", "first"), label=("is_critical", "max"))
        .reset_index()
    )
    assignment: dict[str, str] = {}
    for (_source, _label), stratum in patient_table.groupby(["source_dataset", "label"]):
        ids = stratum["patient_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(round(n * test_fraction))) if n >= 3 else 0
        remaining = n - n_test
        n_val = max(1, int(round(remaining * val_fraction / max(1.0 - test_fraction, 1e-9)))) if remaining >= 2 else 0
        test_ids = ids[:n_test]
        val_ids = ids[n_test : n_test + n_val]
        train_ids = ids[n_test + n_val :]
        for pid in train_ids:
            assignment[pid] = "train"
        for pid in val_ids:
            assignment[pid] = "validation"
        for pid in test_ids:
            assignment[pid] = "test"
    return assignment


def build_healthcare_fusion_frame(
    features_path: Path,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    embedding_dim: int = 4,
    seed: int = 42,
    max_rows_per_patient: int | None = 600,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(features_path)
    required = {"patient_id", "timestamp", "source_dataset", "is_critical"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GridPulse features parquet missing columns: {sorted(missing)}")

    # Throttle per-patient row counts so 81 patients * many timestamps stays tractable.
    if max_rows_per_patient is not None:
        keep_indices: list[int] = []
        for _patient, group in df.groupby("patient_id", sort=False):
            stride = max(1, int(np.ceil(len(group) / max_rows_per_patient)))
            keep_indices.extend(group.index[::stride].tolist())
        df = df.loc[keep_indices].reset_index(drop=True)

    patient_split = _patient_stratified_assignment(
        df, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
    )
    df = df.assign(fusion_split=df["patient_id"].map(patient_split).fillna("train"))
    train_mask = (df["fusion_split"].to_numpy() == "train")

    # Build per-domain score and embedding columns.
    domain_scores: dict[str, np.ndarray] = {}
    domain_embeddings: dict[str, np.ndarray] = {}
    for domain, base in DOMAIN_TO_BASE_FEATURE.items():
        cols = _domain_feature_columns(df, base)
        if base not in df.columns:
            raise ValueError(f"GridPulse features parquet missing base column '{base}' for domain '{domain}'.")
        # Score: deviation magnitude of the base feature from the train-fold mean.
        base_values = df[base].to_numpy(dtype=np.float32)
        train_values = base_values[train_mask & ~np.isnan(base_values)]
        center = float(np.nanmean(train_values)) if train_values.size else 0.0
        scale = float(np.nanstd(train_values)) if train_values.size else 1.0
        scale = scale if scale > 1e-6 else 1.0
        deviations = np.abs(base_values - center) / scale
        domain_scores[domain] = _minmax_clip(deviations, train_mask)

        # Embedding: minmax-clipped lag/rolling statistics, padded/truncated to embedding_dim.
        feature_matrix = df[cols].to_numpy(dtype=np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        if feature_matrix.shape[1] >= embedding_dim:
            feature_matrix = feature_matrix[:, :embedding_dim]
        else:
            pad = np.zeros((feature_matrix.shape[0], embedding_dim - feature_matrix.shape[1]), dtype=np.float32)
            feature_matrix = np.hstack([feature_matrix, pad])
        scaled = np.column_stack(
            [
                _minmax_clip(feature_matrix[:, j], train_mask)
                for j in range(feature_matrix.shape[1])
            ]
        )
        domain_embeddings[domain] = scaled

    rows: list[dict] = []
    timestamps = df["timestamp"].astype(str).to_numpy()
    patients = df["patient_id"].astype(str).to_numpy()
    categories = df["source_dataset"].astype(str).to_numpy()
    fusion_splits = df["fusion_split"].astype(str).to_numpy()
    labels = df["is_critical"].astype(int).to_numpy()

    for idx in range(len(df)):
        sample_id = hashlib.md5(f"{patients[idx]}::{timestamps[idx]}".encode()).hexdigest()[:16]
        for domain in DOMAIN_ORDER:
            row = {
                "sample_id": sample_id,
                "patient_key": patients[idx],
                "timestamp": timestamps[idx],
                "category": categories[idx],
                "fusion_split": fusion_splits[idx],
                "domain": domain,
                "label": int(labels[idx]),
                "score": float(domain_scores[domain][idx]),
                "confidence": float(np.clip(2.0 * abs(float(domain_scores[domain][idx]) - 0.5), 0.0, 1.0)),
            }
            embedding = domain_embeddings[domain][idx]
            for emb_idx, value in enumerate(embedding):
                row[f"embedding_{emb_idx}"] = float(value)
            rows.append(row)

    frame = pd.DataFrame(rows)
    sample_frame = frame.groupby("sample_id").first()

    metadata = {
        "benchmark_type": "naturally_paired_gridpulse_clinical_fusion",
        "natural_pairing": True,
        "pairing_unit": "(patient_id, timestamp) co-observed across four physiological channels",
        "domain_order": DOMAIN_ORDER,
        "embedding_dim": int(embedding_dim),
        "samples": int(len(sample_frame)),
        "rows": int(len(frame)),
        "positive_fraction_actual": float(sample_frame["label"].mean()),
        "categories": sorted(frame["category"].unique().tolist()),
        "fusion_splits": frame["fusion_split"].value_counts().to_dict(),
        "patients_per_split": (
            frame.drop_duplicates("patient_key")
            .groupby("fusion_split")
            .size()
            .to_dict()
        ),
        "label_distribution_per_split": {
            split: frame[frame["fusion_split"] == split]
            .drop_duplicates("sample_id")["label"]
            .value_counts()
            .to_dict()
            for split in sorted(frame["fusion_split"].unique())
        },
        "score_protocol": {
            "score_fit_split": "train",
            "score_definition": "|x - mean_train(x)| / std_train(x), min-max clipped against train p0..p95",
            "embedding_normalization_split": "train",
        },
        "score_features": {domain: base for domain, base in DOMAIN_TO_BASE_FEATURE.items()},
    }
    return frame, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the naturally-paired GridPulse clinical fusion benchmark")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/raw/healthcare/gridpulse/processed/features.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/fusion/healthcare_paired_inputs.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("experiments/fusion/healthcare_paired_metadata.json"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--embedding-dim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-patient", type=int, default=600)
    args = parser.parse_args()

    frame, metadata = build_healthcare_fusion_frame(
        args.features_path,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        max_rows_per_patient=args.max_rows_per_patient,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    metadata["output"] = str(args.output)
    args.metadata.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
