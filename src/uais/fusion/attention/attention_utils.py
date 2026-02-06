"""Utility helpers for attention fusion data handling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import hashlib
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn import metrics


def load_fusion_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def infer_feature_columns(
    df: pd.DataFrame,
    score_column: str = "score",
    confidence_column: str | None = "confidence",
    embedding_prefix: str = "embedding_",
    feature_columns: Sequence[str] | None = None,
    include_confidence: bool = True,
) -> List[str]:
    if feature_columns:
        return list(feature_columns)
    columns = []
    if score_column in df.columns:
        columns.append(score_column)
    if include_confidence and confidence_column and confidence_column in df.columns:
        columns.append(confidence_column)
    embedding_cols = sorted([c for c in df.columns if c.startswith(embedding_prefix)])
    columns.extend(embedding_cols)
    if not columns:
        raise ValueError("No feature columns found. Provide feature_columns or embedding_prefix.")
    return columns


def validate_fusion_schema(
    df: pd.DataFrame,
    id_column: str = "sample_id",
    domain_column: str = "domain",
    score_column: str = "score",
    label_column: str | None = "label",
    confidence_column: str | None = "confidence",
    embedding_prefix: str = "embedding_",
    timestamp_column: str | None = None,
) -> dict:
    required = {id_column, domain_column, score_column}
    if timestamp_column:
        required.add(timestamp_column)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    def _check_numeric(col_name: str) -> int:
        if col_name not in df.columns:
            return 0
        numeric = pd.to_numeric(df[col_name], errors="coerce")
        bad = numeric.isna() & df[col_name].notna()
        if bad.any():
            raise ValueError(f"Non-numeric values found in column '{col_name}'.")
        return int(bad.sum())

    _check_numeric(score_column)
    if confidence_column:
        _check_numeric(confidence_column)

    embedding_cols = [c for c in df.columns if c.startswith(embedding_prefix)]
    for col in embedding_cols:
        _check_numeric(col)

    score_vals = df[score_column].to_numpy()
    score_oob = int(((score_vals < 0) | (score_vals > 1)).sum())
    if score_oob:
        warnings.warn(f"{score_oob} '{score_column}' values outside [0, 1].", RuntimeWarning)

    confidence_oob = 0
    if confidence_column and confidence_column in df.columns:
        conf_vals = df[confidence_column].to_numpy()
        confidence_oob = int(((conf_vals < 0) | (conf_vals > 1)).sum())
        if confidence_oob:
            warnings.warn(f"{confidence_oob} '{confidence_column}' values outside [0, 1].", RuntimeWarning)

    stats = {
        "rows": int(len(df)),
        "embedding_dim": int(len(embedding_cols)),
        "score_out_of_range": score_oob,
        "confidence_out_of_range": confidence_oob,
    }
    if label_column and label_column in df.columns:
        stats["label_nulls"] = int(df[label_column].isna().sum())
    return stats


def hash_file(path: str | Path, chunk_size: int = 8192) -> str:
    p = Path(path)
    hasher = hashlib.md5()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare_fusion_dataframe(
    df: pd.DataFrame,
    id_column: str,
    domain_column: str,
    feature_columns: Sequence[str],
    label_column: str | None = "label",
    timestamp_column: str | None = None,
) -> Tuple[pd.DataFrame, dict]:
    required = {id_column, domain_column, *feature_columns}
    if timestamp_column:
        required.add(timestamp_column)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df_clean = df.copy()
    before_rows = len(df_clean)
    drop_subset = [id_column, domain_column]
    if timestamp_column:
        drop_subset.append(timestamp_column)
    df_clean = df_clean.dropna(subset=drop_subset)
    dropped_rows = before_rows - len(df_clean)

    if label_column and label_column in df_clean.columns:
        label_counts = df_clean.groupby(id_column)[label_column].nunique(dropna=True)
        conflicts = label_counts[label_counts > 1]
        if not conflicts.empty:
            raise ValueError(f"Conflicting labels for {len(conflicts)} sample_ids.")

    dup_cols = [id_column, domain_column]
    if timestamp_column:
        dup_cols.append(timestamp_column)
    dupes = df_clean.duplicated(subset=dup_cols).any()
    if dupes:
        agg = {col: "mean" for col in feature_columns}
        if label_column and label_column in df_clean.columns:
            agg[label_column] = "max"
        df_clean = df_clean.groupby(dup_cols, as_index=False).agg(agg)

    sample_count = df_clean[id_column].nunique()
    coverage = (
        df_clean.groupby(domain_column)[id_column].nunique().div(sample_count).sort_index().to_dict()
        if sample_count > 0
        else {}
    )
    stats = {
        "rows_before": before_rows,
        "rows_after": len(df_clean),
        "dropped_rows": dropped_rows,
        "sample_count": int(sample_count),
        "domain_coverage": {str(k): float(v) for k, v in coverage.items()},
    }
    return df_clean, stats


def build_fusion_tensors(
    df: pd.DataFrame,
    id_column: str = "sample_id",
    domain_column: str = "domain",
    label_column: str = "label",
    score_column: str = "score",
    confidence_column: str | None = "confidence",
    embedding_prefix: str = "embedding_",
    timestamp_column: str | None = None,
    domain_order: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    validate_schema: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, List[str], List[str]]:
    if id_column not in df.columns:
        raise ValueError(f"Missing id column: {id_column}")
    if domain_column not in df.columns:
        raise ValueError(f"Missing domain column: {domain_column}")
    if validate_schema:
        validate_fusion_schema(
            df,
            id_column=id_column,
            domain_column=domain_column,
            score_column=score_column,
            label_column=label_column,
            confidence_column=confidence_column,
            embedding_prefix=embedding_prefix,
            timestamp_column=timestamp_column,
        )
    if domain_order is None:
        domain_order = sorted(df[domain_column].dropna().unique().tolist())
    domain_to_idx = {domain: idx for idx, domain in enumerate(domain_order)}

    feature_columns = list(feature_columns or [])
    if not feature_columns:
        raise ValueError("feature_columns must be provided or inferred before building tensors.")
    df_work = df
    id_key = id_column
    if timestamp_column and timestamp_column in df.columns:
        df_work = df.copy()
        df_work["_fusion_key"] = (
            df_work[id_column].astype(str).fillna("") + "::" + df_work[timestamp_column].astype(str).fillna("")
        )
        id_key = "_fusion_key"
    df_clean, _ = prepare_fusion_dataframe(
        df_work,
        id_column=id_key,
        domain_column=domain_column,
        feature_columns=feature_columns,
        label_column=label_column,
        timestamp_column=timestamp_column,
    )
    sample_ids = df_clean[id_key].dropna().unique().tolist()
    sample_ids = sorted(sample_ids)

    num_samples = len(sample_ids)
    num_domains = len(domain_order)
    feature_dim = len(feature_columns)
    features = np.zeros((num_samples, num_domains, feature_dim), dtype=np.float32)
    key_padding_mask = np.ones((num_samples, num_domains), dtype=bool)
    labels = None
    if label_column in df.columns:
        labels = np.zeros(num_samples, dtype=np.float32)

    for idx, sample_id in enumerate(sample_ids):
        rows = df_clean[df_clean[id_key] == sample_id]
        if labels is not None:
            label_vals = rows[label_column].dropna().unique()
            if len(label_vals) > 1:
                raise ValueError(f"Conflicting labels for sample_id={sample_id}")
            if len(label_vals) == 1:
                labels[idx] = float(label_vals[0])
        for _, row in rows.iterrows():
            domain = row[domain_column]
            if domain not in domain_to_idx:
                continue
            domain_idx = domain_to_idx[domain]
            features[idx, domain_idx, :] = row[feature_columns].to_numpy(dtype=np.float32)
            key_padding_mask[idx, domain_idx] = False

    return features, key_padding_mask, labels, sample_ids, list(domain_order)


class FusionDataset(Dataset):
    """Simple dataset wrapper for attention fusion."""

    def __init__(self, features: np.ndarray, masks: np.ndarray, labels: np.ndarray | None) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.masks = torch.as_tensor(masks, dtype=torch.bool)
        self.labels = torch.as_tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.features[idx], self.masks[idx]
        return self.features[idx], self.masks[idx], self.labels[idx]


def apply_domain_dropout(mask: torch.Tensor, drop_prob: float, generator: torch.Generator | None = None) -> torch.Tensor:
    if drop_prob <= 0:
        return mask
    available = ~mask
    drop = (torch.rand(mask.shape, device=mask.device, generator=generator) < drop_prob) & available
    new_mask = mask | drop
    all_masked = new_mask.all(dim=1)
    if all_masked.any():
        rows = torch.where(all_masked)[0]
        for row_idx in rows.tolist():
            available_idx = torch.where(available[row_idx])[0]
            if len(available_idx) == 0:
                continue
            keep_idx = available_idx[torch.randint(0, len(available_idx), (1,), device=mask.device)].item()
            new_mask[row_idx, keep_idx] = False
    return new_mask


def compute_domain_reliability(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    domain_order: Sequence[str],
    score_index: int | None,
) -> dict:
    if score_index is None:
        return {domain: float("nan") for domain in domain_order}
    reliabilities = {}
    for domain_idx, domain in enumerate(domain_order):
        present = ~masks[:, domain_idx]
        if present.sum() < 2:
            reliabilities[domain] = float("nan")
            continue
        scores = features[present, domain_idx, score_index]
        y_true = labels[present]
        try:
            reliabilities[domain] = float(metrics.roc_auc_score(y_true, scores))
        except ValueError:
            reliabilities[domain] = float("nan")
    return reliabilities


__all__ = [
    "load_fusion_dataframe",
    "infer_feature_columns",
    "hash_file",
    "validate_fusion_schema",
    "prepare_fusion_dataframe",
    "build_fusion_tensors",
    "FusionDataset",
    "apply_domain_dropout",
    "compute_domain_reliability",
]
