"""Helper to build fusion dataset from score files and train meta-model.

This is a thin wrapper around `train_fusion_meta_model` in train_fusion_model.py
so notebooks/scripts can easily load score CSVs and fit the stacker.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from uais.fusion.train_fusion_model import train_fusion_meta_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_fusion_dataset(score_paths: dict[str, Path] | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load per-domain score CSVs and align lengths.

    Expects each CSV to have at least a `score` column; `label` is optional.
    Uses the shortest length across domains to align scores.
    """
    if score_paths is None:
        score_paths = {
            "fraud": PROJECT_ROOT / "experiments" / "fraud" / "scores.csv",
            "cyber": PROJECT_ROOT / "experiments" / "cyber" / "scores.csv",
            "behavior": PROJECT_ROOT / "experiments" / "behavior" / "scores.csv",
            "vision": PROJECT_ROOT / "experiments" / "vision" / "scores.csv",
        }

    scores = {}
    labels = None
    sample_id_sets = []
    score_frames: dict[str, pd.DataFrame] = {}
    for domain, path in score_paths.items():
        # Allow relative paths to be resolved from project root for notebooks/scripts.
        if not path.is_absolute():
            candidate = PROJECT_ROOT / path
            path = candidate if candidate.exists() else path

        if not path.exists():
            print(f"[warn] Score file missing for {domain}: {path}")
            continue
        df = pd.read_csv(path)
        if "score" not in df.columns:
            raise ValueError(f"Expected 'score' column in {path}")
        scores[domain] = df["score"].to_numpy()
        if labels is None and "label" in df.columns:
            labels = df["label"].to_numpy()
        if "sample_id" in df.columns:
            score_frames[domain] = df.copy()

    if not scores:
        raise FileNotFoundError("No score files found for fusion.")

    if score_frames and len(score_frames) == len(scores):
        has_timestamp = all("timestamp" in frame.columns for frame in score_frames.values())
        key_columns = ["sample_id", "timestamp"] if has_timestamp else ["sample_id"]
        if not has_timestamp and any("timestamp" in frame.columns for frame in score_frames.values()):
            print("[warn] Timestamp missing in some domains; aligning on sample_id only.")
        for domain, frame in score_frames.items():
            cols = key_columns + ["score"] + (["label"] if "label" in frame.columns else [])
            frame = frame[cols].copy()
            if frame.duplicated(subset=key_columns).any():
                agg = {"score": "mean"}
                if "label" in frame.columns:
                    agg["label"] = "max"
                frame = frame.groupby(key_columns, as_index=False).agg(agg)
            score_frames[domain] = frame
            keys = (
                frame[key_columns[0]].astype(str)
                if len(key_columns) == 1
                else frame[key_columns[0]].astype(str) + "::" + frame[key_columns[1]].astype(str)
            )
            sample_id_sets.append(set(keys))
        common_ids = sorted(set.intersection(*sample_id_sets)) if sample_id_sets else []
        if not common_ids:
            raise ValueError("No overlapping sample_id values found across domains.")
        labels = None
        for domain, df in score_frames.items():
            if len(key_columns) == 1:
                df = df.set_index(df["sample_id"].astype(str))
            else:
                df = df.set_index(df["sample_id"].astype(str) + "::" + df["timestamp"].astype(str))
            scores[domain] = df.loc[common_ids, "score"].to_numpy()
            if "label" in df.columns:
                domain_labels = df.loc[common_ids, "label"].to_numpy()
                if labels is None:
                    labels = domain_labels
                elif not np.array_equal(labels, domain_labels):
                    raise ValueError("Conflicting labels found across aligned domains.")
        if labels is None:
            labels = np.zeros(len(common_ids), dtype=int)
    else:
        min_len = min(len(v) for v in scores.values())
        scores = {k: v[:min_len] for k, v in scores.items()}
        if labels is None:
            labels = np.zeros(min_len, dtype=int)
        else:
            labels = labels[:min_len]

    return scores, labels


def train_fusion_model(score_paths: dict[str, Path] | None = None, config: dict | None = None):
    score_dict, labels = build_fusion_dataset(score_paths)
    config = config or {"data": {"test_size": 0.2}, "seed": 42}
    return train_fusion_meta_model(score_dict, labels, config)


__all__ = ["build_fusion_dataset", "train_fusion_model"]
