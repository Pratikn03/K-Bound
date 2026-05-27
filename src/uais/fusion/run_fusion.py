"""CLI to train a fusion meta-model from per-domain scores.

It ingests score files defined in configs/fusion_baseline.yaml (or a supplied YAML),
aligns them by minimum length, and trains a simple logistic regression stacker.
If scores/labels are missing, it falls back to synthetic data so the pipeline still runs.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from uais.fusion.build_embeddings import generate_meta_features
from uais.utils.metrics import classification_metrics
from uais.utils.paths import domain_paths

DEFAULT_CONFIG = Path("configs/fusion_baseline.yaml")


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def _read_scores(
    path: Path,
    score_col: str,
    label_col: str | None,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame | None]:
    if not path.exists():
        raise FileNotFoundError(f"Score file not found: {path}")

    if path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=True)
        if isinstance(data, dict):
            scores = np.asarray(data.get(score_col, list(data.values())[0]))
            labels = data.get(label_col) if label_col else None
            labels = np.asarray(labels) if labels is not None else None
        else:
            scores = np.asarray(data).ravel()
            labels = None
        return scores, labels, None
    else:
        df = pd.read_csv(path)
        if score_col not in df.columns:
            # try common fallbacks
            for candidate in ["anomaly_score", "score", "prob", "probability"]:
                if candidate in df.columns:
                    score_col = candidate
                    break
        scores = df[score_col].to_numpy()
        labels = df[label_col].to_numpy() if label_col and label_col in df.columns else None
        if "sample_id" in df.columns:
            cols = ["sample_id", score_col]
            if "timestamp" in df.columns:
                cols.insert(1, "timestamp")
            if label_col and label_col in df.columns:
                cols.append(label_col)
            df = df[cols].copy()
            df.rename(columns={score_col: "score"}, inplace=True)
            if label_col and label_col in df.columns:
                df.rename(columns={label_col: "label"}, inplace=True)
            return scores, labels, df
    return scores, labels, None


def _align_scores(score_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    min_len = min(len(v) for v in score_dict.values())
    return {k: v[:min_len] for k, v in score_dict.items()}


def _load_inputs(cfg_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    cfg = _load_yaml(cfg_path)
    fusion_cfg = cfg.get("fusion", {}) if isinstance(cfg, dict) else {}
    score_col = fusion_cfg.get("score_column", "score")
    label_col = fusion_cfg.get("label_column")
    score_paths = fusion_cfg.get("scores", {})
    labels_path = fusion_cfg.get("labels_path")

    if not score_paths:
        raise ValueError("No score paths provided in fusion config.")

    scores = {}
    labels = None
    score_frames: dict[str, pd.DataFrame] = {}
    for domain, path in score_paths.items():
        s, labels_for_domain, frame = _read_scores(Path(path), score_col, label_col)
        scores[domain] = s
        if labels_for_domain is not None:
            labels = labels_for_domain
        if frame is not None:
            score_frames[domain] = frame

    if score_frames and len(score_frames) == len(scores):
        has_timestamp = all("timestamp" in frame.columns for frame in score_frames.values())
        key_columns = ["sample_id", "timestamp"] if has_timestamp else ["sample_id"]
        if not has_timestamp and any("timestamp" in frame.columns for frame in score_frames.values()):
            print("[warn] Timestamp missing in some domains; aligning on sample_id only.")
        key_sets = []
        for domain, frame in score_frames.items():
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
            key_sets.append(set(keys))
        common_ids = sorted(set.intersection(*key_sets)) if key_sets else []
        if not common_ids:
            raise ValueError("No overlapping sample_id values found across domains.")
        labels = None
        for domain, frame in score_frames.items():
            if len(key_columns) == 1:
                frame = frame.set_index(frame["sample_id"].astype(str))
            else:
                frame = frame.set_index(frame["sample_id"].astype(str) + "::" + frame["timestamp"].astype(str))
            scores[domain] = frame.loc[common_ids, "score"].to_numpy()
            if "label" in frame.columns:
                domain_labels = frame.loc[common_ids, "label"].to_numpy()
                if labels is None:
                    labels = domain_labels
                elif not np.array_equal(labels, domain_labels):
                    raise ValueError("Conflicting labels found across aligned domains.")
        if labels is None:
            labels = np.zeros(len(common_ids))
    else:
        scores = _align_scores(scores)
        if labels is not None:
            labels = labels[: len(next(iter(scores.values())))]
        elif labels_path:
            labels = np.load(labels_path)
            labels = labels[: len(next(iter(scores.values())))]
        else:
            labels = np.zeros(len(next(iter(scores.values()))))

    return scores, labels


def train_fusion(cfg_path: Path = DEFAULT_CONFIG):
    try:
        score_dict, labels = _load_inputs(cfg_path)
    except Exception as exc:
        # fallback synthetic
        rng = np.random.default_rng(42)
        n = 200
        score_dict = {
            "fraud": rng.uniform(0, 1, size=n),
            "cyber": rng.uniform(0, 1, size=n),
            "behavior": rng.uniform(0, 1, size=n),
        }
        labels = (rng.random(n) < 0.3).astype(int)
        print(f"Fusion fallback: {exc}. Using synthetic scores.")

    X = generate_meta_features(score_dict)
    y = np.asarray(labels)[: X.shape[0]]

    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    fusion_cfg = cfg.get("fusion", {}) if isinstance(cfg, dict) else {}
    test_size = fusion_cfg.get("test_size", 0.2)
    random_state = fusion_cfg.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y if len(np.unique(y)) > 1 else None, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    meta = LogisticRegression(max_iter=500, class_weight="balanced")
    meta.fit(X_train, y_train)

    y_proba = meta.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, y_proba, threshold=0.5)

    paths = domain_paths("fusion")
    paths["models"].mkdir(parents=True, exist_ok=True)
    out_path = paths["models"] / "fusion_meta_model.pkl"
    joblib.dump({"model": meta, "scaler": scaler}, out_path)

    print("Fusion metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Saved fusion model to {out_path}")
    return meta, metrics


if __name__ == "__main__":  # pragma: no cover
    train_fusion()
