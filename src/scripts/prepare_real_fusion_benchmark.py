"""Prepare a real-domain score-level fusion benchmark for CRAF.

The available local datasets are not naturally co-observed by the same entity or
timestamp. This script therefore builds a controlled real-domain benchmark:
domain observations are sampled from real fraud, cyber, behavior, and text
datasets, aligned by binary label, and scored with out-of-fold domain models.

The output follows docs/research/data/FUSION_SCHEMA.md and can be consumed by
src/scripts/run_breakthrough_experiment.py.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EMBEDDING_DIM = 8
DOMAIN_ORDER = ["fraud", "cyber", "behavior", "nlp"]


@dataclass(frozen=True)
class DomainSource:
    name: str
    frame: pd.DataFrame
    label_column: str
    feature_columns: Sequence[str]
    text_column: str | None = None


def _balanced_sample(
    df: pd.DataFrame,
    label_column: str,
    max_rows: int,
    positive_fraction: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = df[label_column].astype(int).to_numpy()
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    n_pos = min(len(pos_idx), max(1, int(max_rows * positive_fraction)))
    n_neg = min(len(neg_idx), max_rows - n_pos)
    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    chosen = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(chosen)
    return df.iloc[chosen].reset_index(drop=True)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _tabular_pipeline(df: pd.DataFrame, feature_columns: Sequence[str], seed: int) -> Pipeline:
    categorical = [
        col
        for col in feature_columns
        if pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_bool_dtype(df[col])
        or isinstance(df[col].dtype, pd.CategoricalDtype)
    ]
    numeric = [col for col in feature_columns if col not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", max_categories=48),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers)),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    alpha=1e-4,
                    max_iter=1500,
                    tol=1e-3,
                    random_state=seed,
                ),
            ),
        ]
    )


def _text_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "hash",
                HashingVectorizer(
                    n_features=2**16,
                    alternate_sign=False,
                    norm="l2",
                    ngram_range=(1, 2),
                    lowercase=True,
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    alpha=5e-5,
                    max_iter=1200,
                    tol=1e-3,
                    random_state=seed,
                ),
            ),
        ]
    )


def _subsample_train_indices(train_idx: np.ndarray, y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Subsample a stratified fraction of training indices for harder scorers."""
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("fraction must be in (0, 1].")
    if fraction >= 1.0:
        return train_idx

    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(y[train_idx]):
        class_idx = train_idx[y[train_idx] == label]
        n_take = max(1, int(np.floor(len(class_idx) * fraction)))
        selected.append(rng.choice(class_idx, size=n_take, replace=False))
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def _oof_scores(source: DomainSource, seed: int, folds: int, train_fraction: float = 1.0) -> tuple[np.ndarray, dict]:
    y = source.frame[source.label_column].astype(int).to_numpy()
    scores = np.zeros(len(source.frame), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    base_model = _text_pipeline(seed) if source.text_column else _tabular_pipeline(source.frame, source.feature_columns, seed)
    X = (
        source.frame[source.text_column].fillna("").astype(str)
        if source.text_column
        else source.frame[list(source.feature_columns)]
    )
    train_sizes = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        train_idx = _subsample_train_indices(train_idx, y, fraction=train_fraction, seed=seed + fold_idx)
        train_sizes.append(int(len(train_idx)))
        model = clone(base_model)
        model.fit(X.iloc[train_idx], y[train_idx])
        scores[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1].astype(np.float32)
    return scores, {
        "rows": int(len(source.frame)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "oof_roc_auc": _safe_auc(y, scores),
        "oof_pr_auc": _safe_pr_auc(y, scores),
        "scorer_train_fraction": float(train_fraction),
        "mean_fold_train_rows": float(np.mean(train_sizes)) if train_sizes else 0.0,
    }


def _minmax_columns(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = np.nanmin(values, axis=0)
    hi = np.nanmax(values, axis=0)
    denom = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
    return np.nan_to_num((values - lo) / denom, nan=0.0, posinf=1.0, neginf=0.0)


def _tabular_embeddings(df: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    numeric = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric:
        numeric_frame = pd.DataFrame({"constant": np.zeros(len(df), dtype=np.float32)})
    else:
        variances = df[numeric].var(numeric_only=True).sort_values(ascending=False)
        selected = list(variances.head(EMBEDDING_DIM).index)
        numeric_frame = df[selected]
    arr = numeric_frame.to_numpy(dtype=np.float32)
    if arr.shape[1] < EMBEDDING_DIM:
        pad = np.zeros((arr.shape[0], EMBEDDING_DIM - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return _minmax_columns(arr[:, :EMBEDDING_DIM])


def _text_embeddings(text: pd.Series) -> np.ndarray:
    values = []
    punctuation = set(string.punctuation)
    url_pattern = re.compile(r"https?://|www\\.")
    for item in text.fillna("").astype(str):
        chars = max(len(item), 1)
        words = item.split()
        word_count = max(len(words), 1)
        values.append(
            [
                len(item),
                word_count,
                sum(len(w) for w in words) / word_count,
                sum(ch.isdigit() for ch in item) / chars,
                sum(ch.isupper() for ch in item) / chars,
                sum(ch in punctuation for ch in item) / chars,
                len(set(words)) / word_count,
                1.0 if url_pattern.search(item) else 0.0,
            ]
        )
    return _minmax_columns(np.asarray(values, dtype=np.float32))


def _domain_frame(source: DomainSource, seed: int, folds: int, train_fraction: float = 1.0) -> tuple[pd.DataFrame, dict]:
    scores, metrics = _oof_scores(source, seed=seed, folds=folds, train_fraction=train_fraction)
    embeddings = (
        _text_embeddings(source.frame[source.text_column])
        if source.text_column
        else _tabular_embeddings(source.frame, source.feature_columns)
    )
    labels = source.frame[source.label_column].astype(int).to_numpy()
    out = pd.DataFrame(
        {
            "domain_source_id": np.arange(len(source.frame), dtype=int),
            "label": labels,
            "score": np.clip(scores, 0.0, 1.0),
            "confidence": np.clip(2.0 * np.abs(scores - 0.5), 0.0, 1.0),
        }
    )
    for idx in range(EMBEDDING_DIM):
        out[f"embedding_{idx}"] = embeddings[:, idx]
    return out, metrics


def _read_sources(root: Path, seed: int, max_rows_per_domain: int) -> list[DomainSource]:
    fraud = pd.read_csv(root / "data/raw/fraud/creditcard.csv")
    fraud = _balanced_sample(fraud, "Class", max_rows=max_rows_per_domain, positive_fraction=0.25, seed=seed)
    fraud_features = [col for col in fraud.columns if col != "Class"]

    cyber_train = pd.read_csv(root / "data/raw/cyber/UNSW_NB15_training-set.csv")
    cyber_test = pd.read_csv(root / "data/raw/cyber/UNSW_NB15_testing-set.csv")
    cyber = pd.concat([cyber_train, cyber_test], ignore_index=True)
    cyber = _balanced_sample(cyber, "label", max_rows=max_rows_per_domain, positive_fraction=0.5, seed=seed + 1)
    cyber_features = [col for col in cyber.columns if col not in {"label", "attack_cat", "id"}]

    behavior = pd.read_csv(root / "data/raw/behavior/online_shoppers_intention.csv")
    behavior["label"] = behavior["Revenue"].astype(int)
    behavior = _balanced_sample(behavior, "label", max_rows=max_rows_per_domain, positive_fraction=0.35, seed=seed + 2)
    behavior_features = [col for col in behavior.columns if col not in {"label", "Revenue"}]

    nlp_path = root / "data/raw/nlp/fakenews/fake_news_labeled.csv"
    nlp = pd.read_csv(nlp_path)
    nlp = _balanced_sample(nlp, "label", max_rows=max_rows_per_domain, positive_fraction=0.5, seed=seed + 3)

    return [
        DomainSource("fraud", fraud, "Class", fraud_features),
        DomainSource("cyber", cyber, "label", cyber_features),
        DomainSource("behavior", behavior, "label", behavior_features),
        DomainSource("nlp", nlp, "label", [], text_column="content"),
    ]


def _build_fusion_rows(
    domain_frames: dict[str, pd.DataFrame],
    n_samples: int,
    positive_fraction: float,
    missing_probability: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_samples) < positive_fraction).astype(int)
    rows = []
    feature_cols = ["score", "confidence", *[f"embedding_{idx}" for idx in range(EMBEDDING_DIM)]]
    by_label = {
        domain: {
            label: frame.index[frame["label"].to_numpy() == label].to_numpy()
            for label in [0, 1]
        }
        for domain, frame in domain_frames.items()
    }
    for sample_idx, label in enumerate(labels):
        sample_id = f"real_fusion_{sample_idx:06d}"
        keep = rng.random(len(DOMAIN_ORDER)) >= missing_probability
        if not keep.any():
            keep[rng.integers(0, len(DOMAIN_ORDER))] = True
        for domain_idx, domain in enumerate(DOMAIN_ORDER):
            if not keep[domain_idx]:
                continue
            candidates = by_label[domain][int(label)]
            chosen = int(rng.choice(candidates))
            src = domain_frames[domain].loc[chosen]
            row = {
                "sample_id": sample_id,
                "domain": domain,
                "label": int(label),
                "source_row": int(src["domain_source_id"]),
            }
            for col in feature_cols:
                row[col] = float(src[col])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real-domain CRAF fusion inputs")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("experiments/fusion/real_domain_fusion_inputs.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("experiments/fusion/real_domain_fusion_metadata.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-rows-per-domain", type=int, default=20000)
    parser.add_argument(
        "--scorer-train-fraction",
        type=float,
        default=1.0,
        help="Stratified fraction of each OOF training fold used to fit domain scorers; use 0.05-0.10 for harder benchmarks.",
    )
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--positive-fraction", type=float, default=0.3)
    parser.add_argument("--missing-probability", type=float, default=0.12)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    metadata_path = args.metadata if args.metadata.is_absolute() else root / args.metadata

    sources = _read_sources(root, seed=args.seed, max_rows_per_domain=args.max_rows_per_domain)
    domain_frames: dict[str, pd.DataFrame] = {}
    domain_metrics: dict[str, dict] = {}
    for source in sources:
        frame, metrics = _domain_frame(
            source,
            seed=args.seed,
            folds=args.folds,
            train_fraction=args.scorer_train_fraction,
        )
        domain_frames[source.name] = frame
        domain_metrics[source.name] = metrics

    fusion_df = _build_fusion_rows(
        domain_frames,
        n_samples=args.samples,
        positive_fraction=args.positive_fraction,
        missing_probability=args.missing_probability,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fusion_df.to_csv(output, index=False)

    coverage = fusion_df.groupby("domain")["sample_id"].nunique().div(args.samples).to_dict()
    metadata = {
        "benchmark_type": "label_aligned_real_domain_score_fusion",
        "natural_pairing": False,
        "pairing_unit": "binary-label-aligned composite sample",
        "important_limitation": (
            "Domains are sampled from real datasets and aligned by binary label; "
            "they are not naturally co-observed entities."
        ),
        "seed": args.seed,
        "folds": args.folds,
        "scorer_train_fraction": args.scorer_train_fraction,
        "samples": args.samples,
        "positive_fraction_requested": args.positive_fraction,
        "positive_fraction_actual": float(fusion_df.groupby("sample_id")["label"].first().mean()),
        "missing_probability": args.missing_probability,
        "domain_order": DOMAIN_ORDER,
        "embedding_dim": EMBEDDING_DIM,
        "domain_coverage": {str(k): float(v) for k, v in coverage.items()},
        "domain_scorer_metrics": domain_metrics,
        "output": str(output),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
