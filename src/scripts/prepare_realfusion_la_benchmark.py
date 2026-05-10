"""Prepare the RealFusion-LA benchmark from four public tabular datasets.

RealFusion-LA (Label-Aligned multimodal anomaly benchmark):
  4 domains  — credit_card_fraud | network_intrusion | online_shoppers | news_text
  ~8,000 composite samples (exact count depends on source data availability)
  ~30.7% positive rate | ~11.8% synthetic missingness

Each composite sample is built by pairing one row per domain that shares the
same ground-truth label (anomalous or normal), creating a label-aligned
multi-domain observation without requiring naturally co-occurring events.

Output: data/real_fusion/realfusion_la.csv
Schema: sample_id, domain, label, score, confidence, embedding_0 .. embedding_N

Usage
-----
# Minimal — downloads what it can and uses the rest from local cache:
python src/scripts/prepare_realfusion_la_benchmark.py

# Point to locally downloaded files:
python src/scripts/prepare_realfusion_la_benchmark.py \\
    --creditcard  data/raw/creditcard.csv \\
    --unswnb15    data/raw/UNSW_NB15_training-set.csv \\
    --shoppers    data/raw/online_shoppers_intention.csv \\
    --agnews      data/raw/ag_news_test.csv \\
    --output      data/real_fusion/realfusion_la.csv \\
    --n-composite 8000 \\
    --missing-rate 0.118 \\
    --seed 42

Dataset download links (free, no registration required for most):
  Credit Card Fraud : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
                      (requires Kaggle account; download creditcard.csv)
  UNSW-NB15         : https://research.unsw.edu.au/projects/unsw-nb15-dataset
                      (direct download links on the page)
  Online Shoppers   : https://archive.ics.uci.edu/dataset/468
                      (online_shoppers_intention.csv)
  AG News           : Loaded automatically via sklearn fetch_20newsgroups
                      (binary: comp.* = normal, sci.med = anomaly)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Per-domain feature extractors
# ---------------------------------------------------------------------------

def _load_creditcard(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X [N, 29], y [N]) from Kaggle creditcard.csv."""
    df = pd.read_csv(path)
    y = df["Class"].values.astype(int)
    X = df.drop(columns=["Class", "Time"]).values.astype(np.float32)
    return X, y


def _load_unswnb15(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) from UNSW-NB15 training or full CSV.

    Handles both the 45-column training-set CSV and the raw 49-column CSV.
    Label column is 'label' or 'attack_cat' (binary: 0=normal, 1=attack).
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    if "label" in df.columns:
        y = df["label"].astype(int).values
    elif "attack_cat" in df.columns:
        y = (df["attack_cat"].str.strip() != "").astype(int).values
    else:
        raise ValueError("Cannot find label column in UNSW-NB15 file.")

    # Drop non-numeric and identifier columns
    drop_cols = {"label", "attack_cat", "proto", "service", "state",
                 "id", "srcip", "dstip", "sport", "dsport"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = pd.to_numeric(
        df[feature_cols].stack(), errors="coerce"
    ).unstack().fillna(0.0).values.astype(np.float32)
    return X, y


def _load_online_shoppers(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) from UCI Online Shoppers Intention dataset.

    Label: Revenue == True → 1 (purchase = anomaly in fraud context).
    """
    df = pd.read_csv(path)
    y = df["Revenue"].astype(int).values

    # Encode categoricals + drop target
    df = df.drop(columns=["Revenue"])
    df = pd.get_dummies(df, columns=["Month", "VisitorType", "Weekend"],
                        drop_first=True)
    X = df.values.astype(np.float32)
    return X, y


def _load_news_text(path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Return TF-IDF features + binary labels from AG News or 20Newsgroups.

    If path is None or missing, falls back to sklearn fetch_20newsgroups
    (downloaded automatically, ~14 MB).

    Binary label: 1 = anomalous category (sci.med), 0 = normal (comp.*).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if path is not None and path.exists():
        # Assume CSV with columns: text, label (or class)
        df = pd.read_csv(path)
        if "description" in df.columns:
            texts = df["description"].fillna("").tolist()
        elif "text" in df.columns:
            texts = df["text"].fillna("").tolist()
        else:
            texts = df.iloc[:, 0].fillna("").tolist()
        if "label" in df.columns:
            y = df["label"].astype(int).values
        elif "Class Index" in df.columns:
            # AG News: 1=World, 2=Sports, 3=Business, 4=Sci/Tech
            # Treat Sci/Tech (4) as anomaly
            y = (df["Class Index"] == 4).astype(int).values
        else:
            y = np.zeros(len(texts), dtype=int)
    else:
        from sklearn.datasets import fetch_20newsgroups
        normal_cats = ["comp.graphics", "comp.os.ms-windows.misc",
                       "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"]
        anomaly_cats = ["sci.med"]
        data_norm = fetch_20newsgroups(
            subset="all", categories=normal_cats, remove=("headers", "footers", "quotes")
        )
        data_anom = fetch_20newsgroups(
            subset="all", categories=anomaly_cats, remove=("headers", "footers", "quotes")
        )
        texts = data_norm.data + data_anom.data
        y = np.array([0] * len(data_norm.data) + [1] * len(data_anom.data), dtype=int)

    vec = TfidfVectorizer(max_features=500, sublinear_tf=True)
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Score + embedding computation
# ---------------------------------------------------------------------------

def _compute_score_and_embedding(
    X: np.ndarray,
    y: np.ndarray,
    n_embedding_dims: int = 16,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (scores, confidences, embeddings) for all samples.

    - score: calibrated anomaly probability from a held-out LR classifier
    - confidence: |score - 0.5| * 2  (sharpness)
    - embedding: PCA-reduced feature vector (n_embedding_dims dims)
    """
    rng = np.random.default_rng(random_seed)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Score: 5-fold cross-val LR predictions (avoids train-set leakage)
    from sklearn.model_selection import cross_val_predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=300,
            random_state=random_seed, solver="lbfgs",
        )
        scores = cross_val_predict(lr, X_s, y, cv=5, method="predict_proba")[:, 1]

    confidences = np.clip(np.abs(scores - 0.5) * 2.0, 0.0, 1.0)

    # Embedding: PCA
    n_dims = min(n_embedding_dims, X_s.shape[1], X_s.shape[0] - 1)
    pca = PCA(n_components=n_dims, random_state=random_seed)
    embeddings = pca.fit_transform(X_s)

    return scores.astype(np.float32), confidences.astype(np.float32), embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Label-alignment: pair samples by class to create composite observations
# ---------------------------------------------------------------------------

def _label_align(
    domain_data: dict,        # {domain_name: (scores, confidences, embeddings, labels)}
    n_composite: int,
    positive_rate: float,
    missing_rate: float,
    random_seed: int,
) -> pd.DataFrame:
    """Build the long-format fusion CSV by label-aligning across domains.

    Each composite sample_id has exactly one row per domain (or is marked
    as missing if the domain is randomly dropped).
    """
    rng = np.random.default_rng(random_seed)
    domain_names = list(domain_data.keys())

    n_pos = int(round(n_composite * positive_rate))
    n_neg = n_composite - n_pos

    # Separate indices per class per domain
    pos_idx: dict = {}
    neg_idx: dict = {}
    for dname, (_, _, _, labels) in domain_data.items():
        pos_idx[dname] = np.where(labels == 1)[0]
        neg_idx[dname] = np.where(labels == 0)[0]

    # Determine achievable sample counts (limited by smallest class)
    max_pos = min(len(v) for v in pos_idx.values())
    max_neg = min(len(v) for v in neg_idx.values())
    if max_pos < n_pos:
        print(f"[warn] Requested {n_pos} positives but only {max_pos} available; "
              f"reducing n_composite accordingly.")
        n_pos = max_pos
    if max_neg < n_neg:
        print(f"[warn] Requested {n_neg} negatives but only {max_neg} available; "
              f"reducing n_composite accordingly.")
        n_neg = max_neg
    n_composite = n_pos + n_neg

    # Sample without replacement for each domain
    chosen_pos: dict = {d: rng.choice(pos_idx[d], size=n_pos, replace=False) for d in domain_names}
    chosen_neg: dict = {d: rng.choice(neg_idx[d], size=n_neg, replace=False) for d in domain_names}

    # Composite label vector: first n_pos are positive
    composite_labels = np.array([1] * n_pos + [0] * n_neg, dtype=int)

    # Build missingness mask: [n_composite, n_domains] bool
    n_domains = len(domain_names)
    missing = rng.random((n_composite, n_domains)) < missing_rate
    # Ensure at least one domain is present per sample
    all_missing_rows = missing.all(axis=1)
    if all_missing_rows.any():
        for row in np.where(all_missing_rows)[0]:
            keep = rng.integers(0, n_domains)
            missing[row, keep] = False

    rows = []
    for sample_id in range(n_composite):
        label = int(composite_labels[sample_id])
        for di, dname in enumerate(domain_names):
            if missing[sample_id, di]:
                continue  # missing row — omit from CSV

            # Fetch the pre-sampled index for this domain+label
            if label == 1:
                raw_idx = chosen_pos[dname][sample_id]
            else:
                raw_idx = chosen_neg[dname][sample_id - n_pos]

            scores, confs, embeddings, _ = domain_data[dname]
            row = {
                "sample_id": sample_id,
                "domain": dname,
                "label": label,
                "score": float(scores[raw_idx]),
                "confidence": float(confs[raw_idx]),
            }
            for ei, ev in enumerate(embeddings[raw_idx]):
                row[f"embedding_{ei}"] = float(ev)
            rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder columns for readability
    embed_cols = sorted([c for c in df.columns if c.startswith("embedding_")])
    base_cols = ["sample_id", "domain", "label", "score", "confidence"]
    df = df[base_cols + embed_cols]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_benchmark(
    creditcard_path: Optional[Path] = None,
    unswnb15_path: Optional[Path] = None,
    shoppers_path: Optional[Path] = None,
    agnews_path: Optional[Path] = None,
    output_path: Path = Path("data/real_fusion/realfusion_la.csv"),
    n_composite: int = 8000,
    positive_rate: float = 0.307,
    missing_rate: float = 0.118,
    n_embedding_dims: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """Build RealFusion-LA and save to output_path. Returns the DataFrame."""
    domain_loaders = {
        "credit_card_fraud": (creditcard_path, _load_creditcard),
        "network_intrusion": (unswnb15_path,   _load_unswnb15),
        "online_shoppers":   (shoppers_path,   _load_online_shoppers),
        "news_text":         (agnews_path,      _load_news_text),
    }

    domain_data = {}
    for dname, (path, loader) in domain_loaders.items():
        print(f"Loading {dname} ...", end=" ", flush=True)
        try:
            if loader is _load_news_text:
                X, y = loader(path)
            elif path is None or not path.exists():
                print(f"SKIP — file not found: {path}")
                continue
            else:
                X, y = loader(path)
            scores, confs, embeddings = _compute_score_and_embedding(
                X, y, n_embedding_dims=n_embedding_dims, random_seed=seed
            )
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            print(f"OK ({len(X)} samples, {n_pos} pos / {n_neg} neg)")
            domain_data[dname] = (scores, confs, embeddings, y)
        except Exception as exc:
            print(f"FAIL — {exc}")

    if len(domain_data) < 2:
        print(
            "\n[ERROR] Fewer than 2 domains loaded. Cannot build benchmark.\n"
            "Please download the missing datasets and rerun with --creditcard, "
            "--unswnb15, --shoppers, and/or --agnews flags.\n"
            "See the module docstring for download links."
        )
        sys.exit(1)

    if len(domain_data) < 4:
        print(f"[warn] Only {len(domain_data)}/4 domains available. "
              f"Benchmark will have fewer domains than the paper.")

    print(f"\nBuilding {n_composite} label-aligned composite samples "
          f"({positive_rate:.1%} positive, {missing_rate:.1%} missing) ...")
    df = _label_align(
        domain_data,
        n_composite=n_composite,
        positive_rate=positive_rate,
        missing_rate=missing_rate,
        random_seed=seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_samples = df["sample_id"].nunique()
    n_rows = len(df)
    actual_pos_rate = df.drop_duplicates("sample_id")["label"].mean()
    actual_missing = 1.0 - n_rows / (n_samples * len(domain_data))
    print(
        f"\nSaved {n_rows} rows ({n_samples} composite samples, "
        f"{len(domain_data)} domains) to {output_path}\n"
        f"  Positive rate : {actual_pos_rate:.3f}\n"
        f"  Missing rate  : {actual_missing:.3f}\n"
        f"  Embedding dims: {sum(1 for c in df.columns if c.startswith('embedding_'))}"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the RealFusion-LA benchmark CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--creditcard", type=Path, default=None,
                        help="Path to Kaggle creditcard.csv")
    parser.add_argument("--unswnb15", type=Path, default=None,
                        help="Path to UNSW-NB15 CSV (training-set or full)")
    parser.add_argument("--shoppers", type=Path, default=None,
                        help="Path to online_shoppers_intention.csv (UCI)")
    parser.add_argument("--agnews", type=Path, default=None,
                        help="Path to AG News CSV (optional; falls back to "
                             "sklearn fetch_20newsgroups)")
    parser.add_argument("--output", type=Path,
                        default=Path("data/real_fusion/realfusion_la.csv"),
                        help="Output path for fusion CSV")
    parser.add_argument("--n-composite", type=int, default=8000,
                        help="Number of composite samples to create (default: 8000)")
    parser.add_argument("--positive-rate", type=float, default=0.307,
                        help="Target fraction of positive samples (default: 0.307)")
    parser.add_argument("--missing-rate", type=float, default=0.118,
                        help="Fraction of domain observations to randomly drop "
                             "(default: 0.118)")
    parser.add_argument("--embedding-dims", type=int, default=16,
                        help="PCA embedding dimensions per domain (default: 16)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_benchmark(
        creditcard_path=args.creditcard,
        unswnb15_path=args.unswnb15,
        shoppers_path=args.shoppers,
        agnews_path=args.agnews,
        output_path=args.output,
        n_composite=args.n_composite,
        positive_rate=args.positive_rate,
        missing_rate=args.missing_rate,
        n_embedding_dims=args.embedding_dims,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
