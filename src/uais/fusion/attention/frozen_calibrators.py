"""Load and apply validation-fitted per-domain isotonic score calibrators."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import QuantileTransformer

__all__ = [
    "FrozenCalibratorBundle",
    "fit_isotonic_calibrators_from_csv",
    "load_calibrator_bundle",
    "apply_calibrators_to_features",
    "write_calibrator_lock",
]


@dataclass
class FrozenCalibratorBundle:
    """Per-domain isotonic maps frozen on a source validation split."""

    dataset_key: str
    domain_order: list[str]
    score_index: int
    models: dict[str, IsotonicRegression] = field(default_factory=dict)
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    frozen_utc: str | None = None

    def apply_scores(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return apply_calibrators_to_features(features, masks, self.domain_order, self, self.score_index)


def fit_isotonic_calibrators_from_csv(
    csv_path: Path,
    *,
    dataset_key: str,
    split_col: str = "split",
    val_values: tuple[str, ...] = ("validation",),
    domain_col: str = "domain",
    score_col: str = "score",
    label_col: str = "label",
    domain_order: list[str] | None = None,
    score_index: int = 0,
) -> FrozenCalibratorBundle:
    df = pd.read_csv(csv_path)
    val = df[df[split_col].isin(val_values)]
    if domain_order is None:
        domain_order = sorted(val[domain_col].astype(str).unique().tolist())
    models: dict[str, IsotonicRegression] = {}
    meta: dict[str, dict[str, Any]] = {}
    for domain in domain_order:
        grp = val[val[domain_col].astype(str) == domain]
        if grp.empty:
            meta[domain] = {"status": "skipped", "reason": "no rows"}
            continue
        y = grp[label_col].astype(int).values
        s = grp[score_col].astype(float).values
        if len(s) < 20:
            meta[domain] = {"status": "skipped", "reason": "insufficient validation data", "n_val": int(len(s))}
            continue
        if len(np.unique(y)) < 2:
            n_q = min(100, max(10, len(s) // 5))
            qt = QuantileTransformer(
                n_quantiles=n_q,
                output_distribution="uniform",
                subsample=min(len(s), 10_000),
                random_state=0,
            )
            qt.fit(s.reshape(-1, 1))
            models[domain] = qt
            meta[domain] = {
                "status": "fitted_quantile",
                "n_val": int(len(s)),
                "reason": "validation single-class; unsupervised quantile map",
            }
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(s, y)
        models[domain] = iso
        meta[domain] = {"status": "fitted_isotonic", "n_val": int(len(s))}
    return FrozenCalibratorBundle(
        dataset_key=dataset_key,
        domain_order=list(domain_order),
        score_index=int(score_index),
        models=models,
        meta=meta,
    )


def apply_calibrators_to_features(
    features: np.ndarray,
    masks: np.ndarray,
    domain_order: list[str],
    bundle: FrozenCalibratorBundle,
    score_index: int | None = None,
) -> np.ndarray:
    """Replace raw domain scores with frozen isotonic calibrated scores."""
    out = np.array(features, copy=True)
    idx = int(score_index if score_index is not None else bundle.score_index)
    for d, domain in enumerate(domain_order):
        model = bundle.models.get(domain)
        if model is None:
            continue
        present = ~masks[:, d]
        if not np.any(present):
            continue
        raw = out[present, d, idx].astype(float)
        if isinstance(model, IsotonicRegression):
            calibrated = model.predict(raw)
        elif isinstance(model, QuantileTransformer):
            calibrated = model.transform(raw.reshape(-1, 1)).ravel()
        else:
            raise TypeError(f"Unknown calibrator type for domain {domain}: {type(model)}")
        out[present, d, idx] = calibrated.astype(np.float32)
    return out


def write_calibrator_lock(
    root: Path,
    bundles: dict[str, FrozenCalibratorBundle],
    *,
    rel_paths: dict[str, str] | None = None,
    merge_existing: bool = True,
) -> Path:
    out_dir = root / "elara_master_c/models/calibrators"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    lock = out_dir / "calibrator_lock_v1.json"
    manifest: dict[str, Any] = {"version": 2, "frozen_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    if merge_existing and lock.is_file():
        try:
            prior = json.loads(lock.read_text(encoding="utf-8"))
            manifest["datasets"] = dict(prior.get("datasets") or {})
        except json.JSONDecodeError:
            pass
    for key, bundle in bundles.items():
        pkl = out_dir / f"{key}_isotonic.pkl"
        with pkl.open("wb") as f:
            pickle.dump(
                {
                    "domain_order": bundle.domain_order,
                    "score_index": bundle.score_index,
                    "models": bundle.models,
                    "meta": bundle.meta,
                },
                f,
            )
        manifest["datasets"][key] = {
            "path": str((rel_paths or {}).get(key, pkl.relative_to(root))),
            "domain_order": bundle.domain_order,
            "score_index": bundle.score_index,
            "domains": bundle.meta,
        }
    lock.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return lock


def load_calibrator_bundle(root: Path, dataset_key: str, lock_path: Path | None = None) -> FrozenCalibratorBundle:
    lock = lock_path or (root / "elara_master_c/models/calibrators/calibrator_lock_v1.json")
    manifest = json.loads(lock.read_text(encoding="utf-8"))
    entry = manifest["datasets"].get(dataset_key)
    if entry is None:
        raise KeyError(f"dataset_key '{dataset_key}' not in calibrator lock {lock}")
    pkl = root / entry["path"]
    payload = pickle.load(pkl.open("rb"))
    return FrozenCalibratorBundle(
        dataset_key=dataset_key,
        domain_order=list(payload["domain_order"]),
        score_index=int(payload["score_index"]),
        models=dict(payload["models"]),
        meta=dict(payload.get("meta") or {}),
        frozen_utc=manifest.get("frozen_utc"),
    )
