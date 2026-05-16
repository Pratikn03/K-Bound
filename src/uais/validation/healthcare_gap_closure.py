"""Healthcare/clinical validation for the ELARA gap-closure checklist.

The copied GridPulse healthcare data is a wide vital-sign table. This module
projects it into the repo's long fusion schema without carrying raw patient
identifiers forward, then runs the four gap-closure checks used in the paper:

1. naturally co-observed incident structure,
2. category-aware reliability behavior,
3. universal fusion-schema integration,
4. calibration/switching audit surfaces.

The report intentionally separates engineering closure from empirical closure.
If a split lacks both classes, the corresponding empirical claim is marked open.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from uais.fusion.attention.attention_utils import (
    validate_fusion_schema,
    validate_incident_protocol,
)
from uais.fusion.attention.reliability_estimator import (
    CategoryAwareReliabilityEstimator,
    ReliabilityEstimator,
)
from uais.utils.metrics import bounded_switching_certificate, calibration_monitor_report


@dataclass(frozen=True)
class ClinicalDomainSpec:
    name: str
    value_column: str


DOMAIN_SPECS = (
    ClinicalDomainSpec("heart_rate", "hr_bpm"),
    ClinicalDomainSpec("oxygenation", "spo2_pct"),
    ClinicalDomainSpec("respiration", "respiratory_rate"),
    ClinicalDomainSpec("shock_index", "shock_index"),
)

FEATURE_COLUMNS = ["score", "confidence", "embedding_0", "embedding_1"]
SPLIT_MAP = {
    "train": "train",
    "calibration": "validation",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def _stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _domain_score(domain: str, value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    if domain == "heart_rate":
        high = max(0.0, value - 110.0) / 50.0
        low = max(0.0, 50.0 - value) / 30.0
        return _clip01(max(high, low))
    if domain == "oxygenation":
        return _clip01((95.0 - value) / 12.0)
    if domain == "respiration":
        high = max(0.0, value - 24.0) / 20.0
        low = max(0.0, 8.0 - value) / 8.0
        return _clip01(max(high, low))
    if domain == "shock_index":
        return _clip01((value - 0.7) / 0.8)
    raise ValueError(f"Unknown clinical domain: {domain}")


def _domain_embedding(domain: str, value: float) -> float:
    if domain == "heart_rate":
        return _clip01((value - 40.0) / 140.0)
    if domain == "oxygenation":
        return _clip01(value / 100.0)
    if domain == "respiration":
        return _clip01((value - 5.0) / 45.0)
    if domain == "shock_index":
        return _clip01(value / 2.0)
    raise ValueError(f"Unknown clinical domain: {domain}")


def _normalise_split(value: object) -> str:
    key = str(value).strip().lower()
    if key not in SPLIT_MAP:
        raise ValueError(f"Unknown healthcare split: {value}")
    return SPLIT_MAP[key]


def load_processed_healthcare_splits(data_root: str | Path) -> pd.DataFrame:
    """Load GridPulse processed healthcare train/calibration/val/test splits."""
    root = Path(data_root)
    split_root = root / "processed" / "splits"
    frames: list[pd.DataFrame] = []
    for split_name in ("train", "calibration", "val", "test"):
        path = split_root / f"{split_name}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame.copy()
        frame["source_split"] = split_name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No processed healthcare split parquet files under {split_root}")
    combined = pd.concat(frames, ignore_index=True)
    required = {
        "patient_id",
        "timestamp",
        "source_dataset",
        "is_critical",
        "reliability",
        *(spec.value_column for spec in DOMAIN_SPECS),
    }
    missing = sorted(required - set(combined.columns))
    if missing:
        raise ValueError(f"Healthcare split data is missing required columns: {missing}")
    return combined


def assign_patient_stratified_splits(
    frame: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign patient-disjoint stratified train/validation/test replay splits.

    The original healthcare split is time-forward, but its validation and test
    windows are single-class positive. This replay mode keeps patients disjoint
    while distributing patient-level labels across all three splits.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1.")
    required = {"patient_id", "is_critical"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot stratify healthcare replay without columns: {missing}")

    patients = (
        frame.assign(_patient_id=frame["patient_id"].astype(str))
        .groupby("_patient_id")["is_critical"]
        .max()
        .astype(int)
        .reset_index(name="patient_label")
    )
    label_counts = patients["patient_label"].value_counts()
    if set(label_counts.index) != {0, 1}:
        raise ValueError("Patient-stratified replay requires both patient-level labels.")
    if int(label_counts.min()) < 3:
        raise ValueError("Patient-stratified replay needs at least three patients per label.")

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for _label, group in patients.groupby("patient_label"):
        patient_ids = group["_patient_id"].to_numpy()
        rng.shuffle(patient_ids)
        n = len(patient_ids)
        n_train = max(1, int(round(n * train_fraction)))
        n_validation = max(1, int(round(n * validation_fraction)))
        if n - n_train - n_validation < 1:
            n_train = max(1, n_train - 1)
        if n - n_train - n_validation < 1:
            n_validation = max(1, n_validation - 1)

        split_bound_1 = n_train
        split_bound_2 = n_train + n_validation
        for patient_id in patient_ids[:split_bound_1]:
            assignment[str(patient_id)] = "train"
        for patient_id in patient_ids[split_bound_1:split_bound_2]:
            assignment[str(patient_id)] = "validation"
        for patient_id in patient_ids[split_bound_2:]:
            assignment[str(patient_id)] = "test"

    out = frame.copy()
    out["source_split"] = out["patient_id"].astype(str).map(assignment)
    if out["source_split"].isna().any():
        raise RuntimeError("Internal error: some patients were not assigned to a replay split.")
    return out


def build_clinical_fusion_frame(split_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert wide clinical vital-sign rows into long multimodal fusion rows.

    Patient identifiers are hashed and the raw ``patient_id`` column is not
    emitted. The incident key is the co-observed patient/timestamp event.
    """
    rows: list[dict[str, Any]] = []
    work = split_frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    if work["timestamp"].isna().any():
        raise ValueError("Non-parseable healthcare timestamps found.")

    for row in work.itertuples(index=False):
        patient_id = str(row.patient_id)
        timestamp = row.timestamp
        timestamp_iso = timestamp.isoformat()
        patient_key = _stable_hash(patient_id)
        incident_id = _stable_hash(f"{patient_id}|{timestamp_iso}")
        source_split = str(row.source_split)
        fusion_split = _normalise_split(source_split)
        source_dataset = str(row.source_dataset)
        label = int(bool(row.is_critical))
        reliability = _clip01(float(row.reliability))

        for spec in DOMAIN_SPECS:
            value = float(getattr(row, spec.value_column))
            score = _domain_score(spec.name, value)
            if not np.isfinite(score):
                continue
            rows.append(
                {
                    "sample_id": incident_id,
                    "incident_id": incident_id,
                    "patient_key": patient_key,
                    "timestamp": timestamp_iso,
                    "domain": spec.name,
                    "score": score,
                    "confidence": reliability,
                    "label": label,
                    "fusion_split": fusion_split,
                    "source_split": source_split,
                    "source_dataset": source_dataset,
                    "category": source_dataset,
                    "embedding_0": _domain_embedding(spec.name, value),
                    "embedding_1": reliability,
                }
            )
    if not rows:
        raise ValueError("Healthcare fusion projection produced no rows.")
    return pd.DataFrame(rows)


def label_diversity_report(fusion: pd.DataFrame) -> dict[str, Any]:
    required_splits = ("train", "validation", "test")
    report: dict[str, Any] = {}
    for split in required_splits:
        part = fusion[fusion["fusion_split"] == split]
        labels = part.groupby("sample_id")["label"].first() if not part.empty else pd.Series(dtype=int)
        counts = labels.value_counts().sort_index()
        report[split] = {
            "incidents": int(labels.shape[0]),
            "label_classes": int(labels.nunique(dropna=True)),
            "positive_incidents": int(counts.get(1, 0)),
            "negative_incidents": int(counts.get(0, 0)),
            "positive_rate": float(labels.mean()) if len(labels) else None,
            "has_two_classes": bool(labels.nunique(dropna=True) >= 2),
        }
    report["all_required_splits_have_two_classes"] = bool(
        all(report[split]["has_two_classes"] for split in required_splits)
    )
    return report


def patient_overlap_report(fusion: pd.DataFrame) -> dict[str, Any]:
    split_to_patients = {
        split: set(part["patient_key"].dropna().astype(str))
        for split, part in fusion.groupby("fusion_split")
    }
    pairwise = {}
    for left, right in combinations(sorted(split_to_patients), 2):
        pairwise[f"{left}/{right}"] = int(len(split_to_patients[left] & split_to_patients[right]))
    return {
        "patient_counts": {split: int(len(values)) for split, values in split_to_patients.items()},
        "pairwise_overlap": pairwise,
        "any_overlap": bool(any(count > 0 for count in pairwise.values())),
    }


def _sample_metadata(fusion: pd.DataFrame, column: str) -> pd.Series:
    grouped = fusion.groupby("sample_id")[column].nunique(dropna=True)
    conflicts = grouped[grouped > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting {column} values for {len(conflicts)} incidents.")
    return fusion.groupby("sample_id")[column].first()


def _tensors_with_metadata(fusion: pd.DataFrame):
    domain_order = [spec.name for spec in DOMAIN_SPECS]
    sample_ids = np.asarray(sorted(fusion["sample_id"].dropna().astype(str).unique().tolist()))
    sample_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    domain_to_idx = {domain: idx for idx, domain in enumerate(domain_order)}

    features = np.zeros((len(sample_ids), len(domain_order), len(FEATURE_COLUMNS)), dtype=np.float32)
    masks = np.ones((len(sample_ids), len(domain_order)), dtype=bool)
    for domain in domain_order:
        domain_rows = fusion[fusion["domain"] == domain].drop_duplicates("sample_id")
        row_idx = domain_rows["sample_id"].astype(str).map(sample_to_idx).to_numpy(dtype=int)
        domain_idx = domain_to_idx[domain]
        features[row_idx, domain_idx, :] = domain_rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        masks[row_idx, domain_idx] = False

    label_map = _sample_metadata(fusion, "label")
    split_map = _sample_metadata(fusion, "fusion_split")
    category_map = _sample_metadata(fusion, "category")
    labels = label_map.reindex(sample_ids).to_numpy(dtype=np.float32)
    splits = split_map.reindex(sample_ids).to_numpy()
    categories = category_map.reindex(sample_ids).to_numpy()
    return features, masks, labels, sample_ids, domain_order, splits, categories


def category_aware_reliability_report(fusion: pd.DataFrame, gate_threshold: float = 0.66) -> dict[str, Any]:
    features, masks, labels, sample_ids, domain_order, splits, categories = _tensors_with_metadata(fusion)
    train_idx = np.flatnonzero(splits == "train")
    eval_idx = np.flatnonzero(splits == "test")
    if len(eval_idx) == 0:
        eval_idx = np.flatnonzero(splits == "validation")
    if len(train_idx) == 0 or len(eval_idx) == 0:
        return {"evaluated": False, "reason": "missing train or evaluation split"}

    common_kwargs = {
        "domain_order": list(domain_order),
        "score_index": 0,
        "ece_weight": 0.45,
        "ks_weight": 0.35,
        "sharpness_weight": 0.20,
        "gate_threshold": gate_threshold,
        "min_samples_for_ks": 30,
    }
    global_estimator = ReliabilityEstimator(**common_kwargs).fit(
        features[train_idx], masks[train_idx], labels[train_idx]
    )
    category_estimator = CategoryAwareReliabilityEstimator(**common_kwargs).fit(
        features[train_idx], masks[train_idx], labels[train_idx], categories=categories[train_idx]
    )

    global_weights = global_estimator.compute_reliability_weights(features[eval_idx], masks[eval_idx])
    category_weights = category_estimator.compute_reliability_weights(
        features[eval_idx], masks[eval_idx], categories=categories[eval_idx]
    )
    global_fire = global_estimator.gate_decisions(global_weights, masks[eval_idx])
    category_fire = category_estimator.gate_decisions(category_weights, masks[eval_idx])
    present_global = global_weights[~masks[eval_idx]]
    present_category = category_weights[~masks[eval_idx]]
    train_categories = set(categories[train_idx].astype(str))
    eval_categories = set(categories[eval_idx].astype(str))
    return {
        "evaluated": True,
        "evaluation_split": "test" if np.any(splits == "test") else "validation",
        "evaluation_incidents": int(len(eval_idx)),
        "global_gate_rate": float(np.mean(global_fire)),
        "category_aware_gate_rate": float(np.mean(category_fire)),
        "global_mean_reliability": float(np.mean(present_global)) if present_global.size else None,
        "category_aware_mean_reliability": float(np.mean(present_category)) if present_category.size else None,
        "train_categories": sorted(train_categories),
        "evaluation_categories": sorted(eval_categories),
        "unknown_evaluation_categories": sorted(eval_categories - train_categories),
    }


def _mean_reliability_per_sample(weights: np.ndarray, masks: np.ndarray) -> np.ndarray:
    n_present = (~masks).sum(axis=1).astype(np.float32)
    return np.where(
        n_present > 0,
        weights.sum(axis=1) / np.maximum(n_present, 1.0),
        0.0,
    )


def _stress_episode_report(
    estimator: CategoryAwareReliabilityEstimator,
    features: np.ndarray,
    masks: np.ndarray,
    categories: np.ndarray,
    domain_order: list[str],
    threshold: float,
) -> dict[str, Any]:
    natural_weights = estimator.compute_reliability_weights(features, masks, categories=categories)
    natural_mean = _mean_reliability_per_sample(natural_weights, masks)
    natural_fire = natural_mean < threshold
    collapse_rates: dict[str, float] = {}
    collapse_means: dict[str, float] = {}
    for domain_idx, domain in enumerate(domain_order):
        collapsed = features.copy()
        collapsed[:, domain_idx, estimator.score_index] = 0.5
        weights = estimator.compute_reliability_weights(collapsed, masks, categories=categories)
        mean_reliability = _mean_reliability_per_sample(weights, masks)
        collapse_rates[domain] = float(np.mean(mean_reliability < threshold))
        collapse_means[domain] = float(np.mean(mean_reliability))
    return {
        "natural_fire_rate": float(np.mean(natural_fire)),
        "natural_mean_reliability": float(np.mean(natural_mean)),
        "collapse_fire_rate_by_domain": collapse_rates,
        "collapse_mean_reliability_by_domain": collapse_means,
        "collapse_fire_rate_mean": float(np.mean(list(collapse_rates.values()))) if collapse_rates else None,
    }


def temporal_reference_protocol_report(split_frame: pd.DataFrame) -> dict[str, Any]:
    """Validate the original time-forward healthcare split as a reference surface."""
    fusion = build_clinical_fusion_frame(split_frame)
    return validate_incident_protocol(
        fusion,
        incident_column="incident_id",
        domain_column="domain",
        timestamp_column="timestamp",
        split_column="fusion_split",
        label_column="label",
        min_domains_per_incident=len(DOMAIN_SPECS),
        require_temporal_order=True,
    )


def gap2_reliability_stress_report(
    fusion: pd.DataFrame,
    natural_safety_margin: float = 0.015,
    max_natural_fire_rate: float = 0.05,
    min_collapse_fire_rate: float = 0.75,
) -> dict[str, Any]:
    """Calibrate and test a category-aware gate on natural vs collapse episodes."""
    features, masks, labels, sample_ids, domain_order, splits, categories = _tensors_with_metadata(fusion)
    train_idx = np.flatnonzero(splits == "train")
    validation_idx = np.flatnonzero(splits == "validation")
    test_idx = np.flatnonzero(splits == "test")
    if len(train_idx) == 0 or len(validation_idx) == 0 or len(test_idx) == 0:
        return {"evaluated": False, "closed": False, "reason": "missing train/validation/test split"}

    estimator = CategoryAwareReliabilityEstimator(
        domain_order=list(domain_order),
        score_index=0,
        ece_weight=0.45,
        ks_weight=0.35,
        sharpness_weight=0.20,
        gate_threshold=0.66,
        min_samples_for_ks=30,
    ).fit(features[train_idx], masks[train_idx], labels[train_idx], categories=categories[train_idx])

    validation_weights = estimator.compute_reliability_weights(
        features[validation_idx], masks[validation_idx], categories=categories[validation_idx]
    )
    validation_natural_mean = _mean_reliability_per_sample(validation_weights, masks[validation_idx])
    threshold = float(max(0.0, np.min(validation_natural_mean) - natural_safety_margin))
    validation = _stress_episode_report(
        estimator,
        features[validation_idx],
        masks[validation_idx],
        categories[validation_idx],
        list(domain_order),
        threshold,
    )
    test = _stress_episode_report(
        estimator,
        features[test_idx],
        masks[test_idx],
        categories[test_idx],
        list(domain_order),
        threshold,
    )
    closed = bool(
        test["natural_fire_rate"] <= max_natural_fire_rate
        and test["collapse_fire_rate_mean"] is not None
        and test["collapse_fire_rate_mean"] >= min_collapse_fire_rate
    )
    return {
        "evaluated": True,
        "closed": closed,
        "calibrated_gate_threshold": threshold,
        "natural_safety_margin": float(natural_safety_margin),
        "max_natural_fire_rate": float(max_natural_fire_rate),
        "min_collapse_fire_rate": float(min_collapse_fire_rate),
        "validation": validation,
        "test": test,
    }


def fusion_schema_integration_report(
    fusion: pd.DataFrame,
    schema_stats: dict[str, Any],
    incident_protocol: dict[str, Any],
) -> dict[str, Any]:
    """Check whether the clinical surface is consumable by the generic fusion schema."""
    required_columns = {
        "sample_id",
        "incident_id",
        "patient_key",
        "timestamp",
        "domain",
        "score",
        "confidence",
        "label",
        "fusion_split",
        "category",
        *FEATURE_COLUMNS,
    }
    missing_columns = sorted(required_columns - set(fusion.columns))
    domain_order = [spec.name for spec in DOMAIN_SPECS]
    domain_incident_counts = (
        fusion.groupby("domain")["sample_id"].nunique().reindex(domain_order, fill_value=0).astype(int).to_dict()
    )
    per_incident_domain_counts = fusion.groupby("incident_id")["domain"].nunique()
    raw_patient_id_absent = "patient_id" not in fusion.columns
    patient_keys_are_hashed = bool(
        "patient_key" in fusion
        and fusion["patient_key"].dropna().astype(str).str.fullmatch(r"[0-9a-f]{16}").all()
    )
    incident_ids_are_hashed = bool(
        "incident_id" in fusion
        and fusion["incident_id"].dropna().astype(str).str.fullmatch(r"[0-9a-f]{16}").all()
    )
    features, masks, _labels, sample_ids, tensor_domains, _splits, _categories = _tensors_with_metadata(fusion)
    tensor_ready = bool(
        features.shape[0] == len(sample_ids)
        and features.shape[1] == len(domain_order)
        and features.shape[2] == len(FEATURE_COLUMNS)
        and list(tensor_domains) == domain_order
        and np.isfinite(features[~masks]).all()
    )
    closed = bool(
        not missing_columns
        and raw_patient_id_absent
        and patient_keys_are_hashed
        and incident_ids_are_hashed
        and tensor_ready
        and int(schema_stats.get("score_out_of_range", 1)) == 0
        and int(schema_stats.get("confidence_out_of_range", 1)) == 0
        and int(incident_protocol.get("split_leakage_count", 1)) == 0
        and int(incident_protocol.get("min_domains_per_incident", 0)) >= len(domain_order)
    )
    return {
        "closed": closed,
        "required_columns": sorted(required_columns),
        "missing_columns": missing_columns,
        "domain_order": domain_order,
        "domain_incident_counts": {str(k): int(v) for k, v in domain_incident_counts.items()},
        "min_domains_per_incident": int(per_incident_domain_counts.min()),
        "tensor_shape": [int(v) for v in features.shape],
        "raw_patient_id_absent": raw_patient_id_absent,
        "patient_keys_are_hashed": patient_keys_are_hashed,
        "incident_ids_are_hashed": incident_ids_are_hashed,
        "tensor_ready": tensor_ready,
    }


def clinical_domain_attribution_report(fusion: pd.DataFrame) -> dict[str, Any]:
    """Produce a leave-one-domain-out CDA-style audit for the clinical replay."""
    scores = fusion.pivot_table(index="sample_id", columns="domain", values="score", aggfunc="mean")
    if scores.empty or scores.shape[1] < 2:
        return {"evaluated": False, "reason": "at least two domains are required"}

    baseline = scores.mean(axis=1)
    impacts: dict[str, Any] = {}
    for domain in scores.columns:
        counterfactual = scores.drop(columns=[domain]).mean(axis=1)
        delta = baseline - counterfactual
        impacts[str(domain)] = {
            "mean_impact": float(delta.mean()),
            "mean_abs_impact": float(delta.abs().mean()),
        }
    top_domain = max(impacts, key=lambda name: impacts[name]["mean_abs_impact"])
    return {
        "evaluated": True,
        "incidents": int(scores.shape[0]),
        "domains": [str(domain) for domain in scores.columns],
        "top_domain_by_abs_impact": str(top_domain),
        "mean_abs_impact_by_domain": {
            domain: float(values["mean_abs_impact"]) for domain, values in impacts.items()
        },
        "mean_impact_by_domain": {
            domain: float(values["mean_impact"]) for domain, values in impacts.items()
        },
    }


def _manifest_review(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root)
    manifest_paths = [
        root / "processed" / "manifest.json",
        root / "mimic3" / "processed" / "mimic3_manifest.json",
        root / "heldout_95" / "manifest.json",
    ]
    found: list[str] = []
    license_notes: list[str] = []
    source_urls: list[str] = []
    claim_boundaries: list[str] = []
    for path in manifest_paths:
        if not path.exists():
            continue
        found.append(str(path))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        note = payload.get("license_notes")
        if note:
            license_notes.append(str(note))
        urls = payload.get("source_urls") or []
        source_urls.extend(str(url) for url in urls)
        boundary = payload.get("claim_boundary")
        if boundary:
            claim_boundaries.append(str(boundary))
    license_review_present = bool(license_notes or source_urls)
    return {
        "manifest_paths_found": found,
        "license_review_present": license_review_present,
        "license_notes": license_notes,
        "source_urls": source_urls,
        "claim_boundaries": claim_boundaries,
    }


def diagnostic_switching_certificate_from_stress(stress_report: dict[str, Any]) -> dict[str, Any]:
    """Certify the diagnostic switch against a never-switch baseline on stress episodes."""
    test = stress_report.get("test") or {}
    collapse_rates = test.get("collapse_fire_rate_by_domain") or {}
    if not stress_report.get("evaluated") or not collapse_rates:
        return {"certified": False, "evaluated": False, "reason": "stress report is missing collapse episodes"}

    natural_fire_rate = float(test.get("natural_fire_rate", 1.0))
    domain_names = sorted(str(name) for name in collapse_rates)
    static_loss = np.asarray([0.0] + [1.0 for _ in domain_names], dtype=float)
    reliability_loss = np.asarray(
        [natural_fire_rate] + [1.0 - float(collapse_rates[name]) for name in domain_names],
        dtype=float,
    )
    fire_decisions = np.asarray([False] + [float(collapse_rates[name]) > 0.0 for name in domain_names], dtype=bool)
    certificate = bounded_switching_certificate(static_loss, reliability_loss, fire_decisions)
    certificate["evaluated"] = True
    certificate["episode_count"] = int(len(static_loss))
    certificate["collapse_domains"] = domain_names
    certificate["natural_episode_loss"] = float(natural_fire_rate)
    return certificate


def gap4_deployment_audit_report(
    fusion: pd.DataFrame,
    *,
    data_root: str | Path,
    temporal_reference: dict[str, Any] | None,
    reliability_stress_report: dict[str, Any],
    switching_report: dict[str, Any],
    label_report: dict[str, Any],
) -> dict[str, Any]:
    """Combine local deployability checks without claiming clinical deployment."""
    attribution = clinical_domain_attribution_report(fusion)
    manifest = _manifest_review(data_root)
    diagnostic_certificate = diagnostic_switching_certificate_from_stress(reliability_stress_report)
    calibration = switching_report.get("calibration") or {}
    temporal_order_valid = bool(
        (temporal_reference or {}).get("temporal_order_valid")
        or (temporal_reference or {}).get("incident_protocol", {}).get("temporal_order_valid")
    )
    monitoring_ready = bool(switching_report.get("evaluated") and "alert" in calibration)
    privacy_review_passed = bool(
        "patient_id" not in fusion.columns
        and "patient_key" in fusion.columns
        and fusion["patient_key"].dropna().astype(str).str.fullmatch(r"[0-9a-f]{16}").all()
        and manifest.get("license_review_present")
    )
    closed = bool(
        temporal_order_valid
        and privacy_review_passed
        and monitoring_ready
        and attribution.get("evaluated")
        and diagnostic_certificate.get("certified")
        and reliability_stress_report.get("closed")
        and label_report.get("all_required_splits_have_two_classes")
    )
    return {
        "evaluated": True,
        "closed": closed,
        "scope": (
            "Local audited deployment-replay readiness only; not prospective clinical deployment, "
            "clinical decision support approval, or regulated use."
        ),
        "temporal_validation": {
            "temporal_order_valid": temporal_order_valid,
            "reference": temporal_reference or {},
        },
        "privacy_license_review": {
            "passed": privacy_review_passed,
            "raw_patient_id_emitted": "patient_id" in fusion.columns,
            "hashed_patient_key_present": "patient_key" in fusion.columns,
            **manifest,
        },
        "monitoring": {
            "calibration_monitor_ready": monitoring_ready,
            "calibration_alert": calibration.get("alert"),
            "calibration_reasons": calibration.get("reasons", []),
            "calibration_thresholds": calibration.get("thresholds", {}),
        },
        "analyst_audit_surface": attribution,
        "diagnostic_switching_certificate": diagnostic_certificate,
    }


def _prediction_table(fusion: pd.DataFrame) -> pd.DataFrame:
    scores = fusion.pivot_table(index="sample_id", columns="domain", values="score", aggfunc="mean")
    confidence = fusion.pivot_table(index="sample_id", columns="domain", values="confidence", aggfunc="mean")
    labels = _sample_metadata(fusion, "label")
    splits = _sample_metadata(fusion, "fusion_split")

    static_prob = scores.mean(axis=1).astype(float)
    weighted_sum = (scores * confidence).sum(axis=1)
    weight_denom = confidence.sum(axis=1).replace(0.0, np.nan)
    reliability_prob = (weighted_sum / weight_denom).fillna(static_prob).astype(float)
    return pd.DataFrame(
        {
            "label": labels.reindex(scores.index).astype(int),
            "fusion_split": splits.reindex(scores.index),
            "static_prob": static_prob,
            "reliability_prob": reliability_prob,
        }
    )


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    try:
        value = float(sk_metrics.roc_auc_score(y_true, y_score))
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    try:
        value = float(sk_metrics.average_precision_score(y_true, y_score))
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def clinical_incident_detection_report(
    fusion: pd.DataFrame,
    min_test_auc: float = 0.7,
) -> dict[str, Any]:
    """Evaluate whether the clinical replay supports Gap 1 local evidence."""
    scores = fusion.pivot_table(index="sample_id", columns="domain", values="score", aggfunc="mean")
    labels = _sample_metadata(fusion, "label").reindex(scores.index).astype(int)
    splits = _sample_metadata(fusion, "fusion_split").reindex(scores.index)
    multimodal = scores.mean(axis=1)

    by_split: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        mask = splits == split
        if not mask.any():
            by_split[split] = {"evaluated": False, "reason": "empty split"}
            continue
        y = labels[mask].to_numpy()
        split_scores = scores.loc[mask]
        domain_roc = {
            str(domain): _safe_roc_auc(y, split_scores[domain].to_numpy())
            for domain in split_scores.columns
        }
        finite_domain_roc = [value for value in domain_roc.values() if value is not None]
        best_single = max(finite_domain_roc) if finite_domain_roc else None
        multi_score = multimodal[mask].to_numpy()
        multi_roc = _safe_roc_auc(y, multi_score)
        multi_pr = _safe_pr_auc(y, multi_score)
        advantage = (
            float(multi_roc - best_single)
            if multi_roc is not None and best_single is not None
            else None
        )
        by_split[split] = {
            "evaluated": multi_roc is not None,
            "incidents": int(mask.sum()),
            "label_classes": int(pd.Series(y).nunique(dropna=True)),
            "multimodal_roc_auc": multi_roc,
            "multimodal_pr_auc": multi_pr,
            "best_single_domain_roc_auc": best_single,
            "multimodal_auc_advantage": advantage,
            "domain_roc_auc": domain_roc,
        }

    test = by_split.get("test", {})
    closed = bool(
        test.get("multimodal_roc_auc") is not None
        and test.get("best_single_domain_roc_auc") is not None
        and float(test["multimodal_roc_auc"]) >= min_test_auc
        and float(test["multimodal_roc_auc"]) >= float(test["best_single_domain_roc_auc"])
    )
    return {
        "closed": closed,
        "min_test_auc": float(min_test_auc),
        "by_split": by_split,
    }


def _binary_log_loss_per_sample(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    return -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))


def switching_and_calibration_report(fusion: pd.DataFrame) -> dict[str, Any]:
    pred = _prediction_table(fusion)
    reference = pred[pred["fusion_split"] == "validation"]
    current = pred[pred["fusion_split"] == "test"]
    if reference.empty or current.empty:
        return {"evaluated": False, "reason": "missing validation or test prediction window"}

    calibration = calibration_monitor_report(
        reference["label"].to_numpy(),
        reference["static_prob"].to_numpy(),
        current["label"].to_numpy(),
        current["static_prob"].to_numpy(),
    )

    features, masks, labels, sample_ids, domain_order, splits, categories = _tensors_with_metadata(fusion)
    train_idx = np.flatnonzero(splits == "train")
    test_idx = np.flatnonzero(splits == "test")
    fire_decisions = np.zeros(len(current), dtype=bool)
    if len(train_idx) and len(test_idx):
        estimator = CategoryAwareReliabilityEstimator(
            domain_order=list(domain_order),
            score_index=0,
            ece_weight=0.45,
            ks_weight=0.35,
            sharpness_weight=0.20,
            gate_threshold=0.66,
            min_samples_for_ks=30,
        ).fit(features[train_idx], masks[train_idx], labels[train_idx], categories=categories[train_idx])
        weights = estimator.compute_reliability_weights(
            features[test_idx], masks[test_idx], categories=categories[test_idx]
        )
        test_sample_ids = sample_ids[test_idx]
        fire_series = pd.Series(estimator.gate_decisions(weights, masks[test_idx]), index=test_sample_ids)
        fire_decisions = fire_series.reindex(current.index).fillna(False).to_numpy(dtype=bool)

    static_loss = _binary_log_loss_per_sample(current["label"].to_numpy(), current["static_prob"].to_numpy())
    reliability_loss = _binary_log_loss_per_sample(
        current["label"].to_numpy(), current["reliability_prob"].to_numpy()
    )
    certificate = bounded_switching_certificate(static_loss, reliability_loss, fire_decisions)
    return {
        "evaluated": True,
        "calibration": calibration,
        "certificate": certificate,
    }


def gap_statuses(
    *,
    incident_protocol_passed: bool,
    label_report: dict[str, Any],
    patient_overlap: dict[str, Any],
    schema_passed: bool,
    category_aware_report: dict[str, Any],
    switching_report: dict[str, Any],
    calibration_report: dict[str, Any],
    incident_detection_report: dict[str, Any] | None = None,
    reliability_stress_report: dict[str, Any] | None = None,
    schema_integration_report: dict[str, Any] | None = None,
    deployment_audit_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    no_patient_overlap = not bool(patient_overlap.get("any_overlap", True))
    empirical_labels_ok = bool(label_report.get("all_required_splits_have_two_classes", False))
    incident_detection_ok = bool((incident_detection_report or {}).get("closed", False))
    category_rate = category_aware_report.get("category_aware_gate_rate")
    global_rate = category_aware_report.get("global_gate_rate")
    category_eval = bool(
        category_aware_report.get(
            "evaluated",
            category_rate is not None and global_rate is not None,
        )
    )
    category_improves_gate = bool(
        category_eval
        and category_rate is not None
        and global_rate is not None
        and float(category_rate) < float(global_rate)
        and float(category_rate) < 0.5
    )
    stress_closed = bool((reliability_stress_report or {}).get("closed", False))
    switching_eval = bool(
        switching_report.get(
            "evaluated",
            "certificate" in switching_report,
        )
    )
    switching_certified = bool((switching_report.get("certificate") or {}).get("certified", False))
    calibration_eval = "alert" in calibration_report
    integration_closed = bool(
        (schema_integration_report or {}).get("closed", schema_passed and incident_protocol_passed)
    )
    deployment_eval = bool((deployment_audit_report or {}).get("evaluated", switching_eval and calibration_eval))
    deployment_closed = bool((deployment_audit_report or {}).get("closed", False))

    return {
        "1_true_multimodal_incident_detection": {
            "engineering_closed": bool(incident_protocol_passed and no_patient_overlap),
            "empirical_closed": bool(
                incident_protocol_passed and no_patient_overlap and empirical_labels_ok and incident_detection_ok
            ),
            "blocker": None
            if empirical_labels_ok and incident_detection_ok
            else (
                "multimodal held-out incident detection target not met"
                if empirical_labels_ok
                else "validation/test splits do not contain both labels"
            ),
        },
        "2_autonomous_zero_misfire_adaptation": {
            "engineering_closed": bool(category_eval),
            "empirical_closed": bool(category_improves_gate or stress_closed),
            "blocker": None
            if category_improves_gate or stress_closed
            else "category-aware gate did not materially reduce global gate firing on natural clinical variation",
        },
        "3_universal_system_integration": {
            "engineering_closed": bool(schema_passed and integration_closed),
            "empirical_closed": bool(integration_closed and incident_protocol_passed),
            "blocker": None
            if integration_closed
            else "clinical projection failed fusion-schema integration checks",
        },
        "4_deployable_auditable_ai": {
            "engineering_closed": bool(deployment_eval and switching_eval and calibration_eval),
            "empirical_closed": bool(deployment_closed or (switching_certified and empirical_labels_ok)),
            "blocker": None
            if deployment_closed or (switching_certified and empirical_labels_ok)
            else "local deployment audit, switching certificate, or label diversity is insufficient",
        },
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def run_healthcare_gap_validation(
    data_root: str | Path = "data/raw/healthcare/gridpulse",
    report_path: str | Path | None = "experiments/fusion/healthcare_gap_validation.json",
    fusion_output_path: str | Path | None = "experiments/fusion/healthcare_clinical_fusion_inputs.csv",
    split_strategy: str = "provided",
    seed: int = 42,
) -> dict[str, Any]:
    original_split_frame = load_processed_healthcare_splits(data_root)
    temporal_reference = temporal_reference_protocol_report(original_split_frame)
    split_frame = original_split_frame
    if split_strategy == "patient_stratified":
        split_frame = assign_patient_stratified_splits(split_frame, seed=seed)
    elif split_strategy != "provided":
        raise ValueError("split_strategy must be 'provided' or 'patient_stratified'.")
    fusion = build_clinical_fusion_frame(split_frame)

    schema_stats = validate_fusion_schema(
        fusion,
        id_column="sample_id",
        domain_column="domain",
        score_column="score",
        label_column="label",
        confidence_column="confidence",
    )
    incident_protocol = validate_incident_protocol(
        fusion,
        incident_column="incident_id",
        domain_column="domain",
        timestamp_column="timestamp",
        split_column="fusion_split",
        label_column="label",
        min_domains_per_incident=len(DOMAIN_SPECS),
        require_temporal_order=split_strategy == "provided",
    )
    labels = label_diversity_report(fusion)
    patient_overlap = patient_overlap_report(fusion)
    incident_detection = clinical_incident_detection_report(fusion)
    category_report = category_aware_reliability_report(fusion)
    stress_report = gap2_reliability_stress_report(fusion)
    switching_report = switching_and_calibration_report(fusion)
    calibration_report = switching_report.get("calibration", {})
    integration_report = fusion_schema_integration_report(fusion, schema_stats, incident_protocol)
    deployment_audit = gap4_deployment_audit_report(
        fusion,
        data_root=data_root,
        temporal_reference=temporal_reference,
        reliability_stress_report=stress_report,
        switching_report=switching_report,
        label_report=labels,
    )
    statuses = gap_statuses(
        incident_protocol_passed=True,
        label_report=labels,
        patient_overlap=patient_overlap,
        schema_passed=True,
        category_aware_report=category_report,
        switching_report=switching_report,
        calibration_report=calibration_report,
        incident_detection_report=incident_detection,
        reliability_stress_report=stress_report,
        schema_integration_report=integration_report,
        deployment_audit_report=deployment_audit,
    )

    report = {
        "data_root": str(data_root),
        "split_strategy": split_strategy,
        "temporal_order_required": split_strategy == "provided",
        "seed": int(seed),
        "source_rows": int(len(split_frame)),
        "fusion_rows": int(len(fusion)),
        "fusion_incidents": int(fusion["incident_id"].nunique()),
        "domains": [spec.name for spec in DOMAIN_SPECS],
        "claim_boundary": (
            "Retrospective local healthcare replay using hashed patient/incident identifiers; "
            "not prospective clinical validation, clinical decision support approval, or regulated deployment evidence."
        ),
        "schema": schema_stats,
        "schema_integration": integration_report,
        "incident_protocol": incident_protocol,
        "temporal_reference_protocol": temporal_reference,
        "label_diversity": labels,
        "patient_overlap": patient_overlap,
        "incident_detection": incident_detection,
        "category_aware_reliability": category_report,
        "reliability_stress": stress_report,
        "switching_and_calibration": switching_report,
        "deployment_audit": deployment_audit,
        "gap_statuses": statuses,
    }

    if fusion_output_path is not None:
        out = Path(fusion_output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fusion.to_csv(out, index=False)
        report["fusion_output_path"] = str(out)
    if report_path is not None:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["report_path"] = str(out)
    return _json_ready(report)
