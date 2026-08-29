#!/usr/bin/env python3
"""Two-stage analysis for the randomized-condition exact-conformal branch.

``decide`` sees labels only for estimator-fit and residual-calibration units.
Test evidence is label free, and the written decision artifact contains no test
benefit or loss. ``evaluate`` joins a separate test-label file after decisions
have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from kga.certificate import split_conformal_rank_radius

LABEL_FIELDS = {
    "delta",
    "B",
    "risk_freeze",
    "risk_adapt",
    "oracle_action",
    "regret",
    "label",
    "labels",
    "y",
    "y_true",
    "balanced_accuracy",
    "macro_f1",
}
MODEL_CONFIG = {
    "class": "sklearn.ensemble.GradientBoostingRegressor",
    "n_estimators": 250,
    "max_depth": 2,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 0,
}


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        records = value.get("records", value) if isinstance(value, dict) else value
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"{path} must contain a list of record objects")
    return records


def load_manifest(path: Path, *, require_sealed: bool) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if expected != canonical_sha256(body):
        raise ValueError("manifest hash mismatch")
    if require_sealed and manifest.get("status") != "SEALED":
        raise ValueError("confirmation manifest is not SEALED")
    return manifest


def _matrix(records: list[dict[str, Any]], *, labels: bool) -> tuple[np.ndarray, np.ndarray | None]:
    if not records:
        raise ValueError("record set is empty")
    rows = []
    targets = []
    width = None
    for row in records:
        z = np.asarray(row.get("Z"), dtype=float)
        if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
            raise ValueError(f"invalid Z for unit {row.get('unit_id')!r}")
        width = z.size if width is None else width
        if z.size != width:
            raise ValueError("evidence vectors do not have a common dimension")
        rows.append(z)
        if labels:
            delta = row.get("delta")
            if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
                raise ValueError(f"missing finite delta for unit {row.get('unit_id')!r}")
            targets.append(float(delta))
    return np.stack(rows), np.asarray(targets, dtype=float) if labels else None


def _index_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    units = manifest.get("units", [])
    index = {unit["unit_id"]: unit for unit in units}
    if len(index) != len(units):
        raise ValueError("manifest contains duplicate unit IDs")
    return index


def _require_exact_role(
    records: list[dict[str, Any]], index: dict[str, dict[str, Any]], role: str
) -> None:
    seen = [row.get("unit_id") for row in records]
    if len(seen) != len(set(seen)):
        raise ValueError(f"duplicate unit IDs in {role} records")
    expected = {unit_id for unit_id, unit in index.items() if unit.get("role") == role}
    actual = set(seen)
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(f"{role} unit mismatch; missing={missing}, extra={extra}")


def _action(delta_hat: float, epsilon: float) -> str:
    if math.isinf(epsilon):
        return "ABSTAIN"
    if delta_hat - epsilon > 0.0:
        return "ADAPT"
    if delta_hat + epsilon < 0.0:
        return "FREEZE"
    return "ABSTAIN"


def decide(
    manifest_path: Path,
    fit_cal_path: Path,
    test_evidence_path: Path,
    *,
    require_sealed: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, require_sealed=require_sealed)
    index = _index_manifest(manifest)
    fit_cal = load_records(fit_cal_path)
    fit = [row for row in fit_cal if row.get("role") == "estimator_fit"]
    calibration = [row for row in fit_cal if row.get("role") == "residual_calibration"]
    test = load_records(test_evidence_path)
    _require_exact_role(fit, index, "estimator_fit")
    _require_exact_role(calibration, index, "residual_calibration")
    _require_exact_role(test, index, "test")
    for row in test:
        forbidden = sorted(LABEL_FIELDS & set(row))
        if forbidden:
            raise ValueError(f"test evidence contains label-bearing fields: {forbidden}")

    x_fit, y_fit = _matrix(fit, labels=True)
    x_cal, y_cal = _matrix(calibration, labels=True)
    x_test, _ = _matrix(test, labels=False)
    if x_fit.shape[1] != x_cal.shape[1] or x_fit.shape[1] != x_test.shape[1]:
        raise ValueError("fit, calibration, and test evidence dimensions differ")
    model = GradientBoostingRegressor(**{key: value for key, value in MODEL_CONFIG.items() if key != "class"})
    model.fit(x_fit, y_fit)
    calibration_predictions = model.predict(x_cal)
    residuals = np.abs(calibration_predictions - y_cal)
    alpha = float(manifest["alpha"])
    epsilon = split_conformal_rank_radius(residuals, alpha, on_infeasible="inf")
    predictions = model.predict(x_test)
    decision_rows = []
    for source, prediction in zip(test, predictions, strict=True):
        unit = index[source["unit_id"]]
        decision_rows.append(
            {
                "unit_id": source["unit_id"],
                "role": "test",
                "environment": {
                    key: unit[key]
                    for key in (
                        "corruption_family",
                        "severity",
                        "composition",
                        "batch_size",
                        "candidate",
                        "model_seed",
                    )
                },
                "delta_hat": float(prediction),
                "epsilon": epsilon,
                "action": _action(float(prediction), epsilon),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "label_free_test_decisions",
        "claim_scope": "confirmatory" if manifest.get("status") == "SEALED" else "smoke_not_evidence",
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "fit_cal_records_sha256": sha256_file(fit_cal_path),
        "test_evidence_sha256": sha256_file(test_evidence_path),
        "model_config": MODEL_CONFIG,
        "model_config_sha256": canonical_sha256(MODEL_CONFIG),
        "calibration_pool_sha256": canonical_sha256(
            [{"unit_id": row["unit_id"], "residual": float(residual)} for row, residual in zip(calibration, residuals, strict=True)]
        ),
        "alpha": alpha,
        "n_fit": len(fit),
        "n_calibration": len(calibration),
        "n_test": len(test),
        "exact_rank": int(math.ceil((len(calibration) + 1) * (1.0 - alpha))),
        "epsilon": epsilon,
        "decision_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decisions": decision_rows,
    }


def evaluate(manifest_path: Path, decisions_path: Path, test_labels_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, require_sealed=False)
    index = _index_manifest(manifest)
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if decisions.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("decision artifact refers to a different manifest")
    decision_rows = decisions.get("decisions", [])
    if any(LABEL_FIELDS & set(row) for row in decision_rows):
        raise ValueError("decision artifact contains label-bearing fields")
    labels = load_records(test_labels_path)
    _require_exact_role(labels, index, "test")
    label_by_id = {row["unit_id"]: row for row in labels}
    if {row["unit_id"] for row in decision_rows} != set(label_by_id):
        raise ValueError("decision and label unit IDs differ")

    evaluated = []
    for row in decision_rows:
        truth = label_by_id[row["unit_id"]]
        delta = float(truth["delta"])
        risk_freeze = float(truth["risk_freeze"])
        risk_adapt = float(truth["risk_adapt"])
        if not np.isclose(delta, risk_freeze - risk_adapt, atol=1e-10, rtol=1e-8):
            raise ValueError(f"delta/risk identity fails for {row['unit_id']}")
        action = row["action"]
        deployed = risk_adapt if action == "ADAPT" else risk_freeze
        oracle = min(risk_freeze, risk_adapt)
        evaluated.append(
            {
                "unit_id": row["unit_id"],
                "delta": delta,
                "risk_freeze": risk_freeze,
                "risk_adapt": risk_adapt,
                "oracle_action": "ADAPT" if delta > 0.0 else "FREEZE",
                "action": action,
                "regret": deployed - oracle,
                "false_adapt": action == "ADAPT" and delta <= 0.0,
                "interval_covers": abs(float(row["delta_hat"]) - delta) <= float(row["epsilon"]),
            }
        )
    n = len(evaluated)
    adapt = sum(row["action"] == "ADAPT" for row in evaluated)
    freeze = sum(row["action"] == "FREEZE" for row in evaluated)
    false_adapt = sum(row["false_adapt"] for row in evaluated)
    return {
        "schema_version": 1,
        "artifact_type": "offline_test_evaluation",
        "claim_scope": decisions.get("claim_scope", "unknown"),
        "manifest_sha256": manifest["manifest_sha256"],
        "decisions_sha256": sha256_file(decisions_path),
        "test_labels_sha256": sha256_file(test_labels_path),
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "n": n,
            "adapt_count": adapt,
            "freeze_count": freeze,
            "abstain_count": n - adapt - freeze,
            "adapt_rate": adapt / n,
            "decision_coverage": (adapt + freeze) / n,
            "fa_u": false_adapt / n,
            "fa_c": false_adapt / adapt if adapt else None,
            "interval_coverage": sum(row["interval_covers"] for row in evaluated) / n,
            "regret": {
                "kga": float(np.mean([row["regret"] for row in evaluated])),
                "always_adapt": float(np.mean([max(0.0, -row["delta"]) for row in evaluated])),
                "always_freeze": float(np.mean([max(0.0, row["delta"]) for row in evaluated])),
            },
            "beats_both_inference": "pending_predeclared_paired_analysis",
        },
        "records": evaluated,
    }


def write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    decide_parser = sub.add_parser("decide")
    decide_parser.add_argument("--manifest", type=Path, required=True)
    decide_parser.add_argument("--fit-cal-records", type=Path, required=True)
    decide_parser.add_argument("--test-evidence", type=Path, required=True)
    decide_parser.add_argument("--output", type=Path, required=True)
    decide_parser.add_argument("--allow-unsealed-smoke", action="store_true")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--manifest", type=Path, required=True)
    eval_parser.add_argument("--decisions", type=Path, required=True)
    eval_parser.add_argument("--test-labels", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "decide":
        payload = decide(
            args.manifest,
            args.fit_cal_records,
            args.test_evidence,
            require_sealed=not args.allow_unsealed_smoke,
        )
    else:
        payload = evaluate(args.manifest, args.decisions, args.test_labels)
    write_new(args.output, payload)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
