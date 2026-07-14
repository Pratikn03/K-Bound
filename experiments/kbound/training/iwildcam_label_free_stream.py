#!/usr/bin/env python3
"""Fit, calibrate, and replay an iWildCam stream gate without target labels.

Development and calibration windows may use their revealed offline benefits.
Held-out evidence and outcomes must be supplied in different files. Decisions
are serialized before the outcome files are opened for offline evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from uniform_multiseed import GBR_CONFIG, route_without_target_labels

FEATURES = (
    "tent_mean_entropy",
    "tent_pred_hist_entropy",
    "tent_pred_n_unique",
    "tent_mean_grad_l2",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_key(window: dict, index: int) -> tuple[str, int]:
    stream = str(window.get("session_id", window.get("stream_id", "")))
    endpoint = int(window.get("window_end_batch", index))
    return stream, endpoint


def load_evidence(paths: list[Path]) -> tuple[np.ndarray, list[dict], list[tuple[str, int]]]:
    evidence = []
    identities = []
    keys = []
    for path in paths:
        payload = json.loads(path.read_text())
        windows = payload.get("windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"{path}: missing non-empty windows")
        for index, window in enumerate(windows):
            evidence.append([float(window[name]) for name in FEATURES])
            key = window_key(window, index)
            keys.append(key)
            identities.append({
                "source": str(path),
                "source_sha256": sha256(path),
                "window_index": index,
                "session_id": key[0],
                "window_end_batch": key[1],
            })
    return np.asarray(evidence, dtype=float), identities, keys


def load_outcomes(paths: list[Path], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    frozen = []
    adapted = []
    keys = []
    frozen_key = f"frozen_window_{metric}"
    adapted_key = f"tent_window_{metric}"
    for path in paths:
        payload = json.loads(path.read_text())
        windows = payload.get("windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"{path}: missing non-empty windows")
        for index, window in enumerate(windows):
            frozen.append(float(window[frozen_key]))
            adapted.append(float(window[adapted_key]))
            keys.append(window_key(window, index))
    frozen_array = np.asarray(frozen, dtype=float)
    adapted_array = np.asarray(adapted, dtype=float)
    return adapted_array - frozen_array, frozen_array, adapted_array, keys


def load_training_windows(paths: list[Path], metric: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    evidence, identities, evidence_keys = load_evidence(paths)
    benefit, _, _, outcome_keys = load_outcomes(paths, metric)
    if evidence_keys != outcome_keys:
        raise ValueError("training evidence and outcomes are misaligned")
    return evidence, benefit, identities


def ensure_disjoint(*groups: list[Path]) -> None:
    resolved = [[path.resolve() for path in group] for group in groups]
    flattened = [path for group in resolved for path in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("development, calibration, and held-out files must be disjoint")


def offline_score(decisions: np.ndarray, benefit: np.ndarray, frozen: np.ndarray, adapted: np.ndarray) -> dict:
    adapt = decisions == "ADAPT"
    committed = decisions != "ABSTAIN"
    chosen = np.where(adapt, adapted, frozen)
    oracle = np.maximum(frozen, adapted)
    harmful = benefit <= 0.0
    return {
        "n_windows": int(len(benefit)),
        "regret": {
            "kga": float(np.mean(oracle - chosen)),
            "always_adapt": float(np.mean(oracle - adapted)),
            "always_freeze": float(np.mean(oracle - frozen)),
        },
        "FA_u": float(np.mean(adapt & harmful)),
        "FA_c": float(np.mean(harmful[adapt])) if np.any(adapt) else 0.0,
        "adapt_rate": float(np.mean(adapt)),
        "decision_coverage": float(np.mean(committed)),
        "mean_metric": {
            "kga": float(np.mean(chosen)),
            "always_adapt": float(np.mean(adapted)),
            "always_freeze": float(np.mean(frozen)),
            "oracle": float(np.mean(oracle)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, nargs="+", required=True)
    parser.add_argument("--heldout-evidence", type=Path, nargs="+", required=True)
    parser.add_argument("--heldout-outcomes", type=Path, nargs="+", required=True)
    parser.add_argument("--metric", choices=["f1", "acc"], default="f1")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-output", type=Path)
    args = parser.parse_args(argv)
    ensure_disjoint(
        args.development,
        args.calibration,
        args.heldout_evidence,
        args.heldout_outcomes,
    )

    z_dev, b_dev, dev_ids = load_training_windows(args.development, args.metric)
    z_cal, b_cal, cal_ids = load_training_windows(args.calibration, args.metric)
    z_test, test_ids, test_keys = load_evidence(args.heldout_evidence)
    estimator = GradientBoostingRegressor(**GBR_CONFIG).fit(z_dev, b_dev)
    residuals = np.abs(estimator.predict(z_cal) - b_cal)

    decisions, estimates, epsilon = route_without_target_labels(
        estimator, z_test, residuals, args.alpha
    )
    live_rows = [
        {
            **identity,
            "features": {name: float(value) for name, value in zip(FEATURES, z_test[index])},
            "delta_hat": float(estimates[index]),
            "epsilon": float(epsilon),
            "decision": str(decisions[index]),
        }
        for index, identity in enumerate(test_ids)
    ]
    live_result = {
        "schema": "iwildcam_label_free_live_decisions_v1",
        "alpha": args.alpha,
        "features": list(FEATURES),
        "calibration_radius": float(epsilon),
        "decisions": live_rows,
        "outcomes_opened": False,
    }
    live_output = args.live_output or args.output.with_name(
        f"{args.output.stem}_LIVE{args.output.suffix}"
    )
    live_output.parent.mkdir(parents=True, exist_ok=True)
    live_output.write_text(json.dumps(live_result, indent=2) + "\n")
    live_sha = sha256(live_output)

    b_test, frozen_test, adapted_test, outcome_keys = load_outcomes(
        args.heldout_outcomes, args.metric
    )
    if test_keys != outcome_keys:
        raise ValueError("held-out evidence and outcomes are misaligned")

    result = {
        "schema": "iwildcam_label_free_stream_gate_v1",
        "alpha": args.alpha,
        "metric": args.metric,
        "features": list(FEATURES),
        "split_integrity": {
            "development": dev_ids,
            "calibration": cal_ids,
            "heldout": test_ids,
            "heldout_evidence_files": [str(path) for path in args.heldout_evidence],
            "heldout_outcome_files": [str(path) for path in args.heldout_outcomes],
            "files_disjoint": True,
        },
        "benefit_estimator": {"class": "GradientBoostingRegressor", **GBR_CONFIG},
        "calibration": {
            "n": int(len(residuals)),
            "radius": float(epsilon),
            "rule": "exact finite-sample residual order statistic",
        },
        "live_decisions_artifact": str(live_output),
        "live_decisions_sha256_before_outcome_open": live_sha,
        "offline_evaluation": offline_score(
            decisions, b_test, frozen_test, adapted_test
        ),
        "scope": (
            "Held-out decisions use only the fixed feature vector, fitted estimator, "
            "and calibration radius. The live artifact is written before the separate "
            "held-out outcome files are opened for offline evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
