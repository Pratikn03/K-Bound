#!/usr/bin/env python3
"""Run the locked, episode-level population-transfer procedure.

This is a small, independently sampled *procedure validation* panel.  It is
deliberately separate from benchmark replays: fit and calibration episodes
contain measured benefits, while score episodes expose only their observable
state to the decision path.  Score outcomes are retained in a sealed sidecar
for post-decision coverage scoring and are never accepted by the decision
function.

The output therefore demonstrates that the population procedure can be
instantiated under an explicit episode model.  It does not establish that a
benchmark environment is exchangeable with this synthetic population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Direct invocation from ``docs/research/kbound/scripts`` must still resolve the
# repository's released ``kga`` package rather than relying on the caller's
# working directory.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kga.population_transfer import (  # noqa: E402 - direct-script repository bootstrap
    compose_conditional_population_interval,
    hoeffding_paired_accuracy_radius,
)

SCHEMA = "kbound_locked_episode_population_v1"


def _extended_real_json(value: Any) -> Any:
    """Encode legitimate unbounded output intervals, never nonstandard JSON numbers.

    Input hashing remains strict; this encoding is only for decision outputs.
    NaN is not an interval endpoint and still fails strict JSON serialization.
    """
    if isinstance(value, float) and math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: _extended_real_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_extended_real_json(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _probability(z: int) -> float:
    return {-1: 0.2, 0: 0.5, 1: 0.8}[int(z)]


def _population_benefit(z: int) -> float:
    return 0.6 * float(z)


def _episode(rng: np.random.Generator, episode_id: str, m: int, *, include_outcome: bool) -> dict[str, Any]:
    """Sample one episode; its state is drawn once, then held fixed for m draws."""

    z = int(rng.choice(np.asarray([-1, 0, 1], dtype=int)))
    y = rng.binomial(1, _probability(z), size=int(m))
    benefit = float(np.mean(2 * y - 1))
    row: dict[str, Any] = {
        "episode_id": str(episode_id),
        "Z": [float(z)],
        "sample_size": int(m),
    }
    if include_outcome:
        row["benefit"] = benefit
        row["population_benefit"] = _population_benefit(z)
    return row


def generate_locked_panel(
    *, seed: int, n_fit: int = 24, n_cal: int = 39, n_score: int = 24, m: int = 128
) -> dict[str, Any]:
    """Generate independent fit/calibration/score episode pools.

    A ``SeedSequence`` gives each role an independent child stream.  Score
    outcomes are placed in a separate sealed sidecar and are not copied into
    ``score_episodes``.
    """

    if min(int(n_fit), int(n_cal), int(n_score), int(m)) < 1:
        raise ValueError("episode and sample counts must be positive")
    root = np.random.SeedSequence(int(seed))
    fit_rng, cal_rng, score_rng = [np.random.default_rng(child) for child in root.spawn(3)]
    fit = [_episode(fit_rng, f"fit-{i:04d}", m, include_outcome=True) for i in range(int(n_fit))]
    calibration = [_episode(cal_rng, f"cal-{i:04d}", m, include_outcome=True) for i in range(int(n_cal))]
    score_full = [_episode(score_rng, f"score-{i:04d}", m, include_outcome=True) for i in range(int(n_score))]
    score = [
        {key: value for key, value in row.items() if key not in {"benefit", "population_benefit"}} for row in score_full
    ]
    sealed_outcomes = {row["episode_id"]: row["benefit"] for row in score_full}
    sealed_population = {row["episode_id"]: row["population_benefit"] for row in score_full}
    provenance = {
        "schema": "kbound_locked_episode_provenance_v1",
        "root_seed": int(seed),
        "child_streams": ["fit", "calibration", "score"],
        "episode_level_state": "one Z state sampled once and held fixed per episode",
        "episode_sample_size": int(m),
        "sampling_model": "Z uniform on {-1,0,1}; conditional Bernoulli draws independent within an episode",
        "exchangeability_assumption": (
            "fit/calibration/score episodes are treated as independent draws from the declared episode law; "
            "this assumption is explicit and is not inferred for any benchmark"
        ),
        "score_outcomes_sealed_before_decision": True,
        "decision_input_excludes": ["benefit", "population_benefit", "labels", "targets"],
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    return {
        "schema": SCHEMA,
        "seed": int(seed),
        "fit_episodes": fit,
        "calibration_episodes": calibration,
        "score_episodes": score,
        "sealed_score_outcomes": sealed_outcomes,
        "sealed_population_benefits": sealed_population,
        "provenance": provenance,
    }


def _fit_predictor(fit: list[dict[str, Any]]) -> tuple[float, float]:
    x = np.asarray([[row["Z"][0], 1.0] for row in fit], dtype=float)
    y = np.asarray([row["benefit"] for row in fit], dtype=float)
    if x.shape[0] < 2 or np.linalg.matrix_rank(x) < 2:
        raise ValueError("fit episodes do not identify a two-parameter predictor")
    slope, intercept = np.linalg.lstsq(x, y, rcond=None)[0]
    return float(slope), float(intercept)


def _rank_radius(residuals: list[float], alpha: float) -> float:
    ordered = np.sort(np.asarray(residuals, dtype=float))
    rank = int(math.ceil((len(ordered) + 1) * (1.0 - float(alpha))))
    if rank > len(ordered):
        return float("inf")
    return float(ordered[rank - 1])


def run_locked_population(
    panel: dict[str, Any],
    *,
    alpha_cell: float = 0.05,
    delta_sampling: float = 0.05,
    alpha_population: float = 0.10,
) -> dict[str, Any]:
    """Fit, calibrate, and score a locked panel without reading score outcomes."""

    fit = panel["fit_episodes"]
    calibration = panel["calibration_episodes"]
    score = panel["score_episodes"]
    seen_episodes: set[str] = set()
    for role in (fit, calibration, score):
        for row in role:
            episode_id = row["episode_id"]
            if episode_id in seen_episodes:
                raise ValueError("episode ID overlap or duplicate across fit/calibration/score roles")
            seen_episodes.add(episode_id)
    if any("benefit" not in row for row in fit + calibration):
        raise ValueError("fit and calibration episodes require measured benefits")
    if any("benefit" in row or "population_benefit" in row for row in score):
        raise ValueError("score decision inputs must be outcome-blind")
    slope, intercept = _fit_predictor(fit)
    cal_pred = [slope * row["Z"][0] + intercept for row in calibration]
    residuals = [abs(pred - row["benefit"]) for pred, row in zip(cal_pred, calibration, strict=True)]
    epsilon = _rank_radius(residuals, alpha_cell)
    m_values = {int(row["sample_size"]) for row in score}
    if len(m_values) != 1:
        raise ValueError("all score episodes must use one locked sample size")
    m = next(iter(m_values))
    sampling_radius = hoeffding_paired_accuracy_radius(n=m, delta=delta_sampling)
    decisions: list[dict[str, Any]] = []
    for row in score:
        prediction = float(slope * row["Z"][0] + intercept)
        interval = compose_conditional_population_interval(
            delta_hat=prediction,
            epsilon=epsilon,
            r_samp=sampling_radius,
            alpha_cell=alpha_cell,
            delta_sampling=delta_sampling,
            alpha_population=alpha_population,
        )
        decisions.append(
            {
                "episode_id": row["episode_id"],
                "prediction": prediction,
                "cell_radius": float(epsilon),
                "sampling_radius": float(sampling_radius),
                "population_radius": float(interval.population_radius),
                "threshold": {
                    "lower": float(prediction - interval.population_radius),
                    "upper": float(prediction + interval.population_radius),
                },
                "action": interval.action.value,
            }
        )
    outcomes = panel.get("sealed_score_outcomes", {})
    population = panel.get("sealed_population_benefits", {})
    scored = []
    for row in decisions:
        outcome = float(outcomes[row["episode_id"]])
        truth = float(population[row["episode_id"]])
        scored.append(
            {
                **row,
                "measured_benefit": outcome,
                "population_benefit": truth,
                "population_interval_covers": abs(row["prediction"] - truth) <= row["population_radius"],
            }
        )
    return {
        "schema": "kbound_locked_episode_population_result_v1",
        "protocol": {
            "status": "PASS",
            "decision_is_outcome_blind": True,
            "fit_count": len(fit),
            "calibration_count": len(calibration),
            "score_count": len(score),
            "alpha_cell": float(alpha_cell),
            "delta_sampling": float(delta_sampling),
            "alpha_population": float(alpha_population),
            "predictor": "least-squares affine predictor on episode state Z using fit outcomes only",
            "calibration": "exact split-conformal rank on calibration episodes only",
            "population_interval": "cell radius plus Hoeffding radius for fresh within-episode draws",
            "extended_real_json_encoding": "output infinities are strings +inf and -inf; NaN is rejected",
        },
        "input_sha256": sha256_json({"fit": fit, "calibration": calibration, "score": score}),
        "decisions": decisions,
        "scored_outcomes": scored,
        "decision_path_sha256": sha256_json(_extended_real_json(decisions)),
    }


def outcome_blind_perturbation(
    panel: dict[str, Any],
    *,
    alpha_cell: float = 0.05,
    delta_sampling: float = 0.05,
    alpha_population: float = 0.10,
) -> dict[str, Any]:
    """Replay after changing one sealed score outcome and compare full paths."""

    budgets = {"alpha_cell": alpha_cell, "delta_sampling": delta_sampling, "alpha_population": alpha_population}
    baseline = run_locked_population(panel, **budgets)
    mutated = json.loads(json.dumps(panel))
    key = sorted(mutated["sealed_score_outcomes"])[0]
    original = float(mutated["sealed_score_outcomes"][key])
    mutated["sealed_score_outcomes"][key] = -1.0 if original >= 0 else 1.0
    changed = run_locked_population(mutated, **budgets)
    same = baseline["decisions"] == changed["decisions"]
    return {
        "schema": "kbound_outcome_blind_perturbation_v1",
        "status": "PASS" if same else "FAIL",
        "perturbed_episode_id": key,
        "original_outcome": original,
        "perturbed_outcome": mutated["sealed_score_outcomes"][key],
        "compared_fields": ["prediction", "cell_radius", "population_radius", "threshold", "action"],
        "decision_path_unchanged": same,
        "before_sha256": baseline["decision_path_sha256"],
        "after_sha256": changed["decision_path_sha256"],
    }


def write_panel(out_dir: Path, panel: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("population evidence requires a fresh or empty output directory")
    budgets = {key: result["protocol"][key] for key in ("alpha_cell", "delta_sampling", "alpha_population")}
    verified = run_locked_population(panel, **budgets)
    if result != verified:
        raise ValueError("population result does not match its input panel and risk budgets")
    perturbation = outcome_blind_perturbation(panel, **budgets)
    if perturbation["status"] != "PASS" or perturbation["before_sha256"] != result["decision_path_sha256"]:
        raise ValueError("outcome-blind verification does not match the saved decision path")
    out_dir.mkdir(parents=True, exist_ok=True)
    decision_input = {
        "schema": panel["schema"],
        "seed": panel["seed"],
        "fit_episodes": panel["fit_episodes"],
        "calibration_episodes": panel["calibration_episodes"],
        "score_episodes": panel["score_episodes"],
        "provenance": panel["provenance"],
    }
    # Exclusive creation claims the output before any evidence files are written;
    # an overlapping writer cannot bypass the earlier empty-directory check.
    with (out_dir / "POPULATION_LOCK_MANIFEST.json").open("x", encoding="utf-8") as lock_file:
        lock_file.write(json.dumps(panel["provenance"], indent=2) + "\n")
    (out_dir / "population_panel_input.json").write_text(json.dumps(decision_input, indent=2) + "\n", encoding="utf-8")
    (out_dir / "population_panel_output.json").write_text(
        json.dumps(_extended_real_json(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out_dir / "OUTCOME_BLIND_PERTURBATION_RECEIPT.json").write_text(
        json.dumps(perturbation, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "sealed_score_outcomes.json").write_text(
        json.dumps(
            {
                "schema": "kbound_sealed_score_outcomes_v1",
                "outcomes": panel["sealed_score_outcomes"],
                "population_benefits": panel["sealed_population_benefits"],
                "sha256": sha256_json(
                    {
                        "outcomes": panel["sealed_score_outcomes"],
                        "population_benefits": panel["sealed_population_benefits"],
                    }
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "kbound_locked_population_receipt_v1",
        "status": "PASS",
        "scope": "synthetic independently sampled episode-level procedure validation; no benchmark exchangeability claim",
        "decision_input_sha256": sha256_json(decision_input),
        "decision_path_sha256": result["decision_path_sha256"],
        "files": {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in (
                "POPULATION_LOCK_MANIFEST.json",
                "population_panel_input.json",
                "population_panel_output.json",
                "sealed_score_outcomes.json",
                "OUTCOME_BLIND_PERTURBATION_RECEIPT.json",
            )
        },
        "outcome_blind_perturbation": "verified by test_task2_research_closure.py",
        "outcome_blind_perturbation_receipt": "OUTCOME_BLIND_PERTURBATION_RECEIPT.json",
    }
    (out_dir / "POPULATION_PROVENANCE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--n-fit", type=int, default=24)
    parser.add_argument("--n-cal", type=int, default=39)
    parser.add_argument("--n-score", type=int, default=24)
    parser.add_argument("--m", type=int, default=128)
    args = parser.parse_args()
    panel = generate_locked_panel(seed=args.seed, n_fit=args.n_fit, n_cal=args.n_cal, n_score=args.n_score, m=args.m)
    result = run_locked_population(panel)
    receipt = write_panel(args.out_dir, panel, result)
    print(json.dumps({"status": "PASS", "out_dir": str(args.out_dir), "receipt": receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
