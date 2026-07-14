#!/usr/bin/env python3
"""Uniform held-out-seed KGA scoring for every raw-data runner.

For each target seed, the next seed is used only for residual calibration and
all remaining seeds fit the benefit regressor. The target seed is routed once.
This gives every supported dataset the same fit/calibrate/test logic and keeps
target benefits out of the live decision function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from calibration import exact_rank_radius
from sklearn.ensemble import GradientBoostingRegressor

GBR_CONFIG = {
    "n_estimators": 250,
    "max_depth": 2,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 0,
}


@dataclass(frozen=True)
class Cells:
    seed: int
    method: str
    conditions: tuple[str, ...]
    z_names: tuple[str, ...]
    z: np.ndarray
    benefit: np.ndarray
    frozen: np.ndarray
    adapted: np.ndarray
    source: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_paths(run_dir: Path, dataset: str, method: str, seed: int) -> tuple[Path, ...]:
    name = f"per_condition_{dataset}_{method}_seed{seed}.json"
    return run_dir / f"seed{seed}" / name, run_dir / name


def load_cells(run_dir: Path, dataset: str, method: str, seed: int) -> Cells:
    path = next((item for item in _candidate_paths(run_dir, dataset, method, seed) if item.is_file()), None)
    if path is None:
        searched = ", ".join(str(item) for item in _candidate_paths(run_dir, dataset, method, seed))
        raise FileNotFoundError(f"missing seed artifact; searched: {searched}")
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: records must be a non-empty list")
    z = np.asarray([row["Z"] for row in records], dtype=float)
    benefit = np.asarray([row["B"] for row in records], dtype=float)
    frozen = np.asarray([row["a0"] for row in records], dtype=float)
    adapted = np.asarray([row.get("a_adapted", row.get("aa")) for row in records], dtype=float)
    conditions = tuple(str(row["condition"]) for row in records)
    z_names = tuple(str(name) for name in records[0].get("Z_names", ()))
    if z.ndim != 2 or len(z) != len(benefit):
        raise ValueError(f"{path}: invalid evidence shape {z.shape}")
    if len(set(conditions)) != len(conditions):
        raise ValueError(f"{path}: duplicate condition identifiers")
    if not all(tuple(str(name) for name in row.get("Z_names", ())) == z_names for row in records):
        raise ValueError(f"{path}: evidence names differ within the file")
    if z_names and len(z_names) != z.shape[1]:
        raise ValueError(f"{path}: evidence names do not match evidence width")
    arrays = (z, benefit, frozen, adapted)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError(f"{path}: non-finite evidence or outcomes")
    if not np.allclose(benefit, adapted - frozen, rtol=1e-7, atol=1e-9):
        raise ValueError(f"{path}: B is inconsistent with a_adapted - a0")
    return Cells(seed, method, conditions, z_names, z, benefit, frozen, adapted, path)


def discover_methods(run_dir: Path, dataset: str, seeds: list[int]) -> list[str]:
    methods: set[str] = set()
    for seed in seeds:
        for directory in (run_dir, run_dir / f"seed{seed}"):
            prefix = f"per_condition_{dataset}_"
            suffix = f"_seed{seed}.json"
            for path in directory.glob(f"{prefix}*{suffix}"):
                methods.add(path.name[len(prefix) : -len(suffix)])
    return sorted(methods)


def fit_estimator(cells: list[Cells]) -> GradientBoostingRegressor:
    z = np.concatenate([item.z for item in cells], axis=0)
    benefit = np.concatenate([item.benefit for item in cells], axis=0)
    return GradientBoostingRegressor(**GBR_CONFIG).fit(z, benefit)


def route_without_target_labels(
    estimator: GradientBoostingRegressor,
    target_z: np.ndarray,
    calibration_residuals: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Make target decisions from Z and precomputed calibration residuals only."""
    estimates = estimator.predict(np.asarray(target_z, dtype=float))
    epsilon = exact_rank_radius(calibration_residuals, alpha)
    decisions = np.full(len(estimates), "ABSTAIN", dtype=object)
    decisions[estimates - epsilon > 0.0] = "ADAPT"
    decisions[estimates + epsilon < 0.0] = "FREEZE"
    return decisions, estimates, epsilon


def score(decisions: np.ndarray, cells: Cells) -> dict:
    adapt = decisions == "ADAPT"
    committed = decisions != "ABSTAIN"
    chosen = np.where(adapt, cells.adapted, cells.frozen)
    oracle = np.maximum(cells.frozen, cells.adapted)
    harmful = cells.benefit <= 0.0
    return {
        "n": int(len(cells.benefit)),
        "regret": {
            "kga": float(np.mean(oracle - chosen)),
            "always_adapt": float(np.mean(oracle - cells.adapted)),
            "always_freeze": float(np.mean(oracle - cells.frozen)),
        },
        "mean_accuracy": {
            "kga": float(np.mean(chosen)),
            "always_adapt": float(np.mean(cells.adapted)),
            "always_freeze": float(np.mean(cells.frozen)),
            "oracle": float(np.mean(oracle)),
        },
        "FA_u": float(np.mean(adapt & harmful)),
        "FA_c": float(np.mean(harmful[adapt])) if np.any(adapt) else 0.0,
        "adapt_rate": float(np.mean(adapt)),
        "decision_coverage": float(np.mean(committed)),
        "abstain_rate": float(np.mean(~committed)),
        "decision_counts": {name: int(np.sum(decisions == name)) for name in ("ADAPT", "FREEZE", "ABSTAIN")},
        "false_adapt_indicator": (adapt & harmful).astype(float).tolist(),
        "gain_vs_adapt": (chosen - cells.adapted).tolist(),
        "gain_vs_freeze": (chosen - cells.frozen).tolist(),
    }


def hierarchical_ci(folds: list[dict], field: str, nboot: int, seed: int) -> dict:
    values = [np.asarray(item[field], dtype=float) for item in folds]
    observed = float(np.mean(np.concatenate(values)))
    rng = np.random.default_rng(seed)
    draws = np.empty(nboot, dtype=float)
    for index in range(nboot):
        selected = rng.integers(0, len(values), len(values))
        sample = []
        for fold_index in selected:
            fold = values[int(fold_index)]
            sample.append(fold[rng.integers(0, len(fold), len(fold))])
        draws[index] = float(np.mean(np.concatenate(sample)))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean": observed, "ci95": [float(low), float(high)]}


def evaluate_method(
    run_dir: Path,
    dataset: str,
    method: str,
    seeds: list[int],
    alpha: float,
    nboot: int,
) -> dict:
    if len(seeds) < 3:
        raise ValueError("uniform held-out-seed evaluation requires at least three seeds")
    loaded = {seed: load_cells(run_dir, dataset, method, seed) for seed in seeds}
    dimensions = {cells.z.shape[1] for cells in loaded.values()}
    if len(dimensions) != 1:
        raise ValueError(f"{dataset}/{method}: evidence dimensions differ across seeds: {dimensions}")
    condition_schemas = {cells.conditions for cells in loaded.values()}
    if len(condition_schemas) != 1:
        raise ValueError(f"{dataset}/{method}: condition identifiers or order differ across seeds")
    evidence_schemas = {cells.z_names for cells in loaded.values()}
    if len(evidence_schemas) != 1:
        raise ValueError(f"{dataset}/{method}: evidence names differ across seeds")
    folds = []
    for index, test_seed in enumerate(seeds):
        calibration_seed = seeds[(index + 1) % len(seeds)]
        fit_seeds = [seed for seed in seeds if seed not in (test_seed, calibration_seed)]
        estimator = fit_estimator([loaded[seed] for seed in fit_seeds])
        calibration = loaded[calibration_seed]
        residuals = np.abs(estimator.predict(calibration.z) - calibration.benefit)
        target = loaded[test_seed]
        decisions, estimates, epsilon = route_without_target_labels(estimator, target.z, residuals, alpha)
        metrics = score(decisions, target)
        folds.append({
            "test_seed": test_seed,
            "calibration_seed": calibration_seed,
            "fit_seeds": fit_seeds,
            "epsilon": epsilon,
            "input": str(target.source),
            "input_sha256": _sha256(target.source),
            "estimates": estimates.tolist(),
            "decisions": decisions.tolist(),
            **metrics,
        })
    gain_adapt = hierarchical_ci(folds, "gain_vs_adapt", nboot, seed=731)
    gain_freeze = hierarchical_ci(folds, "gain_vs_freeze", nboot, seed=977)
    fa_u_interval = hierarchical_ci(folds, "false_adapt_indicator", nboot, seed=1223)
    pooled = {
        name: float(np.mean([fold[name] for fold in folds]))
        for name in ("FA_u", "FA_c", "adapt_rate", "decision_coverage", "abstain_rate")
    }
    pooled["regret"] = {
        policy: float(np.mean([fold["regret"][policy] for fold in folds]))
        for policy in ("kga", "always_adapt", "always_freeze")
    }
    pooled["gain_vs_adapt"] = gain_adapt
    pooled["gain_vs_freeze"] = gain_freeze
    pooled["FA_u_interval"] = fa_u_interval
    pooled["beats_both_point"] = (
        pooled["regret"]["kga"] < pooled["regret"]["always_adapt"]
        and pooled["regret"]["kga"] < pooled["regret"]["always_freeze"]
        and pooled["FA_u"] <= alpha
    )
    pooled["beats_both_gain_ci"] = (
        gain_adapt["ci95"][0] > 0.0
        and gain_freeze["ci95"][0] > 0.0
        and pooled["FA_u"] <= alpha
    )
    pooled["beats_both_ci_robust"] = (
        gain_adapt["ci95"][0] > 0.0
        and gain_freeze["ci95"][0] > 0.0
        and fa_u_interval["ci95"][1] <= alpha
    )
    return {"method": method, "seeds": seeds, "folds": folds, "pooled": pooled}


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--nboot", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    methods = args.methods or discover_methods(args.run_dir, args.dataset, args.seeds)
    if not methods:
        raise SystemExit("no complete methods discovered")
    started = time.time()
    root = Path(__file__).resolve().parents[3]
    result = {
        "schema": "kbound_uniform_multiseed_v1",
        "dataset": args.dataset,
        "alpha": args.alpha,
        "protocol": {
            "outer_split": "rotate target seed; next seed calibrates; remaining seeds fit",
            "target_label_use": "offline evaluation only; route_without_target_labels receives only Z and calibration residuals",
            "radius": "exact finite-sample rank order statistic",
            "coverage_scope": "empirical held-out-seed residual calibration; exchangeability is not asserted",
            "benefit_estimator": {"class": "GradientBoostingRegressor", **GBR_CONFIG},
            "bootstrap": "hierarchical seed-then-condition paired bootstrap",
        },
        "software": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "git_head": _git_head(root),
        "methods": {},
    }
    for method in methods:
        result["methods"][method] = evaluate_method(
            args.run_dir, args.dataset, method, args.seeds, args.alpha, args.nboot
        )
    result["wall_seconds"] = round(time.time() - started, 3)
    output = args.output or args.run_dir / "UNIFORM_MULTISEED_RESULTS.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")
    for method, record in result["methods"].items():
        pooled = record["pooled"]
        print(
            f"{method}: regret={pooled['regret']} FA_u={pooled['FA_u']:.4f} "
            f"point={pooled['beats_both_point']} gain_CI={pooled['beats_both_gain_ci']} "
            f"robust_CI={pooled['beats_both_ci_robust']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
