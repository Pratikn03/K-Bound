#!/usr/bin/env python3
"""Calibration-size, batch-regime, and architecture sensitivity analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from uniform_multiseed import (
    evaluate_method,
    fit_estimator,
    hierarchical_ci,
    load_cells,
    route_without_target_labels,
    score,
)


def summarize(folds: list[dict], alpha: float, nboot: int) -> dict:
    regret = {
        policy: float(np.mean([fold["regret"][policy] for fold in folds]))
        for policy in ("kga", "always_adapt", "always_freeze")
    }
    gain_adapt = hierarchical_ci(folds, "gain_vs_adapt", nboot, seed=1729)
    gain_freeze = hierarchical_ci(folds, "gain_vs_freeze", nboot, seed=2718)
    fa_u_interval = hierarchical_ci(folds, "false_adapt_indicator", nboot, seed=3141)
    fa_u = float(np.mean([fold["FA_u"] for fold in folds]))
    return {
        "regret": regret,
        "FA_u": fa_u,
        "FA_c": float(np.mean([fold["FA_c"] for fold in folds])),
        "adapt_rate": float(np.mean([fold["adapt_rate"] for fold in folds])),
        "decision_coverage": float(np.mean([fold["decision_coverage"] for fold in folds])),
        "gain_vs_adapt": gain_adapt,
        "gain_vs_freeze": gain_freeze,
        "FA_u_interval": fa_u_interval,
        "beats_both_point": (
            regret["kga"] < regret["always_adapt"]
            and regret["kga"] < regret["always_freeze"]
            and fa_u <= alpha
        ),
        "beats_both_gain_ci": (
            gain_adapt["ci95"][0] > 0.0
            and gain_freeze["ci95"][0] > 0.0
            and fa_u <= alpha
        ),
        "beats_both_ci_robust": (
            gain_adapt["ci95"][0] > 0.0
            and gain_freeze["ci95"][0] > 0.0
            and fa_u_interval["ci95"][1] <= alpha
        ),
    }


def _condition_has_regime(condition: str, regime: str) -> bool:
    return regime in condition.split("|")


def variant_folds(
    run_dir: Path,
    dataset: str,
    method: str,
    seeds: list[int],
    alpha: float,
    calibration_size: int | None = None,
    batch_regime: str | None = None,
) -> list[dict]:
    loaded = {seed: load_cells(run_dir, dataset, method, seed) for seed in seeds}
    folds = []
    for index, test_seed in enumerate(seeds):
        calibration_seed = seeds[(index + 1) % len(seeds)]
        fit_seeds = [seed for seed in seeds if seed not in (test_seed, calibration_seed)]
        estimator = fit_estimator([loaded[seed] for seed in fit_seeds])
        calibration = loaded[calibration_seed]
        residuals = np.abs(estimator.predict(calibration.z) - calibration.benefit)
        if calibration_size is not None and calibration_size < len(residuals):
            rng = np.random.default_rng(10000 + test_seed + calibration_size)
            chosen = np.sort(rng.choice(len(residuals), calibration_size, replace=False))
            residuals = residuals[chosen]
        target = loaded[test_seed]
        if batch_regime is not None:
            keep = np.asarray([_condition_has_regime(item, batch_regime) for item in target.conditions])
            if not np.any(keep):
                raise ValueError(f"no {batch_regime} conditions in {target.source}")
            target = type(target)(
                target.seed,
                target.method,
                tuple(np.asarray(target.conditions, dtype=object)[keep]),
                target.z_names,
                target.z[keep],
                target.benefit[keep],
                target.frozen[keep],
                target.adapted[keep],
                target.source,
            )
        decisions, _, epsilon = route_without_target_labels(estimator, target.z, residuals, alpha)
        folds.append({
            "test_seed": test_seed,
            "calibration_seed": calibration_seed,
            "calibration_n": int(len(residuals)),
            "epsilon": epsilon,
            **score(decisions, target),
        })
    return folds


def parse_architecture(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise argparse.ArgumentTypeError("architecture must be NAME=RUN_DIR")
    name, path = item.split("=", 1)
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--nboot", type=int, default=5000)
    parser.add_argument("--calibration-sizes", type=int, nargs="*", default=[16, 32, 64, 128])
    parser.add_argument("--batch-regimes", nargs="*", default=["large_iid", "small", "tiny"])
    parser.add_argument("--architecture", action="append", default=[], type=parse_architecture)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if len(args.seeds) < 3:
        raise SystemExit("at least three seeds are required")

    baseline = evaluate_method(
        args.run_dir, args.dataset, args.method, args.seeds, args.alpha, args.nboot
    )
    output = {
        "schema": "kbound_multiseed_ablation_v1",
        "dataset": args.dataset,
        "method": args.method,
        "seeds": args.seeds,
        "alpha": args.alpha,
        "baseline": baseline["pooled"],
        "calibration_size": {},
        "batch_regime": {},
        "architecture": {},
    }
    for size in args.calibration_sizes:
        folds = variant_folds(
            args.run_dir, args.dataset, args.method, args.seeds, args.alpha, calibration_size=size
        )
        output["calibration_size"][str(size)] = summarize(folds, args.alpha, args.nboot)
    for regime in args.batch_regimes:
        try:
            folds = variant_folds(
                args.run_dir, args.dataset, args.method, args.seeds, args.alpha, batch_regime=regime
            )
        except ValueError as exc:
            output["batch_regime"][regime] = {"status": "not_present", "reason": str(exc)}
        else:
            output["batch_regime"][regime] = summarize(folds, args.alpha, args.nboot)
    for name, path in args.architecture:
        output["architecture"][name] = evaluate_method(
            path, args.dataset, args.method, args.seeds, args.alpha, args.nboot
        )["pooled"]
    destination = args.output or args.run_dir / "MULTISEED_ABLATIONS.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
