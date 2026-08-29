#!/usr/bin/env python3
"""Cluster-aware inference for the current exact-rank CIFAR-10-C policies.

The canonical compact panel stores one row per condition and run seed.  Run
seeds share an archived checkpoint, so they are nested repetitions rather than
independent model draws.  This analysis therefore averages the paired regret
gaps within corruption family and uses the six corruption families as the
inference units.

The reported contrast is always

    regret(fixed baseline) - regret(KGA),

so positive values favor KGA.  Percentile intervals resample whole corruption
families.  P-values are exact one-sided sign-flip tests over the family-level
paired gaps and are Holm-adjusted across the two fixed-policy comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from kga.policy import decide_kga  # noqa: E402

DEFAULT_SOURCE_DIR = (
    ROOT / "experiments/kbound/results/reconciled_panels_v1/source/cifar10c"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/kbound/results/reconciled_panels_v1/"
    "current_policy_cluster_inference.json"
)
BASELINES = ("always_adapt", "always_freeze")
CI_CONVENTION = "baseline_regret_minus_kga_regret; positive values favor KGA"
PROTOCOL_LOCK = ROOT / "research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_candidate(source_dir: Path, candidate: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = sorted(source_dir.glob(f"per_condition_cifar10c_{candidate}_seed*.json"))
    if not files:
        raise ValueError(f"no canonical files found for candidate {candidate!r} in {source_dir}")

    all_records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    expected_conditions: tuple[str, ...] | None = None
    seen_seeds: set[int] = set()

    for path in files:
        payload = json.loads(path.read_text())
        records = payload.get("records")
        metadata = payload.get("metadata", {})
        if payload.get("schema") != "kbound-compact-panel-source-v1":
            raise ValueError(f"unexpected schema in {path}")
        if not isinstance(records, list) or not records:
            raise ValueError(f"missing records in {path}")
        method = str(metadata.get("method", payload.get("method", ""))).lower()
        if method != candidate:
            raise ValueError(f"candidate mismatch in {path}: {method!r} != {candidate!r}")
        seed = int(metadata.get("seed", payload.get("seed")))
        if seed in seen_seeds:
            raise ValueError(f"duplicate run seed {seed} for {candidate}")
        seen_seeds.add(seed)

        conditions = tuple(str(row["condition"]) for row in records)
        if len(set(conditions)) != len(conditions):
            raise ValueError(f"duplicate condition within {path}")
        if expected_conditions is None:
            expected_conditions = conditions
        elif conditions != expected_conditions:
            raise ValueError(f"condition order/set differs in {path}")

        benefit = np.asarray([row["B"] for row in records], dtype=float)
        prediction = np.asarray([row["b_hat"] for row in records], dtype=float)
        current_epsilon, current_decisions = decide_kga(
            prediction, benefit, alpha=0.10, calibration="loo"
        )

        for index, row in enumerate(records):
            if int(row["seed"]) != seed:
                raise ValueError(f"record seed mismatch in {path}")
            decision = str(row["kga_decision"]).upper()
            if decision not in {"ADAPT", "FREEZE", "ABSTAIN"}:
                raise ValueError(f"invalid decision {decision!r} in {path}")
            values = [row["a0"], row["a_adapted"], row["B"], row["b_hat"], row["eps_conformal"]]
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"non-finite numeric value in {path}")
            replayed = dict(row)
            replayed["current_policy_epsilon"] = float(current_epsilon[index])
            replayed["current_policy_decision"] = str(current_decisions[index])
            all_records.append(replayed)

        sources.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "run_seed": seed,
                "records": len(records),
            }
        )

    return all_records, sources


def _regret_gaps(row: dict[str, Any]) -> dict[str, float]:
    a0 = float(row["a0"])
    adapted = float(row["a_adapted"])
    oracle = max(a0, adapted)
    decision = str(row["current_policy_decision"]).upper()
    kga_accuracy = adapted if decision == "ADAPT" else a0
    kga_regret = oracle - kga_accuracy
    return {
        "always_adapt": (oracle - adapted) - kga_regret,
        "always_freeze": (oracle - a0) - kga_regret,
    }


def family_effects(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    nested: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {baseline: [] for baseline in BASELINES}
    )
    for row in records:
        condition = str(row["condition"])
        family = condition.split("|", 1)[0]
        if not family:
            raise ValueError(f"cannot parse corruption family from {condition!r}")
        gaps = _regret_gaps(row)
        for baseline in BASELINES:
            nested[family][baseline].append(gaps[baseline])
    return {
        family: {baseline: float(np.mean(values)) for baseline, values in by_base.items()}
        for family, by_base in sorted(nested.items())
    }


def cluster_bootstrap_ci(
    effects: np.ndarray, *, n_boot: int, seed: int, ci_level: float
) -> dict[str, Any]:
    if effects.ndim != 1 or len(effects) < 2:
        raise ValueError("cluster bootstrap requires at least two one-dimensional effects")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(effects), size=(n_boot, len(effects)))
    draws = effects[indices].mean(axis=1)
    tail = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "point": float(effects.mean()),
        "ci": [float(lo), float(hi)],
        "ci_level": ci_level,
        "replicates": n_boot,
        "random_seed": seed,
    }


def exact_sign_flip_pvalue(effects: np.ndarray) -> float:
    """Exact one-sided p-value for the alternative mean(effect) > 0."""

    if effects.ndim != 1 or len(effects) == 0:
        raise ValueError("sign-flip test requires a non-empty one-dimensional vector")
    observed = float(effects.mean())
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(effects)):
        null.append(float(np.mean(effects * np.asarray(signs, dtype=float))))
    tolerance = 1e-15
    return float(np.mean(np.asarray(null) >= observed - tolerance))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        running = max(running, (m - rank) * pvalue)
        adjusted[name] = min(1.0, running)
    return adjusted


def analyze_candidate(
    source_dir: Path,
    candidate: str,
    *,
    n_boot: int,
    seed: int,
    ci_level: float,
) -> dict[str, Any]:
    records, sources = load_candidate(source_dir, candidate)
    effects_by_family = family_effects(records)
    families = sorted(effects_by_family)
    if len(families) != 6:
        raise ValueError(f"expected six corruption families, found {len(families)}")

    comparisons: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for offset, baseline in enumerate(BASELINES):
        effects = np.asarray([effects_by_family[f][baseline] for f in families], dtype=float)
        bootstrap = cluster_bootstrap_ci(
            effects, n_boot=n_boot, seed=seed + offset, ci_level=ci_level
        )
        raw_p[baseline] = exact_sign_flip_pvalue(effects)
        comparisons[baseline] = {
            **bootstrap,
            "family_effects": {
                family: float(effects_by_family[family][baseline]) for family in families
            },
            "p_value_one_sided_exact_sign_flip": raw_p[baseline],
        }

    adjusted = holm_adjust(raw_p)
    for baseline in BASELINES:
        comparisons[baseline]["p_value_holm_within_candidate_posthoc"] = adjusted[baseline]
        comparisons[baseline]["holm_within_candidate_posthoc_reject_at_0.05"] = (
            adjusted[baseline] <= 0.05
        )

    decisions = Counter(str(row["current_policy_decision"]).upper() for row in records)
    recorded_decisions = Counter(str(row["kga_decision"]).upper() for row in records)
    replay_disagreements = sum(
        str(row["current_policy_decision"]).upper()
        != str(row["kga_decision"]).upper()
        for row in records
    )
    n = len(records)
    both_ci_positive = all(comparisons[b]["ci"][0] > 0.0 for b in BASELINES)
    both_holm = all(
        comparisons[b]["holm_within_candidate_posthoc_reject_at_0.05"]
        for b in BASELINES
    )
    return {
        "candidate": candidate,
        "grain": {
            "record": "candidate x run_seed x stress-grid condition",
            "inference_unit": "corruption_family",
            "nested_repetitions": "run seeds and condition cells are averaged within family",
            "n_records": n,
            "n_run_seeds": len({int(row["seed"]) for row in records}),
            "n_conditions_per_seed": n // len({int(row["seed"]) for row in records}),
            "n_inference_units": len(families),
            "families": families,
        },
        "decision_counts": {
            action: decisions[action] for action in ("ADAPT", "FREEZE", "ABSTAIN")
        },
        "recorded_historical_decision_counts": {
            action: recorded_decisions[action]
            for action in ("ADAPT", "FREEZE", "ABSTAIN")
        },
        "current_vs_recorded_decision_disagreements": replay_disagreements,
        "adapt_exposure": decisions["ADAPT"] / n,
        "freeze_exposure": decisions["FREEZE"] / n,
        "strict_decision_coverage": (decisions["ADAPT"] + decisions["FREEZE"]) / n,
        "comparisons": comparisons,
        "gate": {
            "both_pointwise_95pct_cluster_bootstrap_intervals_positive": both_ci_positive,
            "both_one_sided_sign_flip_tests_survive_within_candidate_posthoc_holm_0.05": (
                both_holm
            ),
            "posthoc_within_candidate_cluster_sensitivity_pass": bool(
                both_ci_positive and both_holm
            ),
        },
        "current_policy_replay": {
            "entry_point": "kga.policy.decide_kga",
            "alpha": 0.10,
            "calibration": "loo",
            "radius": "exact split-conformal rank radius, leave-one-cell-out",
            "stored_kga_decision_used_for_scoring": False,
        },
        "sources": sources,
    }


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    candidates = [
        analyze_candidate(
            args.source_dir,
            candidate,
            n_boot=args.n_boot,
            seed=args.seed + 10 * index,
            ci_level=args.ci_level,
        )
        for index, candidate in enumerate(args.candidates)
    ]
    protocol_family_raw_p = {
        f"{row['candidate']}::{baseline}": row["comparisons"][baseline][
            "p_value_one_sided_exact_sign_flip"
        ]
        for row in candidates
        for baseline in BASELINES
    }
    protocol_family_adjusted = holm_adjust(protocol_family_raw_p)
    for row in candidates:
        protocol_rejects = []
        for baseline in BASELINES:
            key = f"{row['candidate']}::{baseline}"
            adjusted = protocol_family_adjusted[key]
            row["comparisons"][baseline][
                "p_value_holm_preregistered_six_comparison_family"
            ] = adjusted
            row["comparisons"][baseline][
                "holm_preregistered_six_comparison_reject_at_0.05"
            ] = adjusted <= 0.05
            protocol_rejects.append(adjusted <= 0.05)
        row["gate"][
            "both_sign_flip_tests_survive_preregistered_six_comparison_holm_0.05"
        ] = all(protocol_rejects)
        row["gate"]["preregistered_six_comparison_cluster_sensitivity_pass"] = bool(
            row["gate"]["both_pointwise_95pct_cluster_bootstrap_intervals_positive"]
            and all(protocol_rejects)
        )

    policy_path = ROOT / "kga/policy.py"
    certificate_path = ROOT / "kga/certificate.py"
    return {
        "schema": "kbound-current-policy-cluster-inference-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "analysis_script": Path(__file__).relative_to(ROOT).as_posix(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "live_code_bindings": {
            "policy": {
                "path": policy_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(policy_path),
            },
            "certificate": {
                "path": certificate_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(certificate_path),
            },
            "preregistered_protocol": {
                "path": PROTOCOL_LOCK.relative_to(ROOT).as_posix(),
                "sha256": sha256(PROTOCOL_LOCK),
            },
        },
        "source_scope": "canonical current-policy exact-rank CIFAR-10-C compact panel",
        "contrast_convention": CI_CONVENTION,
        "inference": {
            "bootstrap": (
                "paired percentile bootstrap of corruption-family mean gaps; whole families "
                "resampled with replacement"
            ),
            "hypothesis_test": (
                "exact one-sided sign-flip test over six paired corruption-family effects"
            ),
            "multiplicity": (
                "Primary audit: Holm adjustment over the preregistered six candidate-by-"
                "baseline comparisons. A two-comparison within-candidate Holm result is also "
                "shown and is explicitly post hoc."
            ),
            "holm_applies_to": "p-values only; confidence intervals are unadjusted",
            "ci_level": args.ci_level,
            "bootstrap_replicates": args.n_boot,
            "random_seed_base": args.seed,
        },
        "claim_boundary": {
            "supports": (
                "retrospective current-policy family sensitivity on the controlled CIFAR-10-C "
                "stress grid; positive pointwise intervals may be described conditionally"
            ),
            "does_not_support": [
                "a preregistered cluster-robust win when six-comparison Holm fails",
                "simultaneous confidence intervals",
                "independent-checkpoint population inference",
                "prospective confirmation",
                "natural-shift generalization",
                "official-code POEM or AETTA superiority",
            ],
            "few_cluster_warning": (
                "Only six corruption families are available; intervals and sign-flip tests have "
                "low resolution and must be reported with the family-level effects."
            ),
            "multiplicity_warning": (
                "The preregistered family contains all six candidate-by-baseline comparisons; "
                "within-candidate Holm values are retrospective diagnostics only."
            ),
        },
        "preregistered_six_comparison_holm": {
            "raw_p_values": protocol_family_raw_p,
            "adjusted_p_values": protocol_family_adjusted,
            "family_size": len(protocol_family_raw_p),
            "alpha": 0.05,
        },
        "candidates": {row["candidate"]: row for row in candidates},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", nargs="+", default=["tent", "eata", "sar"])
    parser.add_argument("--n-boot", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_827)
    parser.add_argument("--ci-level", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_boot < 1:
        raise SystemExit("--n-boot must be positive")
    if not 0.0 < args.ci_level < 1.0:
        raise SystemExit("--ci-level must lie strictly between 0 and 1")
    artifact = build_artifact(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=False, allow_nan=False) + "\n")
    print(args.output)
    for candidate, row in artifact["candidates"].items():
        print(candidate, row["gate"])


if __name__ == "__main__":
    main()
