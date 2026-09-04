#!/usr/bin/env python3
"""Cluster-aware inference for the current exact-rank CIFAR-10-C policies.

The canonical compact source stores one row per condition and run seed, while
``canonical_panel_results.json`` stores the current cross-fitted per-cell
prediction, radius, and action.  Historical ``b_hat`` and actions in the compact
source are audit fields only.  Run seeds share an archived checkpoint, so they
are nested repetitions rather than independent model draws.  This analysis
therefore averages the paired regret gaps within corruption family and uses the
six corruption families as the inference units.

The reported contrast is always

    regret(fixed baseline) - regret(KGA),

so positive values favor KGA.  Percentile intervals resample whole corruption
families. P-values are exact one-sided sign-flip tests over the family-level
paired gaps. The six candidate-by-fixed-policy contrasts were prospectively
named, but this exact-rank replay, sign-flip analysis, and six-way Holm
adjustment are retrospective and non-confirmatory.
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

DEFAULT_SOURCE_DIR = (
    ROOT / "experiments/kbound/results/reconciled_panels_v1/source/cifar10c"
)
DEFAULT_CANONICAL = (
    ROOT / "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/kbound/results/reconciled_panels_v1/"
    "current_policy_cluster_inference.json"
)
BASELINES = ("always_adapt", "always_freeze")
CI_CONVENTION = "baseline_regret_minus_kga_regret; positive values favor KGA"
PROTOCOL_LOCK = ROOT / "research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml"
CURRENT_POLICY_BINDING_PATHS = {
    "crossfit": "kga/crossfit.py",
    "policy": "kga/policy.py",
    "certificate": "kga/certificate.py",
    "numeric_validation": "kga/_validation.py",
    "reconciliation": "scripts/reconcile_result_panels.py",
    "preregistered_protocol": PROTOCOL_LOCK.relative_to(ROOT).as_posix(),
}
SCHEMA = "kbound-current-policy-cluster-inference-v3"
FAMILY_FIELD = "retrospective_holm_over_six_prospectively_named_contrasts"
COMPARISON_P_FIELD = "p_value_retrospective_holm_six_prospectively_named_contrasts"
COMPARISON_REJECT_FIELD = "retrospective_holm_six_contrasts_reject_at_0_05"
GATE_REJECTS_BOTH_FIELD = "both_sign_flip_tests_survive_retrospective_six_contrast_holm_0_05"
GATE_PASS_FIELD = "retrospective_six_contrast_cluster_sensitivity_pass"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def controlled_grid_sample_id(*, track: str, candidate: str, seed: int, condition: str) -> str:
    identity = {"track": track, "candidate": candidate, "seed": seed, "condition": condition}
    payload = (json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return f"grid-cell-sha256:{hashlib.sha256(payload).hexdigest()}"


def controlled_grid_input_sha256(Z: Any, B: Any, sample_ids: list[str]) -> dict[str, str]:
    return {
        "Z": _compact_sha256(np.asarray(Z, dtype=float).tolist()),
        "B": _compact_sha256(np.asarray(B, dtype=float).tolist()),
        "sample_ids": _compact_sha256([str(value) for value in sample_ids]),
    }


def relative_or_name(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def current_policy_code_bindings() -> dict[str, dict[str, str]]:
    """Seal the executed replay primitives, including mask-aware coercion."""
    return {
        name: {"path": relative_path, "sha256": sha256(ROOT / relative_path)}
        for name, relative_path in CURRENT_POLICY_BINDING_PATHS.items()
    }


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=2
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _canonical_candidate_files(canonical_path: Path, candidate: str) -> dict[int, dict[str, Any]]:
    payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    try:
        files = payload["panels"]["cifar10c"]["panel"]["candidates"][candidate]["per_file"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"canonical cell authority is missing candidate {candidate!r}") from exc
    if not isinstance(files, list) or not files:
        raise ValueError(f"canonical cell authority has no files for candidate {candidate!r}")
    by_seed: dict[int, dict[str, Any]] = {}
    for row in files:
        seed = int(row["seed"])
        if seed in by_seed:
            raise ValueError(f"duplicate canonical run seed {seed} for {candidate}")
        authority = row.get("current_cell_authority")
        if not isinstance(authority, dict) or authority.get("schema") != "kbound-controlled-grid-cell-authority-v1":
            raise ValueError(f"missing canonical per-cell authority for {candidate} seed {seed}")
        cells = authority.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError(f"empty canonical per-cell authority for {candidate} seed {seed}")
        expected_hash = hashlib.sha256(
            (json.dumps(cells, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        ).hexdigest()
        if authority.get("sha256") != expected_hash:
            raise ValueError(f"canonical per-cell authority hash mismatch for {candidate} seed {seed}")
        predictions = [cell.get("prediction") for cell in cells]
        radii = [cell.get("radius") for cell in cells]
        actions = [cell.get("action") for cell in cells]
        for name, values in (("prediction", predictions), ("radius", radii), ("action", actions)):
            value_hash = hashlib.sha256(
                (json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            ).hexdigest()
            if row.get(f"current_{name}_sha256") != value_hash:
                raise ValueError(f"canonical current-{name} hash mismatch for {candidate} seed {seed}")
        protocol = row.get("crossfit_protocol")
        if not isinstance(protocol, dict) or protocol.get("status") != "ok":
            raise ValueError(f"canonical cross-fit protocol is not successful for {candidate} seed {seed}")
        expected_settings = {
            "schema": "kga-controlled-grid-crossfit-v1",
            "alpha": 0.1,
            "requested_n_folds": 5,
            "calibration_fraction_target": 0.3,
            "gbrt": {
                "n_estimators": 250,
                "max_depth": 2,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "random_state": 0,
            },
        }
        if any(protocol.get(key) != value for key, value in expected_settings.items()):
            raise ValueError(f"canonical cross-fit protocol settings mismatch for {candidate} seed {seed}")
        expected_dependencies = {
            name: sha256(ROOT / CURRENT_POLICY_BINDING_PATHS[name])
            for name in ("crossfit", "certificate", "policy", "numeric_validation")
        }
        if protocol.get("implementation_sha256") != expected_dependencies:
            raise ValueError(f"canonical cross-fit executable hashes mismatch for {candidate} seed {seed}")
        by_seed[seed] = row
    return by_seed


def load_candidate(
    source_dir: Path,
    candidate: str,
    *,
    canonical_path: Path = DEFAULT_CANONICAL,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = sorted(source_dir.glob(f"per_condition_cifar10c_{candidate}_seed*.json"))
    if not files:
        raise ValueError(f"no canonical files found for candidate {candidate!r} in {source_dir}")
    canonical_by_seed = _canonical_candidate_files(canonical_path, candidate)

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

        canonical_file = canonical_by_seed.get(seed)
        if canonical_file is None:
            raise ValueError(f"canonical per-cell authority is missing {candidate} seed {seed}")
        authority = canonical_file["current_cell_authority"]
        canonical_cells = authority["cells"]
        canonical_conditions = tuple(str(cell["condition"]) for cell in canonical_cells)
        if canonical_conditions != conditions:
            raise ValueError(f"canonical per-cell condition order differs in {path}")

        sample_ids = [
            controlled_grid_sample_id(
                track=str(row.get("benchmark") or row.get("dataset")),
                candidate=str(row.get("method") or row.get("candidate")),
                seed=int(row["seed"]),
                condition=str(row["condition"]),
            )
            for row in records
        ]
        if sample_ids != [str(cell.get("sample_id")) for cell in canonical_cells]:
            raise ValueError(f"canonical stable sample IDs differ in {path}")
        expected_inputs = controlled_grid_input_sha256(
            [row["Z"] for row in records], [row["B"] for row in records], sample_ids
        )
        if canonical_file["crossfit_protocol"].get("input_sha256") != expected_inputs:
            raise ValueError(f"canonical cross-fit input hash mismatch in {path}")

        for index, row in enumerate(records):
            if int(row["seed"]) != seed:
                raise ValueError(f"record seed mismatch in {path}")
            decision = str(row["kga_decision"]).upper()
            if decision not in {"ADAPT", "FREEZE", "ABSTAIN"}:
                raise ValueError(f"invalid decision {decision!r} in {path}")
            cell = canonical_cells[index]
            current_prediction = float(cell["prediction"])
            current_epsilon = float(cell["radius"])
            current_decision = str(cell["action"]).upper()
            if current_decision not in {"ADAPT", "FREEZE", "ABSTAIN"}:
                raise ValueError(f"invalid canonical decision {current_decision!r} in {canonical_path}")
            values = [row["a0"], row["a_adapted"], row["B"], current_prediction, current_epsilon]
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"non-finite numeric value in {path}")
            replayed = dict(row)
            replayed["current_policy_prediction"] = current_prediction
            replayed["current_policy_epsilon"] = current_epsilon
            replayed["current_policy_decision"] = current_decision
            all_records.append(replayed)

        sources.append(
            {
                "path": relative_or_name(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "run_seed": seed,
                "records": len(records),
                "canonical_path": relative_or_name(canonical_path),
                "canonical_sha256": sha256(canonical_path),
                "canonical_cell_authority_sha256": authority["sha256"],
            }
        )

    if seen_seeds != set(canonical_by_seed):
        raise ValueError(
            f"source/canonical run-seed mismatch for {candidate}: "
            f"source={sorted(seen_seeds)}, canonical={sorted(canonical_by_seed)}"
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
    canonical_path: Path = DEFAULT_CANONICAL,
    n_boot: int,
    seed: int,
    ci_level: float,
) -> dict[str, Any]:
    records, sources = load_candidate(source_dir, candidate, canonical_path=canonical_path)
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
            "entry_point": "canonical_panel_results.json per-file current_cell_authority",
            "alpha": 0.10,
            "calibration": "cell-outcome-disjoint fit/calibrate/score cross-fit",
            "radius": "exact split-conformal rank radius on the score-fold complement calibration subset",
            "stored_b_hat_used_for_scoring": False,
            "stored_kga_decision_used_for_scoring": False,
        },
        "sources": sources,
    }


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    candidates = [
        analyze_candidate(
            args.source_dir,
            candidate,
            canonical_path=getattr(args, "canonical_path", DEFAULT_CANONICAL),
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
            row["comparisons"][baseline][COMPARISON_P_FIELD] = adjusted
            row["comparisons"][baseline][COMPARISON_REJECT_FIELD] = adjusted <= 0.05
            protocol_rejects.append(adjusted <= 0.05)
        row["gate"][GATE_REJECTS_BOTH_FIELD] = all(protocol_rejects)
        row["gate"][GATE_PASS_FIELD] = bool(
            row["gate"]["both_pointwise_95pct_cluster_bootstrap_intervals_positive"]
            and all(protocol_rejects)
        )

    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "analysis_script": Path(__file__).relative_to(ROOT).as_posix(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "live_code_bindings": current_policy_code_bindings(),
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
                "Retrospective Holm adjustment over the six prospectively named contrasts. "
                "The exact-rank replay, sign-flip tests, and Holm analysis are retrospective "
                "and non-confirmatory. A two-comparison within-candidate Holm result is also "
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
                "a confirmatory cluster-robust win",
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
                "The six contrasts were prospectively named, but the exact-rank replay, "
                "sign-flip tests, and Holm adjustment are retrospective and non-confirmatory; "
                "within-candidate Holm values are additional post-hoc diagnostics."
            ),
        },
        FAMILY_FIELD: {
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
    parser.add_argument("--canonical", dest="canonical_path", type=Path, default=DEFAULT_CANONICAL)
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
