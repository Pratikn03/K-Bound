#!/usr/bin/env python3
"""Describe current-policy interval inclusion on the already-opened CIFAR grid.

This is a deterministic, no-training replay, not a new experiment. Only the
15 named compact CIFAR-10-C source files and their existing numerical
authorities are read. Every candidate and corruption family is retained.
Historical ``eps_conformal`` and ``kga_decision`` fields are never scored.

The same residual collection is reused across leave-one-cell-out calibration
pools. Its aggregate inclusion rate is therefore strongly rank-constrained,
not an independent validation of calibration. These dependent-cell summaries
establish neither exchangeability nor selection-conditional, held-out-family,
independent-checkpoint, natural-shift, or population-risk coverage.

All scientific replay comparisons must match the resident canonical and
current-policy artifacts exactly before any diagnostic is written. Missing,
nonresident, malformed, duplicate, or incomplete inputs fail closed; this
script has no restore, download, fitting, or target-access path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

SCHEMA = "kbound-current-policy-interval-diagnostics-v1"
ALPHA = 0.10
CANDIDATES = ("tent", "eata", "sar")
SEEDS = (0, 1, 2, 3, 4)
FAMILIES = ("contrast", "defocus_blur", "fog", "gaussian_noise", "jpeg_compression", "pixelate")
CONDITIONS_PER_SEED = 432
ACTIONS = ("ADAPT", "FREEZE", "ABSTAIN")
BASELINES = ("always_adapt", "always_freeze")
RESULT_REL = Path("experiments/kbound/results/reconciled_panels_v1")
SOURCE_REL = RESULT_REL / "source/cifar10c"
CANONICAL_REL = RESULT_REL / "canonical_panel_results.json"
INFERENCE_REL = RESULT_REL / "current_policy_cluster_inference.json"
MANIFEST_REL = RESULT_REL / "source_manifest.json"
OUTPUT_REL = Path("docs/research/kbound/paper/generated/current_policy_interval_diagnostics")
CODE_PATHS = {
    "policy": "kga/policy.py",
    "certificate": "kga/certificate.py",
    "numeric_validation": "kga/_validation.py",
    "preregistered_protocol": "research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml",
}
DATALESS_FLAG = 0x40000000
BENEFIT_IDENTITY_ATOL = 1e-7  # Archived accuracy tensors may have float32 rounding.


def require_resident(path: Path) -> None:
    """Check metadata before reading; never implicitly hydrate an iCloud file."""
    info = path.stat(follow_symlinks=False)
    if getattr(info, "st_flags", 0) & DATALESS_FLAG:
        raise ValueError(f"nonresident/dataless input; no restoration attempted: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"input must be a resident regular file, not a symlink: {path}")


def resident_bytes(path: Path) -> bytes:
    require_resident(path)
    return path.read_bytes()


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_json(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = resident_bytes(path)
    payload = json.loads(data, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload, {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _released_policy():
    # Importing kga executes its public package initializer. Preflight package
    # files as well as the three numerical dependencies before that import.
    package_files = sorted((ROOT / "kga").glob("*.py"))
    if not package_files:
        raise ValueError("released kga package is unavailable")
    for path in package_files:
        require_resident(path)
    from kga.policy import decide_kga

    return decide_kga


def _finite_number(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number; no record may be dropped or imputed")
    return float(value)


def replay_records(records: list[dict[str, Any]], candidate: str, seed: int) -> list[dict[str, Any]]:
    """Replay one intact candidate/run-seed pool through released decide_kga."""
    if not isinstance(records, list) or not records:
        raise ValueError("a nonempty, intact record pool is required")
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("each record must be an object")
        if type(row.get("seed")) is not int or row["seed"] != seed:
            raise ValueError("record seed mismatch")
        if row.get("method") != candidate or row.get("benchmark") != "cifar10c":
            raise ValueError("record candidate/benchmark mismatch")
        condition = row.get("condition")
        if not isinstance(condition, str) or "|" not in condition:
            raise ValueError("record requires a parseable condition identity")
        family = condition.split("|", 1)[0]
        if family not in FAMILIES:
            raise ValueError(f"unexpected corruption family: {family!r}")
        if condition in seen:
            raise ValueError(f"duplicate condition in candidate/seed pool: {condition}")
        seen.add(condition)
        values = {key: _finite_number(row, key) for key in ("B", "b_hat", "a0", "a_adapted")}
        if not all(0 <= values[key] <= 1 for key in ("a0", "a_adapted")):
            raise ValueError("accuracy values must lie in [0, 1]")
        if abs(values["B"] - (values["a_adapted"] - values["a0"])) > BENEFIT_IDENTITY_ATOL:
            raise ValueError("B disagrees with adapted minus frozen accuracy")
        clean.append({"candidate": candidate, "seed": seed, "condition": condition, "family": family, **values})

    prediction = np.asarray([row["b_hat"] for row in clean], dtype=float)
    benefit = np.asarray([row["B"] for row in clean], dtype=float)
    epsilon, decisions = _released_policy()(prediction, benefit, alpha=ALPHA, calibration="loo")
    if epsilon.shape != prediction.shape or decisions.shape != prediction.shape:
        raise ValueError("released replay returned a changed record shape")
    for row, radius, action in zip(clean, epsilon, decisions, strict=True):
        if math.isnan(float(radius)) or radius < 0 or str(action) not in ACTIONS:
            raise ValueError("released replay returned an invalid radius or action")
        row["epsilon"] = float(radius)
        row["decision"] = str(action)
    return clean


def _fraction(count: int, denominator: int, reason: str | None = None) -> dict[str, Any]:
    return {
        "numerator": count,
        "denominator": denominator,
        "value": count / denominator if denominator else None,
        "defined": denominator != 0,
        "undefined_reason": None if denominator else reason,
    }


def _extended_value(value: float) -> dict[str, Any]:
    if math.isnan(value) or value < 0:
        raise ValueError("invalid interval width")
    return {"value": value if math.isfinite(value) else None, "status": "finite" if math.isfinite(value) else "positive_infinity"}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe all cells; zero-exposure conditional rates remain undefined."""
    if not records:
        raise ValueError("cannot summarize an empty group")
    n = len(records)
    prediction = np.asarray([row["b_hat"] for row in records], dtype=float)
    benefit = np.asarray([row["B"] for row in records], dtype=float)
    epsilon = np.asarray([row["epsilon"] for row in records], dtype=float)
    decisions = np.asarray([row["decision"] for row in records], dtype=object)
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(benefit)):
        raise ValueError("nonfinite prediction/outcome cannot enter an inclusion denominator")
    if np.any(np.isnan(epsilon)) or np.any(epsilon < 0) or not set(decisions) <= set(ACTIONS):
        raise ValueError("invalid radius or decision in summary")
    finite = np.isfinite(epsilon)
    if np.any(epsilon[finite] > np.finfo(float).max / 2):
        raise ValueError("finite full interval width would overflow")
    width = 2 * epsilon
    included = np.abs(prediction - benefit) <= epsilon
    counts = {action: int(np.sum(decisions == action)) for action in ACTIONS}
    false_adapt = int(np.sum((decisions == "ADAPT") & (benefit <= 0)))
    false_freeze = int(np.sum((decisions == "FREEZE") & (benefit >= 0)))
    return {
        "n": n,
        "nominal_inclusion_target": 1 - ALPHA,
        "observed_inclusion": _fraction(int(np.sum(included)), n),
        "finite_interval_count": int(np.sum(finite)),
        "infinite_interval_count": int(np.sum(~finite)),
        "finite_interval_inclusion": _fraction(int(np.sum(included & finite)), int(np.sum(finite)), "no finite intervals"),
        "full_interval_width": {
            "definition": "2 * epsilon, unclipped; accuracy proportion units",
            "mean": _extended_value(float(np.mean(width))),
            "median": _extended_value(float(np.median(width))),
            "minimum": _extended_value(float(np.min(width))),
            "maximum": _extended_value(float(np.max(width))),
        },
        "commitment": _fraction(counts["ADAPT"] + counts["FREEZE"], n),
        "actions": {action: _fraction(counts[action], n) for action in ACTIONS},
        "false_adapt": {
            "event": "ADAPT and measured cell B <= 0",
            "marginal": _fraction(false_adapt, n),
            "conditional": _fraction(false_adapt, counts["ADAPT"], "no ADAPT decisions"),
        },
        "false_freeze": {
            "event": "FREEZE and measured cell B >= 0",
            "marginal": _fraction(false_freeze, n),
            "conditional": _fraction(false_freeze, counts["FREEZE"], "no FREEZE decisions"),
        },
    }


def replay_score(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the canonical scorer's arithmetic and order, without importing its fits."""
    if not records:
        raise ValueError("cannot score an empty pool")
    benefit = np.asarray([row["B"] for row in records], dtype=float)
    frozen = np.asarray([row["a0"] for row in records], dtype=float)
    adapted = np.asarray([row["a_adapted"] for row in records], dtype=float)
    decisions = np.asarray([row["decision"] for row in records], dtype=object)
    oracle = np.maximum(frozen, adapted)
    is_adapt = decisions == "ADAPT"
    counts = Counter(decisions)
    false_adapt = int(np.sum(is_adapt & (benefit <= 0)))
    n = len(records)
    return {
        "n": n,
        "regret": {
            "kga": float(np.mean(oracle - np.where(is_adapt, adapted, frozen))),
            "always_adapt": float(np.mean(oracle - adapted)),
            "always_freeze": float(np.mean(oracle - frozen)),
        },
        "adapt_count": counts["ADAPT"],
        "freeze_count": counts["FREEZE"],
        "abstain_count": counts["ABSTAIN"],
        "false_adapt_count": false_adapt,
        "fa_u": false_adapt / n,
        "fa_c": false_adapt / counts["ADAPT"] if counts["ADAPT"] else None,
        "adapt_rate": counts["ADAPT"] / n,
        "decision_coverage": float(np.mean(decisions != "ABSTAIN")),
    }


def exact_match(checks: list[dict[str, Any]], name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"scientific/provenance mismatch in {name}: {actual!r} != {expected!r}")
    checks.append({"check": name, "comparison": "exact_equality", "observed": actual, "expected": expected, "passed": True})


def verify_scientific_equality(
    records: list[dict[str, Any]], canonical: dict[str, Any], inference: dict[str, Any]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    score = replay_score(records)
    for key, value in score.items():
        exact_match(checks, f"canonical.{key}", value, canonical[key])
    counts = {action: score[f"{action.lower()}_count"] for action in ACTIONS}
    exact_match(checks, "current_policy.decision_counts", counts, inference["decision_counts"])
    exact_match(checks, "current_policy.adapt_exposure", score["adapt_rate"], inference["adapt_exposure"])
    exact_match(checks, "current_policy.freeze_exposure", score["freeze_count"] / len(records), inference["freeze_exposure"])
    exact_match(checks, "current_policy.strict_decision_coverage", score["decision_coverage"], inference["strict_decision_coverage"])
    seeds = sorted({row["seed"] for row in records})
    families = sorted({row["family"] for row in records})
    grain = {
        "n_records": len(records),
        "n_run_seeds": len(seeds),
        "n_conditions_per_seed": len(records) // len(seeds),
        "n_inference_units": len(families),
        "families": families,
    }
    for key, value in grain.items():
        exact_match(checks, f"current_policy.grain.{key}", value, inference["grain"][key])
    by_seed = {row["seed"]: row for row in canonical["per_file"]}
    if len(by_seed) != len(canonical["per_file"]):
        raise ValueError("duplicate seed in canonical per-file authority")
    exact_match(checks, "canonical.per_file.seeds", sorted(by_seed), seeds)
    for seed in seeds:
        subset = [row for row in records if row["seed"] == seed]
        for key, value in replay_score(subset).items():
            exact_match(checks, f"canonical.seed{seed}.score.{key}", value, by_seed[seed]["score"][key])
        epsilon = np.asarray([row["epsilon"] for row in subset], dtype=float)
        for suffix, operation in (("min", np.min), ("mean", np.mean), ("max", np.max)):
            exact_match(checks, f"canonical.seed{seed}.epsilon_{suffix}", float(operation(epsilon)), by_seed[seed][f"epsilon_{suffix}"])
    for baseline in BASELINES:
        effects: dict[str, float] = {}
        for family in families:
            values = []
            for row in records:
                if row["family"] != family:
                    continue
                oracle = max(row["a0"], row["a_adapted"])
                served = row["a_adapted"] if row["decision"] == "ADAPT" else row["a0"]
                fixed = row["a_adapted"] if baseline == "always_adapt" else row["a0"]
                values.append((oracle - fixed) - (oracle - served))
            effects[family] = float(np.mean(values))
        expected = inference["comparisons"][baseline]
        exact_match(checks, f"current_policy.{baseline}.family_effects", effects, expected["family_effects"])
        exact_match(checks, f"current_policy.{baseline}.point", float(np.mean(list(effects.values()))), expected["point"])
    return checks


def build_artifact(root: Path = ROOT) -> dict[str, Any]:
    canonical, canonical_seal = read_json(root / CANONICAL_REL, root)
    inference, inference_seal = read_json(root / INFERENCE_REL, root)
    manifest, manifest_seal = read_json(root / MANIFEST_REL, root)
    checks: list[dict[str, Any]] = []
    exact_match(checks, "canonical.schema", canonical["schema"], "kbound-canonical-panel-results-v2")
    exact_match(checks, "current_policy.schema", inference["schema"], "kbound-current-policy-cluster-inference-v3")
    exact_match(checks, "canonical.alpha", canonical["alpha"], ALPHA)
    exact_match(checks, "canonical.source_manifest_sha256", manifest_seal["sha256"], canonical["source_manifest_sha256"])
    exact_match(checks, "manifest.file_count", len(manifest["files"]), manifest["file_count"])
    exact_match(checks, "canonical.source_file_count", manifest["file_count"], canonical["source_file_count"])
    manifest_files = {row["destination"]: row for row in manifest["files"]}
    if len(manifest_files) != len(manifest["files"]):
        raise ValueError("duplicate destination in source manifest")
    source_dir = root / SOURCE_REL
    expected_names = sorted(f"per_condition_cifar10c_{candidate}_seed{seed}.json" for candidate in CANDIDATES for seed in SEEDS)
    exact_match(checks, "complete_source_census", sorted(path.name for path in source_dir.glob("*.json")), expected_names)

    code_bindings = {}
    exact_match(checks, "current_policy.binding_inventory", sorted(inference["live_code_bindings"]), sorted(CODE_PATHS))
    for name, relative in CODE_PATHS.items():
        data = resident_bytes(root / relative)
        binding = {"path": relative, "sha256": hashlib.sha256(data).hexdigest()}
        exact_match(checks, f"current_policy.code.{name}", binding, inference["live_code_bindings"][name])
        code_bindings[name] = binding

    all_sources = []
    result = {}
    common_conditions: set[str] | None = None
    canonical_panel = canonical["panels"]["cifar10c"]
    canonical_sources = {row["compact_path"]: row for row in canonical_panel["source_provenance"]}
    if len(canonical_sources) != len(canonical_panel["source_provenance"]):
        raise ValueError("duplicate canonical source identity")
    exact_match(checks, "canonical.candidate_inventory", sorted(canonical_panel["panel"]["candidates"]), sorted(CANDIDATES))
    exact_match(checks, "current_policy.candidate_inventory", sorted(inference["candidates"]), sorted(CANDIDATES))
    for candidate in CANDIDATES:
        authority = inference["candidates"][candidate]
        replay_contract = authority["current_policy_replay"]
        for key, expected in (("entry_point", "kga.policy.decide_kga"), ("alpha", ALPHA), ("calibration", "loo"), ("stored_kga_decision_used_for_scoring", False)):
            exact_match(checks, f"{candidate}.replay.{key}", replay_contract[key], expected)
        inference_sources = {row["path"]: row for row in authority["sources"]}
        if len(inference_sources) != len(authority["sources"]) or len(inference_sources) != len(SEEDS):
            raise ValueError("incomplete or duplicate current-policy source identity")
        records = []
        expected_order = None
        for seed in SEEDS:
            path = source_dir / f"per_condition_cifar10c_{candidate}_seed{seed}.json"
            payload, seal = read_json(path, root)
            metadata = payload.get("metadata", {})
            if payload.get("schema") != "kbound-compact-panel-source-v1":
                raise ValueError(f"unexpected source schema: {path}")
            if metadata.get("method") != candidate or metadata.get("benchmark") != "cifar10c":
                raise ValueError(f"source candidate/benchmark mismatch: {path}")
            if type(metadata.get("seed")) is not int or metadata["seed"] != seed:
                raise ValueError(f"source seed mismatch: {path}")
            source_records = payload.get("records")
            if metadata.get("alpha") != ALPHA or metadata.get("n_conditions") != CONDITIONS_PER_SEED or not isinstance(source_records, list) or len(source_records) != CONDITIONS_PER_SEED:
                raise ValueError(f"changed level or incomplete 432-condition source pool: {path}")
            relative = seal["path"]
            source_manifest = manifest_files[relative]
            for key, expected in (("sha256", source_manifest["compact_sha256"]), ("bytes", source_manifest["compact_bytes"])):
                exact_match(checks, f"source_manifest.{relative}.{key}", seal[key], expected)
            exact_match(checks, f"source_manifest.{relative}.records", len(source_records), source_manifest["records"])
            exact_match(checks, f"canonical.source.{relative}.sha256", seal["sha256"], canonical_sources[relative]["compact_sha256"])
            for key, value in {**seal, "records": len(source_records), "run_seed": seed}.items():
                exact_match(checks, f"current_policy.source.{relative}.{key}", value, inference_sources[relative][key])
            replay = replay_records(source_records, candidate, seed)
            conditions = tuple(row["condition"] for row in replay)
            if expected_order is None:
                expected_order = conditions
            elif conditions != expected_order:
                raise ValueError("condition order/set differs across run seeds")
            if common_conditions is None:
                common_conditions = set(conditions)
            elif set(conditions) != common_conditions:
                raise ValueError("condition set differs across candidates")
            exact_match(checks, f"{candidate}.seed{seed}.family_cell_counts", dict(sorted(Counter(row["family"] for row in replay).items())), {family: 72 for family in FAMILIES})
            records.extend(replay)
            all_sources.append({**seal, "candidate": candidate, "run_seed": seed, "records": len(replay)})
        identities = {(row["candidate"], row["seed"], row["condition"]) for row in records}
        if len(identities) != len(records):
            raise ValueError("duplicate candidate/run-seed/condition identity")
        science_checks = verify_scientific_equality(records, canonical_panel["panel"]["candidates"][candidate], authority)
        result[candidate] = {
            "summary": summarize(records),
            "by_corruption_family": {family: summarize([row for row in records if row["family"] == family]) for family in FAMILIES},
            "unchanged_replay_score": replay_score(records),
            "scientific_equality_checks": science_checks,
        }

    script = Path(__file__).resolve()
    return {
        "schema": SCHEMA,
        "analysis_script": script.relative_to(ROOT).as_posix(),
        "analysis_script_sha256": hashlib.sha256(resident_bytes(script)).hexdigest(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "scope": "Retrospective descriptive interval inclusion on already-opened dependent CIFAR-10-C cells; no fitting or new data access.",
        "calibration": {
            "entry_point": "kga.policy.decide_kga",
            "method": "leave-one-cell-out empirical order-statistic calibration",
            "alpha": ALPHA,
            "nominal_inclusion_target": 1 - ALPHA,
            "pool": "other 431 cells of the same candidate and run seed; unchanged across family summaries",
            "historical_fields_ignored": ["eps_conformal", "kga_decision"],
            "reuse_warning": "LOO residual pools reuse the same scored collection. Aggregate inclusion is strongly rank-constrained, not independent validation of calibration.",
        },
        "units": {
            "record": "candidate x run seed x controlled condition",
            "repetitions": "5 run/stream seeds conditional on the archived checkpoint, not 5 independent checkpoints",
            "groups": "3 candidates; 6 corruption families each; 2160 cells/candidate; 360 cells/family/candidate",
            "weighting": "each saved cell has equal weight; no aggregation across candidate policies",
            "outcome": "measured cell accuracy benefit B = adapted accuracy - frozen accuracy; not population benefit",
            "interval_inclusion": "abs(b_hat - B) <= epsilon, including equality and all +infinity intervals",
            "interval_width": "2 * epsilon (full, unclipped width), accuracy proportion units, not percentage points",
            "commitment": "(ADAPT + FREEZE) / all cells; different from interval inclusion and adapt rate",
            "false_adapt": "ADAPT and B <= 0; marginal denominator all cells; conditional denominator ADAPT cells",
            "false_freeze": "FREEZE and B >= 0; marginal denominator all cells; conditional denominator FREEZE cells",
            "undefined_values": "null plus defined=false/reason for zero-denominator rates; positive_infinity status for unbounded width",
            "benefit_identity_tolerance": BENEFIT_IDENTITY_ATOL,
            "scientific_authority_comparisons": "exact Python numeric/object equality; no tolerance or rounded-table comparisons",
        },
        "does_not_establish": [
            "exchangeability or independence of the related grid cells",
            "selection-conditional coverage or conditional false-commitment control",
            "independent held-out-family or held-out-environment calibration",
            "independent-checkpoint inference or natural-shift generalization",
            "coverage of population risk or a new confirmatory experiment",
        ],
        "input_authorities": [canonical_seal, inference_seal, manifest_seal],
        "code_bindings": code_bindings,
        "sources": all_sources,
        "provenance_and_completeness_checks": checks,
        "all_scientific_checks_passed": True,
        "candidates": result,
    }


def _number(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"


def _width(row: dict[str, Any], name: str) -> str:
    value = row["full_interval_width"][name]
    return r"$\infty$" if value["status"] == "positive_infinity" else _number(value["value"])


def render_latex(artifact: dict[str, Any]) -> str:
    """Concise three-candidate table; widths are full and rates are fractions."""
    lines = [
        "% Generated by build_current_policy_interval_diagnostics.py; do not edit.",
        "% RETROSPECTIVE dependent-cell description, not independent calibration validation.",
        "% LOO pool reuse strongly rank-constrains pooled inclusion. All rates are fractions.",
        "% Full widths are 2*epsilon, unclipped, in accuracy proportion units. -- is undefined.",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Candidate & $n$ & Nominal & Inclusion & Mean width & Median width & Commitment & $\mathrm{FA}_{u}/\mathrm{FA}_{c}$ & $\mathrm{FF}_{u}/\mathrm{FF}_{c}$ \\",
        r"\midrule",
    ]
    for candidate in CANDIDATES:
        row = artifact["candidates"][candidate]["summary"]
        errors = ["/".join(_number(row[event][kind]["value"]) for kind in ("marginal", "conditional")) for event in ("false_adapt", "false_freeze")]
        columns = [candidate.upper(), str(row["n"]), _number(row["nominal_inclusion_target"]), _number(row["observed_inclusion"]["value"]), _width(row, "mean"), _width(row, "median"), _number(row["commitment"]["value"]), *errors]
        lines.append(" & ".join(columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_groups_latex(artifact: dict[str, Any]) -> str:
    """All eighteen candidate/family groups, with exact error denominators."""
    lines = [
        "% Generated by build_current_policy_interval_diagnostics.py; do not edit.",
        "% RETROSPECTIVE family diagnostics; no within-family recalibration is performed.",
        "% Pooled LOO inclusion is rank-constrained, not independent calibration validation.",
        "% All rates are fractions; width is the mean full 2*epsilon in accuracy units.",
        "% Conditional errors include exact false-action/action counts; -- is undefined.",
        r"\begin{tabular}{llrrrrll}",
        r"\toprule",
        r"Candidate & Family & $n$ & Inclusion & Mean width & Commitment & False A/A; $\mathrm{FA}_{c}$ & False F/F; $\mathrm{FF}_{c}$ \\",
        r"\midrule",
    ]
    for candidate in CANDIDATES:
        for family, row in artifact["candidates"][candidate]["by_corruption_family"].items():
            errors = []
            for event in ("false_adapt", "false_freeze"):
                rate = row[event]["conditional"]
                errors.append(f"{rate['numerator']}/{rate['denominator']}; {_number(rate['value'])}")
            columns = [candidate.upper(), family.replace("_", r"\_"), str(row["n"]), _number(row["observed_inclusion"]["value"]), _width(row, "mean"), _number(row["commitment"]["value"]), *errors]
            lines.append(" & ".join(columns) + r" \\")
        if candidate != CANDIDATES[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_outputs(
    outputs: dict[Path, str], *, check: bool = False, refresh_existing: bool = False,
    protected_paths: tuple[Path, ...] = (),
) -> None:
    """Preflight the entire output set, then refresh only recognized generated files.

    The caller must finish all scientific/input checks before calling this
    function. Refresh never permits symlinks, source/code collisions, or
    replacement of an unrecognized file. Existing generated files are replaced
    atomically, and a concurrent edit fails instead of being overwritten.
    """
    if check and refresh_existing:
        raise ValueError("--check and --refresh-existing are mutually exclusive")
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output paths must be distinct")
    protected = {path.resolve() for path in protected_paths}
    before: dict[Path, bytes | None] = {}
    encoded = {path: contents.encode("utf-8") for path, contents in outputs.items()}
    for path, contents in encoded.items():
        if path.resolve() in protected:
            raise ValueError(f"output would overwrite an input authority or code file: {path}")
        if path.suffix not in {".json", ".tex"}:
            raise ValueError(f"unexpected diagnostic output extension: {path}")
        if any(component.is_symlink() for component in (path, *path.parents)):
            raise ValueError(f"diagnostic output path must not contain symlinks: {path}")
        old = resident_bytes(path) if path.exists() else None
        before[path] = old
        if check:
            if old != contents:
                raise ValueError(f"generated output is absent or differs: {path}")
        elif old is not None and old != contents:
            if not refresh_existing:
                raise ValueError(f"output differs; use a new path or explicit --refresh-existing: {path}")
            if path.suffix == ".json":
                existing = json.loads(old, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
                if not isinstance(existing, dict) or existing.get("schema") != SCHEMA or existing.get("analysis_script") != "docs/research/kbound/scripts/build_current_policy_interval_diagnostics.py":
                    raise ValueError(f"refusing to refresh an unrecognized JSON artifact: {path}")
            elif not old.startswith(b"% Generated by build_current_policy_interval_diagnostics.py; do not edit.\n"):
                raise ValueError(f"refusing to refresh unrecognized TeX: {path}")
    if check:
        return
    for path, contents in encoded.items():
        old = before[path]
        if old == contents:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if old is None:
            with path.open("xb") as stream:
                stream.write(contents)
            continue
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(contents)
            if resident_bytes(path) != old:
                raise ValueError(f"output changed after preflight; refusing to overwrite: {path}")
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=ROOT / OUTPUT_REL.with_suffix(".json"))
    parser.add_argument("--output-tex", type=Path, default=ROOT / OUTPUT_REL.with_suffix(".tex"))
    parser.add_argument("--output-groups-tex", type=Path, default=ROOT / OUTPUT_REL.with_name(OUTPUT_REL.name + "_groups").with_suffix(".tex"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify generated outputs byte-for-byte, without writing")
    mode.add_argument("--refresh-existing", action="store_true", help="refresh recognized generated outputs only after every scientific/input check passes")
    args = parser.parse_args()
    try:
        artifact = build_artifact()
        outputs = {
            args.output_json: json.dumps(artifact, indent=2, allow_nan=False) + "\n",
            args.output_tex: render_latex(artifact),
            args.output_groups_tex: render_groups_latex(artifact),
        }
        if len({path.resolve() for path in (args.output_json, args.output_tex, args.output_groups_tex)}) != 3:
            raise ValueError("JSON and TeX outputs must be three different files")
        protected = tuple(
            ROOT / entry["path"]
            for entry in [*artifact["input_authorities"], *artifact["sources"], *artifact["code_bindings"].values()]
        ) + (Path(__file__).resolve(),)
        write_outputs(outputs, check=args.check, refresh_existing=args.refresh_existing, protected_paths=protected)
        print("All current-policy scientific equality checks passed; retrospective interval diagnostics verified.")
        for candidate in CANDIDATES:
            row = artifact["candidates"][candidate]["summary"]
            print(f"{candidate}: n={row['n']}, inclusion={row['observed_inclusion']['value']:.6f}, commitment={row['commitment']['value']:.6f}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Interval diagnostics blocked: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
