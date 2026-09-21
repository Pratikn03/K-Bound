#!/usr/bin/env python3
"""Lock, prepare, decide, then score a retrospective saved-panel comparison.

The decide phase does not open the held-out outcome authority. Each fitted fold
excludes its scored semantic groups from all fit/tune/calibration roles. Hashes
protect each phase boundary.
This is not prospective data collection, independent deployment, or raw-run recovery.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from kga.calibration_value import (  # noqa: E402 - direct source-tree CLI execution
    build_folds,
    calibrate_fold,
    decide,
    digest,
    matched_exposure_ids,
    score_decisions,
    score_semantic_groups,
    semantic_group,
    sha256_file,
    verify_sha256,
    write_fresh_json,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_gz(path, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with Path(path).open("xb") as f:
        f.write(gzip.compress(payload, mtime=0))


def read_json(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_text())


def verify_phase(output, phase):
    receipt = read_json(output / f"{phase}_receipt.json")
    for name, sha in receipt["outputs"].items():
        verify_sha256(output / name, sha)
    return receipt


def prepare(protocol_path, output, dataset_filter=None):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"fresh output directory required: {output}")
    protocol = read_json(protocol_path)
    output.mkdir(parents=True)
    source = Path(protocol["source_root"])
    implementations = {
        str(Path(__file__).relative_to(ROOT)): sha256_file(__file__),
        "kga/calibration_value.py": sha256_file(ROOT / "kga/calibration_value.py"),
    }
    # This is written before the new fit/tune/calibration/score experiment.
    write_fresh_json(
        output / "protocol_seal.json",
        {
            "sealed_at": now(),
            "protocol_sha256": sha256_file(protocol_path),
            "protocol": protocol,
            "implementation_sha256": implementations,
            "status": "LOCKED_RETROSPECTIVE",
            "execution_dataset_filter": dataset_filter,
        },
    )
    for item in protocol["inputs"]:
        verify_sha256(source / item["path"], item["sha256"])
    authorities = read_json(source / protocol["compact_manifest"])
    source_auth = {r["destination"]: r["compact_sha256"] for r in authorities["files"]}
    datasets = defaultdict(list)
    for item in protocol["inputs"]:
        if item["role"] == "manifest":
            continue
        path = item["path"]
        data = read_json(source / path)
        if item["role"] == "compact_panel":
            if source_auth.get(path) != item["sha256"]:
                raise ValueError("compact manifest disagrees with locked hash")
            for row in data["records"]:
                name = f"{row['benchmark']}/{row['method']}"
                record = {
                    "id": digest([name, row["seed"], row["condition"]]),
                    "group": semantic_group(row["condition"]),
                    "environment": row["condition"].split("|")[0],
                    "checkpoint": "historical_shared_source_checkpoint",
                    "features": row["Z"],
                    "frozen_score": row["a0"],
                    "candidate_score": row["a_adapted"],
                }
                if abs(record["candidate_score"] - record["frozen_score"] - row["B"]) > 1e-8:
                    raise ValueError("benefit arithmetic mismatch")
                datasets[name].append(record)
        elif item["role"] == "mixed_checkpoint_panel":
            for model, rows in data.items():
                for row in rows:
                    record = {
                        "id": digest(["mixed", model, row["condition"]]),
                        "group": semantic_group(row["condition"]),
                        "environment": row["condition"].split("|")[0],
                        "checkpoint": model,
                        "features": row["Z"],
                        "frozen_score": row["a0"],
                        "candidate_score": row["aa"],
                    }
                    if abs(record["candidate_score"] - record["frozen_score"] - row["B"]) > 1e-8:
                        raise ValueError("mixed benefit arithmetic mismatch")
                    datasets["mixed_checkpoints/tent"].append(record)
    prep = []
    score = []
    for dataset, rows in sorted(datasets.items()):
        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate panel IDs")
        for kind in ["checkpoint", "joint"] if dataset.startswith("mixed") else ["environment"]:
            for fold in build_folds(rows, kind):
                outcomes = fold.pop("score_outcomes")
                fold["dataset"] = dataset
                fold["fold_id"] = digest([dataset, fold["name"]])
                prep.append(fold)
                score.append({"fold_id": fold["fold_id"], "dataset": dataset, "kind": kind, "outcomes": outcomes})
    write_gz(output / "prepared_decision_inputs.json.gz", prep)
    write_gz(output / "heldout_score_outcomes.json.gz", score)
    write_fresh_json(
        output / "prepare_receipt.json",
        {
            "completed_at": now(),
            "protocol_seal_sha256": sha256_file(output / "protocol_seal.json"),
            "dataset_rows": {k: len(v) for k, v in datasets.items()},
            "fold_count": len(prep),
            "implementation_sha256": implementations,
            "seconds": time.perf_counter() - started,
            "outputs": {
                name: sha256_file(output / name)
                for name in ["prepared_decision_inputs.json.gz", "heldout_score_outcomes.json.gz"]
            },
        },
    )


def decide_phase(output):
    started = time.perf_counter()
    # Verify preparation without opening outcome data in the decision process.
    receipt = read_json(output / "prepare_receipt.json")
    verify_sha256(output / "protocol_seal.json", receipt["protocol_seal_sha256"])
    verify_sha256(output / "prepared_decision_inputs.json.gz", receipt["outputs"]["prepared_decision_inputs.json.gz"])
    for path, sha in receipt["implementation_sha256"].items():
        verify_sha256(ROOT / path, sha)
    sealed = read_json(output / "protocol_seal.json")
    config = sealed["protocol"]["config"]
    prepared = read_json(output / "prepared_decision_inputs.json.gz")
    results = []
    fits = []
    for i, fold in enumerate(prepared):
        trained = calibrate_fold(fold["fit"], fold["tune"], fold["cal"], config)
        t = time.perf_counter()
        decisions = decide(trained, fold["score_features"])
        trained["receipt"]["timing"]["decision_seconds"] = time.perf_counter() - t
        by_arm = defaultdict(list)
        for d in decisions:
            by_arm[d["arm"]].append(d)
        matched = []
        for arm, rows in by_arm.items():
            if not arm.startswith(("constant_", "scaled_cell", "environment_max")):
                continue
            count = sum(r["action"] == "ADAPT" for r in rows)
            point_ids = matched_exposure_ids([r["prediction"] for r in rows], [r["id"] for r in rows], count)
            gate_ids = sorted(r["id"] for r in rows if r["action"] == "ADAPT")
            constant = not arm.startswith("scaled_cell")
            if constant and sorted(point_ids) != gate_ids:
                raise AssertionError("constant-radius exposure ranking mismatch")
            matched.append(
                {
                    "arm": arm,
                    "exposure": count,
                    "constant_radius": constant,
                    "same_selection": sorted(point_ids) == gate_ids,
                    "point_ids": point_ids,
                }
            )
        fits.append(
            {
                "fold_id": fold["fold_id"],
                "dataset": fold["dataset"],
                "kind": fold["kind"],
                "name": fold["name"],
                "fit": trained["receipt"],
                "matched_exposure": matched,
            }
        )
        results.append(
            {"fold_id": fold["fold_id"], "dataset": fold["dataset"], "kind": fold["kind"], "decisions": decisions}
        )
        print(f"decided {i + 1}/{len(prepared)} {fold['dataset']} {fold['name']}", flush=True)
    write_gz(output / "sealed_decisions.json.gz", results)
    write_fresh_json(output / "fit_and_calibration_receipts.json", fits)
    write_fresh_json(
        output / "decide_receipt.json",
        {
            "completed_at": now(),
            "seconds": time.perf_counter() - started,
            "prepare_receipt_sha256": sha256_file(output / "prepare_receipt.json"),
            "output_scope": "all held-out decisions generated before any scoring phase",
            "outputs": {
                name: sha256_file(output / name)
                for name in ["sealed_decisions.json.gz", "fit_and_calibration_receipts.json"]
            },
        },
    )


def score_phase(output):
    started = time.perf_counter()
    dec = verify_phase(output, "decide")
    prep = verify_phase(output, "prepare")
    verify_sha256(output / "prepare_receipt.json", dec["prepare_receipt_sha256"])
    truth = {f["fold_id"]: f for f in read_json(output / "heldout_score_outcomes.json.gz")}
    fits = {f["fold_id"]: f for f in read_json(output / "fit_and_calibration_receipts.json")}
    folds = read_json(output / "sealed_decisions.json.gz")
    grouped = defaultdict(lambda: {"decisions": [], "outcomes": [], "folds": []})
    per_fold = []
    matched_metrics = []
    for f in folds:
        outcomes = truth[f["fold_id"]]["outcomes"]
        by_arm = defaultdict(list)
        for d in f["decisions"]:
            by_arm[d["arm"]].append(d)
        for arm, decisions in by_arm.items():
            metrics = score_decisions(decisions, outcomes)
            per_fold.append(
                {"fold_id": f["fold_id"], "dataset": f["dataset"], "kind": f["kind"], "arm": arm, "metrics": metrics}
            )
            group = grouped[(f["dataset"], f["kind"], arm)]
            group["decisions"] += decisions
            group["outcomes"] += outcomes
            group["folds"].append(f["fold_id"])
        for matched in fits[f["fold_id"]]["matched_exposure"]:
            arm = matched["arm"]
            ids = set(matched["point_ids"])
            ranking_decisions = [
                dict(
                    d,
                    action="ADAPT" if d["id"] in ids else "FREEZE",
                    lower=None,
                    upper=None,
                    interval_status="not_applicable",
                )
                for d in by_arm[arm]
            ]
            matched_metrics.append(
                {
                    "fold_id": f["fold_id"],
                    "dataset": f["dataset"],
                    "kind": f["kind"],
                    "arm": arm,
                    "constant_radius": matched["constant_radius"],
                    "same_selection": matched["same_selection"],
                    "exposure": matched["exposure"],
                    "gate": score_decisions(by_arm[arm], outcomes),
                    "point_at_matched_exposure": score_decisions(ranking_decisions, outcomes),
                }
            )
    summary = []
    group_metrics = []
    for (dataset, kind, arm), data in sorted(grouped.items()):
        metrics = score_decisions(data["decisions"], data["outcomes"])
        cluster = "checkpoint" if dataset.startswith("mixed") else "environment"
        group_ids = sorted({r[cluster] for r in data["outcomes"]})
        id_truth = {r["id"]: r for r in data["outcomes"]}
        cluster_loss = []
        point_cluster_loss = []
        point = grouped[(dataset, kind, "point")]
        point_map = {d["id"]: d for d in point["decisions"]}
        for g in group_ids:
            ds = [d for d in data["decisions"] if id_truth[d["id"]][cluster] == g]
            ys = [id_truth[d["id"]] for d in ds]
            gm = score_decisions(ds, ys)
            pm = score_decisions([point_map[d["id"]] for d in ds], ys)
            group_metrics.append(
                {"dataset": dataset, "kind": kind, "arm": arm, "cluster_unit": cluster, "cluster": g, "metrics": gm}
            )
            cluster_loss.append([gm["harm_weighted_losses"][str(w)] for w in [1, 5, 20]])
            point_cluster_loss.append([pm["harm_weighted_losses"][str(w)] for w in [1, 5, 20]])
        delta = np.asarray(cluster_loss) - np.asarray(point_cluster_loss)
        rng = np.random.default_rng(20260921)
        resampled = rng.integers(0, len(group_ids), (1000, len(group_ids)))
        boot = delta[resampled].mean(axis=1)
        ci = {
            str(w): {
                "mean_equal_cluster_delta": float(delta[:, j].mean()),
                "descriptive_95_percentile": [float(x) for x in np.quantile(boot[:, j], [0.025, 0.975])],
            }
            for j, w in enumerate([1, 5, 20])
        }
        summary.append(
            {
                "dataset": dataset,
                "kind": kind,
                "arm": arm,
                "metrics": metrics,
                "cluster_unit": cluster,
                "cluster_count": len(group_ids),
                "paired_loss_minus_point": ci,
            }
        )
    write_fresh_json(
        output / "results_summary.json",
        {
            "status": "COMPLETED_RETROSPECTIVE",
            "rows": summary,
            "caveats": read_json(output / "protocol_seal.json")["protocol"]["limitations"],
            "preparation": prep,
            "decisions": dec,
            "bootstrap": "1000 paired resamples of entire corruption environments or checkpoints; descriptive, low N, no population CI claim",
        },
    )
    write_fresh_json(output / "per_fold_metrics.json", per_fold)
    write_fresh_json(output / "per_group_metrics.json", group_metrics)
    write_fresh_json(output / "matched_exposure_metrics.json", matched_metrics)
    names = ["results_summary.json", "per_fold_metrics.json", "per_group_metrics.json", "matched_exposure_metrics.json"]
    write_fresh_json(
        output / "score_receipt.json",
        {
            "completed_at": now(),
            "seconds": time.perf_counter() - started,
            "decide_receipt_sha256": sha256_file(output / "decide_receipt.json"),
            "outputs": {name: sha256_file(output / name) for name in names},
        },
    )
    write_fresh_json(
        output / "runtime_receipt.json",
        {
            "command": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "measured_scope": "saved aggregate features only; no neural adaptation, inference, or image-level evidence regenerated",
        },
    )


def rescore_phase(source, output):
    """A reporting-only correction retains all sealed decisions and fit receipts."""
    if output.exists():
        raise FileExistsError(f"fresh report directory required: {output}")
    verify_phase(source, "decide")
    verify_phase(source, "prepare")
    verify_phase(source, "score")
    output.mkdir(parents=True)
    names = [
        "protocol_seal.json",
        "prepare_receipt.json",
        "prepared_decision_inputs.json.gz",
        "heldout_score_outcomes.json.gz",
        "decide_receipt.json",
        "sealed_decisions.json.gz",
        "fit_and_calibration_receipts.json",
    ]
    for name in names:
        shutil.copy2(source / name, output / name)
    write_fresh_json(
        output / "reporting_correction_receipt.json",
        {
            "created_at": now(),
            "source_attempt": str(source),
            "source_score_receipt_sha256": sha256_file(source / "score_receipt.json"),
            "changed": "Directional false FREEZE now counts committed FREEZE with B >= 0. Helpful non-ADAPT forgone count is separate. Negative ADAPT and zero-benefit ADAPT are separated.",
            "unchanged": "All model fits, thresholds, calibrated radii, splits, input hashes and sealed decisions are byte-identical. No refitting or outcome-based tuning.",
            "executed_decision_source": "See original attempt executed_source directory and original protocol seal.",
            "scoring_implementation_sha256": {
                str(Path(__file__).relative_to(ROOT)): sha256_file(__file__),
                "kga/calibration_value.py": sha256_file(ROOT / "kga/calibration_value.py"),
            },
            "copied_authorities": {name: sha256_file(output / name) for name in names},
        },
    )
    score_phase(output)


def group_risk_phase(source, output):
    """Add descriptive held-out group risk without changing any old authority."""
    if output.exists():
        raise FileExistsError(f"fresh group-risk directory required: {output}")
    canonical_path = source / "FINAL_RESULTS_SUMMARY.json"
    canonical_sha = sha256_file(canonical_path)
    canonical = read_json(canonical_path)
    existing = {(r["dataset"], r["kind"], r["arm"]): r for r in canonical["rows"]}
    for name, sha in canonical["source_summary_sha256"].items():
        verify_sha256(ROOT / name, sha)
    grouped = defaultdict(lambda: {"decisions": [], "outcomes": []})
    per_fold = []
    input_hashes = {str(canonical_path): canonical_sha}
    for source_name in canonical["authorities"].values():
        authority = ROOT / source_name
        for phase in ["prepare", "decide", "score"]:
            receipt = verify_phase(authority, phase)
            input_hashes[str(authority / f"{phase}_receipt.json")] = sha256_file(authority / f"{phase}_receipt.json")
            input_hashes.update({str(authority / name): sha for name, sha in receipt["outputs"].items()})
        truth = {f["fold_id"]: f for f in read_json(authority / "heldout_score_outcomes.json.gz")}
        for fold in read_json(authority / "sealed_decisions.json.gz"):
            selected_authority = canonical["authorities"][
                "imagenetc" if fold["dataset"].startswith("imagenetc/") else "cifar10c_and_mixed_checkpoints"
            ]
            if source_name != selected_authority:
                continue
            outcomes = [dict(r, fold_id=fold["fold_id"]) for r in truth[fold["fold_id"]]["outcomes"]]
            arms = defaultdict(list)
            for decision in fold["decisions"]:
                arms[decision["arm"]].append(dict(decision, fold_id=fold["fold_id"]))
            for arm, decisions in arms.items():
                key = (fold["dataset"], fold["kind"], arm)
                if key not in existing:
                    raise ValueError("unexpected cohort/arm absent from canonical results")
                grouped[key]["decisions"].extend(decisions)
                grouped[key]["outcomes"].extend(outcomes)
                per_fold.append(
                    {
                        "fold_id": fold["fold_id"],
                        "dataset": fold["dataset"],
                        "kind": fold["kind"],
                        "arm": arm,
                        "group_metrics": score_semantic_groups(decisions, outcomes),
                    }
                )
    if set(grouped) != set(existing):
        raise ValueError("group report does not cover the complete canonical cohort/arm set")
    rows = []
    for (dataset, kind, arm), data in sorted(grouped.items()):
        cell = existing[(dataset, kind, arm)]["metrics"]
        rows.append(
            {
                "dataset": dataset,
                "kind": kind,
                "arm": arm,
                "heldout_group_metrics": score_semantic_groups(data["decisions"], data["outcomes"]),
                "equal_cell_comparison": {
                    key: cell[key]
                    for key in [
                        "n",
                        "adapt",
                        "false_adapt_count",
                        "false_adapt_conditional",
                        "false_adapt_unconditional",
                        "harm_weighted_losses",
                    ]
                },
            }
        )
    verify_sha256(canonical_path, canonical_sha)
    output.mkdir(parents=True)
    write_fresh_json(
        output / "heldout_semantic_group_risk.json",
        {
            "status": "COMPLETE_DESCRIPTIVE_GROUP_REPORT",
            "rows": rows,
            "unit": "(fold_id, semantic_group)",
            "exposure": "At least one ADAPT cell in the group.",
            "error": "At least one ADAPT cell with B <= 0 in the group.",
            "conditional_denominator": "Adapted groups only; null when zero groups adapt.",
            "unconditional_denominator": "All held-out groups, including those with no ADAPT.",
            "utility": "Mean cell loss inside each group, then equal mean across all groups; distinct from equal-cell utility.",
            "limits": "Descriptive evaluation on related historical groups; does not establish exchangeability, independent deployments, or population risk control.",
        },
    )
    write_fresh_json(output / "per_fold_semantic_group_risk.json", per_fold)
    implementations = {
        str(Path(__file__).relative_to(ROOT)): sha256_file(__file__),
        "kga/calibration_value.py": sha256_file(ROOT / "kga/calibration_value.py"),
    }
    for name, sha in implementations.items():
        target = output / "scoring_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        verify_sha256(target, sha)
    write_fresh_json(
        output / "GROUP_RISK_CORRECTION_RECEIPT.json",
        {
            "completed_at": now(),
            "command": sys.argv,
            "correction": "Adds previously omitted held-out semantic-group error/exposure and equal-group utility alongside unchanged equal-cell results.",
            "no_refitting_or_deciding": True,
            "original_canonical_authority_unchanged": True,
            "inputs_sha256": input_hashes,
            "scoring_implementation_sha256": implementations,
            "outputs_sha256": {
                name: sha256_file(output / name)
                for name in ["heldout_semantic_group_risk.json", "per_fold_semantic_group_risk.json"]
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "decide", "score", "run", "rescore", "group-risk"])
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/research/kbound/next_phase/calibration_value_protocol.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-output", type=Path)
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()
    if args.phase == "group-risk":
        if args.source_output is None:
            parser.error("--source-output required for group-risk")
        group_risk_phase(args.source_output, args.output)
    if args.phase == "rescore":
        if args.source_output is None:
            parser.error("--source-output required for rescore")
        rescore_phase(args.source_output, args.output)
    if args.phase in ["prepare", "run"]:
        prepare(args.protocol, args.output, args.datasets)
    if args.phase in ["decide", "run"]:
        decide_phase(args.output)
    if args.phase in ["score", "run"]:
        score_phase(args.output)


if __name__ == "__main__":
    main()
