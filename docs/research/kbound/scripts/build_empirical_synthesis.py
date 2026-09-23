#!/usr/bin/env python3
"""Recompute a descriptive synthesis from frozen, already opened evidence.

No training, network access, target pixels, original runner imports, or source
mutations. This module intentionally computes metrics independently of the
historical scoring implementation. The output is a new retrospective analysis;
it does not replace the frozen authorities or manufacture missing provenance.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CALIBRATION = "output/next_phase/calibration_value_v1"
JOURNAL = "docs/research/kbound/journal_revision"
BOOTSTRAP_SEED = 20260921
BOOTSTRAP_RESAMPLES = 1000
SCHEMA = "kbound-empirical-synthesis-v1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read_bound_input(root, relative, bindings, expected_sha256=None):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("input path escapes repository")
    data = path.read_bytes()
    actual = sha256(data)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    bindings[relative] = {"path": relative, "bytes": len(data), "sha256": actual}
    if relative.endswith(".gz"):
        data = gzip.decompress(data)
    return json.loads(data)


def checked_cells(cells):
    if not cells:
        raise ValueError("empty cell collection")
    result = list(cells)
    keys = [(r["fold_id"], r["cell_id"]) for r in result]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate fold/cell identity")
    for r in result:
        if r["action"] not in {"ADAPT", "FREEZE", "ABSTAIN"}:
            raise ValueError("unknown action")
        for k in ("frozen_accuracy", "candidate_accuracy"):
            if not math.isfinite(r[k]) or not 0 <= r[k] <= 1:
                raise ValueError("accuracy must be finite and in [0, 1]")
        lower, upper = r.get("lower"), r.get("upper")
        if (lower is None) != (upper is None):
            raise ValueError("both interval endpoints must be present or absent")
        if lower is not None and (not math.isfinite(lower) or not math.isfinite(upper) or lower > upper):
            raise ValueError("invalid finite interval")
    return result


def benefit(row):
    return row["candidate_accuracy"] - row["frozen_accuracy"]


def cell_loss(row, weight=5):
    value = benefit(row)
    return weight * max(-value, 0) if row["action"] == "ADAPT" else max(value, 0)


def summarize_cells(cells):
    cells = checked_cells(cells)
    n = len(cells)
    counts = Counter(r["action"] for r in cells)
    accepted = [r for r in cells if r["action"] == "ADAPT"]
    groups = defaultdict(list)
    for r in cells:
        groups[(r["fold_id"], r["group"])].append(r)
    finite = [r for r in cells if r.get("lower") is not None]
    complete = [g for g in groups.values() if all(r.get("lower") is not None for r in g)]

    def included(r):
        return r["lower"] <= benefit(r) <= r["upper"]

    exposed_groups = [g for g in groups.values() if any(r["action"] == "ADAPT" for r in g)]
    error_groups = sum(any(r["action"] == "ADAPT" and benefit(r) <= 0 for r in g) for g in exposed_groups)
    strictly_harmful_groups = sum(any(r["action"] == "ADAPT" and benefit(r) < 0 for r in g) for g in exposed_groups)
    nonpositive = sum(benefit(r) <= 0 for r in accepted)
    helpful = sum(benefit(r) > 0 for r in cells)
    helpful_accepted = sum(benefit(r) > 0 for r in accepted)
    policy = [r["candidate_accuracy"] if r["action"] == "ADAPT" else r["frozen_accuracy"] for r in cells]
    return {
        "n_cells": n,
        "n_groups": len(groups),
        "n_clusters": len({r["cluster"] for r in cells}),
        "cell_weighting": "equal recorded cell; not deployment frequency or pooled image weighting",
        "action_counts": {a: counts[a] for a in ("ADAPT", "FREEZE", "ABSTAIN")},
        "exposure_rate": len(accepted) / n,
        "abstention_rate": counts["ABSTAIN"] / n,
        "strict_decision_rate": (counts["ADAPT"] + counts["FREEZE"]) / n,
        "accuracy_percent": {
            "frozen": 100 * mean(r["frozen_accuracy"] for r in cells),
            "candidate": 100 * mean(r["candidate_accuracy"] for r in cells),
            "policy": 100 * mean(policy),
        },
        "pooled_image_accuracy_percent": None,
        "pooled_image_accuracy_reason": "Not the declared estimand; image-level counts and cross-cell overlap are not resolved by aggregate accuracies.",
        "macro_f1": None,
        "macro_f1_reason": "No aggregate macro-F1 estimand is declared here; aggregate accuracy cannot recover class precision, recall or F1.",
        "mean_policy_gain_pp": 100 * mean(p - r["frozen_accuracy"] for p, r in zip(policy, cells)),
        "oracle_regret_pp": 100 * mean(cell_loss(r, 1) for r in cells),
        "weighted_loss_5_pp": 100 * mean(cell_loss(r, 5) for r in cells),
        "weighted_loss_sensitivity_pp": {str(w): 100 * mean(cell_loss(r, w) for r in cells) for w in (1, 5, 20)},
        "mean_positive_benefit_forgone_pp": 100 * sum(max(benefit(r), 0) for r in cells if r["action"] != "ADAPT") / n,
        "mean_accepted_harm_magnitude_pp": 100 * sum(max(-benefit(r), 0) for r in accepted) / n,
        "helpful_opportunities": helpful,
        "helpful_accepted_count": helpful_accepted,
        "helpful_opportunities_missed": helpful - helpful_accepted,
        "helpful_acceptance_fraction": helpful_accepted / helpful if helpful else None,
        "false_freeze_count": sum(r["action"] == "FREEZE" and benefit(r) >= 0 for r in cells),
        "nonpositive_accepted_count": nonpositive,
        "harmful_accepted_count": sum(benefit(r) < 0 for r in accepted),
        "zero_benefit_accepted_count": sum(benefit(r) == 0 for r in accepted),
        "nonpositive_acceptance_conditional": nonpositive / len(accepted) if accepted else None,
        "conditional_error_unavailable_reason": None
        if accepted
        else "No ADAPT exposure; conditional acceptance error is undefined.",
        "nonpositive_acceptance_unconditional": nonpositive / n,
        "n_exposed_groups": len(exposed_groups),
        "n_error_groups": error_groups,
        "n_strictly_harmful_groups": strictly_harmful_groups,
        "n_zero_only_error_groups": error_groups - strictly_harmful_groups,
        "group_nonpositive_acceptance_conditional": error_groups / len(exposed_groups) if exposed_groups else None,
        "group_conditional_error_unavailable_reason": None
        if exposed_groups
        else "No exposed semantic groups; conditional group error is undefined.",
        "group_nonpositive_acceptance_unconditional": error_groups / len(groups),
        "equal_group_weighted_loss_5_pp": 100 * mean(mean(cell_loss(r) for r in g) for g in groups.values()),
        "intervals": {
            "n_finite_cells": len(finite),
            "n_included_finite_cells": sum(included(r) for r in finite),
            "cell_inclusion": mean(included(r) for r in finite) if finite else None,
            "n_unbounded_cells": sum(r.get("interval_status") == "unbounded" for r in cells),
            "mean_finite_width_pp": 100 * mean(r["upper"] - r["lower"] for r in finite) if finite else None,
            "n_complete_finite_groups": len(complete),
            "n_simultaneously_included_groups": sum(all(included(r) for r in g) for g in complete),
            "simultaneous_group_inclusion": mean(all(included(r) for r in g) for g in complete) if complete else None,
            "denominators": "Cell inclusion uses finite cells only; simultaneous inclusion uses groups for which every cell has finite endpoints. Null denotes unavailable, not zero. Unbounded intervals are counted separately.",
        },
    }


def paired_cluster_comparison(cells, baseline, metric="weighted_loss_5_pp"):
    cells, baseline = checked_cells(cells), checked_cells(baseline)
    left = {(r["fold_id"], r["cell_id"]): r for r in cells}
    right = {(r["fold_id"], r["cell_id"]): r for r in baseline}
    if left.keys() != right.keys():
        raise ValueError("pairing identity mismatch")
    clusters = defaultdict(list)
    for key, r in left.items():
        b = right[key]
        if any(r[k] != b[k] for k in ("cluster", "group", "frozen_accuracy", "candidate_accuracy")):
            raise ValueError("paired outcome or cluster mismatch")
        if metric == "weighted_loss_5_pp":
            delta = 100 * (cell_loss(r) - cell_loss(b))
        elif metric == "oracle_regret_pp":
            delta = 100 * (cell_loss(r, 1) - cell_loss(b, 1))
        else:
            raise ValueError("unsupported paired metric")
        clusters[r["cluster"]].append(delta)
    labels = sorted(clusters)
    delta = np.array([mean(clusters[c]) for c in labels])
    n = len(labels)
    if n >= 2:
        indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, (BOOTSTRAP_RESAMPLES, n))
        interval = [float(x) for x in np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])]
    else:
        interval = None
    return {
        "metric": metric,
        "direction": "policy minus comparator; negative favors policy",
        "n_clusters": n,
        "n_paired_cells": len(cells),
        "mean_equal_cell_delta_pp": mean(v for values in clusters.values() for v in values),
        "mean_equal_cluster_delta_pp": float(delta.mean()),
        "descriptive_paired_cluster_bootstrap_95_percentile_pp": interval,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unavailable_reason": None
        if n >= 2
        else "Fewer than two clusters cannot estimate between-cluster uncertainty.",
        "bootstrap_scope": "Retrospective descriptive sensitivity over named, related historical clusters; not a population confidence guarantee or a multiplicity-corrected significance test.",
        "per_cluster_delta_pp": [
            {"cluster": c, "n_cells": len(clusters[c]), "delta_pp": float(v)} for c, v in zip(labels, delta)
        ],
        "leave_one_cluster_out": [
            {
                "omitted_cluster": c,
                "remaining_clusters": n - 1,
                "mean_equal_cluster_delta_pp": float(np.delete(delta, i).mean()) if n > 1 else None,
            }
            for i, c in enumerate(labels)
        ],
        "influence_scope": "New exploratory leave-one-cluster-out diagnostic; no clusters are removed from the primary estimate.",
    }


def matched_exposure_cells(cells):
    cells = checked_cells(cells)
    folds = defaultdict(list)
    for r in cells:
        if not math.isfinite(r["prediction"]):
            raise ValueError("prediction must be finite")
        folds[r["fold_id"]].append(r)
    result = []
    for group in folds.values():
        n = sum(r["action"] == "ADAPT" for r in group)
        selected = {r["cell_id"] for r in sorted(group, key=lambda r: (-r["prediction"], r["cell_id"]))[:n]}
        result.extend(
            dict(
                r,
                action="ADAPT" if r["cell_id"] in selected else "FREEZE",
                lower=None,
                upper=None,
                interval_status="not_applicable",
            )
            for r in group
        )
    return result


def verify_metric(actual, expected, label):
    if actual is None or expected is None:
        same = actual is None and expected is None
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        same = (
            math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11)
        )
    else:
        same = actual == expected
    if not same:
        raise ValueError(f"authority mismatch {label}: {actual!r} != {expected!r}")
    return 1


def verify_secondary_macro_f1(row):
    values = []
    for counts in row["per_class"]:
        for key in ("tp", "fp", "fn"):
            if not isinstance(counts[key], int) or counts[key] < 0:
                raise ValueError("invalid class counts")
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        value = 2 * counts["tp"] / denominator if denominator else row["zero_denominator_convention"]
        verify_metric(value, counts["f1"], "class F1")
        values.append(value)
    verify_metric(len(values), row["n_output_indicators"], "class count")
    value = mean(values)
    verify_metric(value, row["macro_f1"], "macro-F1")
    return value


def verify_historical_summary(metrics, authority, label):
    fields = {
        "n_cells": "n",
        "oracle_regret_pp": "oracle_regret",
        "mean_policy_gain_pp": "score_gain",
        "false_freeze_count": "false_freeze_count",
        "helpful_opportunities_missed": "forgone_helpful_nonadapt_count",
        "harmful_accepted_count": "harmful_adapt_count",
        "nonpositive_accepted_count": "false_adapt_count",
        "zero_benefit_accepted_count": "zero_benefit_adapt_count",
        "nonpositive_acceptance_conditional": "false_adapt_conditional",
        "nonpositive_acceptance_unconditional": "false_adapt_unconditional",
    }
    count = 0
    for ours, theirs in fields.items():
        value = metrics[ours] / 100 if ours.endswith("_pp") else metrics[ours]
        count += verify_metric(value, authority[theirs], f"{label}/{theirs}")
    for action, value in metrics["action_counts"].items():
        count += verify_metric(value, authority[action.lower()], f"{label}/{action}")
    for weight, value in metrics["weighted_loss_sensitivity_pp"].items():
        count += verify_metric(value / 100, authority["harm_weighted_losses"][weight], f"{label}/loss{weight}")
    for ours, theirs in [
        ("cell_inclusion", "interval_inclusion"),
        ("n_finite_cells", "finite_interval_count"),
        ("n_unbounded_cells", "unbounded_interval_count"),
    ]:
        count += verify_metric(metrics["intervals"][ours], authority[theirs], f"{label}/{theirs}")
    return count


def normalized_cell(decision, outcome, fold, cluster_key):
    return {
        "cell_id": decision["id"],
        "fold_id": fold["fold_id"],
        "group": decision["group"],
        "cluster": outcome[cluster_key],
        "frozen_accuracy": outcome["frozen_score"],
        "candidate_accuracy": outcome["candidate_score"],
        "action": decision["action"],
        "lower": decision.get("lower"),
        "upper": decision.get("upper"),
        "interval_status": decision.get("interval_status", "not_applicable"),
        "prediction": decision["prediction"],
    }


def calibration_synthesis(root, bindings):
    def read(path, expected=None):
        return read_bound_input(root, path, bindings, expected)

    protocol = read("docs/research/kbound/next_phase/calibration_value_protocol.json")
    original = {}
    for item in protocol["inputs"]:
        data = read(f"{CALIBRATION}/original_inputs/{item['path']}", item["sha256"])
        if item["role"] == "compact_panel":
            for r in data["records"]:
                identity = digest([f"{r['benchmark']}/{r['method']}", r["seed"], r["condition"]])
                original[identity] = (r["a0"], r["a_adapted"], r["Z"])
                verify_metric(r["a_adapted"] - r["a0"], r["B"], "compact benefit")
        elif item["role"] == "mixed_checkpoint_panel":
            for model, records in data.items():
                for r in records:
                    original[digest(["mixed", model, r["condition"]])] = (r["a0"], r["aa"], r["Z"])
                    verify_metric(r["aa"] - r["a0"], r["B"], "checkpoint benefit")
    final = read(f"{CALIBRATION}/FINAL_RESULTS_SUMMARY.json")
    group_source = read(f"{CALIBRATION}/group_risk_report_05/heldout_semantic_group_risk.json")
    groups = {(x["dataset"], x["kind"], x["arm"]): x["heldout_group_metrics"] for x in group_source["rows"]}
    cells, matched = defaultdict(list), {}
    source_paths, fit_evidence = {}, []
    linked_outcomes = 0
    for directory, use_imagenet in [("report_03", False), ("attempt_04_imagenet_schema_corrected", True)]:
        prefix = f"{CALIBRATION}/{directory}"
        prepare = read(f"{prefix}/prepare_receipt.json")
        decide = read(f"{prefix}/decide_receipt.json")
        verify_metric(
            bindings[f"{prefix}/prepare_receipt.json"]["sha256"],
            decide["prepare_receipt_sha256"],
            "prepare receipt SHA",
        )
        score = read(f"{prefix}/score_receipt.json")
        verify_metric(
            bindings[f"{prefix}/decide_receipt.json"]["sha256"], score["decide_receipt_sha256"], "decision receipt SHA"
        )
        summary = read(
            f"{prefix}/results_summary.json", final["source_summary_sha256"][f"{prefix}/results_summary.json"]
        )
        outcomes = {
            f["fold_id"]: f
            for f in read(
                f"{prefix}/heldout_score_outcomes.json.gz", prepare["outputs"]["heldout_score_outcomes.json.gz"]
            )
        }
        decisions = read(f"{prefix}/sealed_decisions.json.gz", decide["outputs"]["sealed_decisions.json.gz"])
        fit = read(
            f"{prefix}/fit_and_calibration_receipts.json", decide["outputs"]["fit_and_calibration_receipts.json"]
        )
        fit_evidence.extend(f for f in fit if f["dataset"].startswith("imagenetc") == use_imagenet)
        for f in decisions:
            if f["dataset"].startswith("imagenetc") != use_imagenet:
                continue
            truth = {r["id"]: r for r in outcomes[f["fold_id"]]["outcomes"]}
            if len(truth) != len(outcomes[f["fold_id"]]["outcomes"]):
                raise ValueError("duplicate held-out outcome")
            for r in truth.values():
                verify_metric(
                    (r["frozen_score"], r["candidate_score"], r["features"]),
                    original[r["id"]],
                    "original score/feature link",
                )
                linked_outcomes += 1
            for d in f["decisions"]:
                key = (f["dataset"], f["kind"], d["arm"])
                y = truth[d["id"]]
                verify_metric(d["group"], y["group"], "semantic group identity")
                cells[key].append(
                    normalized_cell(d, y, f, "checkpoint" if f["dataset"].startswith("mixed") else "environment")
                )
                source_paths[key] = prefix
        for record in read(
            f"{prefix}/matched_exposure_metrics.json", score["outputs"]["matched_exposure_metrics.json"]
        ):
            if record["dataset"].startswith("imagenetc") == use_imagenet:
                matched[record["dataset"], record["kind"], record["arm"], record["fold_id"]] = record
        # Every published final row must occur verbatim in the selected correction authority.
        selected = {(r["dataset"], r["kind"], r["arm"]): r for r in summary["rows"]}
        for r in final["rows"]:
            if r["dataset"].startswith("imagenetc") == use_imagenet:
                verify_metric(r, selected[r["dataset"], r["kind"], r["arm"]], "final selected authority")
    rows, checks, matched_checks, constant_checks = [], 0, 0, 0
    for saved in final["rows"]:
        key = saved["dataset"], saved["kind"], saved["arm"]
        data = cells[key]
        metrics = summarize_cells(data)
        checks += verify_historical_summary(metrics, saved["metrics"], "/".join(key))
        ga = groups[key]
        for ours, theirs in [
            ("n_groups", "n_all_groups"),
            ("n_exposed_groups", "n_adapted_groups"),
            ("n_error_groups", "n_error_groups"),
            ("n_strictly_harmful_groups", "n_strictly_harmful_adapted_groups"),
            ("n_zero_only_error_groups", "n_zero_only_error_groups"),
            ("group_nonpositive_acceptance_conditional", "false_adapt_conditional"),
            ("group_nonpositive_acceptance_unconditional", "false_adapt_unconditional"),
        ]:
            checks += verify_metric(metrics[ours], ga[theirs], "/".join(key) + "/" + theirs)
        checks += verify_metric(
            metrics["equal_group_weighted_loss_5_pp"] / 100,
            ga["equal_group_harm_weighted_losses"]["5"],
            "equal-group loss",
        )
        comparisons = {
            arm: paired_cluster_comparison(data, cells[key[:2] + (arm,)]) for arm in ("point", "fixed_margin_harm5")
        }
        point = comparisons["point"]
        checks += verify_metric(
            point["mean_equal_cluster_delta_pp"] / 100,
            saved["paired_loss_minus_point"]["5"]["mean_equal_cluster_delta"],
            "point cluster delta",
        )
        for actual, expected in zip(
            point["descriptive_paired_cluster_bootstrap_95_percentile_pp"],
            saved["paired_loss_minus_point"]["5"]["descriptive_95_percentile"],
        ):
            checks += verify_metric(actual / 100, expected, "published point bootstrap")
        matched_report = None
        applicable = key[2].startswith(("constant_", "scaled_", "environment_max"))
        if applicable:
            control = matched_exposure_cells(data)
            control_by_fold = defaultdict(list)
            data_by_fold = defaultdict(list)
            for r in control:
                control_by_fold[r["fold_id"]].append(r)
            for r in data:
                data_by_fold[r["fold_id"]].append(r)
            same = True
            for fold, rows_in_fold in data_by_fold.items():
                recorded = matched[key + (fold,)]
                comparison_rows = control_by_fold[fold]
                m = summarize_cells(rows_in_fold)
                p = summarize_cells(comparison_rows)
                checks += verify_historical_summary(m, recorded["gate"], "matched gate")
                checks += verify_historical_summary(p, recorded["point_at_matched_exposure"], "matched ranking")
                selected = {r["cell_id"] for r in rows_in_fold if r["action"] == "ADAPT"}
                ranked = {r["cell_id"] for r in comparison_rows if r["action"] == "ADAPT"}
                checks += verify_metric(selected == ranked, recorded["same_selection"], "matched selection")
                same &= selected == ranked
                matched_checks += 1
                if recorded["constant_radius"]:
                    checks += verify_metric(selected, ranked, "constant-radius equivalence")
                    constant_checks += 1
            matched_report = {
                "metrics": summarize_cells(control),
                "same_selected_cells": same,
                "comparison": paired_cluster_comparison(data, control),
                "exposure_contract": "Within each held-out fold, choose exactly the gate ADAPT count by descending shared predicted benefit, with cell-ID tie breaking; no outcome ranking.",
            }
        rows.append(
            {
                "study": "shared_predictor_retrospective",
                "dataset": key[0],
                "design": key[1],
                "arm": key[2],
                "cluster_unit": saved["cluster_unit"],
                "independent_deployments": None,
                "independent_deployments_reason": "Historical related corruption/checkpoint units are not independent newly executed deployments.",
                "accuracy_contract": "ordinary classification accuracy, equal recorded-cell mean",
                "authority": source_paths[key],
                "metrics": metrics,
                "comparisons": comparisons,
                "matched_exposure": matched_report,
            }
        )
    if len(rows) != 160 or len({(r["dataset"], r["design"]) for r in rows}) != 8:
        raise ValueError("expected all eight cohort/designs and 160 arm rows")
    return rows, {
        "status": "PASS",
        "original_input_files": len(protocol["inputs"]),
        "unique_original_records": len(original),
        "scored_fold_cell_links": linked_outcomes,
        "scored_arm_decisions": sum(map(len, cells.values())),
        "cohort_arm_rows": len(rows),
        "heldout_group_rows": len(groups),
        "matched_fold_arm_rows": matched_checks,
        "constant_radius_equivalence_checks": constant_checks,
        "scalar_authority_comparisons": checks,
        "selected_fold_count": len(fit_evidence),
        "independent_calibration_environment_counts": sorted(
            {f["fit"]["independent_calibration_environment_count"] for f in fit_evidence}
        ),
        "maximum_calibration_group_count": max(len(f["fit"]["role_groups"]["cal"]) for f in fit_evidence),
        "scope": "Stored record arithmetic, input hashes, selected correction authorities, decisions and descriptive uncertainty. Does not authenticate missing original producer execution.",
    }


def principal_cifar(root, bindings):
    prefix = "experiments/kbound/results/reconciled_panels_v1"
    canonical_path = f"{prefix}/canonical_panel_results.json"
    canonical = read_bound_input(root, canonical_path, bindings)
    source_manifest = read_bound_input(
        root, f"{prefix}/source_manifest.json", bindings, canonical["source_manifest_sha256"]
    )
    hashes = {r["destination"]: r["compact_sha256"] for r in source_manifest["files"]}
    rows, checks = [], 0
    for candidate, panel in sorted(canonical["panels"]["cifar10c"]["panel"]["candidates"].items()):
        cells = []
        for f in panel["per_file"]:
            source = f"{prefix}/source/cifar10c/per_condition_cifar10c_{candidate}_seed{f['seed']}.json"
            compact = read_bound_input(root, source, bindings, hashes[source])
            actions = {r["sample_id"]: r for r in f["current_cell_authority"]["cells"]}
            seed_cells = []
            for r in compact["records"]:
                identity = f"{candidate}|seed={f['seed']}|condition={r['condition']}"
                d = actions[identity]
                group = r["condition"].rsplit("|", 1)[0]
                seed_cells.append(
                    {
                        "cell_id": identity,
                        "fold_id": "principal",
                        "group": group,
                        "cluster": r["condition"].split("|")[0],
                        "seed": f["seed"],
                        "frozen_accuracy": r["a0"],
                        "candidate_accuracy": r["a_adapted"],
                        "action": d["action"],
                        "prediction": d["prediction"],
                        "lower": d["prediction"] - d["radius"],
                        "upper": d["prediction"] + d["radius"],
                        "interval_status": "finite",
                    }
                )
                checks += verify_metric(benefit(seed_cells[-1]), r["B"], "principal CIFAR benefit")
            for arm, action in [("kga", None), ("always_adapt", "ADAPT"), ("always_freeze", "FREEZE")]:
                ds = seed_cells if action is None else [dict(r, action=action) for r in seed_cells]
                checks += verify_metric(
                    summarize_cells(ds)["oracle_regret_pp"] / 100,
                    f["score"]["regret"][arm],
                    "principal per-seed regret",
                )
            cells.extend(seed_cells)
        arms = {
            "kga": cells,
            "always_adapt": [
                dict(r, action="ADAPT", lower=None, upper=None, interval_status="not_applicable") for r in cells
            ],
            "always_freeze": [
                dict(r, action="FREEZE", lower=None, upper=None, interval_status="not_applicable") for r in cells
            ],
        }
        for arm, ds in arms.items():
            m = summarize_cells(ds)
            checks += verify_metric(m["oracle_regret_pp"] / 100, panel["regret"][arm], "principal pooled regret")
            rows.append(
                {
                    "study": "principal_cifar_current_policy",
                    "dataset": f"cifar10c/{candidate}",
                    "design": "historical_current_cell_policy",
                    "arm": arm,
                    "authority": canonical_path,
                    "metrics": m,
                    "accuracy_contract": "ordinary classification accuracy, equal recorded-cell mean; not a new raw inference run",
                    "cluster_unit": "corruption_type",
                    "run_seed_count": len(panel["per_file"]),
                    "independent_deployments": None,
                    "independent_deployments_reason": "Five stream seeds on six related corruptions do not identify five independent trained models or deployments.",
                    "comparisons": {
                        a: paired_cluster_comparison(ds, arms[a], metric="oracle_regret_pp")
                        for a in ("always_adapt", "always_freeze")
                    },
                    "seed_policy_minus_always_adapt_accuracy_pp": [
                        {
                            "seed": s,
                            "delta_pp": mean(
                                100
                                * (
                                    (r["candidate_accuracy"] if r["action"] == "ADAPT" else r["frozen_accuracy"])
                                    - r["candidate_accuracy"]
                                )
                                for r in ds
                                if r["seed"] == s
                            ),
                        }
                        for s in sorted({r["seed"] for r in ds})
                    ],
                }
            )
    return rows, checks


def principal_cct(root, bindings):
    release_path = "docs/research/kbound/paper/generated/cct20_release_manifest.json"
    release = read_bound_input(root, release_path, bindings)
    bundle_path = "docs/research/kbound/release/cct20_public_evidence_bundle.zip"
    data = (root / bundle_path).read_bytes()
    bindings[bundle_path] = {"path": bundle_path, "bytes": len(data), "sha256": sha256(data)}
    source = release["upstream_artifacts"]["one_shot_score"]
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        manifest_bytes = bundle.read("manifest.json")
        manifest = json.loads(manifest_bytes)
        descriptor = [
            o
            for o in manifest["objects"]
            if o["content_role"] == "upstream_artifacts.one_shot_score" and o["source_sha256"] == source["sha256"]
        ]
        if len(descriptor) != 1:
            raise ValueError("ambiguous or missing CCT score object")
        descriptor = descriptor[0]
        payload = bundle.read(descriptor["bundle_path"])
        verify_metric(sha256(payload), descriptor["published_sha256"], "CCT published object SHA")
        score = json.loads(payload)
    bindings[bundle_path]["consulted_members"] = [
        {"path": "manifest.json", "bytes": len(manifest_bytes), "sha256": sha256(manifest_bytes)},
        {
            "path": descriptor["bundle_path"],
            "bytes": len(payload),
            "sha256": sha256(payload),
            "normalization": descriptor["normalization"],
            "original_score_sha256": source["sha256"],
        },
    ]
    cells, secondary, checks = [], [], 0
    for r in score["cells"]:
        accuracy = r["set_membership_top1_accuracy"]
        cell = {
            "cell_id": f"seed{r['checkpoint_seed']}_location{r['location_id']}",
            "fold_id": "cct_target",
            "group": str(r["location_id"]),
            "cluster": str(r["location_id"]),
            "frozen_accuracy": accuracy["always_freeze"],
            "candidate_accuracy": accuracy["always_adapt"],
            "action": r["decision"],
            "lower": None,
            "upper": None,
            "interval_status": "not_reconstructed",
            "checkpoint_seed": r["checkpoint_seed"],
            "n_evaluation_images": r["n_evaluation_images"],
        }
        checks += verify_metric(benefit(cell), r["adaptation_benefit"], "CCT benefit")
        checks += verify_metric(
            accuracy["kga"],
            accuracy["always_adapt"] if cell["action"] == "ADAPT" else accuracy["always_freeze"],
            "CCT policy accuracy",
        )
        for arm in ("kga", "always_adapt", "always_freeze"):
            checks += verify_metric(
                max(accuracy["always_adapt"], accuracy["always_freeze"]) - accuracy[arm],
                r["regret_to_better_fixed_action"][arm],
                "CCT cell regret",
            )
        cells.append(cell)
        secondary.append(
            {
                "cell_id": cell["cell_id"],
                "checkpoint_seed": r["checkpoint_seed"],
                "location_id": r["location_id"],
                "metric_contract": score["label_contract"]["secondary"],
                "macro_f1_by_policy": {a: verify_secondary_macro_f1(v) for a, v in r["multilabel_macro_f1"].items()},
                "scope": "Archived descriptive cell-level secondary metric, checked from class TP/FP/FN counts; not a locked aggregate or inference claim.",
            }
        )
    arms = {
        "kga": cells,
        "always_adapt": [dict(r, action="ADAPT") for r in cells],
        "always_freeze": [dict(r, action="FREEZE") for r in cells],
    }
    rows = []
    for arm, ds in arms.items():
        metrics = summarize_cells(ds)
        rows.append(
            {
                "study": "principal_cct20",
                "dataset": "cct20/tent",
                "design": "5_checkpoints_x_9_camera_locations",
                "arm": arm,
                "authority": release_path,
                "score_member": descriptor["bundle_path"],
                "metrics": metrics,
                "accuracy_contract": score["label_contract"]["primary"],
                "cluster_unit": "camera_location",
                "checkpoint_count": score["checkpoint_count"],
                "location_count": score["location_count"],
                "independent_deployments": None,
                "independent_deployments_reason": "Cross-classified checkpoints and shared locations are not 45 independent deployments.",
                "weighted_loss_scope": "New descriptive harm-weight-5 synthesis, not the original locked endpoint.",
                "prospective_disclosure": release["prospective_disclosure"],
            }
        )
    kga = rows[0]["metrics"]
    for comparator, target in [("always_adapt", rows[1]), ("always_freeze", rows[2])]:
        label = "versus_" + comparator
        contrast = (target["metrics"]["oracle_regret_pp"] - kga["oracle_regret_pp"]) / 100
        checks += verify_metric(
            contrast, release["primary_comparisons"][label]["point_estimate"], "CCT primary contrast"
        )
    for action, value in kga["action_counts"].items():
        checks += verify_metric(value, release["action_exposure"]["counts"][action], "CCT exposure")
    return (
        rows,
        secondary,
        {
            "scalar_authority_checks": checks,
            "original_primary_comparisons": release["primary_comparisons"],
            "original_inference_scope": "Published location sign-flip and cross-classified checkpoint/location interval results carried verbatim, not recomputed or replaced by cell bootstrap.",
            "secondary_scope": release["secondary_outcome_reporting"],
            "retention_endpoint_met": release["safe_utility"]["passes"],
            "strong_routing_success": release["strong_success_checks"]["protocol_strong_success"],
        },
    )


def metric_contracts():
    return {
        "schema": SCHEMA,
        "benefit": "B = candidate accuracy minus frozen accuracy, fraction units; positive means helpful.",
        "accuracy": "100 times equal-cell mean of deployed model accuracy; ABSTAIN and FREEZE deploy frozen. No image-pooling or deployment-frequency claim.",
        "cct_accuracy": "Top-1 correct iff predicted class belongs to the complete distinct annotated category set; not ordinary single-label accuracy or exact multilabel-set accuracy.",
        "oracle_regret": "100 * mean(max(B,0) - B * 1{ADAPT}); unweighted opportunity loss in percentage points.",
        "weighted_loss_5": "100 * mean(max(B,0)*1{not ADAPT} + 5*max(-B,0)*1{ADAPT}); preserve published harm weight 5. It is not accuracy or F1.",
        "nonpositive_acceptance": "ADAPT with B <= 0; strictly harmful uses B < 0. Zero benefit contributes an event but no harm magnitude.",
        "conditional_acceptance_error": "Divide errors by ADAPT count, null if no ADAPT. Unconditional divides by all cells.",
        "group_error": "Group key is (fold, semantic group); any accepted nonpositive cell creates an error. Conditional denominator is exposed groups.",
        "helpful_opportunities_missed": "B > 0 with FREEZE or ABSTAIN. Explicit false FREEZE uses FREEZE with B >= 0, matching archived scoring convention.",
        "cell_inclusion": "Fraction of finite intervals containing B, over finite cell intervals. No nominal transfer guarantee.",
        "simultaneous_group_inclusion": "Fraction of complete finite groups for which every cell interval contains its B. Group-max cell inclusion is separately reported and cannot be substituted.",
        "macro_f1": "Cannot be recovered from aggregate accuracy. CCT archived cell-level 16-indicator macro-F1 is recomputed from class counts with zero denominator => 0, without adding a post-hoc aggregate claim.",
        "uncertainty": "Shared-predictor study: 1000 fixed-seed paired resamples of whole corruption types or source checkpoints; descriptive only. New primary CIFAR comparisons and leave-one-cluster-out diagnostics are retrospective.",
        "matched_exposure": "Rank shared predicted benefit within each scored fold and accept exactly the gate acceptance count; ties by cell ID. Outcome is never used for selection.",
        "null_semantics": "Unavailable or undefined with explicit scope; not zero, not a failed observation, not evidence of safety.",
        "provenance": "Hashes establish inspected bytes and arithmetic lineage; they do not reconstruct missing producer logs, fit data, checkpoint identities or original executions.",
        "excluded_claims": [
            "independent deployment validation",
            "guaranteed 90% natural-shift coverage",
            "general calibration superiority",
            "production safety",
            "unrecorded classifier F1",
        ],
    }


def render_markdown(result):
    lines = [
        "# Empirical synthesis from frozen evidence",
        "",
        "This versioned analysis recomputes preserved results without refitting, training, opening targets, or replacing historical authorities. All new uncertainty and influence analyses are retrospective and descriptive.",
        "",
        "## Shared-predictor comparisons",
        "",
        "Values are **100L5**, percentage points of harm-weight-5 decision loss, lower is better. They are not accuracy or F1.",
        "",
        "| Cohort / hold-out | Cells | Clusters | Point | Margin | Constant 90% | Scaled 90% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    indexed = {(r["dataset"], r["design"], r["arm"]): r for r in result["calibration_rows"]}
    for dataset, design in sorted({k[:2] for k in indexed}):
        ds = [
            indexed[dataset, design, arm]
            for arm in ["point", "fixed_margin_harm5", "constant_cell_alpha0.1", "scaled_cell_alpha0.1"]
        ]
        lines.append(
            f"| {dataset} / {design} | {ds[0]['metrics']['n_cells']} | {ds[0]['metrics']['n_clusters']} | "
            + " | ".join(f"{r['metrics']['weighted_loss_5_pp']:.4f}" for r in ds)
            + " |"
        )
    lines += [
        "",
        "The eight rows are cohort/design summaries, not eight independent deployments. CIFAR has six corruption clusters; ImageNet and the checkpoint designs each have three clusters. The 108 checkpoint records recur in the checkpoint and joint designs.",
        "",
        "## Coverage and the unit of inclusion",
        "",
        "| CIFAR method | Constant-cell inclusion | Scaled-cell inclusion | Group-max cell inclusion | Group-max simultaneous-group inclusion |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in ["eata", "sar", "tent"]:
        ds = [
            indexed[f"cifar10c/{candidate}", "environment", a]["metrics"]["intervals"]
            for a in ["constant_cell_alpha0.1", "scaled_cell_alpha0.1", "constant_group_alpha0.1"]
        ]
        lines.append(
            f"| {candidate} | {100 * ds[0]['cell_inclusion']:.2f}% | {100 * ds[1]['cell_inclusion']:.2f}% | {100 * ds[2]['cell_inclusion']:.2f}% | {100 * ds[2]['simultaneous_group_inclusion']:.2f}% |"
        )
    lines += [
        "",
        "All intervals above were nominally 90%. These observed rates do not support nominal transfer coverage. The independent-environment arms have no eligible calibration environments. Zero exposure in the 10% risk arm leaves conditional error undefined.",
        "",
        "## Tent sensitivity",
        "",
    ]
    tent = indexed["cifar10c/tent", "environment", "scaled_cell_alpha0.1"]
    for label, comp in list(tent["comparisons"].items()) + [
        ("point at matched exposure", tent["matched_exposure"]["comparison"])
    ]:
        interval = comp["descriptive_paired_cluster_bootstrap_95_percentile_pp"]
        jpeg = next(x for x in comp["leave_one_cluster_out"] if x["omitted_cluster"] == "jpeg_compression")
        lines.append(
            f"- Versus {label}: scaled-minus-comparator loss {comp['mean_equal_cluster_delta_pp']:.6f} pp; descriptive 95% bootstrap range [{interval[0]:.6f}, {interval[1]:.6f}] pp. Omitting JPEG for sensitivity only gives {jpeg['mean_equal_cluster_delta_pp']:.6f} pp."
        )
    lines += [
        "",
        "All-cluster estimates remain primary. These low-N, related historical clusters do not establish population significance. Equal-exposure differences isolate ranking from acceptance rate; the much larger comparison against the unthresholded point arm is not an accuracy improvement.",
        "",
        "## Principal CIFAR policies",
        "",
        "| Candidate | Frozen accuracy | Adapt accuracy | KGA accuracy | KGA minus adapt | KGA regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in result["principal_rows"]:
        if r["study"] == "principal_cifar_current_policy" and r["arm"] == "kga":
            m = r["metrics"]
            a = m["accuracy_percent"]
            lines.append(
                f"| {r['dataset']} | {a['frozen']:.4f}% | {a['candidate']:.4f}% | {a['policy']:.4f}% | {a['policy'] - a['candidate']:.4f} pp | {m['oracle_regret_pp']:.4f} pp |"
            )
    lines += [
        "",
        "These are means over recorded condition/seed cells. Their aggregation weights differ from image-pooled accuracy or deployment frequency. SAR remains unfavorable to KGA versus always-adapt.",
        "",
        "## CCT-20 retention and routing",
        "",
    ]
    cct = [r for r in result["principal_rows"] if r["study"] == "principal_cct20" and r["arm"] == "kga"][0]["metrics"]
    a = cct["accuracy_percent"]
    lines += [
        f"Across 45 checkpoint/location cells (five checkpoints, nine locations), equal-cell set-membership top-1 accuracy is {a['frozen']:.4f}% frozen, {a['candidate']:.4f}% always-adapt and {a['policy']:.4f}% KGA. KGA makes {cct['action_counts']['ADAPT']} ADAPT, {cct['action_counts']['FREEZE']} FREEZE and {cct['action_counts']['ABSTAIN']} ABSTAIN decisions. It misses {cct['helpful_opportunities_missed']}/{cct['helpful_opportunities']} helpful opportunities; conditional acceptance error is undefined at zero exposure.",
        "",
        "The locked retention endpoint is met; stronger routing success is not. Published cross-classified interval and location-test results are preserved separately in JSON. Archived 16-indicator cell-level macro-F1 is checked from class counts and retained as secondary descriptive evidence. Aggregate macro-F1 and an aggregate inference claim remain null.",
        "",
        "## Reproduction and boundaries",
        "",
        "Extract the released next_phase_evidence.zip at the repository root first; use the pinned research environment and run:",
        "",
        "```sh",
        "python docs/research/kbound/scripts/build_empirical_synthesis.py",
        "python docs/research/kbound/scripts/build_empirical_synthesis.py --check",
        "```",
        "",
        "The generated input manifest binds every consulted file and CCT bundle member by SHA-256. Inputs are read only. JSON contains all 160 calibration arm rows, all coverage denominators, matched exposure, paired cluster effects, leave-one-cluster-out diagnostics, nine principal CIFAR policy rows and three CCT policy rows. CSV is a flat summary; JSON is authoritative for nested uncertainty and unavailable-value reasons.",
        "",
        "Historical producer-provenance gaps, calibration transfer failures and lack of independent deployment evidence remain. No new protected target was opened and no confirmatory result was manufactured.",
    ]
    return "\n".join(lines) + "\n"


def flattened_csv(result):
    fields = [
        "study",
        "dataset",
        "design",
        "arm",
        "cluster_unit",
        "n_cells",
        "n_clusters",
        "accuracy_policy_percent",
        "oracle_regret_pp",
        "weighted_loss_5_pp",
        "adapt",
        "freeze",
        "abstain",
        "harmful_accepted_count",
        "nonpositive_accepted_count",
        "nonpositive_acceptance_conditional",
        "helpful_opportunities_missed",
        "cell_inclusion",
        "simultaneous_group_inclusion",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for r in result["calibration_rows"] + result["principal_rows"]:
        m = r["metrics"]
        row = {k: r[k] for k in ("study", "dataset", "design", "arm", "cluster_unit")}
        row.update(
            {
                k: m[k]
                for k in (
                    "n_cells",
                    "n_clusters",
                    "oracle_regret_pp",
                    "weighted_loss_5_pp",
                    "harmful_accepted_count",
                    "nonpositive_accepted_count",
                    "nonpositive_acceptance_conditional",
                    "helpful_opportunities_missed",
                )
            }
        )
        row.update(
            accuracy_policy_percent=m["accuracy_percent"]["policy"],
            **{k.lower(): v for k, v in m["action_counts"].items()},
            cell_inclusion=m["intervals"]["cell_inclusion"],
            simultaneous_group_inclusion=m["intervals"]["simultaneous_group_inclusion"],
        )
        writer.writerow({k: "null" if v is None else v for k, v in row.items()})
    return stream.getvalue().encode()


def render_latex_tables(result):
    """Supplement tables; the existing principal manuscript tables stay untouched."""
    indexed = {(r["dataset"], r["design"], r["arm"]): r for r in result["calibration_rows"]}
    lines = [
        "% Generated by build_empirical_synthesis.py from frozen authorities.",
        r"\begin{table}[tb]",
        r"\centering\small",
        r"\caption{Observed inclusion on held-out CIFAR corruption cells and complete semantic groups. "
        r"All four columns use nominal 90\% intervals; simultaneous-group inclusion requires "
        r"every cell interval in the group to contain its benefit. These are descriptive rates "
        r"from six historical corruption types, not a transfer guarantee.}",
        r"\label{tab:journal-inclusion-units}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Constant-cell radius} & \multicolumn{2}{c}{Group-max radius} \\",
        r"Method & Cell (\%) & Group (\%) & Cell (\%) & Group (\%) \\",
        r"\midrule",
    ]
    for candidate in ("eata", "sar", "tent"):
        constant = indexed[f"cifar10c/{candidate}", "environment", "constant_cell_alpha0.1"]["metrics"]["intervals"]
        group = indexed[f"cifar10c/{candidate}", "environment", "constant_group_alpha0.1"]["metrics"]["intervals"]
        values = [
            constant["cell_inclusion"],
            constant["simultaneous_group_inclusion"],
            group["cell_inclusion"],
            group["simultaneous_group_inclusion"],
        ]
        lines.append(candidate.upper() + " & " + " & ".join(f"{100 * v:.2f}" for v in values) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[tb]",
        r"\centering\small",
        r"\caption{Reconstructed means for the principal CIFAR policy, distinct from the "
        r"shared-predictor study. Accuracy is the equally weighted mean of preserved "
        r"condition/seed-cell accuracies. Regret is unweighted oracle regret in percentage "
        r"points; exposure is the percentage of cells with ADAPT. Reconstruction checks "
        r"stored arithmetic and does not authenticate missing original image-level execution.}",
        r"\label{tab:journal-principal-means}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Mean accuracy (\%)} & Regret & ADAPT \\",
        r"Method & Frozen & Adapt & KGA & (pp) & (\%) \\",
        r"\midrule",
    ]
    for row in result["principal_rows"]:
        if row["study"] != "principal_cifar_current_policy" or row["arm"] != "kga":
            continue
        metric = row["metrics"]
        accuracy = metric["accuracy_percent"]
        values = [
            accuracy["frozen"],
            accuracy["candidate"],
            accuracy["policy"],
            metric["oracle_regret_pp"],
            100 * metric["exposure_rate"],
        ]
        lines.append(row["dataset"].split("/")[1].upper() + " & " + " & ".join(f"{v:.4f}" for v in values) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[tb]",
        r"\centering\small",
        r"\caption{Retrospective CIFAR Tent sensitivity, scaled calibration minus each comparator "
        r"in $100L_5$ percentage points; negative favors scaled calibration. The ranges use "
        r"1,000 paired resamples of six historical corruption types and are descriptive, "
        r"not population confidence guarantees. JPEG is omitted only for the influence "
        r"diagnostic; all six types remain in the primary estimate.}",
        r"\label{tab:journal-tent-sensitivity}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Comparator & Mean difference & 95\% descriptive range & Without JPEG \\",
        r"\midrule",
    ]
    tent = indexed["cifar10c/tent", "environment", "scaled_cell_alpha0.1"]
    comparisons = [
        ("Point", tent["comparisons"]["point"]),
        ("Tuned margin", tent["comparisons"]["fixed_margin_harm5"]),
        ("Point, matched exposure", tent["matched_exposure"]["comparison"]),
    ]
    for name, comparison in comparisons:
        low, high = comparison["descriptive_paired_cluster_bootstrap_95_percentile_pp"]
        jpeg = next(v for v in comparison["leave_one_cluster_out"] if v["omitted_cluster"] == "jpeg_compression")
        lines.append(
            f"{name} & {comparison['mean_equal_cluster_delta_pp']:.6f} & "
            f"[{low:.6f}, {high:.6f}] & {jpeg['mean_equal_cluster_delta_pp']:.6f}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines).encode()


def build_outputs(root):
    root = root.resolve()
    bindings = {}
    calibration, verification = calibration_synthesis(root, bindings)
    principal, primary_checks = principal_cifar(root, bindings)
    cct, secondary, cct_record = principal_cct(root, bindings)
    result = {
        "schema": SCHEMA,
        "analysis_status": "RETROSPECTIVE_DESCRIPTIVE_FROM_FROZEN_INPUTS",
        "metric_contracts": metric_contracts(),
        "calibration_rows": calibration,
        "principal_rows": principal + cct,
        "calibration_verification": verification,
        "principal_cifar_scalar_authority_checks": primary_checks,
        "cct_original_inference": cct_record,
        "source_scope": "Frozen historical authorities; no training, target access or original-run reruns.",
    }
    builder = Path(__file__).read_bytes()
    provenance = {
        "schema": SCHEMA,
        "builder_path": str(Path(__file__).relative_to(ROOT)),
        "builder_sha256": sha256(builder),
        "inputs": [bindings[k] for k in sorted(bindings)],
        "n_input_files": len(bindings),
        "runtime": {"python": ".".join(map(str, sys.version_info[:3])), "numpy": np.__version__},
        "determinism": "Sorted JSON, fixed bootstrap seed; no timestamp or absolute input root in generated files.",
    }
    outputs = {
        "empirical_synthesis_v1.json": json_bytes(result),
        "empirical_metric_contracts_v1.json": json_bytes(metric_contracts()),
        "empirical_rows_v1.csv": flattened_csv(result),
        "empirical_tables.tex": render_latex_tables(result),
        "empirical_cct_secondary_cells_v1.json": json_bytes(
            {
                "schema": SCHEMA,
                "cells": secondary,
                "aggregate_macro_f1": None,
                "aggregate_reason": "Frozen protocol reports descriptive cell-level secondary outcomes only; no post-hoc aggregate or inference claim added.",
            }
        ),
        "empirical_synthesis_v1.md": render_markdown(result).encode(),
        "empirical_input_manifest_v1.json": json_bytes(provenance),
    }
    outputs["empirical_synthesis_receipt_v1.json"] = json_bytes(
        {
            "schema": SCHEMA,
            "status": "PASS",
            "builder_sha256": sha256(builder),
            "input_manifest_sha256": sha256(outputs["empirical_input_manifest_v1.json"]),
            "outputs": [
                {"path": name, "bytes": len(data), "sha256": sha256(data)} for name, data in sorted(outputs.items())
            ],
            "calibration_verification": verification,
            "principal_cifar_scalar_authority_checks": primary_checks,
            "cct_scalar_authority_checks": cct_record["scalar_authority_checks"],
            "scope": "Independent arithmetic and reproducible retrospective synthesis, not scientific confirmation.",
        }
    )
    return outputs


def write_outputs(destination, outputs, check=False, replace=False):
    for name in outputs:
        if Path(name).name != name or not name.startswith("empirical_"):
            raise ValueError("output must be a single empirical_ filename")
    # Check every collision before changing anything; frozen input paths are never output targets.
    for name, data in outputs.items():
        path = destination / name
        if path.is_symlink():
            raise ValueError(f"output symlink is not allowed: {name}")
        if check:
            if not path.is_file() or path.read_bytes() != data:
                raise ValueError(f"generated output differs: {name}")
        elif path.exists() and path.read_bytes() != data and not replace:
            raise FileExistsError(
                f"generated output exists with different bytes: {path}; use a new destination or explicit --replace"
            )
    if not check:
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in outputs.items():
            path = destination / name
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--replace", action="store_true", help="Explicitly replace only generated empirical_ outputs, never inputs."
    )
    args = parser.parse_args()
    outputs = build_outputs(args.repo)
    destination = args.output_directory or args.repo / JOURNAL
    write_outputs(destination, outputs, check=args.check, replace=args.replace)
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": "check" if args.check else "build",
                "files": len(outputs),
                "output_directory": str(destination),
            }
        )
    )


if __name__ == "__main__":
    main()
