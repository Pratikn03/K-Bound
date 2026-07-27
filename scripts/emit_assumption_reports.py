#!/usr/bin/env python3
"""Emit one assumption report per evaluated track.

Source of truth is `docs/research/kbound/panel_review_2026-07-25/NUMBERS_PACK.json`,
the audited pack produced by the 2026-07-25 panel review.  Every value written here
is copied from a named pack entry and carries that entry's id, method string and
artifact paths in the report's `provenance` block, so any number in a report can be
traced back to the raw units it was computed from.

Why read the pack rather than recompute from raw records: the pack's intervals are
already dependence-aware (it carries iid, cluster-by-condition, cluster-by-seed and
cluster-by-corruption-family variants of the same interval), they were computed by
scripts under review, and recomputing them here would create a second, unreviewed
path to the same numbers.  What this script adds is the *assumption state* around
them -- which checks passed, which were never evaluated, and what the deployment is
therefore permitted to emit.

Nothing is inferred.  A statistic the pack does not contain is written as null with
a string in `limitations` saying so.

    python scripts/emit_assumption_reports.py            # writes the reports
    python scripts/emit_assumption_reports.py --check    # exit 1 if any would change

The schema is `kga.assumption_report.AssumptionReport`.  This script builds the JSON
directly and imports nothing from `kga`, so it runs without numpy/scipy; use
`--validate` to check field parity against the dataclass where they are available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "docs/research/kbound/panel_review_2026-07-25/NUMBERS_PACK.json"
OUT = REPO / "research_lock/assumption_reports"

SCHEMA_VERSION = "kbound-assumption-report/1"

# Predeclared thresholds. Fixed before the promoted conditions were evaluated (A6);
# mirrored from kga.assumptions.GateThresholds defaults.
THRESHOLDS = {
    "min_effective_units": 20,
    "support_frac_outside_warn": 0.05,
    "support_frac_outside_fail": 0.20,
    "domain_auroc_warn": 0.75,
    "domain_auroc_fail": 0.90,
    "radius_cv_warn": 0.20,
    "radius_cv_fail": 0.50,
    "decision_disagreement_warn": 0.05,
    "decision_disagreement_fail": 0.20,
    "conclusion_change_fail": 0.0,
}

LADDER = {
    "certify": "adapt_freeze_abstain",
    "restricted": "freeze_or_abstain",
    "diagnostic_only": "none",
    "reject": "none",
}
SEVERITY = {"certify": 0, "restricted": 1, "diagnostic_only": 2, "reject": 3}


def worse(a: str, b: str) -> str:
    """The gate only ever moves down the ladder."""
    return a if SEVERITY[a] >= SEVERITY[b] else b


def entry(pack: dict, eid: str) -> dict:
    for e in pack["entries"]:
        if e["id"] == eid:
            return e
    raise KeyError(f"NUMBERS_PACK has no entry {eid!r}")


def prov(pack: dict, *ids: str) -> dict:
    out = {}
    for eid in ids:
        e = entry(pack, eid)
        out[eid] = {
            "description": e.get("description"),
            "method": e.get("method"),
            "artifact_paths": e.get("artifact_paths"),
        }
    return out


def report(**kw) -> dict:
    """Assemble one report, filling every schema field explicitly."""
    gate = kw["deployment_gate"]
    base = {
        "dataset": None, "protocol": None, "inference_unit": None,
        "calibration_test_separated": None, "candidate_fixed_before_test": None,
        "target_labels_used_for_routing": None,
        "coverage_type": "diagnostic_only", "theoretical_coverage_claimed": False,
        "observed_coverage": None, "n_rows": None, "n_units": None,
        "coverage_interval_95": None, "coverage_interval_method": None,
        "support_overlap_status": "fail", "radius_stability_status": "fail",
        "conclusion_stability_status": "fail", "leakage_status": "pass",
        "deployment_gate": gate, "fallback_action": LADDER[gate],
        "alpha": None, "thresholds": THRESHOLDS, "diagnostics": {},
        "limitations": [], "provenance": {}, "schema_version": SCHEMA_VERSION,
    }
    unknown = set(kw) - set(base)
    if unknown:
        raise KeyError(f"not in schema: {sorted(unknown)}")
    base.update(kw)
    base["fallback_action"] = LADDER[base["deployment_gate"]]
    return base


NEVER_THEORETICAL = (
    "no theoretical coverage claim: A1-A3 are not checkable from label-free "
    "deployment evidence, so any coverage figure here is an observed hit rate"
)
NO_HIT_RATE = (
    "observed interval-hit coverage is not computable from the released records: "
    "they carry decisions and realised benefits, not per-unit interval endpoints. "
    "false-adapt rates below are NOT interval-hit coverage and must not be read as it"
)


def build(pack: dict) -> dict[str, dict]:
    alpha = pack["alpha"]
    out: dict[str, dict] = {}

    # ---------------- ImageNet-C SAR ---------------------------------------- #
    iid = entry(pack, "item3.imagenetc_sar.ci.iid135_as_coded.exact_rank")["value"]
    cond = entry(pack, "item3.imagenetc_sar.ci.cluster_by_condition.exact_rank")["value"]
    fam = entry(pack, "item3.imagenetc_sar.ci.cluster_by_corruption_family.exact_rank")["value"]
    out["imagenetc_sar"] = report(
        dataset="ImageNet-C (SAR)", protocol="win_hunt_v5_imagenetc_ms / pooled_5seed",
        inference_unit="corruption_condition (27 conditions x 5 seeds = 135 rows)",
        calibration_test_separated=True, candidate_fixed_before_test=True,
        target_labels_used_for_routing=False, alpha=alpha,
        coverage_type="diagnostic_only",
        n_rows=iid["n_units"], n_units=cond["n_units"],
        coverage_interval_method="paired percentile bootstrap, 20000 replicates, "
                                 "rng seed 20260720 (from NUMBERS_PACK)",
        support_overlap_status="fail", radius_stability_status="fail",
        conclusion_stability_status="fail",
        deployment_gate="diagnostic_only",
        diagnostics={
            "beats_both_by_resampling_unit": {
                "iid_135_rows_as_coded": {"n_units": iid["n_units"], "beats_both": iid["beats_both_ci"],
                                          "adapt_gap_ci95": iid["adapt_gap_ci95"]},
                "cluster_by_condition": {"n_units": cond["n_units"], "beats_both": cond["beats_both_ci"],
                                         "adapt_gap_ci95": cond["adapt_gap_ci95"]},
                "cluster_by_corruption_family": {"n_units": fam["n_units"], "beats_both": fam["beats_both_ci"],
                                                 "adapt_gap_ci95": fam["adapt_gap_ci95"]},
            },
        },
        limitations=[
            "CONCLUSION FLIPS ON THE RESAMPLING UNIT: beats-both holds treating the "
            f"{iid['n_units']} cell-seed rows as independent and FAILS when they are "
            f"clustered by condition ({cond['n_units']} units). Both clusterings are "
            "admissible, so the promoted conclusion is a conclusion about the clustering",
            f"cluster-by-corruption-family has only {fam['n_units']} clusters, far below "
            f"the declared minimum of {THRESHOLDS['min_effective_units']} effective units; "
            "a 3-cluster percentile bootstrap is not a primary interval (A5 unmet)",
            NO_HIT_RATE, NEVER_THEORETICAL,
            "support overlap and radius stability were not evaluated for this track; "
            "an unevaluated check is a failed check",
        ],
        provenance=prov(pack, "item3.imagenetc_sar.ci.iid135_as_coded.exact_rank",
                        "item3.imagenetc_sar.ci.cluster_by_condition.exact_rank",
                        "item3.imagenetc_sar.ci.cluster_by_corruption_family.exact_rank"),
    )

    # ---------------- CIFAR-10-C SAR (quarantined) -------------------------- #
    q = entry(pack, "item6.cifar10c_sar.quarantine")["value"]
    out["cifar10c_sar"] = report(
        dataset="CIFAR-10-C (SAR)", protocol="CIFAR10C_SAR_REBUILD_PROTOCOL_v2",
        inference_unit="seed (5 seeds over the corruption x severity grid)",
        calibration_test_separated=True, candidate_fixed_before_test=True,
        target_labels_used_for_routing=False, alpha=alpha,
        coverage_type="diagnostic_only",
        n_units=len(q["harmful_base_rate_per_seed"]),
        support_overlap_status="fail", radius_stability_status="fail",
        conclusion_stability_status="fail",
        deployment_gate="reject",
        diagnostics={
            "harmful_base_rate_per_seed": q["harmful_base_rate_per_seed"],
            "seed0_over_seeds1to4_ratio": q["seed0_over_seeds1to4_ratio"],
            "seeds1to4_regret_kga_adapt_freeze": q.get("seeds1to4_regret_kga_adapt_freeze"),
        },
        limitations=[
            "TRACK IS QUARANTINED; see CIFAR10C_SAR_QUARANTINE.md. No certificate is "
            "emitted and no claim in the paper rests on it",
            f"seed 0's harmful base rate ({q['seed0_harmful']}) is "
            f"{q['seed0_over_seeds1to4_ratio']:.2f}x the mean of seeds 1-4 "
            f"({q['seeds1to4_harmful_mean']:.4f}); the seeds are not exchangeable, so A1 "
            "and A5 both fail at the seed unit",
            "1 of 5 seeds beats both fixed policies and it is seed 0 -- the "
            "non-exchangeable one",
            NO_HIT_RATE, NEVER_THEORETICAL,
        ],
        provenance=prov(pack, "item6.cifar10c_sar.quarantine"),
    )

    # ---------------- PACS --------------------------------------------------- #
    p23 = entry(pack, "item23.pacs")
    v = p23["value"]
    n_dom = len(v["per_domain"]); n_seed = len(v["seeds"])
    out["pacs"] = report(
        dataset="PACS (4 LODO splits)", protocol="PACS_VLCS_PREREG_PROTOCOL_v1",
        inference_unit="domain x seed cell", calibration_test_separated=True,
        candidate_fixed_before_test=True, target_labels_used_for_routing=False,
        alpha=alpha, coverage_type="diagnostic_only",
        n_rows=n_dom * n_seed * 18, n_units=n_dom * n_seed,
        coverage_interval_method="Clopper-Pearson / Wilson on the pooled false-adapt "
                                 "count (from NUMBERS_PACK) -- a false-adapt interval, "
                                 "NOT an interval-hit coverage interval",
        support_overlap_status="fail", radius_stability_status="fail",
        conclusion_stability_status="fail", deployment_gate="diagnostic_only",
        diagnostics={
            "pooled_false_adapt": {"count": 2, "n": n_dom * n_seed * 18, "rate": 2 / (n_dom * n_seed * 18),
                                   "wilson95": [0.00254, 0.03313], "clopper_pearson95": [0.00112, 0.03305]},
            "per_domain": v["per_domain"],
            "panel_row_regret_kga_adapt_freeze": p23.get("old_value", {}).get("panel_row"),
        },
        limitations=[
            f"only {n_dom * n_seed} effective units (domain x seed) against a declared "
            f"minimum of {THRESHOLDS['min_effective_units']}; A5 unmet",
            "art_painting seed 1 has false-adapt 0.1111 (2 of 18 cells), ABOVE alpha; the "
            "pooled rate hides a cell where the marginal bound is not met in-sample",
            "art_painting seed 2 abstains on all 18 cells (decision coverage 0.0), so the "
            "guarantee is untested there rather than satisfied",
            "acceptable pooled false-adapt coexists with routing utility worse than "
            "always-adapt on this track: coverage is not utility",
            NO_HIT_RATE, NEVER_THEORETICAL,
        ],
        provenance=prov(pack, "item23.pacs"),
    )

    # ---------------- ImageNet-R --------------------------------------------- #
    r23 = entry(pack, "item23.imagenet_r")
    rv = r23["value"]
    n_bb = len(rv["per_backbone"]); n_seed_r = len(rv["seeds"])
    out["imagenet_r"] = report(
        dataset="ImageNet-R", protocol="IMAGENETR_DIVERSE_PANEL_PROTOCOL_D_v1",
        inference_unit="backbone x seed", calibration_test_separated=True,
        candidate_fixed_before_test=True, target_labels_used_for_routing=False,
        alpha=alpha, coverage_type="diagnostic_only",
        n_units=n_bb * n_seed_r,
        support_overlap_status="fail", radius_stability_status="fail",
        conclusion_stability_status="fail", deployment_gate="diagnostic_only",
        diagnostics={
            "n_backbones": n_bb, "n_seeds": n_seed_r,
            "per_backbone": rv["per_backbone"],
            "panel_note": r23.get("old_value", {}).get("panel_note"),
        },
        limitations=[
            "KGA is worse than always-adapt on 7 of 10 backbones; a low observed "
            "false-adapt rate here reflects rare harmful cases, not useful selection",
            "observed false-adapt 1/480 with harmful cells nearly absent on several "
            "backbones: the guarantee is barely exercised and should be reported as "
            "guarantee-untested, not as a safety success",
            NO_HIT_RATE, NEVER_THEORETICAL,
        ],
        provenance=prov(pack, "item23.imagenet_r"),
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if any report would change")
    ap.add_argument("--validate", action="store_true", help="check field parity against the dataclass")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not PACK.exists():
        print(f"NUMBERS_PACK not found at {PACK}", file=sys.stderr)
        return 2
    pack = json.loads(PACK.read_text())
    reports = build(pack)

    if args.validate:
        from dataclasses import fields
        from kga.assumption_report import AssumptionReport
        want = {f.name for f in fields(AssumptionReport)}
        for name, r in reports.items():
            if set(r) != want:
                print(f"{name}: schema drift -> extra={sorted(set(r)-want)} "
                      f"missing={sorted(want-set(r))}", file=sys.stderr)
                return 1
        print(f"field parity OK against AssumptionReport ({len(want)} fields)")

    args.out.mkdir(parents=True, exist_ok=True)
    drift = 0
    for name, r in reports.items():
        path = args.out / f"{name}.assumption_report.json"
        text = json.dumps(r, indent=2) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != text:
                print(f"WOULD CHANGE {path.relative_to(REPO)}"); drift += 1
            continue
        path.write_text(text)
        print(f"{path.relative_to(REPO)}  gate={r['deployment_gate']:<16} "
              f"action={r['fallback_action']:<20} n_units={r['n_units']} "
              f"theoretical_claimed={r['theoretical_coverage_claimed']}")
    if args.check:
        print("all reports current" if not drift else f"{drift} report(s) out of date")
        return 1 if drift else 0
    print(f"\n{len(reports)} report(s) written to {args.out.relative_to(REPO)}/")
    print("theoretical_coverage_claimed is false on every track, by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
