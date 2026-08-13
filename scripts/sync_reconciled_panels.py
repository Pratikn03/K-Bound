#!/usr/bin/env python3
"""Propagate source-replayed panel results into the paper's structured manifests.

The numerical source of truth is
``experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json``.
This script intentionally updates structured JSON only; manuscript prose is audited
separately so a numeric refresh cannot silently change the claim scope.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
SOURCE_MANIFEST = ROOT / "experiments/kbound/results/reconciled_panels_v1/source_manifest.json"
TABLE_PATH = ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json"
LEDGER_PATH = ROOT / "docs/research/kbound/claim_ledger.json"
FRONTIER_PATH = ROOT / "experiments/kbound/frontier_sweep_v1/decision_value_results.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n")


def _regret(score: dict[str, Any]) -> list[float]:
    values = score["regret"]
    return [values["kga"], values["always_adapt"], values["always_freeze"]]


def _decision_counts(score: dict[str, Any]) -> dict[str, int]:
    return {
        "ADAPT": score["adapt_count"],
        "FREEZE": score["freeze_count"],
        "ABSTAIN": score["abstain_count"],
    }


def _ci(score: dict[str, Any], baseline: str) -> list[float] | None:
    bootstrap = score["seed_inference"].get("paired_seed_bootstrap")
    return bootstrap["gaps"][baseline]["ci95"] if bootstrap else None


def _zero_error_cp95(n: int) -> float | None:
    return 1.0 - math.pow(0.05, 1.0 / n) if n else None


def _claim(ledger: dict[str, Any], claim_id: str) -> dict[str, Any]:
    return next(row for row in ledger["claims"] if row["claim_id"] == claim_id)


def _sync_table(panel: dict[str, Any], table: dict[str, Any]) -> None:
    panels = panel["panels"]
    tracks = table["tracks"]
    source = PANEL_PATH.relative_to(ROOT).as_posix()
    source_manifest = SOURCE_MANIFEST.relative_to(ROOT).as_posix()

    imagenetc = panels["imagenetc"]["panel"]["candidates"]
    for candidate in ("tent", "eata", "sar"):
        score = imagenetc[candidate]
        key = f"imagenetc_{candidate}"
        row = tracks.setdefault(key, {})
        row.update(
            {
                "regret": _regret(score),
                "false_adapt": score["fa_u"],
                "false_adapt_count": score["false_adapt_count"],
                "decision_counts": _decision_counts(score),
                "n_cells": score["n"],
                "seeds": panels["imagenetc"]["panel"]["seeds"],
                "quantile_rule": "per-candidate/per-seed exact-rank leave-one-condition-out residual calibration",
                "point_beats_both": score["point_beats_both"],
                "ci_robust_beats_both": score["seed_inference"]["ci_robust_beats_both"],
                "gap_baseline_minus_kga_ci95_seed": {
                    "always_adapt": _ci(score, "always_adapt"),
                    "always_freeze": _ci(score, "always_freeze"),
                },
                "source": source,
                "source_manifest": source_manifest,
                "status": "source-record exact-LOO replay",
            }
        )
    tracks["imagenetc_sar"]["verdict"] = (
        "Pooled point estimate is below both fixed policies, with one false adaptation in 135 cells; "
        "the seed bootstrap touches zero on the freeze side, so CI-robust beats-both is not claimed."
    )

    office_panel = panels["officehome"]
    office = office_panel["primary"]["exact_rank_transfer_score"]
    office_row = tracks["officehome_M_v2"]
    office_row.update(
        {
            "regret": _regret(office),
            "false_adapt": office["fa_u"],
            "false_adapt_count": office["false_adapt_count"],
            "n_test": office["n"],
            "decision_counts": _decision_counts(office),
            "cp95_upper_fa_c": _zero_error_cp95(office["adapt_count"]),
            "point_beats_both": office["point_beats_both"],
            "ci_robust_beats_both": office["seed_inference"]["ci_robust_beats_both"],
            "ci_vs_adapt": _ci(office, "always_adapt"),
            "ci_vs_freeze": _ci(office, "always_freeze"),
            "source": source,
            "source_manifest": source_manifest,
            "quantile_rule": "held-out target test with leave-one-calibration-record-out residuals and exact-rank radius",
            "a7_status": office_panel["primary"]["calibration"]["a7_status"],
            "verdict": (
                "Descriptive no-harm tie with freeze under the locked release runtime; the "
                "predeclared uniform A7 full-fit-versus-LOO stability premise is absent."
            ),
            "source_caveat": (
                "The compact source records and original SHA-256 hashes are released. Decision "
                f"counts are {office['adapt_count']} ADAPT, {office['freeze_count']} FREEZE, and "
                f"{office['abstain_count']} ABSTAIN under the recorded runtime."
            ),
            "independent_seed_replication": {
                "regret": _regret(office_panel["independent_seed_replication"]["exact_rank_transfer_score"]),
                "n_test": office_panel["independent_seed_replication"]["n_test"],
                "seeds": office_panel["independent_seed_replication"]["test_seeds"],
                "decision_counts": _decision_counts(
                    office_panel["independent_seed_replication"]["exact_rank_transfer_score"]
                ),
                "ci_robust_beats_both": office_panel["independent_seed_replication"]
                ["exact_rank_transfer_score"]["seed_inference"]["ci_robust_beats_both"],
                "a7_status": office_panel["independent_seed_replication"]["calibration"]["a7_status"],
            },
        }
    )

    iwild_panel = panels["iwildcam"]["primary"]
    iwild = iwild_panel["exact_rank_transfer_score"]
    tracks["iwildcam_H_v2"].update(
        {
            "regret": _regret(iwild),
            "false_adapt": iwild["fa_u"],
            "false_adapt_count": iwild["false_adapt_count"],
            "n_test": iwild["n"],
            "decision_counts": _decision_counts(iwild),
            "cp95_upper_fa_c": None,
            "point_beats_both": iwild["point_beats_both"],
            "ci_robust_beats_both": iwild["seed_inference"]["ci_robust_beats_both"],
            "source": source,
            "source_manifest": source_manifest,
            "quantile_rule": "seed-0 calibration to seed-1 held-out test with exact-rank radius",
            "a7_status": iwild_panel["calibration"]["a7_status"],
            "verdict": "Descriptive no-harm tie with freeze; zero ADAPT decisions and only one test seed.",
            "guarantee_status": "untested: zero ADAPT decisions",
            "source_caveat": (
                "The compact source records and original SHA-256 hash are released. Decision "
                f"counts are {iwild['adapt_count']} ADAPT, {iwild['freeze_count']} FREEZE, and "
                f"{iwild['abstain_count']} ABSTAIN under the recorded runtime."
            ),
        }
    )

    pacs_panel = panels["pacs"]
    pacs = pacs_panel["pooled_domain_seed_mean"]
    tracks["pacs"].update(
        {
            "completed_seeds": len(pacs_panel["seeds"]),
            "planned_seeds": len(pacs_panel["seeds"]),
            "mean_regret_kga_adapt_freeze": _regret(pacs),
            "reported_false_adapt_mean": pacs["fa_u"],
            "source": source,
            "source_manifest": source_manifest,
            "decision_replay_available": pacs_panel["decision_replay_available"],
            "decision_replay_blocker": pacs_panel["decision_replay_blocker"],
            "verdict": (
                "Completed three-seed null diagnostic. Seed summaries agree exactly, but archived "
                "per-cell files omit b_hat and calibration residuals, so gate decisions cannot be replayed."
            ),
        }
    )

    imagenetr_panel = panels["imagenet_r"]["panel"]
    imagenetr = imagenetr_panel["architecture_panel_aggregate"]
    candidates = imagenetr_panel["candidates"]
    worse = sum(
        row["regret"]["kga"] > row["regret"]["always_adapt"] for row in candidates.values()
    )
    tracks["imagenet_r_D"].update(
        {
            "completed_seeds": imagenetr_panel["seeds"],
            "planned_seed_count": len(imagenetr_panel["seeds"]),
            "mean_regret_kga_adapt_freeze": _regret(imagenetr),
            "observed_false_adapt": f'{imagenetr["false_adapt_count"]}/{imagenetr["n"]}',
            "false_adapt_count": imagenetr["false_adapt_count"],
            "decision_counts": _decision_counts(imagenetr),
            "beats_both_backbones": "0/10",
            "worse_than_always_adapt_backbones": f"{worse}/10",
            "source": source,
            "source_manifest": source_manifest,
            "quantile_rule": "per-backbone/per-seed exact-rank leave-one-condition-out residual calibration",
            "verdict": (
                f"Negative four-seed, ten-backbone diagnostic: KGA is worse than always-adapt on "
                f"{worse}/10 backbones; no architecture has CI-robust beats-both."
            ),
            "per_backbone": {
                name: {
                    "regret": _regret(row),
                    "decision_counts": _decision_counts(row),
                    "false_adapt_count": row["false_adapt_count"],
                }
                for name, row in candidates.items()
            },
        }
    )

    table["reconciliation_source"] = {
        "canonical_panel": source,
        "source_manifest": source_manifest,
        "generator": panel["generator"],
        "generator_sha256": panel["generator_sha256"],
        "runtime": panel.get("runtime"),
    }


def _sync_ledger(ledger: dict[str, Any]) -> None:
    source = PANEL_PATH.relative_to(ROOT).as_posix()
    source_manifest = SOURCE_MANIFEST.relative_to(ROOT).as_posix()

    theorem = _claim(ledger, "KB-CLAIM-001")
    theorem["claim_text"] = (
        "A strict adapt or freeze commitment is uniformly supportable over the declared drift class "
        "iff |M| > beta; on |M| <= beta, abstention is the maximal sound three-way action."
    )
    theorem["allowed_wording"] = "strict-commitment frontier over the declared drift class"
    theorem["forbidden_wording"] = ["assumption-free", "universal", "benefit sign identifiable iff"]

    certificate = _claim(ledger, "KB-CLAIM-003")
    certificate["claim_text"] = (
        "If P(|Delta_hat-Delta| <= epsilon) >= 1-alpha, the KGA interval rule controls the "
        "unconditional false-adapt event FA_u at level alpha."
    )
    certificate["assumptions"] = [
        "valid marginal interval coverage",
        "exchangeability or another justified calibration argument is one route to coverage",
    ]
    certificate["allowed_wording"] = "FA_u <= alpha under interval coverage"

    imagenetc = _claim(ledger, "KB-CLAIM-011")
    imagenetc.update(
        {
            "claim_text": (
                "ImageNet-C SAR exact-LOO panel has a pooled point estimate below both fixed policies, "
                "but the seed-level freeze comparison touches zero and FA_u is 1/135."
            ),
            "status": "no-harm",
            "supporting_artifacts": [source, source_manifest],
            "calibration_method": "exact_rank_leave_one_condition_out_residual_calibration",
            "test_split": "27 cross-fitted conditions per seed x 5 seeds",
            "assumptions": ["controlled grid", "seed is the primary inference unit"],
            "allowed_wording": "pooled point-estimate no-harm; no CI-robust beats-both claim",
            "forbidden_wording": ["zero false adaptation", "CI-supported beats-both", "natural-shift win"],
        }
    )

    office = _claim(ledger, "KB-CLAIM-020")
    office.update(
        {
            "claim_text": (
                "Office-Home M-v2 exact-rank source replay ties always-freeze under the locked "
                "release runtime; the predeclared A7 stability premise is absent."
            ),
            "status": "descriptive",
            "supporting_artifacts": [source, source_manifest],
            "calibration_method": "calibration-record LOO residuals + exact-rank transfer radius",
            "assumptions": ["dev-locked candidate", "A7 full-fit-versus-LOO stability not established"],
            "allowed_wording": "descriptive no-harm tie with zero ADAPT decisions under the locked runtime",
            "forbidden_wording": ["CI-robust beats both", "natural-shift win", "uniform no-harm"],
        }
    )

    iwild = _claim(ledger, "KB-CLAIM-021")
    iwild.update(
        {
            "claim_text": (
                "iWildCam H-v2 exact-rank source replay ties always-freeze with zero ADAPT decisions."
            ),
            "status": "descriptive",
            "supporting_artifacts": [source, source_manifest],
            "calibration_method": "seed-0 calibration to seed-1 test with exact-rank radius",
            "assumptions": ["one held-out test seed", "A7 full-fit-versus-LOO stability not established"],
            "allowed_wording": "descriptive no-harm tie; false-adapt guarantee untested",
            "forbidden_wording": ["beats both", "powered safety result", "multi-seed natural win"],
        }
    )

    mixture = _claim(ledger, "KB-CLAIM-024")
    mixture.update(
        {
            "status": "diagnostic",
            "allowed_wording": (
                "historical researcher-constructed routing aggregate; rerun required under reconciled "
                "per-track decisions"
            ),
            "forbidden_wording": ["promoted beats-both claim", "natural-shift win", "transfer result"],
        }
    )

    pacs = _claim(ledger, "KB-CLAIM-041")
    pacs.update(
        {
            "status": "diagnostic",
            "supporting_artifacts": [source, source_manifest],
            "allowed_wording": "three-seed null diagnostic; seed summaries cross-validated",
            "forbidden_wording": ["decision replay complete", "beats both"],
        }
    )

    imagenetr = _claim(ledger, "KB-CLAIM-042")
    imagenetr.update(
        {
            "status": "diagnostic",
            "supporting_artifacts": [source, source_manifest],
            "calibration_method": "per-backbone/per-seed exact-rank LOO residual calibration",
            "claim_text": (
                "ImageNet-R Protocol D exact-LOO replay is a negative architecture-panel diagnostic: "
                "KGA is worse than always-adapt on 8/10 backbones."
            ),
            "allowed_wording": "negative four-seed architecture-panel diagnostic",
            "forbidden_wording": ["beats both", "single deployable policy", "7/10 backbones"],
        }
    )

    ledger["reconciliation_source"] = {
        "canonical_panel": source,
        "source_manifest": source_manifest,
    }


def _frontier_row(row: dict[str, Any], n: int) -> dict[str, Any]:
    return {
        "regret": row["regret"],
        "adapt_rate": row["adapt_rate"],
        "FA_u": row["fa_u"],
        "FA_c": row["fa_c"],
        "n_adapt": row["adapt_count"],
        "n": n,
        "kappa": row["kappa"],
        "yield": row["yield"],
        "n_freeze": row["freeze_count"],
        "FF_u": 0.0,
    }


def _sync_frontier_dataset(frontier: dict[str, Any], score: dict[str, Any]) -> None:
    n = score["n"]
    sweep = [_frontier_row(row, n) for row in score["kappa_sweep"]]
    frontier["sweep"] = sweep
    frontier["kga_operating_point_kappa1"] = next(row for row in sweep if row["kappa"] == 1.0)
    frontier["trivial_baselines"]["always_adapt"]["regret"] = score["regret"]["always_adapt"]
    frontier["trivial_baselines"]["always_freeze"]["regret"] = score["regret"]["always_freeze"]
    frontier["trivial_baselines"]["always_abstain"]["regret"] = score["regret"]["always_freeze"]


def _sync_frontier(panel: dict[str, Any], frontier_data: dict[str, Any]) -> None:
    panels = panel["panels"]
    for key, panel_key in (("imagenetc", "imagenetc"), ("imagenetr", "imagenet_r")):
        grid = panels[panel_key]["panel"]
        _sync_frontier_dataset(frontier_data["frontier"][key]["__pooled__"], grid["architecture_panel_aggregate"])
        diagnostics = {"__pooled__": grid["architecture_panel_aggregate"]["radius_diagnostics"]}
        diagnostics.update(
            {name: row["radius_diagnostics"] for name, row in grid["candidates"].items()}
        )
        frontier_data["abstention"]["per_cell_tracks"][key] = diagnostics
    frontier_data["reconciliation_source"] = {
        "canonical_panel": PANEL_PATH.relative_to(ROOT).as_posix(),
        "note": "ImageNet-C and ImageNet-R kappa sweeps use source-replayed exact-LOO radii.",
    }


def main() -> None:
    panel = _load(PANEL_PATH)
    table = _load(TABLE_PATH)
    ledger = _load(LEDGER_PATH)
    frontier = _load(FRONTIER_PATH)
    _sync_table(panel, table)
    _sync_ledger(ledger)
    _sync_frontier(panel, frontier)
    _write(TABLE_PATH, table)
    _write(LEDGER_PATH, ledger)
    _write(FRONTIER_PATH, frontier)
    print(f"updated {TABLE_PATH.relative_to(ROOT)}")
    print(f"updated {LEDGER_PATH.relative_to(ROOT)}")
    print(f"updated {FRONTIER_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
