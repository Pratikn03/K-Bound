#!/usr/bin/env python3
"""Generate manuscript macros from the receipt-bound So2Sat development stop.

This builder is intentionally narrow: it accepts only the released two-candidate,
nine-city development authority and emits no target score.  Any receipt, study-shape,
candidate-ID, feasibility, or target-access drift stops generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SELECTION = (
    ROOT
    / "experiments/kbound/results/so2sat_lcz42_prospective_v1/"
    "development_mps_bn_fix_v1/so2sat_candidate_selection.json"
)
RECEIPT = SELECTION.with_suffix(".json.receipt.json")
OUTPUT = ROOT / "docs/research/kbound/paper/generated/so2sat_numbers.tex"

EXPECTED_CANDIDATES = {
    "tent_adam_bn_affine_probe_transfer_v1": "Tent",
    "sar_sam_bn_affine_probe_transfer_v1": "Sar",
}
EXPECTED_CHECKS = {
    "at_least_two_harmful_cities",
    "at_least_two_helpful_cities",
    "loco_routed_gain_over_best_fixed",
    "loco_sign_accuracy",
    "meaningful_adapt_cell_exposure",
    "meaningful_adapt_city_exposure",
    "meaningful_freeze_cell_exposure",
    "meaningful_freeze_city_exposure",
    "nontrivial_oracle_routing_gap",
}


class AuthorityError(ValueError):
    """Raised when the release authority cannot support deterministic macros."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            AuthorityError(f"non-standard JSON constant: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise AuthorityError(f"JSON authority must be an object: {path}")
    return value


def load_validated_numbers(
    selection_path: Path = SELECTION, receipt_path: Path = RECEIPT
) -> dict[str, Any]:
    if not selection_path.is_file() or not receipt_path.is_file():
        raise AuthorityError("So2Sat selection authority and receipt are both required")
    selection = strict_json(selection_path)
    receipt = strict_json(receipt_path)
    if (
        receipt.get("schema") != "kbound_so2sat_artifact_receipt_v2"
        or receipt.get("artifact_sha256") != sha256(selection_path)
        or receipt.get("artifact_bytes") != selection_path.stat().st_size
    ):
        raise AuthorityError("So2Sat selection receipt does not bind the authority bytes")
    candidate_ids = selection.get("candidate_ids")
    summaries = selection.get("candidate_summaries")
    gate_fit_cities = selection.get("study_binding", {}).get("gate_fit_cities")
    if (
        selection.get("schema") != "kbound_so2sat_adapter_candidate_selection_v1"
        or selection.get("status")
        != "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL"
        or selection.get("selected_candidate_id") is not None
        or not isinstance(candidate_ids, list)
        or set(candidate_ids) != set(EXPECTED_CANDIDATES)
        or not isinstance(summaries, dict)
        or set(summaries) != set(EXPECTED_CANDIDATES)
        or not isinstance(gate_fit_cities, list)
        or len(gate_fit_cities) != 9
        or len(set(gate_fit_cities)) != 9
        or selection.get("gate_cal_rows_read_before_selection") != 0
        or selection.get("target_inputs") != []
        or selection.get("target_pixels_read") != 0
        or selection.get("target_labels_read") != 0
    ):
        raise AuthorityError("So2Sat authority is not the sealed two-candidate no-target stop")

    numbers: dict[str, Any] = {
        "CandidateCount": 2,
        "DevelopmentCityCount": 9,
        "GateCalRowsRead": 0,
        "TargetInputCount": 0,
        "TargetPixelsRead": 0,
        "TargetLabelsRead": 0,
        "FeasibilityCheckCount": len(EXPECTED_CHECKS),
    }
    shapes: set[tuple[int, int, int]] = set()
    opened_outcomes = 0
    for candidate_id, macro_prefix in EXPECTED_CANDIDATES.items():
        feasibility = summaries[candidate_id].get("feasibility", {})
        city_count = feasibility.get("city_count")
        checkpoint_count = feasibility.get("checkpoint_count")
        cell_count = feasibility.get("cell_count")
        checks = feasibility.get("checks")
        if (
            feasibility.get("schema") != "kbound_so2sat_candidate_feasibility_v1"
            or feasibility.get("data_role") != "gate_fit_only"
            or feasibility.get("feasible") is not False
            or city_count != len(gate_fit_cities)
            or not isinstance(checkpoint_count, int)
            or checkpoint_count <= 0
            or cell_count != city_count * checkpoint_count
            or set(feasibility.get("city_mean_benefit", {})) != set(gate_fit_cities)
            or not isinstance(checks, dict)
            or set(checks) != EXPECTED_CHECKS
            or any(type(value) is not bool for value in checks.values())
        ):
            raise AuthorityError(f"So2Sat candidate study shape drifted: {candidate_id}")
        shapes.add((city_count, checkpoint_count, cell_count))
        opened_outcomes += cell_count
        helpful = feasibility.get("helpful_cities")
        harmful = feasibility.get("harmful_cities")
        oracle_gap = feasibility.get("oracle_routing_gap")
        loco_gain = feasibility.get("loco_routed_gain_over_best_fixed")
        loco_sign_accuracy = feasibility.get("loco_sign_accuracy")
        loco_adapt_cells = feasibility.get("loco_adapt_cells")
        if (
            not isinstance(helpful, list)
            or not isinstance(harmful, list)
            or len(set(helpful)) != len(helpful)
            or len(set(harmful)) != len(harmful)
            or not set(helpful) <= set(gate_fit_cities)
            or not set(harmful) <= set(gate_fit_cities)
            or not set(helpful).isdisjoint(harmful)
            or not isinstance(oracle_gap, (int, float))
            or oracle_gap < 0
            or not isinstance(loco_gain, (int, float))
            or loco_gain >= 0
            or not isinstance(loco_sign_accuracy, (int, float))
            or not 0.0 <= loco_sign_accuracy <= 1.0
            or not isinstance(loco_adapt_cells, int)
            or not 0 <= loco_adapt_cells <= cell_count
        ):
            raise AuthorityError(f"So2Sat candidate metrics drifted: {candidate_id}")
        numbers.update(
            {
                f"{macro_prefix}HelpfulCityCount": len(helpful),
                f"{macro_prefix}HarmfulCityCount": len(harmful),
                f"{macro_prefix}OracleGainPP": f"{100.0 * oracle_gap:.4f}",
                f"{macro_prefix}LossBestFixedPP": f"{100.0 * abs(loco_gain):.4f}",
                f"{macro_prefix}PassedCheckCount": sum(checks.values()),
            }
        )
        if macro_prefix == "Tent":
            numbers["TentLocoSignPct"] = f"{100.0 * loco_sign_accuracy:.2f}"
        else:
            numbers["SarAdaptCellCount"] = loco_adapt_cells

    if len(shapes) != 1:
        raise AuthorityError("So2Sat candidates do not share one development study shape")
    _, checkpoint_count, cell_count = shapes.pop()
    if checkpoint_count != 5 or cell_count != 45 or opened_outcomes != 90:
        raise AuthorityError("So2Sat released study must remain 9 x 5 x 2 opened outcomes")
    numbers.update(
        {
            "CheckpointCount": checkpoint_count,
            "CellCountPerCandidate": cell_count,
            "OpenedOutcomeCount": opened_outcomes,
        }
    )
    return numbers


def render_numbers_tex(numbers: dict[str, Any]) -> str:
    ordered = (
        "CandidateCount",
        "DevelopmentCityCount",
        "CheckpointCount",
        "CellCountPerCandidate",
        "OpenedOutcomeCount",
        "TentHelpfulCityCount",
        "TentHarmfulCityCount",
        "TentOracleGainPP",
        "TentLocoSignPct",
        "TentLossBestFixedPP",
        "TentPassedCheckCount",
        "SarHelpfulCityCount",
        "SarHarmfulCityCount",
        "SarOracleGainPP",
        "SarAdaptCellCount",
        "SarLossBestFixedPP",
        "SarPassedCheckCount",
        "FeasibilityCheckCount",
        "GateCalRowsRead",
        "TargetInputCount",
        "TargetPixelsRead",
        "TargetLabelsRead",
    )
    if set(numbers) != set(ordered):
        raise AuthorityError("So2Sat macro set is incomplete or contains an unknown value")
    lines = [
        "% AUTO-GENERATED by scripts/build_so2sat_numbers.py. Do not edit by hand."
    ]
    lines.extend(
        rf"\newcommand{{\SoTwo{name}}}{{{numbers[name]}}}" for name in ordered
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    rendered = render_numbers_tex(load_validated_numbers(args.selection, args.receipt))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="ascii")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
