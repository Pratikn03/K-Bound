"""Run the healthcare/clinical gap-closure validation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uais.validation.healthcare_gap_closure import run_healthcare_gap_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/healthcare/gridpulse")
    parser.add_argument("--report", default="experiments/fusion/healthcare_gap_validation.json")
    parser.add_argument("--fusion-output", default="experiments/fusion/healthcare_clinical_fusion_inputs.csv")
    parser.add_argument(
        "--split-strategy",
        choices=["provided", "patient_stratified"],
        default="provided",
        help="Use provided GridPulse splits or create a patient-disjoint two-class replay split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fusion-output", action="store_true")
    args = parser.parse_args()

    report = run_healthcare_gap_validation(
        data_root=Path(args.data_root),
        report_path=Path(args.report),
        fusion_output_path=None if args.no_fusion_output else Path(args.fusion_output),
        split_strategy=args.split_strategy,
        seed=args.seed,
    )
    summary = {
        "report_path": report.get("report_path"),
        "fusion_output_path": report.get("fusion_output_path"),
        "fusion_incidents": report.get("fusion_incidents"),
        "gap_statuses": report.get("gap_statuses"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
