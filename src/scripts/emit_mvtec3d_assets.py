"""Render MVTec 3D-AD paper assets with an ``mvtec3d_`` filename prefix.

This is a thin shim around :mod:`scripts.generate_craf_paper_assets`. The
existing writer functions hard-code the ``elara_`` filename prefix; rather than
refactor them, this script invokes them against a temporary output directory
and then renames every produced file from ``elara_*`` to ``mvtec3d_*``.

Use it after :mod:`scripts.run_breakthrough_experiment` produces the
naturally paired MVTec 3D-AD results JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from scripts.generate_craf_paper_assets import (
    _set_style,
    plot_adversarial_delta,
    plot_calibration,
    plot_cda_impacts,
    plot_clean_benchmark,
    plot_drift_curves,
    plot_failure_cases,
    plot_missing_modality,
    write_adversarial_table,
    write_calibration_table,
    write_category_aware_table,
    write_clean_ci_table,
    write_clean_table,
    write_component_ablation_table,
    write_drift_table,
    write_failure_case_table,
    write_missing_table,
    write_paired_benchmark_metadata_table,
    write_tau_sweep_table,
)

WRITERS_TAKING_DATA = [
    write_clean_table,
    write_clean_ci_table,
    write_missing_table,
    write_drift_table,
    write_calibration_table,
    write_adversarial_table,
    write_tau_sweep_table,
    write_component_ablation_table,
    write_category_aware_table,
    write_failure_case_table,
]


PLOTTERS = [
    plot_clean_benchmark,
    plot_missing_modality,
    plot_drift_curves,
    plot_calibration,
    plot_cda_impacts,
    plot_adversarial_delta,
    plot_failure_cases,
]


def _rename_prefixed_files(src_dir: Path, dst_dir: Path, src_prefix: str, dst_prefix: str) -> list[Path]:
    moved: list[Path] = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.iterdir()):
        if not path.is_file():
            continue
        if not path.name.startswith(src_prefix):
            continue
        new_name = dst_prefix + path.name[len(src_prefix) :]
        target = dst_dir / new_name
        if target.exists():
            target.unlink()
        shutil.move(str(path), str(target))
        moved.append(target)
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MVTec3D paper assets with mvtec3d_ prefix")
    parser.add_argument("--input", required=True, type=Path, help="Path to MVTec3D results JSON")
    parser.add_argument("--metadata", type=Path, default=None, help="Path to MVTec3D fusion metadata JSON")
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--tables-dir", required=True, type=Path)
    parser.add_argument(
        "--asset-prefix",
        default="mvtec3d_",
        help="Filename prefix for emitted assets (use mvtec3d_patchcore_ to keep Gap 3 assets distinct).",
    )
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_tables = Path(tmp) / "tables"
        tmp_figs = Path(tmp) / "figures"
        tmp_tables.mkdir(parents=True, exist_ok=True)
        tmp_figs.mkdir(parents=True, exist_ok=True)

        for writer in WRITERS_TAKING_DATA:
            writer(data, tmp_tables)

        _set_style()
        for plotter in PLOTTERS:
            plotter(data, tmp_figs)

        moved_tables = _rename_prefixed_files(tmp_tables, args.tables_dir, "elara_", args.asset_prefix)
        moved_figs = _rename_prefixed_files(tmp_figs, args.figures_dir, "elara_", args.asset_prefix)

    if metadata is not None:
        write_paired_benchmark_metadata_table(metadata, args.tables_dir)

    print(f"Wrote {len(moved_tables)} MVTec3D tables to {args.tables_dir}")
    for path in moved_tables:
        print(f"  {path.name}")
    print(f"Wrote {len(moved_figs)} MVTec3D figures to {args.figures_dir}")
    for path in moved_figs:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
