"""Phase 2.2A — registry-driven Family-A cell driver.

Looks up benchmark / protocol / pairing-strength / config from the v2
registry and runs the audited 30-seed pilot for any A-POWERED-N cell.

Refuses to:
- run any non-A-POWERED-* experiment_id (rejects Family B/C/D IDs);
- accept a free-form benchmark/protocol on the command line;
- override the locked primary comparator (static_attention);
- overwrite the historical A-POWERED-1 K=10 SECONDARY_ALL_COMPARATOR
  pilot-audit outputs.

Outputs land under separate paths from the historical pilot — they
land in /experiments/phase2/predictions/<experiment_id>__<bench>__<proto>/
(same layout the historical pilot used for A-POWERED-1, but for the
other cells; the historical A-POWERED-1 directory is left alone).

This driver is the SAME training path as run_phase2_powered_audited_pilot.py
— it imports `run_one_seed()` from that module. Only the registry
lookup and the cell-by-cell dispatch are new.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Reuse the exact training path from the pilot driver — same code path.
from scripts.run_phase2_powered_audited_pilot import run_one_seed  # noqa: E402
from elara.evaluation.prediction_archive import PredictionArchive  # noqa: E402


REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"

# Locked map from registry (benchmark, protocol) -> config path. The map
# is verified against the registry at startup; if the registry lists a
# (benchmark, protocol) pair that has no config file, the driver
# refuses to run.
CONFIG_MAP = {
    ("MVTec 3D-AD", "PatchCore supervised-paired"):
        "configs/attention_mvtec3d_patchcore_supervised_paired.yaml",
    ("MVTec 3D-AD", "PatchCore held-out category"):
        "configs/attention_mvtec3d_patchcore_heldout.yaml",
    ("MVTec LOCO-AD", "PatchCore supervised-paired"):
        "configs/attention_mvtec_loco_patchcore_supervised_paired.yaml",
    ("VisA", "RGB+edge supervised-paired"):
        "configs/attention_visa_supervised_paired.yaml",
    ("UNSW-NB15", "flow/conn/context"):
        "configs/attention_unsw_paired.yaml",
}


def _registry_row(experiment_id: str) -> dict[str, str]:
    if not REGISTRY_V2.exists():
        raise SystemExit(f"missing v2 registry: {REGISTRY_V2}")
    with REGISTRY_V2.open() as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] == experiment_id:
                return row
    raise SystemExit(
        f"experiment_id {experiment_id!r} not present in v2 registry; "
        "Family-A driver rejects unknown IDs."
    )


def _validate_cell(row: dict[str, str]) -> None:
    fam = row["analysis_family"]
    if fam != "A":
        raise SystemExit(
            f"{row['experiment_id']} is family {fam!r}; this driver runs Family A only "
            "(Family B/C/D IDs rejected)."
        )
    if not row["experiment_id"].startswith("A-POWERED-"):
        raise SystemExit(
            f"{row['experiment_id']} is not an A-POWERED-* cell; rejected."
        )
    if row["primary_comparator"] != "static_attention":
        raise SystemExit(
            f"{row['experiment_id']}: registry primary_comparator={row['primary_comparator']!r}; "
            "Phase-2 v2 locks Family-A primary comparator as static_attention."
        )


def _config_for(row: dict[str, str]) -> Path:
    key = (row["benchmark"], row["protocol"])
    if key not in CONFIG_MAP:
        raise SystemExit(
            f"{row['experiment_id']}: no config registered for "
            f"benchmark={row['benchmark']!r} protocol={row['protocol']!r}"
        )
    cfg_path = ROOT / CONFIG_MAP[key]
    if not cfg_path.exists():
        raise SystemExit(f"config file missing: {cfg_path}")
    return cfg_path


def _cell_dir_slug(experiment_id: str, benchmark: str, protocol: str) -> str:
    return f"{experiment_id}__{benchmark.replace(' ', '_')}__{protocol.replace(' ', '_')}"


def _verify_not_overwriting_a1_historical(
    experiment_id: str, seed_metrics_out: Path, selection_log_out: Path
) -> None:
    """Hard protection: refuse to write to the historical A-POWERED-1
    K=10 secondary-pilot-audit CSVs unless we're being explicitly told
    to and the experiment IS A-POWERED-1."""
    historical_metrics = ROOT / "experiments" / "phase2" / "statistics" / "family_a_powered_seed_metrics.csv"
    historical_selection = ROOT / "experiments" / "phase2" / "statistics" / "family_a_selection_log.csv"
    if experiment_id != "A-POWERED-1":
        for p, hist in (
            (seed_metrics_out, historical_metrics),
            (selection_log_out, historical_selection),
        ):
            if p.resolve() == hist.resolve():
                raise SystemExit(
                    f"refusing to write {experiment_id} metrics to historical "
                    f"A-POWERED-1 file {hist}"
                )


def run_cell(
    experiment_id: str,
    seeds: int,
    seed_start: int,
    seed_metrics_out: Path,
    selection_log_out: Path,
    archive_root: Path,
) -> int:
    row = _registry_row(experiment_id)
    _validate_cell(row)
    cfg_path = _config_for(row)
    _verify_not_overwriting_a1_historical(experiment_id, seed_metrics_out, selection_log_out)

    # If seeds=0, this is a validation-only invocation (used by the
    # registry/static-reference test suite). Return success without
    # creating output files.
    if int(seeds) <= 0:
        print(f"[v2-family-a {experiment_id}] validation-only invocation (seeds=0); exiting OK")
        return 0

    cfg = yaml.safe_load(cfg_path.read_text())
    archive = PredictionArchive(root=archive_root)
    seed_metrics_out.parent.mkdir(parents=True, exist_ok=True)
    selection_log_out.parent.mkdir(parents=True, exist_ok=True)

    seeds_planned = list(range(int(seed_start), int(seed_start) + int(seeds)))

    metrics_fields = [
        "experiment_id", "benchmark", "protocol", "seed", "n_test_samples",
        "val_auc_router", "val_auc_boost", "chosen_head",
        "chosen_val_auc", "chosen_test_auc",
        "static_test_auc", "craf_test_auc",
        "router_test_auc", "boost_test_auc",
    ]
    selection_fields = [
        "experiment_id", "benchmark", "protocol", "seed",
        "candidate", "val_auc", "test_auc", "selection_used_test_metrics",
    ]
    metrics_new = not seed_metrics_out.exists()
    selection_new = not selection_log_out.exists()
    metrics_f = seed_metrics_out.open("a", newline="")
    selection_f = selection_log_out.open("a", newline="")
    metrics_w = csv.DictWriter(metrics_f, fieldnames=metrics_fields)
    selection_w = csv.DictWriter(selection_f, fieldnames=selection_fields)
    if metrics_new:
        metrics_w.writeheader()
    if selection_new:
        selection_w.writeheader()

    benchmark = row["benchmark"]
    protocol = row["protocol"]
    pairing_strength = row["pairing_strength"]

    for s in seeds_planned:
        print(f"[v2-family-a {experiment_id}] seed={s} starting", flush=True)
        result = run_one_seed(
            cfg, s,
            archive=archive,
            experiment_id=experiment_id,
            benchmark=benchmark,
            protocol=protocol,
            pairing_strength=pairing_strength,
            cell_dir_slug=_cell_dir_slug(experiment_id, benchmark, protocol),
        )
        metrics_w.writerow({
            "experiment_id": experiment_id,
            "benchmark": benchmark,
            "protocol": protocol,
            "seed": result["seed"],
            "n_test_samples": result["n_test_samples"],
            "val_auc_router": result["val_auc_router"],
            "val_auc_boost": result["val_auc_boost"],
            "chosen_head": result["chosen_head"],
            "chosen_val_auc": result["chosen_val_auc"],
            "chosen_test_auc": result["chosen_test_auc"],
            "static_test_auc": result["static_test_auc"],
            "craf_test_auc": result["craf_test_auc"],
            "router_test_auc": result["router_test_auc"],
            "boost_test_auc": result["boost_test_auc"],
        })
        metrics_f.flush()
        common = {"experiment_id": experiment_id, "benchmark": benchmark,
                  "protocol": protocol, "seed": result["seed"],
                  "selection_used_test_metrics": False}
        selection_w.writerow({**common, "candidate": "rga_meta_router",
                              "val_auc": result["val_auc_router"],
                              "test_auc": result["router_test_auc"]})
        selection_w.writerow({**common, "candidate": "rga_boosted_fusion",
                              "val_auc": result["val_auc_boost"],
                              "test_auc": result["boost_test_auc"]})
        for name, vauc in result["baseline_val_aucs"].items():
            selection_w.writerow({**common, "candidate": name,
                                  "val_auc": vauc,
                                  "test_auc": result["baseline_test_aucs"].get(name)})
        selection_f.flush()
        print(f"[v2-family-a {experiment_id}] seed={s} done  "
              f"chosen={result['chosen_head']}  "
              f"chosen_test_auc={result['chosen_test_auc']:.4f}", flush=True)

    metrics_f.close()
    selection_f.close()
    print(f"[v2-family-a {experiment_id}] {len(seeds_planned)} seeds complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True,
                        help="A-POWERED-N — must exist in the v2 registry")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--registry", type=Path, default=REGISTRY_V2,
                        help="(documented; the driver always reads the v2 registry path)")
    parser.add_argument("--archive-root", type=Path,
                        default=Path("experiments/phase2/predictions"))
    parser.add_argument("--seed-metrics-out", type=Path,
                        default=None,
                        help="default: experiments/phase2/statistics/family_a_v2_<EID>_seed_metrics.csv")
    parser.add_argument("--selection-log-out", type=Path,
                        default=None,
                        help="default: experiments/phase2/statistics/family_a_v2_<EID>_selection_log.csv")
    args = parser.parse_args()

    eid = args.experiment_id
    if args.seed_metrics_out is None:
        args.seed_metrics_out = ROOT / "experiments" / "phase2" / "statistics" / f"family_a_v2_{eid}_seed_metrics.csv"
    if args.selection_log_out is None:
        args.selection_log_out = ROOT / "experiments" / "phase2" / "statistics" / f"family_a_v2_{eid}_selection_log.csv"

    return run_cell(
        experiment_id=eid,
        seeds=args.seeds,
        seed_start=args.seed_start,
        seed_metrics_out=args.seed_metrics_out,
        selection_log_out=args.selection_log_out,
        archive_root=args.archive_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
