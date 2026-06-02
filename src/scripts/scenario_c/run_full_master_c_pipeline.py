#!/usr/bin/env python3
"""Run the full Master Scenario C training + audit pipeline (T0–T7).

Stages:
  T0  Governance, split hashes, calibrator/baseline freeze, Gate A
  T1  (optional) Prepare fusion inputs for all development datasets
  T3  Fusion training: M0 ELARA-Bench-LA, M1 MVTec supervised-paired (multi-seed)
  T3+ Extended benchmarks (visa, loco, unsw, eyecandies) if --extended
  T4  Base RGA mechanism replication
  T5  RGA+ powered audited pilot (Family A path)
  T6  Gate decision rule E2E + theorem table emit
  T7  Confirmatory: M1 (5 seeds), M2 external (5 seeds if not --skip-m2-external)
  Final: confirmatory statistics, checklist, theorem stack validator

Usage:
  PYTHONPATH=.:src .venv/bin/python src/scripts/scenario_c/run_full_master_c_pipeline.py \\
    2>&1 | tee /tmp/master_c_full_pipeline.log
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _run(cmd: list[str], root: Path, label: str) -> int:
    print(f"\n{'=' * 72}\n=== {label} ===\n$ {' '.join(cmd)}\n", flush=True)
    env = __import__("os").environ.copy()
    env["PYTHONPATH"] = f"{root}:{root / 'src'}"
    return subprocess.call(cmd, cwd=root, env=env)


def _merge_seed_results(runs: list[dict]) -> dict:
    table = []
    for run in runs:
        for row in run.get("table_1_clean_performance") or []:
            table.append(row)
    base = runs[-1].copy()
    seeds = []
    for row in table:
        s = row.get("seed")
        if s is not None:
            seeds.append(int(s))
    base["seeds"] = seeds or base.get("seeds")
    base["table_1_clean_performance"] = table
    base["n_seeds"] = len(table)
    methods = [
        "rga_boosted_fusion",
        "static_attention",
        "craf_attention",
        "sar_score_adapter",
        "tent_score_adapter",
    ]
    summary: dict = {}
    for method in methods:
        vals = [
            float((row.get(method) or {}).get("roc_auc"))
            for row in table
            if isinstance((row.get(method) or {}).get("roc_auc"), (int, float))
        ]
        if vals:
            import statistics

            summary[method] = {
                "roc_auc": {
                    "mean": statistics.mean(vals),
                    "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
            }
    base["clean_metric_summary"] = summary
    return base


def _fusion_multi_seed(
    root: Path,
    py: str,
    *,
    config: str,
    merged_out: Path,
    label: str,
    seeds: list[int],
    archive_root: str | None,
    force: bool,
) -> int:
    if merged_out.is_file() and not force:
        print(f"SKIP {label}: {merged_out} exists (use --force-fusion to retrain)")
        return 0
    all_results: list[dict] = []
    rc = 0
    for seed in seeds:
        seed_out = merged_out.with_name(f"{merged_out.stem}_seed{seed}{merged_out.suffix}")
        cmd = [
            py,
            "src/scripts/run_breakthrough_experiment.py",
            "--config",
            config,
            "--output",
            str(seed_out),
            "--seed",
            str(seed),
        ]
        if archive_root:
            cmd.extend(["--archive-root", archive_root])
        rc = _run(cmd, root, f"{label} seed={seed}") or rc
        if seed_out.is_file():
            all_results.append(json.loads(seed_out.read_text(encoding="utf-8")))
    if all_results:
        merged_out.write_text(json.dumps(_merge_seed_results(all_results), indent=2), encoding="utf-8")
        print(f"Merged -> {merged_out}")
    return rc if all_results else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71])
    parser.add_argument("--prepare-t1", action="store_true", help="Run all T1 prepare scripts (slow)")
    parser.add_argument("--skip-mvtec-upgrade", action="store_true", help="Skip HF/Kaggle expert download")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip T4/T5 phase-2 scripts")
    parser.add_argument("--skip-m2-external", action="store_true", help="Skip M2 3D-ADAM confirmatory retrain")
    parser.add_argument("--skip-m2-proxy", action="store_true", help="Skip inverted-MVTec M2 proxy")
    parser.add_argument("--force-fusion", action="store_true", help="Re-run fusion even if outputs exist")
    parser.add_argument("--infra-only", action="store_true", help="T0 + audits only, no training")
    args = parser.parse_args()

    root = _repo_root()
    py = sys.executable
    rc = 0
    started = datetime.now(timezone.utc).isoformat()
    print(f"Master C full pipeline started {started}")

    # --- T0 ---
    for cmd, label in [
        ([py, "src/scripts/scenario_c/validate_master_c_governance.py"], "T0 governance"),
        ([py, "src/scripts/scenario_c/generate_split_hashes.py"], "T0 split hashes"),
        ([py, "src/scripts/scenario_c/freeze_domain_calibrators.py"], "T2 calibrator freeze"),
        ([py, "src/scripts/scenario_c/freeze_strongest_baselines.py"], "T3 strongest baseline freeze"),
        ([py, "src/scripts/scenario_c/qualify_upstream_experts.py"], "T1 Gate A qualification"),
    ]:
        rc = _run(cmd, root, label) or rc

    if not args.skip_mvtec_upgrade:
        rc = _run([py, "src/scripts/scenario_c/upgrade_mvtec_experts.py"], root, "T1 MVTec expert v2") or rc

    rc = _run(
        [py, "-m", "pytest", "tests/test_master_c_governance.py", "tests/test_master_c_leakage_splits.py", "-q"],
        root,
        "T0 governance tests",
    ) or rc

    if args.prepare_t1:
        reg = root / "elara_master_c/configs/training_stage_registry.yaml"
        import yaml

        stage = yaml.safe_load(reg.read_text(encoding="utf-8"))["stages"]["T1_data_splits"]
        for rel in stage.get("scripts", []):
            if (root / rel).is_file():
                rc = _run([py, rel], root, f"T1 prepare {rel}") or rc

    if args.infra_only:
        return _finalize(root, py, rc, started)

    # --- T3 core fusion ---
    fusion_jobs = [
        (
            "configs/attention_real_fusion.yaml",
            root / "experiments/fusion/master_c_real_domain_results.json",
            "T3 M0 ELARA-Bench-LA",
            args.seeds,
            "elara_master_c/predictions/development",
        ),
        (
            "configs/attention_mvtec3d_patchcore_supervised_paired.yaml",
            root / "experiments/fusion/master_c_mvtec_supervised_paired_results.json",
            "T3/T5 M1 MVTec supervised-paired",
            args.seeds,
            "elara_master_c/predictions/development",
        ),
        (
            "configs/attention_visa_supervised_paired.yaml",
            root / "experiments/fusion/visa_supervised_paired_master_c_results.json",
            "T3 VisA supervised-paired",
            args.seeds,
            None,
        ),
        (
            "configs/attention_mvtec_loco_patchcore_supervised_paired.yaml",
            root / "experiments/fusion/mvtec_loco_supervised_paired_master_c_results.json",
            "T3 MVTec LOCO supervised-paired",
            args.seeds,
            None,
        ),
        (
            "configs/attention_unsw_paired.yaml",
            root / "experiments/fusion/unsw_paired_master_c_results.json",
            "T3 UNSW paired",
            args.seeds,
            None,
        ),
        (
            "configs/attention_real3d_fusion.yaml",
            root / "experiments/fusion/eyecandies_master_c_results.json",
            "T3 Eyecandies (development)",
            args.seeds,
            None,
        ),
    ]

    for cfg, out, label, seeds, arch in fusion_jobs:
        if not (root / cfg).is_file():
            print(f"SKIP missing config: {cfg}")
            continue
        rc = (
            _fusion_multi_seed(
                root,
                py,
                config=cfg,
                merged_out=out,
                label=label,
                seeds=seeds,
                archive_root=arch,
                force=args.force_fusion,
            )
            or rc
        )

    # M1 confirmatory copy for statistics
    m1_conf = root / "experiments/fusion/m1_confirmatory_t5_results.json"
    m1_master = root / "experiments/fusion/master_c_mvtec_supervised_paired_results.json"
    if m1_master.is_file():
        m1_conf.write_text(m1_master.read_text(encoding="utf-8"))

    if not args.skip_phase2:
        rc = _run(
            [
                py,
                "src/scripts/run_phase2_mechanism_replication.py",
                "--experiment-id",
                "B-MECH-1",
                "--seeds",
                "30",
                "--seed-start",
                "42",
            ],
            root,
            "T4 mechanism replication (B-MECH-1)",
        ) or rc
        pilot = root / "src/scripts/run_phase2_powered_audited_pilot.py"
        if pilot.is_file():
            rc = _run([py, str(pilot.relative_to(root))], root, "T5 RGA+ powered pilot") or rc

    for rel, label in [
        ("src/scripts/audit_gate_decision_rule_e2e.py", "T6 GDR E2E audit"),
        ("src/scripts/emit_gate_decision_rule_table.py", "T6 emit GDR table"),
        ("src/scripts/emit_risk_dominance_t4_table.py", "Emit T4 table"),
        ("src/scripts/emit_ks_power_t6_table.py", "Emit T6 KS table"),
        ("src/scripts/emit_meta_router_pac_t7_table.py", "Emit T7 PAC table"),
        ("src/scripts/emit_theory_experiment_mapping.py", "Emit theory mapping"),
    ]:
        if (root / rel).is_file():
            rc = _run([py, rel], root, label) or rc

    # --- T7 confirmatory ---
    if not args.skip_m2_external:
        ext_out = root / "experiments/fusion/m2_external_3d_adam_confirmatory_results.json"
        if ext_out.is_file() and not args.force_fusion:
            print(f"SKIP M2 external: {ext_out} exists")
        else:
            rc = (
                _fusion_multi_seed(
                    root,
                    py,
                    config="configs/attention_m2_external_3d_adam_sealed.yaml",
                    merged_out=ext_out,
                    label="T7 M2 external 3D-ADAM",
                    seeds=args.seeds,
                    archive_root="elara_master_c/predictions/confirmation",
                    force=args.force_fusion,
                )
                or rc
            )

    if not args.skip_m2_proxy:
        proxy_out = root / "experiments/fusion/m2_confirmatory_sealed_results.json"
        rc = (
            _fusion_multi_seed(
                root,
                py,
                config="configs/attention_m2_confirmatory_sealed.yaml",
                merged_out=proxy_out,
                label="T7 M2 proxy inverted-MVTec",
                seeds=args.seeds,
                archive_root="elara_master_c/predictions/confirmation",
                force=args.force_fusion,
            )
            or rc
        )

    return _finalize(root, py, rc, started)


def _finalize(root: Path, py: str, rc: int, started: str) -> int:
    rc = _run([py, "src/scripts/scenario_c/run_one_class_degradation_sweep.py"], root, "One-class degradation sweep") or rc
    rc = _run([py, "src/scripts/emit_mvtec3d_sota_demarcation.py"], root, "Emit SOTA demarcation") or rc
    rc = _run([py, "src/scripts/scenario_c/confirmatory_statistics.py"], root, "Confirmatory statistics") or rc
    rc = _run([py, "src/scripts/scenario_c/audit_checklist_progress.py"], root, "Checklist audit") or rc
    rc = _run([py, "src/scripts/validate_theorem_stack.py"], root, "Theorem stack validator") or rc

    manifest = {
        "pipeline": "run_full_master_c_pipeline",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": rc,
    }
    out_dir = root / "elara_master_c/audits/stage_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"full_pipeline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nPipeline finished exit={rc}. Manifest: {path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
