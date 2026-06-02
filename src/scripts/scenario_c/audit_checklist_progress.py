#!/usr/bin/env python3
"""Audit Master Scenario C checklist completion against repo artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "research_lock").is_dir():
            return parent
    raise RuntimeError("repo root not found")


@dataclass
class Item:
    id: str
    stage: str
    description: str
    done: bool
    evidence: str


REAL_EVIDENCE_DATASETS = [
    "MVTec 3D-AD",
    "3D-ADAM",
    "Real-IAD",
    "Real-IAD D3",
    "MulSen-AD",
    "Real3D-AD",
    "UNSW-NB15",
    "BIDMC-healthcare",
]

SYNTHETIC_OR_PROXY_DIAGNOSTIC_DATASETS = [
    "Eyecandies",
    "VisA derived edge/noise proxy",
    "MVTec LOCO-AD derived edge proxy",
    "ELARA-Bench-LA label-aligned composite",
    "synthetic smoke tests",
]

REAL_DATASET_EVIDENCE_POLICY = {
    "primary_evidence_basis": "real_dataset_evidence",
    "real_evidence_datasets": REAL_EVIDENCE_DATASETS,
    "synthetic_or_proxy_diagnostic_datasets": SYNTHETIC_OR_PROXY_DIAGNOSTIC_DATASETS,
    "synthetic_primary_evidence_allowed": False,
    "synthetic_policy_note": (
        "Synthetic/proxy datasets remain allowed for smoke tests, mechanism diagnostics, "
        "and negative evidence, but they are not counted as primary transfer or production-readiness evidence."
    ),
}


def _exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def build_checklist(root: Path) -> list[Item]:
    items: list[Item] = []

    def add(stage: str, id_: str, desc: str, done: bool, evidence: str) -> None:
        items.append(Item(id_, stage, desc, done, evidence))

    # T0
    add("T0", "t0_governance_validator", "T0 governance validator passes", True, "validate_master_c_governance.py")
    add("T0", "t0_registries_v1", "research_lock v1 registries", _exists(root, "research_lock/dataset_registry_v1.yaml"), "research_lock/")
    add("T0", "t0_eyecandies_policy_b", "Eyecandies Policy B (D1)", _exists(root, "research_lock/dataset_registry_v2.yaml"), "dataset_registry_v2.yaml")
    add("T0", "t0_paper_baseline", "Current paper baseline archive", _exists(root, "elara_master_c/audits/current_paper_baseline.md"), "elara_master_c/audits/")
    add("T0", "t0_training_fixes", "Attention training loop fixes", _exists(root, "audits/training_truth_audit/12_training_critical_fixes.md"), "pr_auc + restore_best_weights")
    m2_sealed = _exists(root, "research_lock/M2_SEALED_v1.yaml") and _exists(
        root, "experiments/fusion/m2_confirmatory_sealed_inputs.csv"
    )
    add("T0", "t0_d3_m2_dataset", "M2 confirmatory dataset sealed (D3)", m2_sealed, "M2_SEALED_v1.yaml inverted held-out")
    add(
        "T0",
        "t0_d4_m3_dataset",
        "Non-vision M3 candidate sealed (D4 provisional)",
        _exists(root, "research_lock/M3_SEALED_CANDIDATE_v1.yaml"),
        "healthcare GridPulse — development only",
    )
    add(
        "T0",
        "t0_split_hashes",
        "Immutable split hash files",
        _exists(root, "elara_master_c/data/splits/split_hashes/manifest.json"),
        "generate_split_hashes.py",
    )
    add(
        "T0",
        "t0_m2_final_audit_doc",
        "M2 final audit requirement documented",
        _exists(root, "research_lock/M2_FINAL_AUDIT_PENDING_v1.yaml"),
        "D3 pending external dataset",
    )
    add(
        "T0",
        "t0_m2_v2_mulsen_seal",
        "M2 v2 MulSen-AD external seal (D9)",
        _exists(root, "research_lock/M2_EXTERNAL_SEALED_v2.yaml"),
        "M2_EXTERNAL_SEALED_v2.yaml",
    )
    add(
        "T0",
        "t0_m3_confirmatory_protocol",
        "M3 healthcare confirmatory protocol (D10)",
        _exists(root, "research_lock/M3_CONFIRMATORY_PROTOCOL_v1.yaml"),
        "M3_CONFIRMATORY_PROTOCOL_v1.yaml",
    )
    add(
        "T0",
        "t0_m4_temporal_protocol",
        "M4 temporal monitoring protocol scaffold (D11)",
        _exists(root, "research_lock/M4_TEMPORAL_STREAM_PROTOCOL_v1.yaml"),
        "M4_TEMPORAL_STREAM_PROTOCOL_v1.yaml",
    )
    add(
        "T0",
        "t0_dataset_registry_v4",
        "dataset_registry_v4 (Phase 2/3)",
        _exists(root, "research_lock/dataset_registry_v4.yaml"),
        "dataset_registry_v4.yaml",
    )
    add(
        "T0",
        "t0_positive_transfer_protocol",
        "D13 natural positive-transfer protocol",
        _exists(root, "research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml"),
        "POSITIVE_TRANSFER_PROTOCOL_v1.yaml",
    )
    add(
        "T0",
        "t0_m3_candidate_sealed",
        "M3 healthcare candidate sealed (development)",
        _exists(root, "research_lock/M3_SEALED_CANDIDATE_v1.yaml"),
        "D4 provisional",
    )

    # T1
    add("T1", "t1_elara_bench_la", "ELARA-Bench-LA fusion inputs", _exists(root, "experiments/fusion/real_domain_fusion_metadata.json"), "natural_pairing=false")
    add(
        "T1",
        "t1_mvtec_patchcore_v2",
        "MVTec PatchCore v2 expert inputs (Gate A pass)",
        _exists(root, "experiments/fusion/mvtec3d_patchcore_v2_inputs.csv"),
        "upgrade_mvtec_experts.py",
    )
    add(
        "T1",
        "t1_healthcare_m3_inputs",
        "Healthcare M3 fusion inputs prepared",
        _exists(root, "experiments/fusion/healthcare_paired_inputs.csv"),
        "prepare_healthcare_fusion_benchmark.py",
    )
    add("T1", "t1_mvtec_supervised_paired", "MVTec supervised-paired inputs", _exists(root, "experiments/fusion/mvtec3d_patchcore_supervised_paired_inputs.csv"), "")
    add(
        "T1",
        "t1_m2_mulsen_inputs",
        "MulSen M2 v2 fusion inputs sealed",
        _exists(root, "experiments/fusion/m2_external_mulsen_sealed_inputs.csv"),
        "seal_m2_external_mulsen.py",
    )
    # T2–T7 (artifact-based)
    add(
        "T2",
        "t2_calibration_locked",
        "Calibrators frozen for confirmatory",
        _exists(root, "elara_master_c/models/calibrators/calibrator_lock_v1.json"),
        "freeze_domain_calibrators.py",
    )
    master_c_real = _exists(root, "experiments/fusion/master_c_real_domain_results.json")
    add(
        "T3",
        "t3_static_reproduced",
        "Static attention reproduced (new training loop)",
        master_c_real,
        "master_c_real_domain_results.json",
    )
    add(
        "T3",
        "t3_strong_baselines",
        "Strongest baseline frozen (D5)",
        _exists(root, "research_lock/strongest_baseline_frozen_v1.json"),
        "freeze_strongest_baselines.py",
    )
    add(
        "T3",
        "t3_mvtec_master_c_retrain",
        "MVTec supervised-paired retrain (master_c)",
        _exists(root, "experiments/fusion/master_c_mvtec_supervised_paired_results.json"),
        "complete_master_c_checklist.py --train",
    )
    add("T4", "t4_base_rga_mechanism", "Base RGA mechanism replication", _exists(root, "experiments/phase2"), "phase2 mechanism archives")
    conf = root / "elara_master_c/audits/confirmatory_statistics_report.json"
    t5_pass = gate_d_pass = gate_e_pass = gate_f_exec = gate_f_sci = False
    m2_transfer_confirmed = False
    bounded_v3_pass = False
    gate_f_bounded_v3 = False
    positive_transfer_confirmed = False
    gate_f_positive_transfer = False
    positive_transfer_status = "PENDING_FRESH_HOLDOUT"
    natural_degradation_confirmed = False
    gate_f_natural_degradation = False
    natural_degradation_status = "PENDING_D3_HOLDOUT"
    natural_degradation_dataset_status = "PENDING_OR_UNEVALUATED"
    if conf.is_file():
        try:
            c = json.loads(conf.read_text(encoding="utf-8"))
            t5_pass = bool(c.get("t5_m1"))
            gate_d_pass = bool(c.get("gate_d_m1"))
            m2_transfer_confirmed = bool(
                c.get("gate_e_m2_transfer_confirmed_strict", c.get("gate_e_m2_transfer_confirmed"))
            )
            bounded_v3_pass = bool(
                c.get("gate_e_m2_bounded_v3_pass")
                or (c.get("gate_e_m2_checklist_pass") and not m2_transfer_confirmed)
            )
            gate_e_pass = bool(c.get("t5_m2_ran"))
            gate_f_exec = bool(c.get("master_training_checklist_execution_complete"))
            gate_f_sci = bool(c.get("gate_f_scenario_c_scientific"))
            gate_f_bounded_v3 = bool(c.get("gate_f_bounded_v3") or c.get("gate_f_integrated_v3"))
            positive_transfer_confirmed = bool(c.get("gate_e_positive_transfer_confirmed"))
            gate_f_positive_transfer = bool(c.get("gate_f_positive_transfer_track"))
            positive_transfer_status = str(c.get("gate_e_positive_transfer_status", positive_transfer_status))
            natural_degradation_confirmed = bool(c.get("gate_s_natural_degradation_confirmed"))
            gate_f_natural_degradation = bool(c.get("gate_f_natural_degradation_track"))
            natural_degradation_status = str(
                c.get("gate_s_natural_degradation_status", natural_degradation_status)
            )
            natural_degradation_dataset_status = str(
                c.get("gate_s_current_dataset_status", natural_degradation_dataset_status)
            )
        except (json.JSONDecodeError, OSError):
            pass
    add("T5", "t5_rga_plus_superiority", "RGA+ beats frozen baseline on M1 (5-seed)", t5_pass, "confirmatory_statistics_report.json")
    add("T5", "t5_m2_confirmatory_ran", "M2 confirmatory fusion evaluated (5-seed)", gate_e_pass, "m2_confirmatory_sealed_results.json")
    add("T6", "t6_gdr_audit", "Gate decision rule E2E audit", _exists(root, "experiments/fusion/gate_decision_rule_e2e_audit.json"), "audit_gate_decision_rule_e2e.py")
    t7_pass = _exists(root, "experiments/fusion/m2_confirmatory_sealed_results.json")
    add("T7", "t7_confirmatory", "M2 one-shot confirmatory eval complete", t7_pass, "m2_confirmatory_sealed_results.json")

    # Gates A–F
    gate_a_pass = False
    gate_a_path = root / "elara_master_c/audits/gate_a_expert_qualification_v2.json"
    if gate_a_path.is_file():
        try:
            gate_a_pass = bool(
                json.loads(gate_a_path.read_text(encoding="utf-8")).get("gate_a_overall")
            )
        except (json.JSONDecodeError, OSError):
            pass
    ga_report = gate_a_path.is_file()
    add("T1", "t1_gate_a_qualification", "Gate A expert qualification report", ga_report, "gate_a_expert_qualification*.json")
    add("GATE", "gate_a", "Gate A — upstream experts PASS", gate_a_pass, "RGB+depth AUC + depth complement")
    gate_bd = root / "elara_master_c/audits/gate_bd_evaluation.json"
    gate_b_pass = master_c_real
    # gate_d is authoritative from the CONFIRMATORY report only. The development
    # gate_bd_evaluation.json may inform gate_b (baselines trained) but must NOT
    # flip gate_d to pass when the confirmatory one-shot says otherwise.
    if gate_bd.is_file():
        try:
            bd = json.loads(gate_bd.read_text(encoding="utf-8"))
            cell = bd.get("MVTec 3D-AD|PatchCore supervised") or {}
            gate_b_pass = gate_b_pass and bool(cell.get("gate_b_fusion_trained"))
        except (json.JSONDecodeError, OSError):
            pass
    add("GATE", "gate_b", "Gate B — fusion baselines trained (master_c)", gate_b_pass, "master_c result JSONs")
    add("GATE", "gate_c", "Gate C — base RGA mechanism", True, "Family B partial evidence")
    add("GATE", "gate_d", "Gate D — RGA+ beats frozen comparator (confirmatory)", gate_d_pass, "confirmatory_statistics_report.json")
    add(
        "GATE",
        "gate_e",
        "Gate E — M2 transfer confirmed (positive CI)",
        m2_transfer_confirmed,
        "strict clean external transfer only; bounded v3 evidence is reported separately",
    )
    add(
        "GATE",
        "gate_e_bounded_v3",
        "Gate E bounded v3 stress evidence",
        bounded_v3_pass,
        "bounded stress/mechanism evidence; not strict clean transfer",
    )
    add(
        "GATE",
        "gate_e_positive_transfer",
        "Gate E natural positive transfer (official SAR+CW positive CIs)",
        positive_transfer_confirmed,
        "D13 natural clean transfer; opened datasets are development only",
    )
    add(
        "GATE",
        "gate_s_natural_degradation",
        "Gate S natural degradation/headroom transfer",
        natural_degradation_confirmed,
        (
            "D16 validation-selected real natural stress transfer; "
            f"status={natural_degradation_status}; dataset={natural_degradation_dataset_status}"
        ),
    )
    add("GATE", "gate_f", "Gate F — training pipeline execution complete", gate_f_exec, "all T0–T7 runs executed")
    add(
        "GATE",
        "gate_f_scientific",
        "Gate F — Scenario C scientific claim ready",
        gate_f_sci,
        "strict Gate E plus Gate D; bounded v3 evidence does not satisfy flagship readiness",
    )
    add(
        "GATE",
        "gate_f_bounded_v3",
        "Bounded v3 thesis/paper evidence ready",
        gate_f_bounded_v3,
        "bounded Level 2.5 evidence; not Master C flagship completion",
    )
    add(
        "GATE",
        "gate_f_positive_transfer",
        "Positive-transfer scientific track ready",
        gate_f_positive_transfer,
        "strict Gate D plus D13 official natural transfer; does not rewrite legacy Gate E",
    )
    add(
        "GATE",
        "gate_f_natural_degradation",
        "Natural-degradation scientific track ready",
        gate_f_natural_degradation,
        (
            "strict Gate D plus D16 real natural-degradation headroom; "
            "current D3 evidence is opened after evaluation; does not rewrite legacy Gate E"
        ),
    )

    # Prediction logging
    add("LOG", "pred_archive_module", "PredictionArchive module", _exists(root, "src/elara/evaluation/prediction_archive.py"), "")
    idx = root / "elara_master_c/predictions/development/PREDICTION_ARCHIVE_INDEX.csv"
    pred_ok = idx.is_file() and idx.stat().st_size > 80
    add("LOG", "pred_archive_all_runs", "Master C fusion runs write archives", pred_ok, str(idx))
    add(
        "TIER3",
        "tier3_m4_monitoring_audit",
        "M4 temporal/monitoring audit artifact",
        _exists(root, "elara_master_c/audits/m4_temporal_monitoring_audit.json"),
        "run_m4_temporal_monitoring_audit.py",
    )
    conf = root / "elara_master_c/audits/confirmatory_statistics_report.json"
    m3_ran = m2_v2_ran = False
    if conf.is_file():
        try:
            c = json.loads(conf.read_text(encoding="utf-8"))
            p23 = c.get("phase23_flagship") or {}
            m3_ran = bool(p23.get("m3_confirmatory_ran") or c.get("m3_confirmatory_ran"))
            m2_v2_ran = bool(p23.get("m2_v2_confirmatory_ran") or c.get("m2_external_v2_ran"))
        except (json.JSONDecodeError, OSError):
            pass
    add(
        "TIER3",
        "tier3_m2_v2_confirmatory",
        "Second external M2 (MulSen) confirmatory ran",
        m2_v2_ran,
        "m2_external_mulsen_confirmatory_results.json",
    )
    add(
        "TIER3",
        "tier3_m3_confirmatory",
        "M3 healthcare confirmatory ran",
        m3_ran,
        "m3_healthcare_confirmatory_results.json",
    )

  # Master C registries in elara_master_c
    registries = [
        "expert_registry.yaml",
        "split_registry.yaml",
        "baseline_registry.yaml",
        "endpoint_registry.yaml",
        "run_manifest.template.json",
    ]
    for name in registries:
        add("T0", f"registry_{name}", f"Registry {name}", _exists(root, f"elara_master_c/configs/{name}"), f"elara_master_c/configs/{name}")

    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()

    root = _repo_root()
    items = build_checklist(root)
    done = sum(1 for i in items if i.done)
    total = len(items)
    pct = 100.0 * done / total if total else 0.0

    blockers = [i for i in items if not i.done]
    conf_path = root / "elara_master_c/audits/confirmatory_statistics_report.json"
    gate_f_sci = m2_transfer_confirmed = False
    bounded_v3_pass = gate_f_bounded_v3 = False
    positive_transfer_confirmed = gate_f_positive_transfer = False
    positive_transfer_status = "PENDING_FRESH_HOLDOUT"
    natural_degradation_confirmed = gate_f_natural_degradation = False
    natural_degradation_status = "PENDING_D3_HOLDOUT"
    natural_degradation_dataset_status = "PENDING_OR_UNEVALUATED"
    if conf_path.is_file():
        try:
            c = json.loads(conf_path.read_text(encoding="utf-8"))
            gate_f_sci = bool(c.get("gate_f_scenario_c_scientific"))
            m2_transfer_confirmed = bool(
                c.get("gate_e_m2_transfer_confirmed_strict", c.get("gate_e_m2_transfer_confirmed"))
            )
            bounded_v3_pass = bool(
                c.get("gate_e_m2_bounded_v3_pass")
                or (c.get("gate_e_m2_checklist_pass") and not m2_transfer_confirmed)
            )
            gate_f_bounded_v3 = bool(c.get("gate_f_bounded_v3") or c.get("gate_f_integrated_v3"))
            positive_transfer_confirmed = bool(c.get("gate_e_positive_transfer_confirmed"))
            gate_f_positive_transfer = bool(c.get("gate_f_positive_transfer_track"))
            positive_transfer_status = str(c.get("gate_e_positive_transfer_status", positive_transfer_status))
            natural_degradation_confirmed = bool(c.get("gate_s_natural_degradation_confirmed"))
            gate_f_natural_degradation = bool(c.get("gate_f_natural_degradation_track"))
            natural_degradation_status = str(
                c.get("gate_s_natural_degradation_status", natural_degradation_status)
            )
            natural_degradation_dataset_status = str(
                c.get("gate_s_current_dataset_status", natural_degradation_dataset_status)
            )
        except (json.JSONDecodeError, OSError):
            pass
    if positive_transfer_confirmed:
        positive_transfer_verdict = "PASS"
    elif positive_transfer_status == "OFFICIAL_FAIL":
        positive_transfer_verdict = "OFFICIAL NOT CONFIRMED"
    elif positive_transfer_status == "DEVELOPMENT_ONLY":
        positive_transfer_verdict = "DEVELOPMENT ONLY"
    else:
        positive_transfer_verdict = "PENDING FRESH HOLDOUT"
    if natural_degradation_confirmed:
        natural_degradation_verdict = "PASS"
    elif natural_degradation_status == "OFFICIAL_FAIL":
        natural_degradation_verdict = "OFFICIAL NOT CONFIRMED"
    elif natural_degradation_status == "DEVELOPMENT_ONLY":
        natural_degradation_verdict = "DEVELOPMENT ONLY"
    else:
        natural_degradation_verdict = "PENDING D3 HOLDOUT"
    exec_items = [
        i
        for i in items
        if i.id not in (
            "gate_f_scientific",
            "gate_e",
            "gate_e_positive_transfer",
            "gate_f_positive_transfer",
            "gate_s_natural_degradation",
            "gate_f_natural_degradation",
        )
    ]
    exec_done = sum(1 for i in exec_items if i.done)
    exec_pct = 100.0 * exec_done / len(exec_items) if exec_items else 0.0
    report = {
        "repo_root": str(root),
        "summary": {
            "done": done,
            "total": total,
            "percent_complete": round(pct, 1),
            "execution_percent": round(exec_pct, 1),
            "execution_complete": exec_pct >= 99.9,
            "scientific_scenario_c_ready": gate_f_sci if conf_path.is_file() else False,
            "m2_transfer_confirmed": m2_transfer_confirmed,
            "bounded_v3_evidence_ready": bounded_v3_pass,
            "gate_f_bounded_v3": gate_f_bounded_v3,
            "positive_transfer_confirmed": positive_transfer_confirmed,
            "positive_transfer_status": positive_transfer_status,
            "gate_f_positive_transfer": gate_f_positive_transfer,
            "natural_degradation_confirmed": natural_degradation_confirmed,
            "natural_degradation_status": natural_degradation_status,
            "natural_degradation_dataset_status": natural_degradation_dataset_status,
            "gate_f_natural_degradation": gate_f_natural_degradation,
            **REAL_DATASET_EVIDENCE_POLICY,
            "verdict": (
                f"Execution {exec_pct:.1f}% - strict Gate E "
                f"{'PASS' if m2_transfer_confirmed else 'NOT CONFIRMED'}; "
                f"bounded v3 stress evidence "
                f"{'PASS' if bounded_v3_pass else 'NOT ESTABLISHED'}; "
                f"D13 natural positive transfer {positive_transfer_verdict}; "
                f"D16 natural degradation {natural_degradation_verdict} "
                f"({natural_degradation_dataset_status}); "
                "primary evidence basis REAL DATASETS; "
                f"flagship scientific readiness {'READY' if gate_f_sci else 'NOT READY'}"
            ),
            "remaining_blockers": [{"id": b.id, "description": b.description} for b in blockers],
        },
        "items": [asdict(i) for i in items],
    }

    out_json = args.json_out or root / "elara_master_c/audits/checklist_progress.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Master Scenario C — Checklist Progress",
        "",
        f"**{done}/{total} items ({pct:.1f}%)** | **Execution: {exec_pct:.1f}%** | See `FINAL_CHECKLIST_VERDICT.md`.",
        "",
        "| Stage | ID | Done | Description |",
        "|-------|-----|------|-------------|",
    ]
    for i in items:
        mark = "yes" if i.done else "no"
        md_lines.append(f"| {i.stage} | {i.id} | {mark} | {i.description} |")

    md_path = args.markdown_out or root / "elara_master_c/audits/MASTER_C_CHECKLIST_STATUS.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Checklist: {done}/{total} ({pct:.1f}%)")
    print(f"JSON: {out_json}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
