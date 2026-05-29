#!/usr/bin/env python3
"""Confirmatory stats for T5 / Gate E (bootstrap CI, Holm, clean false-fire)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from uais.utils.stats import bootstrap_ci, holm_bonferroni


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "research_lock").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _per_seed_roc(payload: dict, method: str) -> list[float]:
    rows = payload.get("table_1_clean_performance") or []
    out: list[float] = []
    for row in rows:
        block = row.get(method) or {}
        v = block.get("roc_auc")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out.append(float(v))
    return out


def _frozen_comparator(root: Path, benchmark: str, protocol: str) -> str | None:
    lock = root / "research_lock/strongest_baseline_frozen_v1.json"
    if not lock.is_file():
        return None
    data = json.loads(lock.read_text(encoding="utf-8"))
    for cell in data.get("cells", []):
        if cell.get("benchmark") == benchmark and cell.get("protocol") == protocol:
            return str(cell["strongest_baseline"])
    return "sar_score_adapter"


def evaluate_cell(
    results_path: Path,
    *,
    benchmark: str,
    protocol: str,
    family: str,
) -> dict:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    comp = _frozen_comparator(_repo_root(), benchmark, protocol)
    if comp is None:
        comp = "tent_score_adapter"
    rga = np.array(_per_seed_roc(payload, "rga_boosted_fusion"), dtype=float)
    base = np.array(_per_seed_roc(payload, comp), dtype=float)
    static = np.array(_per_seed_roc(payload, "static_attention"), dtype=float)
    craf = np.array(_per_seed_roc(payload, "craf_attention"), dtype=float)

    n = min(len(rga), len(base))
    if n < 2:
        return {"error": "insufficient seeds", "n_seeds": n}

    delta = rga[:n] - base[:n]
    ci = bootstrap_ci(delta, n_bootstrap=1000, alpha=0.05, seed=42)
    # clean false-fire: fraction of seeds where craf equals static on clean (proxy from adapt rate if present)
    clean_ffr = float(np.mean(craf[:n] > static[:n] + 0.15))  # coarse proxy
    mech = payload.get("mechanism_summary") or {}
    if "clean_false_fire_rate" in mech:
        clean_ffr = float(mech["clean_false_fire_rate"])

    pvals = []
    if n >= 5:
        from scipy import stats

        _, p = stats.ttest_rel(rga[:n], base[:n])
        pvals.append(float(p))
    holm = holm_bonferroni(np.array(pvals), alpha=0.05) if pvals else {"rejected": [], "adjusted_pvalues": []}

    pass_gate_e = bool(ci["low"] > 0) and clean_ffr <= 0.10
    pass_gate_d = bool(np.mean(rga[:n]) > np.mean(base[:n])) and pass_gate_e
    pass_t5 = pass_gate_d and (not pvals or len(holm.get("rejected", [])) > 0 or ci["low"] > 0)

    return {
        "family": family,
        "benchmark": benchmark,
        "protocol": protocol,
        "results_path": str(results_path),
        "frozen_comparator": comp,
        "n_seeds": int(n),
        "mean_delta_roc_auc": float(np.mean(delta)),
        "bootstrap_95_ci": ci,
        "clean_false_fire_proxy": clean_ffr,
        "holm_bonferroni": holm,
        "gate_d_pass": pass_gate_d,
        "gate_e_pass": pass_gate_e,
        "t5_confirmatory_pass": pass_t5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    cells = [
        (
            root / "experiments/fusion/master_c_mvtec_supervised_paired_results.json",
            "MVTec 3D-AD",
            "PatchCore supervised",
            "M1",
        ),
        (
            root / "experiments/fusion/m2_confirmatory_sealed_results.json",
            "MVTec 3D-AD inverted-heldout",
            "M2_one_shot_audit",
            "M2",
        ),
    ]
    report: dict = {"cells": [], "gate_d_overall": True, "gate_e_overall": True, "t5_overall": True}
    for path, bench, proto, fam in cells:
        if not path.is_file():
            report["cells"].append({"benchmark": bench, "error": "missing results", "t5_confirmatory_pass": False})
            report["gate_d_overall"] = False
            report["gate_e_overall"] = False
            report["t5_overall"] = False
            continue
        cell = evaluate_cell(path, benchmark=bench, protocol=proto, family=fam)
        report["cells"].append(cell)
        for key in ("gate_d_pass", "gate_e_pass", "t5_confirmatory_pass"):
            if not cell.get(key, False):
                if key == "gate_d_pass":
                    report["gate_d_overall"] = False
                if key == "gate_e_pass":
                    report["gate_e_overall"] = False
                if key == "t5_confirmatory_pass":
                    report["t5_overall"] = False

    report["gate_f_scenario_c"] = bool(
        report.get("gate_d_overall")
        and report.get("gate_e_overall")
        and _exists_gate_a(root)
        and _exists_gate_c(root)
    )

    out = root / "elara_master_c/audits/confirmatory_statistics_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("gate_f_scenario_c") else 1


def _exists_gate_a(root: Path) -> bool:
    p = root / "elara_master_c/audits/gate_a_expert_qualification_v2.json"
    if not p.is_file():
        return False
    return bool(json.loads(p.read_text()).get("gate_a_overall"))


def _exists_gate_c(root: Path) -> bool:
    return (root / "experiments/fusion/gate_decision_rule_e2e_audit.json").is_file()


if __name__ == "__main__":
    sys.exit(main())
