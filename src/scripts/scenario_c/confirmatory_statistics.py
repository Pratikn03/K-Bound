#!/usr/bin/env python3
"""Confirmatory stats for T5 / Gate E (bootstrap CI, Holm, clean false-fire)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from uais.utils.stats import holm_bonferroni


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


def _load_m2_external_paired_inference(root: Path) -> dict | None:
    path = root / "experiments/fusion/m2_external_3d_adam_paired_inference.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_paired_into_cell(cell: dict, paired: dict) -> dict:
    """Overlay per-sample DeLong/bootstrap onto M2_external seed-level cell."""
    primary_id = paired.get("primary_comparison", "M2-EXTERNAL-vs-SAR")
    rows = {c["comparison_id"]: c for c in paired.get("comparisons", [])}
    row = rows.get(primary_id)
    if row is None:
        return cell
    ci = {
        "low": float(row["bootstrap_95_ci_low"]),
        "high": float(row["bootstrap_95_ci_high"]),
        "mean": float(row["ensemble_delta_auc"]),
    }
    cell.update(
        {
            "inference_mode": "per_sample_paired_ensemble",
            "paired_inference_path": "experiments/fusion/m2_external_3d_adam_paired_inference.json",
            "n_test_samples": int(row["n_test_samples"]),
            "mean_rga_auc": float(row["ensemble_rga_auc"]),
            "mean_base_auc": float(row["ensemble_comparator_auc"]),
            "mean_delta_roc_auc": float(row["ensemble_delta_auc"]),
            "delong_p_raw": float(row["delong_p_raw"]),
            "delong_p_holm": float(row.get("delong_p_holm", row["delong_p_raw"])),
            "bootstrap_95_ci": ci,
            "bootstrap_ci_width": float(ci["high"] - ci["low"]),
            "bootstrap_ci_excludes_zero": bool(row.get("bootstrap_ci_excludes_zero")),
            "practical_effect_band": str(row.get("practical_effect_band", "")),
            "per_seed_rga_auc": row.get("per_seed_rga_auc"),
            "per_seed_comparator_auc": row.get("per_seed_comparator_auc"),
            "per_seed_delta_auc": row.get("per_seed_delta_auc"),
            "cell_valid": True,
            "validity_reasons": [],
            "holm_bonferroni": {
                "rejected": [bool(row.get("delong_p_holm", 1.0) < 0.05)],
                "p_adjusted": [float(row.get("delong_p_holm", row["delong_p_raw"]))],
            },
        }
    )
    clean_ffr = float(cell.get("clean_false_fire_proxy", 0.0))
    cell["gate_e_pass"] = bool(
        row.get("bootstrap_ci_excludes_zero") and float(row["ensemble_delta_auc"]) > 0
    )
    cell["gate_d_pass"] = bool(float(row["ensemble_delta_auc"]) > 0 and cell["gate_e_pass"])
    cell["t5_confirmatory_pass"] = bool(
        cell["gate_d_pass"] and cell["holm_bonferroni"]["rejected"][0]
    )
    return cell


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
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        boot.append(float(np.mean(delta[idx])))
    ci = {
        "low": float(np.percentile(boot, 2.5)),
        "high": float(np.percentile(boot, 97.5)),
        "mean": float(np.mean(boot)),
    }
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
    holm_raw = holm_bonferroni(np.array(pvals), alpha=0.05) if pvals else {"reject": [], "p_adjusted": []}
    # holm_bonferroni returns the boolean mask under key "reject".
    reject_mask = holm_raw.get("reject", holm_raw.get("rejected", []))
    holm = {
        "rejected": [bool(x) for x in np.asarray(reject_mask).tolist()],
        "p_adjusted": [float(x) for x in np.asarray(holm_raw.get("p_adjusted", [])).tolist()],
    }

    # --- Validity guard (prevents fake-CI / degenerate cells from "passing") ---
    # The reported fusion methods are deterministic per seed, so a seed-level
    # bootstrap can collapse to a point (zero width). Such a CI is NOT a valid
    # interval and must never be read as "excludes zero". Likewise, a cell where
    # the methods score below chance (AUC < 0.5) is degenerate and cannot support
    # a transfer/superiority claim. In both cases the only honest verdict is
    # "not established" -> all gates fail until a real per-sample paired test
    # (DeLong / per-sample bootstrap on archived predictions) is computed.
    mean_rga = float(np.mean(rga[:n]))
    mean_base = float(np.mean(base[:n]))
    min_method_auc = min(mean_rga, mean_base)
    ci_width = float(ci["high"] - ci["low"])
    seed_variance = float(np.var(delta))
    reasons: list[str] = []
    if ci_width < 1e-9 or seed_variance < 1e-12:
        reasons.append(
            "degenerate_ci_zero_seed_variance: per-seed deltas are identical "
            "(deterministic methods); the seed-bootstrap CI is not a valid "
            "interval. Requires a per-sample paired test (DeLong / per-sample "
            "bootstrap on archived predictions)."
        )
    if min_method_auc < 0.5:
        reasons.append(
            f"below_chance_auc: min(rga={mean_rga:.4f}, base={mean_base:.4f}) < 0.5 "
            "-> degenerate cell; not a usable comparison."
        )
    cell_valid = len(reasons) == 0
    holm_rejected = len(holm.get("rejected", [])) > 0 and any(holm["rejected"])

    pass_gate_e = cell_valid and bool(ci["low"] > 0) and clean_ffr <= 0.10
    pass_gate_d = cell_valid and bool(mean_rga > mean_base) and pass_gate_e
    # t5 (superiority) now requires genuine significance, not merely ci_low>0.
    pass_t5 = cell_valid and pass_gate_d and holm_rejected

    return {
        "family": family,
        "benchmark": benchmark,
        "protocol": protocol,
        "results_path": str(results_path),
        "frozen_comparator": comp,
        "n_seeds": int(n),
        "mean_rga_auc": mean_rga,
        "mean_base_auc": mean_base,
        "mean_delta_roc_auc": float(np.mean(delta)),
        "seed_variance": seed_variance,
        "bootstrap_95_ci": ci,
        "bootstrap_ci_width": ci_width,
        "clean_false_fire_proxy": clean_ffr,
        "holm_bonferroni": holm,
        "cell_valid": cell_valid,
        "validity_reasons": reasons,
        "gate_d_pass": pass_gate_d,
        "gate_e_pass": pass_gate_e,
        "t5_confirmatory_pass": pass_t5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    m2_paired = _load_m2_external_paired_inference(root)
    m2_external = root / "experiments/fusion/m2_external_3d_adam_confirmatory_results.json"
    m2_proxy = root / "experiments/fusion/m2_confirmatory_sealed_results.json"
    cells = [
        (
            root / "experiments/fusion/master_c_mvtec_supervised_paired_results.json",
            "MVTec 3D-AD",
            "PatchCore supervised",
            "M1",
        ),
    ]
    # Gate E (P4): authoritative external one-shot when present; else legacy proxy.
    if m2_external.is_file():
        cells.append(
            (
                m2_external,
                "3D-ADAM category-held-out",
                "M2_external_one_shot_audit",
                "M2_external",
            )
        )
    if m2_proxy.is_file():
        cells.append(
            (
                m2_proxy,
                "MVTec 3D-AD inverted-heldout",
                "M2_one_shot_audit_proxy",
                "M2_proxy",
            )
        )
    report: dict = {
        "cells": [],
        "gate_d_m1": False,
        "gate_d_m2": False,
        "gate_e_m2_transfer_confirmed": False,
        "t5_m1": False,
        "t5_m2_ran": False,
    }
    for path, bench, proto, fam in cells:
        if not path.is_file():
            report["cells"].append({"benchmark": bench, "error": "missing results", "t5_confirmatory_pass": False})
            continue
        cell = evaluate_cell(path, benchmark=bench, protocol=proto, family=fam)
        report["cells"].append(cell)
        if fam == "M1":
            report["t5_m1"] = bool(cell.get("t5_confirmatory_pass"))
            report["gate_d_m1"] = bool(cell.get("gate_d_pass"))
        if fam == "M2_external":
            if m2_paired is not None:
                cell = _merge_paired_into_cell(cell, m2_paired)
                report["cells"][-1] = cell
            report["t5_m2_ran"] = True
            report["gate_e_m2_transfer_confirmed"] = bool(cell.get("gate_e_pass"))
            report["gate_d_m2_external"] = bool(cell.get("gate_d_pass"))
            report["m2_external_cell_valid"] = bool(cell.get("cell_valid"))
            if m2_paired is not None:
                report["m2_external_paired_inference"] = m2_paired
        if fam == "M2_proxy":
            report["t5_m2_ran"] = report.get("t5_m2_ran") or True
            report["m2_proxy_ran"] = True
            report["gate_e_m2_proxy"] = bool(cell.get("gate_e_pass"))
            report["gate_d_m2_proxy"] = bool(cell.get("gate_d_pass"))

    report["gate_f_scenario_c_scientific"] = bool(
        report.get("gate_d_m1")
        and report.get("gate_e_m2_transfer_confirmed")
        and _exists_gate_a(root)
        and _exists_gate_c(root)
    )
    report["master_training_checklist_execution_complete"] = bool(
        report.get("t5_m2_ran") and _exists_gate_a(root) and _exists_gate_c(root)
    )

    out = root / "elara_master_c/audits/confirmatory_statistics_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("master_training_checklist_execution_complete") else 1


def _exists_gate_a(root: Path) -> bool:
    p = root / "elara_master_c/audits/gate_a_expert_qualification_v2.json"
    if not p.is_file():
        return False
    return bool(json.loads(p.read_text()).get("gate_a_overall"))


def _exists_gate_c(root: Path) -> bool:
    return (root / "experiments/fusion/gate_decision_rule_e2e_audit.json").is_file()


if __name__ == "__main__":
    sys.exit(main())
