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


def _per_sample_paired_bootstrap_from_archive(
    root: Path,
    benchmark: str,
    protocol: str,
    rga_method: str,
    comparator_method: str,
    n_iter: int = 10000,
    seed: int = 0,
) -> dict | None:
    """Compute a true per-sample paired bootstrap CI on Δ AUC from archived
    predictions. Resolves the degenerate-seed-bootstrap problem flagged in
    `validity_reasons`: instead of bootstrapping over per-seed delta means
    (which collapses to a point when seed variance is zero), we bootstrap
    over per-sample (sample, score_rga, score_static, label) tuples averaged
    across all available seeds.

    Returns None if archive predictions for this cell are absent.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    archive_roots = [
        root / "elara_master_c" / "predictions" / "confirmation",
        root / "elara_master_c" / "predictions" / "development",
    ]
    # Tolerate small protocol-name aliases between the cell definitions and
    # archive metadata (e.g. "PatchCore supervised" vs "PatchCore supervised-paired",
    # "M2_one_shot_audit_proxy" vs "M2_one_shot_audit").
    proto_aliases = {protocol}
    if protocol.endswith("_proxy"):
        proto_aliases.add(protocol[:-len("_proxy")])
    if protocol == "PatchCore supervised":
        proto_aliases.add("PatchCore supervised-paired")
    if protocol == "PatchCore supervised-paired":
        proto_aliases.add("PatchCore supervised")

    for archive_root in archive_roots:
        idx_path = archive_root / "PREDICTION_ARCHIVE_INDEX.csv"
        if not idx_path.is_file():
            continue
        idx = pd.read_csv(idx_path)
        cell = idx[
            (idx["benchmark"] == benchmark)
            & (idx["protocol"].isin(list(proto_aliases)))
            & (idx["split"] == "test")
        ]
        if cell.empty:
            continue
        rga_rows = cell[cell["method"] == rga_method]
        comp_rows = cell[cell["method"] == comparator_method]
        if rga_rows.empty or comp_rows.empty:
            continue

        def _stack(rows):
            seed_to_df = {}
            for _, r in rows.iterrows():
                p = root / r["artifact_path"]
                if not p.is_file() or p.name.startswith("._"):
                    continue
                df = pd.read_parquet(p)
                seed_to_df[int(r["seed"])] = df
            return seed_to_df

        rga_per_seed = _stack(rga_rows)
        comp_per_seed = _stack(comp_rows)
        common_seeds = sorted(set(rga_per_seed) & set(comp_per_seed))
        if not common_seeds:
            continue

        # Build a per-sample score matrix [n_samples x n_seeds] for each method.
        first = rga_per_seed[common_seeds[0]]
        sids = first["sample_id"].to_numpy()
        labels = first["label"].to_numpy().astype(int)
        # Drop unlabeled rows (label=-1)
        keep = labels >= 0
        sids = sids[keep]; labels = labels[keep]
        n_samples = len(sids)
        rga_mat = np.zeros((n_samples, len(common_seeds)), dtype=float)
        comp_mat = np.zeros((n_samples, len(common_seeds)), dtype=float)
        for j, s in enumerate(common_seeds):
            r_df = rga_per_seed[s].set_index("sample_id").reindex(sids)
            c_df = comp_per_seed[s].set_index("sample_id").reindex(sids)
            rga_mat[:, j] = r_df["raw_score"].to_numpy()
            comp_mat[:, j] = c_df["raw_score"].to_numpy()

        # Seed-averaged per-sample score (paired across methods)
        rga_score = rga_mat.mean(axis=1)
        comp_score = comp_mat.mean(axis=1)

        # Skip degenerate label distributions
        if len(np.unique(labels)) < 2:
            return None

        mean_rga_auc = float(roc_auc_score(labels, rga_score))
        mean_comp_auc = float(roc_auc_score(labels, comp_score))
        delta_point = mean_rga_auc - mean_comp_auc

        rng = np.random.default_rng(int(seed))
        deltas = np.empty(n_iter, dtype=float)
        idx_arr = np.arange(n_samples)
        for i in range(n_iter):
            sel = rng.integers(0, n_samples, size=n_samples)
            lbl_b = labels[sel]
            if len(np.unique(lbl_b)) < 2:
                deltas[i] = np.nan
                continue
            r_b = rga_score[sel]; c_b = comp_score[sel]
            try:
                deltas[i] = roc_auc_score(lbl_b, r_b) - roc_auc_score(lbl_b, c_b)
            except ValueError:
                deltas[i] = np.nan
        valid = deltas[~np.isnan(deltas)]
        if valid.size < 100:
            return None
        ci_low = float(np.percentile(valid, 2.5))
        ci_high = float(np.percentile(valid, 97.5))
        return {
            "inference_mode": "per_sample_paired_bootstrap_seed_averaged",
            "n_samples": int(n_samples),
            "n_seeds_pooled": int(len(common_seeds)),
            "mean_rga_auc": mean_rga_auc,
            "mean_comp_auc": mean_comp_auc,
            "delta_point": float(delta_point),
            "bootstrap_95_ci_low": ci_low,
            "bootstrap_95_ci_high": ci_high,
            "bootstrap_n_iter": int(n_iter),
        }
    return None


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
    for rel in (
        "experiments/fusion/m2_external_3d_adam_transfer_v1_paired_inference.json",
        "experiments/fusion/m2_external_3d_adam_paired_inference.json",
    ):
        path = root / rel
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


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

    # --- Per-sample paired-bootstrap fallback for degenerate-seed-variance cells ---
    # When the seed-level bootstrap is degenerate, fall back to a true per-sample
    # paired bootstrap on the archived predictions (the fix recommended in the
    # validity-reason text). If the fallback yields a valid interval, replace
    # the degenerate CI and recompute the gate decisions on the new CI.
    fallback = None
    if not cell_valid and any("degenerate_ci_zero_seed_variance" in r for r in reasons):
        fallback = _per_sample_paired_bootstrap_from_archive(
            _repo_root(), benchmark, protocol,
            rga_method="rga_boosted_fusion",
            comparator_method=comp,
        )
        if fallback is not None:
            ci = {
                "low": fallback["bootstrap_95_ci_low"],
                "high": fallback["bootstrap_95_ci_high"],
                "mean": fallback["delta_point"],
            }
            ci_width = ci["high"] - ci["low"]
            # The per-sample bootstrap supersedes the degenerate seed-bootstrap
            # finding, so drop that specific reason. Below-chance AUC reason
            # (if present) still applies and still blocks the cell.
            reasons = [r for r in reasons if "degenerate_ci_zero_seed_variance" not in r]
            cell_valid = len(reasons) == 0

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
        "per_sample_bootstrap_fallback": fallback,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    m2_paired = _load_m2_external_paired_inference(root)
    m2_external = root / "experiments/fusion/m2_external_3d_adam_transfer_v1_confirmatory_results.json"
    if not m2_external.is_file():
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
