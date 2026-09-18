#!/usr/bin/env python3
"""Protocol B KGA Evaluation Across Independently Trained Source Models (Task 1).

Evaluates Protocol B KGA routing on each of the 3 source models:
- Model 0 (Canonical baseline, clean acc = 0.8737, seed 0)
- Model 101 (Independent model, clean acc = 0.8953, seed 101)
- Model 102 (Independent model, clean acc = 0.8956, seed 102)

Outputs:
- experiments/kbound/results/task1_independent_models_replication/task1_kga_models_summary.json
- docs/research/kbound/paper/generated/tab_task1_independent_models.tex
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import kga
from kga.crossfit import controlled_grid_crossfit, controlled_grid_sample_id

ROOT = Path(__file__).resolve().parents[4]
TASK1_DIR = ROOT / "experiments/kbound/results/task1_independent_models_replication"
SUMMARY_JSON = TASK1_DIR / "replication_summary.json"
OUT_JSON = TASK1_DIR / "task1_kga_models_summary.json"
OUT_TEX = ROOT / "docs/research/kbound/paper/generated/tab_task1_independent_models.tex"


def evaluate_model_kga(
    model_key: str,
    records: list[dict[str, Any]],
    clean_acc: float,
    alpha: float = 0.10,
    n_folds: int = 4,
    random_state: int = 0,
) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        raise ValueError(f"No records for model {model_key}")

    B = np.array([r["delta"] for r in records], dtype=float)
    a0 = np.array([r["acc_frozen"] for r in records], dtype=float)
    aa = np.array([r["acc_adapted"] for r in records], dtype=float)

    sample_ids = [
        controlled_grid_sample_id(
            track="task1_independent_models",
            candidate=model_key,
            seed=r["stream_seed"],
            condition=f"{r['corruption']}_sev{r['severity']}",
        )
        for r in records
    ]

    # Feature matrix Z: frozen accuracy
    Z = a0.reshape(-1, 1)

    crossfit_result = controlled_grid_crossfit(
        Z,
        B,
        sample_ids=sample_ids,
        alpha=alpha,
        n_folds=min(n_folds, n // 2),
        random_state=random_state,
    )

    preds = crossfit_result.prediction
    radii = crossfit_result.radius
    actions = crossfit_result.action

    a_deployed = np.where(actions == "ADAPT", aa, a0)
    a_oracle = np.maximum(aa, a0)
    oracle_regret = a_oracle - a_deployed

    # Point gate (same prediction)
    point_actions = np.where(preds > 0, "ADAPT", "FREEZE")
    point_deployed = np.where(point_actions == "ADAPT", aa, a0)
    point_regret = a_oracle - point_deployed
    point_fa_u = float(np.mean((point_actions == "ADAPT") & (B <= 0)))

    # Counts
    helpful_count = int(np.sum(B > 0))
    harmful_count = int(np.sum(B < 0))
    tied_count = int(np.sum(B == 0))

    adapt_count = int(np.sum(actions == "ADAPT"))
    freeze_count = int(np.sum(actions == "FREEZE"))
    abstain_count = int(np.sum(actions == "ABSTAIN"))

    # Errors
    fa_u = float(np.mean((actions == "ADAPT") & (B <= 0)))
    ff_u = float(np.mean((actions != "ADAPT") & (B > 0)))

    fa_c = float(np.mean(B[actions == "ADAPT"] <= 0)) if adapt_count > 0 else None
    ff_c = float(np.mean(B[actions != "ADAPT"] > 0)) if (freeze_count + abstain_count) > 0 else None

    # Capture rates
    helpful_captured = float(np.sum((actions == "ADAPT") & (B > 0)) / helpful_count) if helpful_count > 0 else 1.0
    pos_benefit_total = float(np.sum(B[B > 0]))
    pos_benefit_captured = float(np.sum(B[(actions == "ADAPT") & (B > 0)]) / pos_benefit_total) if pos_benefit_total > 0 else 1.0

    return {
        "model_key": model_key,
        "clean_acc": clean_acc,
        "n_records": n,
        "helpful_cells": helpful_count,
        "harmful_cells": harmful_count,
        "tied_cells": tied_count,
        "mean_frozen_acc": float(np.mean(a0)),
        "mean_adapted_acc": float(np.mean(aa)),
        "mean_benefit": float(np.mean(B)),
        "always_freeze_regret": float(np.mean(a_oracle - a0)),
        "always_adapt_regret": float(np.mean(a_oracle - aa)),
        "point_gate_acc": float(np.mean(point_deployed)),
        "point_gate_regret": float(np.mean(point_regret)),
        "point_gate_fa_u": point_fa_u,
        "kga_acc": float(np.mean(a_deployed)),
        "kga_regret": float(np.mean(oracle_regret)),
        "kga_fa_u": fa_u,
        "kga_ff_u": ff_u,
        "kga_fa_c": fa_c,
        "kga_ff_c": ff_c,
        "decision_counts": {
            "ADAPT": adapt_count,
            "FREEZE": freeze_count,
            "ABSTAIN": abstain_count,
        },
        "helpful_cell_capture": helpful_captured,
        "positive_benefit_mass_capture": pos_benefit_captured,
    }


def main():
    if not SUMMARY_JSON.exists():
        print(f"Error: {SUMMARY_JSON} does not exist.")
        return

    data = json.loads(SUMMARY_JSON.read_text())
    models_meta = data["models"]

    from docs.research.kbound.scripts.run_independent_source_model_replication import (
        REPRESENTATIVE_CORRUPTIONS, DATA_DIR, evaluate_cell, make_cifar_resnet18, _norm
    )
    import torch

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cifar_c_dir = DATA_DIR / "CIFAR-10-C"
    labels_all = np.load(str(cifar_c_dir / "labels.npy")).astype(int)

    records_by_model: dict[str, list[dict[str, Any]]] = {k: [] for k in models_meta}

    for corr in REPRESENTATIVE_CORRUPTIONS:
        X_all = np.load(str(cifar_c_dir / f"{corr}.npy"))
        for sev in [3, 5]:
            idx_start = (sev - 1) * 10000
            idx_end = sev * 10000
            X_sev = X_all[idx_start:idx_end]
            y_sev = labels_all[idx_start:idx_end]

            for m_key, m_meta in models_meta.items():
                m_obj = make_cifar_resnet18().to(dev)
                m_obj.load_state_dict(torch.load(m_meta["checkpoint_path" if "checkpoint_path" in m_meta else "path"], map_location=dev))
                for str_seed in [0, 1, 2]:
                    res = evaluate_cell(m_obj, X_sev, y_sev, stream_seed=str_seed, dev=dev)
                    rec = {
                        "model_key": m_key,
                        "corruption": corr,
                        "severity": sev,
                        "stream_seed": str_seed,
                        **res,
                    }
                    records_by_model[m_key].append(rec)

    results = {}
    for m_key, recs in records_by_model.items():
        clean_acc = models_meta[m_key]["clean_test_acc"]
        results[m_key] = evaluate_model_kga(m_key, recs, clean_acc)

    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote KGA model evaluation summary to: {OUT_JSON}")

    # Generate LaTeX Table
    tex_lines = [
        r"% AUTO-GENERATED by kga_task1_independent_models.py. Do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Independent Source-Model Protocol B KGA Routing Evaluation (Task 1 Replication).} Evaluation across three independently trained ResNet-18 source checkpoints ($N=36$ cells per model; 6 corruptions $\times$ 2 severities $\times$ 3 stream seeds). Scored record count $n$, clean test accuracy $a_{\mathrm{clean}}$, Always Freeze regret $R_{\mathrm{AF}}$, Always Adapt regret $R_{\mathrm{AA}}$, Point-Gate regret $R_{\mathrm{point}}$, KGA routed accuracy $a_{\mathrm{KGA}}$, KGA oracle regret $R_{\mathrm{KGA}}$, unconditional false adapt rate $\mathrm{FA}_u$, and decision counts (Adapt / Freeze / Abstain) under Protocol B ($\alpha=0.10, \tau=0$).}",
        r"\label{tab:task1_independent_models}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lcccccccccc@{}}",
        r"\toprule",
        r"\textbf{Source Model} & \textbf{Clean Acc} & $n$ & \textbf{Frozen Acc} & \textbf{Adapted Acc} & $\mathbf{R_{\mathrm{AF}}}$ & $\mathbf{R_{\mathrm{AA}}}$ & $\mathbf{R_{\mathrm{point}}}$ & $\mathbf{a_{\mathrm{KGA}}}$ & $\mathbf{R_{\mathrm{KGA}}}$ & \textbf{Decisions (A/F/Abs)} \\",
        r"\midrule",
    ]

    for m_key, r in results.items():
        m_name = "Model 0 (Canonical)" if m_key == "model_0_canonical" else f"Model {models_meta[m_key]['seed']} (Independent)"
        dec_str = f"{r['decision_counts']['ADAPT']}/{r['decision_counts']['FREEZE']}/{r['decision_counts']['ABSTAIN']}"
        tex_lines.append(
            f"{m_name} & {r['clean_acc']*100:.2f}\\% & {r['n_records']} & "
            f"{r['mean_frozen_acc']*100:.2f}\\% & {r['mean_adapted_acc']*100:.2f}\\% & "
            f"{r['always_freeze_regret']*100:.2f}\\% & {r['always_adapt_regret']*100:.2f}\\% & "
            f"{r['point_gate_regret']*100:.2f}\\% & {r['kga_acc']*100:.2f}\\% & "
            f"{r['kga_regret']*100:.2f}\\% & {dec_str} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ])

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(tex_lines))
    print(f"Wrote LaTeX table to: {OUT_TEX}")


if __name__ == "__main__":
    main()
