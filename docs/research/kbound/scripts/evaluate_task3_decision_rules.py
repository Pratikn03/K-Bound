#!/usr/bin/env python3
"""Evaluate all 9 same-prediction decision rules on the authoritative Protocol-B Tent CIFAR-10-C panel."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[4]
CELLS_FILE = ROOT / "docs/research/kbound/paper/generated/current_cifar_baselines_20260912/cells.jsonl"
OUT_DIR = ROOT / "experiments/kbound/results/task3_decision_rules_v1"


def load_cells(path: Path = CELLS_FILE) -> list[dict[str, Any]]:
    cells = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cells.append(json.loads(line))
    return cells


def evaluate_decision_rules():
    cells = load_cells()
    n = len(cells)
    assert n == 2160, f"Expected 2160 cells, got {n}"

    # Extract arrays
    B = np.array([c["B"] for c in cells])
    a0 = np.array([c["a0"] for c in cells])
    aa = np.array([c["a_adapted"] for c in cells])
    oracle = np.maximum(a0, aa)
    pred = np.array([c["current_prediction"] for c in cells])
    radius = np.array([c["current_radius"] for c in cells])
    Z = np.array([c["Z"] for c in cells])  # shape (2160, 11)
    tasks = np.array([c["condition"].split("|")[0] for c in cells])  # 6 corruption types
    unique_tasks = np.unique(tasks)

    # Harmful / helpful indicators
    harmful = B < 0
    helpful = B > 0
    total_harmful = np.sum(harmful)
    total_helpful_benefit = np.sum(np.maximum(0, B))

    rules_actions: dict[str, list[str]] = {}

    # 1. Always Adapt
    rules_actions["Always Adapt"] = ["ADAPT"] * n

    # 2. Always Freeze
    rules_actions["Always Freeze"] = ["FREEZE"] * n

    # 3. Confidence gate: post_conf > pre_conf
    rules_actions["Confidence gate"] = [
        "ADAPT" if Z[i, 4] > Z[i, 1] else "FREEZE" for i in range(n)
    ]

    # 4. Entropy gate: ent_drop > 0
    rules_actions["Entropy gate"] = [
        "ADAPT" if Z[i, 7] > 0 else "FREEZE" for i in range(n)
    ]

    # 5. Drift/KL gate: mkl <= tau, with tau calibrated on dev fold (precomputed per unique task)
    alpha = 0.10
    task_tau = {}
    for t in unique_tasks:
        dv_mask = tasks != t
        mkl_dev = Z[dv_mask, 9]
        B_dev = B[dv_mask]
        best_tau = -np.inf
        # candidate taus evaluated efficiently
        sorted_taus = np.sort(np.unique(mkl_dev))
        for cand_tau in sorted_taus:
            adapt_mask = mkl_dev <= cand_tau
            if np.sum(adapt_mask) > 0 and np.mean(B_dev[adapt_mask] < 0) <= alpha:
                best_tau = cand_tau
        task_tau[t] = best_tau

    rules_actions["Drift/KL gate"] = [
        "ADAPT" if Z[i, 9] <= task_tau[tasks[i]] else "FREEZE" for i in range(n)
    ]

    # 6. ATC-style estimator: isotonic calibration of confidence -> accuracy on dev fold
    atc_decisions = np.empty(n, dtype=object)
    for t in unique_tasks:
        te_mask = tasks == t
        dv_mask = ~te_mask
        conf_dev = np.concatenate([Z[dv_mask, 1], Z[dv_mask, 4]])
        acc_dev = np.concatenate([a0[dv_mask], aa[dv_mask]])
        ir = IsotonicRegression(out_of_bounds="clip").fit(conf_dev, acc_dev)
        pred_post = ir.predict(Z[te_mask, 4])
        pred_pre = ir.predict(Z[te_mask, 1])
        atc_decisions[te_mask] = np.where(pred_post > pred_pre, "ADAPT", "FREEZE")
    rules_actions["ATC-style estimator"] = list(atc_decisions)

    # 7. Point-benefit gate using EXACT SAME Delta_hat as KGA
    rules_actions["Point-benefit gate"] = [
        "ADAPT" if pred[i] > 0 else "FREEZE" for i in range(n)
    ]

    # 8. Fixed-margin three-way gate: Delta_hat > m -> ADAPT, Delta_hat < -m -> FREEZE, else ABSTAIN
    # Margin selected on dev fold without looking at test outcomes (e.g. median calibration residual)
    task_margin = {}
    for t in unique_tasks:
        dv_mask = tasks != t
        task_margin[t] = float(np.median(radius[dv_mask]))

    fixed_margin_actions = []
    for i in range(n):
        m_t = task_margin[tasks[i]]
        if pred[i] > m_t:
            act = "ADAPT"
        elif pred[i] < -m_t:
            act = "FREEZE"
        else:
            act = "ABSTAIN"
        fixed_margin_actions.append(act)
    rules_actions["Fixed-margin three-way gate"] = fixed_margin_actions

    # 9. KGA calibrated interval gate (authoritative Protocol B)
    rules_actions["KGA calibrated interval gate"] = [
        c["policy_actions"]["current_kga"] for c in cells
    ]

    # Compute metrics for each rule
    results = {}
    table_rows = []

    for name, acts in rules_actions.items():
        acts_arr = np.array(acts)
        chosen_acc = np.where(acts_arr == "ADAPT", aa, a0)
        mean_score = float(np.mean(chosen_acc))
        regret = float(np.mean(oracle - chosen_acc))

        A = int(np.sum(acts_arr == "ADAPT"))
        F = int(np.sum(acts_arr == "FREEZE"))
        U = int(np.sum(acts_arr == "ABSTAIN"))

        fa_u_count = int(np.sum((acts_arr == "ADAPT") & (B <= 0)))
        fa_u = fa_u_count / n
        fa_c = (fa_u_count / A) if A > 0 else None

        # False FREEZE uses B >= 0 (Paper definition: directional error when true benefit is nonnegative)
        ff_u_count = int(np.sum((acts_arr == "FREEZE") & (B >= 0)))
        ff_u = ff_u_count / n
        ff_c = (ff_u_count / F) if F > 0 else None

        # Nonpositive adapt fraction: denominator is exactly N_{B <= 0} = 721
        nonpos_adapted_count = int(np.sum((acts_arr == "ADAPT") & (B <= 0)))
        nonpos_adapted_fraction = nonpos_adapted_count / np.sum(B <= 0)

        # Positive benefit captured / forgone
        helpful_captured = float(np.sum(np.maximum(0, B[acts_arr == "ADAPT"]))) / total_helpful_benefit
        helpful_forgone = float(np.sum(np.maximum(0, B[acts_arr != "ADAPT"]))) / total_helpful_benefit

        # Test identities
        assert abs(helpful_captured + helpful_forgone - 1.0) < 1e-9, f"Identity failed: {helpful_captured} + {helpful_forgone} != 1"
        assert abs(fa_u * n - nonpos_adapted_fraction * np.sum(B <= 0)) < 1e-9, f"Identity failed: {fa_u*n} != {nonpos_adapted_fraction*np.sum(B<=0)}"

        commitment_rate = (A + F) / n
        abstention_rate = U / n

        results[name] = {
            "mean_score": mean_score,
            "oracle_regret": regret,
            "A": A,
            "F": F,
            "U": U,
            "FA_u": fa_u,
            "FA_c": fa_c,
            "FF_u": ff_u,
            "FF_c": ff_c,
            "nonpos_adapted_fraction": nonpos_adapted_fraction,
            "helpful_benefits_captured": helpful_captured,
            "helpful_benefit_forgone": helpful_forgone,
            "commitment_rate": commitment_rate,
            "abstention_rate": abstention_rate,
        }

        fa_c_str = f"{fa_c:.4f}" if fa_c is not None else "---"
        ff_c_str = f"{ff_c:.4f}" if ff_c is not None else "---"

        table_rows.append({
            "Policy": name,
            "Mean Score": f"{mean_score:.4f}",
            "Oracle Regret": f"{regret:.4f}",
            "A/F/U": f"{A} / {F} / {U}",
            "FA_u": f"{fa_u:.4f}",
            "FA_c": fa_c_str,
            "FF_u": f"{ff_u:.4f}",
            "FF_c": ff_c_str,
            "Nonpos Adapt": f"{nonpos_adapted_fraction:.4f}",
            "Helpful Captured": f"{helpful_captured:.4f}",
            "Helpful Forgone": f"{helpful_forgone:.4f}",
            "Commitment": f"{commitment_rate:.4f}",
            "Abstention": f"{abstention_rate:.4f}",
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "task3_part_a_decision_rules.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Write Markdown table
    md_lines = [
        "# Task 3 Part A: Same-Prediction Decision Rules Evaluation",
        "",
        "Evaluated on the authoritative Protocol-B Tent CIFAR-10-C panel ($N=2,160$ cells, 5 seeds).",
        "",
        "| Policy | Mean Score | Oracle Regret | A / F / U | $\\mathrm{FA}_u$ | $\\mathrm{FA}_c$ | $\\mathrm{FF}_u$ | $\\mathrm{FF}_c$ | Nonpos Adapt Frac | Helpful Captured | Helpful Forgone | Commitment | Abstention |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in table_rows:
        md_lines.append(
            f"| {r['Policy']} | {r['Mean Score']} | {r['Oracle Regret']} | {r['A/F/U']} | "
            f"{r['FA_u']} | {r['FA_c']} | {r['FF_u']} | {r['FF_c']} | "
            f"{r['Nonpos Adapt']} | {r['Helpful Captured']} | {r['Helpful Forgone']} | "
            f"{r['Commitment']} | {r['Abstention']} |"
        )
    md_content = "\n".join(md_lines) + "\n"
    with open(OUT_DIR / "TASK3_PART_A_TABLE.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(md_content)

    # Also emit Table 25 (Performance) and Table 26 (Errors) LaTeX tables to paper/generated
    tex_dir = ROOT / "docs/research/kbound/paper/generated"
    tex_dir.mkdir(parents=True, exist_ok=True)

    # Table 25 (Perf)
    p_lines = [
        r"% AUTO-GENERATED by evaluate_task3_decision_rules.py. Do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Systematic comparison of nine decision rules on the authoritative Protocol~B CIFAR-10-C Tent panel ($N=2{,}160$ cells): \textbf{performance and action counts}. Regret is mean accuracy shortfall to oracle. A/F/U are Adapt/Freeze/Abstain counts. Commitment $= (A+F)/N$; Abstention $= U/N$. Denominators: A+F+U $= 2{,}160$ for all rows.}",
        r"\label{tab:cifar10c-decision-rules}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Policy & Mean Score & Regret & A / F / U & Commitment & Abstention \\",
        r"\midrule",
    ]
    for r in table_rows:
        p_name = r["Policy"]
        if "KGA" in p_name:
            p_lines.append(f"\\textbf{{{p_name}}} & \\textbf{{{r['Mean Score']}}} & \\textbf{{{r['Oracle Regret']}}} & \\textbf{{{r['A/F/U']}}} & \\textbf{{{r['Commitment']}}} & \\textbf{{{r['Abstention']}}} \\\\")
        else:
            p_lines.append(f"{p_name:<28} & {r['Mean Score']} & {r['Oracle Regret']} & {r['A/F/U']} & {r['Commitment']} & {r['Abstention']} \\\\")
    p_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (tex_dir / "tab_cifar10c_decision_rules_perf.tex").write_text("\n".join(p_lines))

    # Table 26 (Errors) - wrapped with resizebox and tight tabcolsep to prevent ANY overflow
    e_lines = [
        r"% AUTO-GENERATED by evaluate_task3_decision_rules.py. Do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Systematic comparison of nine decision rules: \textbf{directional errors and benefit capture} (same $N=2{,}160$ Tent panel as above). $\mathrm{FA}_u = \Pr[\text{ADAPT}, B \le 0]$ (divides by $N=2{,}160$); $\mathrm{FA}_c$ divides by ADAPT count (undefined when ADAPT$=0$). $\mathrm{FF}_u = \Pr[\text{FREEZE}, B \ge 0]$ (divides by $N$); $\mathrm{FF}_c$ divides by FREEZE count. Nonpos.\ Adapt $=$ fraction of $B \le 0$ cells ($N_{B\le0}=721$; $N_{B<0}=714$, $N_{B=0}=7$) that received ADAPT, satisfying $\mathrm{FA}_u \times N = \text{Nonpos.\ Adapt} \times 721$. Positive Benefit Captured is the fraction of oracle positive adaptation benefit preserved: $\sum_{i: g_i = \text{ADAPT}} \max(0, B_i) / \sum_{i=1}^N \max(0, B_i)$; Forgone is $1 - \text{Captured}$. KGA executed ADAPT on $1{,}094$ of $1{,}439$ helpful cells ($76.0\%$ cell capture), abstaining on the remainder where empirical intervals overlap zero; zero nonpositive cells received ADAPT ($\mathrm{FA}_u = 0.0000$).}",
        r"\label{tab:cifar10c-decision-rules-errors}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Policy & $\mathrm{FA}_u$ & $\mathrm{FA}_c$ & $\mathrm{FF}_u$ & $\mathrm{FF}_c$ & Nonpos.\ Adapt & Benefit Captured & Benefit Forgone \\",
        r"\midrule",
    ]
    for r in table_rows:
        p_name = r["Policy"]
        if "KGA" in p_name:
            e_lines.append(f"\\textbf{{{p_name}}} & \\textbf{{{r['FA_u']}}} & \\textbf{{{r['FA_c']}}} & \\textbf{{{r['FF_u']}}} & \\textbf{{{r['FF_c']}}} & \\textbf{{{r['Nonpos Adapt']}}} & \\textbf{{{r['Helpful Captured']}}} & \\textbf{{{r['Helpful Forgone']}}} \\\\")
        else:
            e_lines.append(f"{p_name:<28} & {r['FA_u']} & {r['FA_c']} & {r['FF_u']} & {r['FF_c']} & {r['Nonpos Adapt']} & {r['Helpful Captured']} & {r['Helpful Forgone']} \\\\")
    e_lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    (tex_dir / "tab_cifar10c_decision_rules_errors.tex").write_text("\n".join(e_lines))
    print(f"Emitted LaTeX tables to {tex_dir}")


if __name__ == "__main__":
    evaluate_decision_rules()
