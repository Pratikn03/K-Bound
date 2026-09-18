#!/usr/bin/env python3
"""Run authenticated AETTA MC-dropout baseline on the Task 1 mixed-sign panel.

Evaluates Model 0 (canonical), Model 101 (seed 101), and Model 102 (seed 102)
across the exact 36 conditions of the mixed-sign panel evaluated in Task 1.

For each condition and model:
- Measures frozen model accuracy a0
- Adapts model with Tent and measures adapted accuracy aa
- Computes AETTA MC-dropout accuracy estimate on adapted model (aetta_ad)
- Computes AETTA MC-dropout accuracy estimate on frozen model (aetta_fr)
- Applies official AETTA recovery/trend decision rule:
    ADAPT if aetta_ad >= aetta_fr and 100 * aetta_ad >= 20.0 else FREEZE
- Reports realized accuracy, oracle regret R_AETTA, false adapt rate FA_u,
  and decision breakdown (A/F/U), directly paired against KGA.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = ROOT / "docs/research/kbound/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from cifar_tent_mps_v2 import (
    AGGRESSIVENESS,
    BATCH_REGIMES,
    _aetta_accuracy_estimate,
    acc_on,
    build_stream_and_eval,
    cifar_c_severity,
    make_cifar_resnet,
    policy_metrics,
    tent_adapt,
)
from run_task1_mixed_sign_replication import PANEL_SPECS, load_cifar_cache

DATA_ROOT = ROOT / "experiments/kbound/cifar"
OUT_DIR = ROOT / "experiments/kbound/results/task3_aetta_mixed_panel"


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=== Running Authenticated AETTA on Task 1 Mixed-Sign Panel ===")
    print(f"Device: {dev}")
    print(f"Panel size: {len(PANEL_SPECS)} conditions")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "aetta_mixed_sign_results.json"

    ckpts = {
        "Model 0 (Canonical)": DATA_ROOT / "resnet18_cifar.pt",
        "Model 101 (Independent)": ROOT / "experiments/kbound/results/task1_independent_models_replication/resnet18_seed101.pt",
        "Model 102 (Independent)": ROOT / "experiments/kbound/results/task1_independent_models_replication/resnet18_seed102.pt",
    }

    models = {}
    model_metadata = {}
    for name, p in ckpts.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {p}")
        m = make_cifar_resnet(10).to(dev)
        m.load_state_dict(torch.load(str(p), map_location=dev))
        m.eval()
        models[name] = m
        model_metadata[name] = {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }

    # Load Task 1 KGA summary for exact comparison
    t1_file = ROOT / "experiments/kbound/results/task1_independent_models_replication/task1_mixed_sign_summary.json"
    t1_data = json.loads(t1_file.read_text()) if t1_file.is_file() else {}

    unique_corrs = sorted(list(set(spec[0] for spec in PANEL_SPECS)))
    cifar_cache, labels_all = load_cifar_cache(unique_corrs)

    records_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in models}

    t0_total = time.time()
    for idx, (corr, sev, brn, comp, agn, rep) in enumerate(PANEL_SPECS):
        cond_str = f"{corr}|s{sev}|{brn}|{comp}|{agn}|r{rep}"
        Xc = cifar_cache[corr]
        sX, sY = cifar_c_severity(Xc, labels_all, sev)
        bs = BATCH_REGIMES[brn]
        ag = AGGRESSIVENESS[agn]

        # Identical deterministic stream RNG per condition
        cond_seed = (int(hashlib.sha256(cond_str.encode()).hexdigest()[:8], 16)) % (2**31 - 1)
        rng = np.random.default_rng(cond_seed)
        stream, ex, ey = build_stream_and_eval(Xc, labels_all, sX, sY, comp, bs, 10, rng, dev)

        print(f"[{idx+1:02d}/{len(PANEL_SPECS):02d}] {cond_str:45s}", end="", flush=True)

        for name, m in models.items():
            a0 = acc_on(m, ex, ey, train_mode=False)
            adapted, un = tent_adapt(m, stream, steps=ag["steps"], lr=ag["lr"])
            aa = acc_on(adapted, ex, ey, train_mode=True)
            delta = aa - a0

            # AETTA MC-dropout estimates
            aetta_fr = _aetta_accuracy_estimate(m, ex, 10, 2048)
            aetta_ad = _aetta_accuracy_estimate(adapted, ex, 10, 2048)

            # Official AETTA decision rule:
            # (1) Non-degradation vs frozen: aetta_ad >= aetta_fr
            # (2) Absolute floor: 100 * aetta_ad >= 20.0
            no_degrade = aetta_ad >= aetta_fr
            above_floor = (100.0 * aetta_ad) >= 20.0
            aetta_dec = "ADAPT" if (no_degrade and above_floor) else "FREEZE"

            records_by_model[name].append({
                "condition": cond_str,
                "a0": float(a0),
                "aa": float(aa),
                "B": float(delta),
                "aetta_acc_est_frozen": float(aetta_fr),
                "aetta_acc_est_adapted": float(aetta_ad),
                "aetta_decision": aetta_dec,
            })
        print(" -> done", flush=True)

    print(f"\nAETTA evaluation finished in {time.time()-t0_total:.1f}s")

    # Compute policy metrics for AETTA
    summary = {}
    print("\n" + "=" * 115)
    print(f"{'Model':24s} | {'Policy':7s} | {'Regret':>7s} | {'Accuracy':>8s} | {'FA_u':>6s} | {'Decisions (A/F/U)':>18s} | {'Beats Both':>10s}")
    print("-" * 115)

    for name, records in records_by_model.items():
        B = np.array([r["B"] for r in records], dtype=float)
        a0 = np.array([r["a0"] for r in records], dtype=float)
        aa = np.array([r["aa"] for r in records], dtype=float)
        aetta_decs = np.array([r["aetta_decision"] for r in records], dtype=object)

        metrics = policy_metrics(aetta_decs, a0, aa, B=B)
        r_aetta = metrics["regret_vs_oracle"]["K_Bound"]
        r_aa = metrics["regret_vs_oracle"]["always_adapt"]
        r_af = metrics["regret_vs_oracle"]["always_freeze"]
        fa_u = metrics["false_adapt_unconditional"]
        acc_aetta = metrics["mean_acc"]["K_Bound"]
        dec_counts = metrics["decision_counts"]
        dec_str = f"{dec_counts['ADAPT']}/{dec_counts['FREEZE']}/{dec_counts['ABSTAIN']}"
        beats_str = "YES" if metrics["beats_both"] else ("YES (regret)" if metrics["beats_both_regret_only"] else "NO")

        print(f"{name:24s} | {'AETTA':7s} | {r_aetta*100:6.2f}% | {acc_aetta*100:7.2f}% | {fa_u:6.4f} | {dec_str:>18s} | {beats_str:>10s}")

        # If KGA metrics available, print KGA row for head-to-head comparison
        if name in t1_data:
            kga_reg = t1_data[name]["regret"]["kga"]
            kga_acc = t1_data[name]["mean_acc"]["K_Bound"]
            kga_fa = t1_data[name]["false_adapt_unconditional"]
            kga_cnts = t1_data[name]["decision_counts"]
            kga_dec_str = f"{kga_cnts['ADAPT']}/{kga_cnts['FREEZE']}/{kga_cnts['ABSTAIN']}"
            kga_beats = "YES" if t1_data[name]["beats_both"] else "NO"
            print(f"{'':24s} | {'KGA':7s} | {kga_reg*100:6.2f}% | {kga_acc*100:7.2f}% | {kga_fa:6.4f} | {kga_dec_str:>18s} | {kga_beats:>10s}")
            print("-" * 115)

        summary[name] = {
            "metadata": model_metadata[name],
            "aetta": {
                "mean_acc": acc_aetta,
                "regret": r_aetta,
                "always_adapt_regret": r_aa,
                "always_freeze_regret": r_af,
                "false_adapt_unconditional": fa_u,
                "decision_counts": dec_counts,
                "beats_both": metrics["beats_both"],
            },
            "records": records,
        }

    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote authenticated AETTA comparison to: {out_json}")


if __name__ == "__main__":
    main()
