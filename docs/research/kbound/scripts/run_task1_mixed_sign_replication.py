#!/usr/bin/env python3
"""Execute Task 1 replication on a mixed-sign panel across 3 independent checkpoints.

Evaluates Model 0 (canonical), Model 101 (independent seed 101), and Model 102 (independent seed 102)
on a 36-condition mixed-sign panel containing:
- 12 harmful adaptation conditions (~33%)
- 6 near-zero conditions (~17%)
- 18 helpful adaptation conditions (~50%)

For each model, records a0, aa, B, and the 11-dimensional evidence vector Z,
then runs Protocol-B exact-rank leave-one-out KGA routing and reports:
R_KGA, R_adapt, R_freeze, FA_u, and A/F/U decisions.
"""
from __future__ import annotations

import argparse
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

import kbound_decide as kb
from cifar_tent_mps_v2 import (
    AGGRESSIVENESS,
    BATCH_REGIMES,
    acc_on,
    build_stream_and_eval,
    cifar_c_severity,
    evidence_vector,
    make_cifar_resnet,
    policy_metrics,
    tent_adapt,
)

DATA_ROOT = ROOT / "experiments/kbound/cifar"
OUT_DIR = ROOT / "experiments/kbound/results/task1_independent_models_replication"

PANEL_SPECS = [
    # 12 Harmful conditions (s1 + aggressive or single_class/imbalanced)
    ("defocus_blur", 1, "large_iid", "imbalanced", "aggressive", 0),
    ("defocus_blur", 1, "large_iid", "single_class", "aggressive", 0),
    ("defocus_blur", 1, "small", "single_class", "aggressive", 0),
    ("defocus_blur", 1, "tiny", "single_class", "aggressive", 0),
    ("fog", 1, "large_iid", "single_class", "aggressive", 0),
    ("fog", 1, "small", "single_class", "aggressive", 0),
    ("contrast", 1, "large_iid", "imbalanced", "aggressive", 0),
    ("contrast", 1, "small", "single_class", "aggressive", 0),
    ("jpeg_compression", 1, "large_iid", "single_class", "aggressive", 0),
    ("jpeg_compression", 1, "small", "single_class", "aggressive", 0),
    ("pixelate", 1, "small", "single_class", "aggressive", 0),
    ("pixelate", 1, "tiny", "single_class", "aggressive", 0),
    # 6 Near-zero conditions (mild / iid or subtle shift)
    ("defocus_blur", 1, "large_iid", "imbalanced", "mild", 0),
    ("defocus_blur", 1, "small", "iid", "mild", 0),
    ("contrast", 1, "large_iid", "iid", "mild", 0),
    ("fog", 1, "large_iid", "iid", "mild", 0),
    ("jpeg_compression", 1, "large_iid", "iid", "mild", 0),
    ("gaussian_noise", 1, "tiny", "imbalanced", "aggressive", 0),
    # 18 Helpful conditions (s5 severe or s3 moderate shifts)
    ("gaussian_noise", 5, "large_iid", "iid", "mild", 0),
    ("gaussian_noise", 5, "large_iid", "iid", "aggressive", 0),
    ("gaussian_noise", 5, "small", "iid", "mild", 0),
    ("gaussian_noise", 5, "tiny", "iid", "mild", 0),
    ("pixelate", 5, "large_iid", "iid", "mild", 0),
    ("pixelate", 5, "large_iid", "iid", "aggressive", 0),
    ("pixelate", 5, "small", "iid", "mild", 0),
    ("pixelate", 5, "tiny", "iid", "mild", 0),
    ("jpeg_compression", 5, "large_iid", "iid", "mild", 0),
    ("jpeg_compression", 5, "small", "iid", "mild", 0),
    ("defocus_blur", 5, "large_iid", "iid", "mild", 0),
    ("defocus_blur", 5, "small", "iid", "mild", 0),
    ("fog", 5, "large_iid", "iid", "mild", 0),
    ("fog", 5, "small", "iid", "mild", 0),
    ("contrast", 5, "large_iid", "iid", "mild", 0),
    ("contrast", 5, "small", "iid", "mild", 0),
    ("gaussian_noise", 1, "large_iid", "iid", "mild", 0),
    ("pixelate", 1, "large_iid", "iid", "mild", 0),
]


def load_cifar_cache(corruptions: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    cifar_c_dir = DATA_ROOT / "CIFAR-10-C"
    labels = np.load(str(cifar_c_dir / "labels.npy")).astype(int)
    cache = {}
    print(f"Preloading {len(corruptions)} CIFAR-10-C corruptions into RAM...")
    t0 = time.time()
    for c in corruptions:
        cache[c] = np.load(str(cifar_c_dir / f"{c}.npy"))
    print(f"Loaded {len(corruptions)} corruptions in {time.time()-t0:.1f}s")
    return cache, labels


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== Running Task 1 Mixed-Sign Independent Model Replication ===")
    print(f"Device: {dev}")
    print(f"Panel size: {len(PANEL_SPECS)} conditions")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = OUT_DIR / "task1_mixed_sign_records_checkpoint.json"

    # Load model checkpoints
    ckpts = {
        "Model 0 (Canonical)": DATA_ROOT / "resnet18_cifar.pt",
        "Model 101 (Independent)": OUT_DIR / "resnet18_seed101.pt",
        "Model 102 (Independent)": OUT_DIR / "resnet18_seed102.pt",
    }

    models = {}
    model_metadata = {}
    for name, p in ckpts.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {name}: {p}")
        m = make_cifar_resnet(10).to(dev)
        m.load_state_dict(torch.load(str(p), map_location=dev))
        m.eval()
        models[name] = m
        model_metadata[name] = {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        print(f"Loaded {name}: {p} (SHA-256: {model_metadata[name]['sha256'][:16]}...)")

    # Resume from checkpoint if available
    records_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    completed_conditions = set()
    if checkpoint_file.is_file():
        try:
            saved_data = json.loads(checkpoint_file.read_text())
            if all(name in saved_data for name in models):
                records_by_model = saved_data
                completed_conditions = set(r["condition"] for r in records_by_model["Model 0 (Canonical)"])
                print(f"Resuming from checkpoint: {len(completed_conditions)}/{len(PANEL_SPECS)} conditions already completed")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    # Preload unique corruptions only if work remains
    if len(completed_conditions) < len(PANEL_SPECS):
        unique_corrs = sorted(list(set(spec[0] for spec in PANEL_SPECS)))
        cifar_cache, labels_all = load_cifar_cache(unique_corrs)

        t0_total = time.time()
        for idx, (corr, sev, brn, comp, agn, rep) in enumerate(PANEL_SPECS):
            cond_str = f"{corr}|s{sev}|{brn}|{comp}|{agn}|r{rep}"
            if cond_str in completed_conditions:
                continue

            Xc = cifar_cache[corr]
            sX, sY = cifar_c_severity(Xc, labels_all, sev)
            bs = BATCH_REGIMES[brn]
            ag = AGGRESSIVENESS[agn]

            # Deterministic stream RNG per condition
            cond_seed = (int(hashlib.sha256(cond_str.encode()).hexdigest()[:8], 16)) % (2**31 - 1)
            rng = np.random.default_rng(cond_seed)
            stream, ex, ey = build_stream_and_eval(Xc, labels_all, sX, sY, comp, bs, 10, rng, dev)

            print(f"[{idx+1:02d}/{len(PANEL_SPECS):02d}] {cond_str:45s}", end="", flush=True)

            deltas_str = []
            for name, m in models.items():
                a0 = acc_on(m, ex, ey, train_mode=False)
                adapted, un = tent_adapt(m, stream, steps=ag["steps"], lr=ag["lr"])
                aa = acc_on(adapted, ex, ey, train_mode=True)
                delta = aa - a0
                Z = evidence_vector(m, adapted, ex, 10, un)
                records_by_model[name].append({
                    "condition": cond_str,
                    "a0": float(a0),
                    "aa": float(aa),
                    "B": float(delta),
                    "update_norm": float(un),
                    "Z": [float(z) for z in Z],
                })
                deltas_str.append(f"{name.split()[0]} {name.split()[1]}:{delta:+.3f}")
            print(f" -> {' | '.join(deltas_str)}", flush=True)

            # Checkpoint incrementally
            checkpoint_file.write_text(json.dumps(records_by_model, indent=2))
        print(f"\nEvaluation phase completed.")

    # Run Protocol-B exact-rank leave-one-out KGA on each model
    results_summary = {}
    print("\n" + "=" * 105)
    print(f"{'Model':25s} | {'R_AF':>7s} | {'R_AA':>7s} | {'R_KGA':>7s} | {'FA_u':>6s} | {'Decisions (A/F/U)':>18s} | {'Harmful %':>9s} | {'Beats Both':>10s}")
    print("-" * 105)

    for name, records in records_by_model.items():
        B = np.array([r["B"] for r in records], dtype=float)
        a0 = np.array([r["a0"] for r in records], dtype=float)
        aa = np.array([r["aa"] for r in records], dtype=float)
        Z = np.array([r["Z"] for r in records], dtype=float)

        # Run leave-one-out KGA
        Bhat, eps, decisions = kb.decide_kga(Z, B, alpha=0.10, calibration="loo")
        metrics = policy_metrics(decisions, a0, aa, B=B)

        harmful_count = int(np.sum(B < 0))
        helpful_count = int(np.sum(B > 0))
        zero_count = int(np.sum(B == 0))

        dec_counts = metrics["decision_counts"]
        dec_str = f"{dec_counts['ADAPT']}/{dec_counts['FREEZE']}/{dec_counts['ABSTAIN']}"

        r_kga = metrics["regret_vs_oracle"]["K_Bound"]
        r_aa = metrics["regret_vs_oracle"]["always_adapt"]
        r_af = metrics["regret_vs_oracle"]["always_freeze"]
        fa_u = metrics["false_adapt_unconditional"]

        harmful_pct = f"{harmful_count}/{len(B)} ({harmful_count/len(B)*100:.1f}%)"
        beats_str = "YES" if metrics["beats_both"] else ("YES (regret)" if metrics["beats_both_regret_only"] else "NO")

        print(f"{name:25s} | {r_af*100:6.2f}% | {r_aa*100:6.2f}% | {r_kga*100:6.2f}% | {fa_u:6.4f} | {dec_str:>18s} | {harmful_pct:>9s} | {beats_str:>10s}")

        results_summary[name] = {
            "metadata": model_metadata[name],
            "n_conditions": len(records),
            "distribution": {
                "harmful": harmful_count,
                "helpful": helpful_count,
                "zero": zero_count,
                "harmful_fraction": float(harmful_count / len(records)),
            },
            "mean_acc": metrics["mean_acc"],
            "regret": {
                "always_freeze": r_af,
                "always_adapt": r_aa,
                "kga": r_kga,
            },
            "false_adapt_unconditional": fa_u,
            "decision_counts": dec_counts,
            "beats_both": metrics["beats_both"],
            "beats_both_regret_only": metrics["beats_both_regret_only"],
            "records": records,
        }

    print("=" * 105)

    # Save output summary
    out_json = OUT_DIR / "task1_mixed_sign_summary.json"
    out_json.write_text(json.dumps(results_summary, indent=2))
    print(f"\nWrote results summary to: {out_json}")


if __name__ == "__main__":
    main()
