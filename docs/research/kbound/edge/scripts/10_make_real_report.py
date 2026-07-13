#!/usr/bin/env python3
"""10 -- assemble a markdown REPORT.md for physical camera validation."""

import argparse
import os
import sys

import _common as C

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from kbound_edge.logging import read_jsonl
from kbound_edge import metrics as M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_real_phone_v1.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)

    results_dir = os.path.normpath(os.path.join(C.EDGE_ROOT, cfg["paths"]["results_dir"]))
    heldout_metrics = C.load_json(os.path.join(results_dir, "heldout_metrics.json"))
    replication_metrics = C.load_json(os.path.join(results_dir, "replication_metrics.json"))
    ablation_results = C.load_json(os.path.join(results_dir, "ablation_results.json"))
    runtime_profile = C.load_json(os.path.join(results_dir, "runtime_profile.json"))
    anti_leakage_audit = C.load_json(os.path.join(results_dir, "anti_leakage_audit.json"))

    L = []
    L.append("# K-Bound Edge -- Physical Camera Validation Report")
    L.append("")
    L.append(f"- **Protocol:** `{cfg['protocol']}`")
    L.append(f"- **Model Version:** `{heldout_metrics['model_version']}`")
    L.append(f"- **Config Hash:** `{heldout_metrics['config_hash']}`")
    L.append(f"- **Epsilon:** `{heldout_metrics['eps']:.4f}`")
    L.append(f"- **Alpha:** `{heldout_metrics['alpha']:.2f}`")
    L.append("")

    L.append("## 1. Anti-Leakage Audit Results")
    L.append("")
    L.append("| Check | Status | Observed |")
    L.append("|---|---|---|")
    for c in anti_leakage_audit["checks"]:
        status = "**PASS**" if c["passed"] else "**FAIL**"
        L.append(f"| {c['check']} | {status} | `{c['observed']}` |")
    L.append("")

    L.append("## 2. Held-Out Replay Results (Phone A)")
    L.append("")
    L.append("```")
    L.append(M.format_comparison_table(heldout_metrics["policy_comparison"]))
    L.append("```")
    L.append("")

    L.append("## 3. External-Device Replication Results (Phone B)")
    L.append("")
    L.append("```")
    L.append(M.format_comparison_table(replication_metrics["policy_comparison"]))
    L.append("```")
    L.append("")

    L.append("## 4. Resource and Live-Runtime Profile")
    L.append("")
    L.append("| Component | Mean (ms) | p95 (ms) |")
    L.append("|---|---|---|")
    for stage in ["frozen_inference", "tent_update", "candidate_inference", "evidence", "gate", "end_to_end", "capture_preprocess"]:
        stats = runtime_profile[stage]
        L.append(f"| {stage} | {stats['mean_ms']:.2f} | {stats['p95_ms']:.2f} |")
    L.append("")

    L.append("## 5. Conformal Gate Ablation Results")
    L.append("")
    L.append("| Variant | Regret | FA_u | Adapt Rate | Abstain Rate | Epsilon |")
    L.append("|---|---|---|---|---|---|")
    for name, stats in ablation_results.items():
        L.append(f"| {name} | {stats['regret']:.4f} | {stats['false_adapt_uncond']:.4f} | {stats['adapt_rate']:.3f} | {stats['abstain_rate']:.3f} | {stats['eps']:.4f} |")
    L.append("")

    out_path = os.path.join(results_dir, "REPORT.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    print(f"[10] Real validation report written -> {out_path}")


if __name__ == "__main__":
    main()
