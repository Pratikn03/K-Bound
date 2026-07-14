#!/usr/bin/env python3
"""08 -- assemble a markdown REPORT.md from the logs + metrics produced by 06/07."""

import argparse
import os
from collections import Counter

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    from kbound_edge.logging import read_jsonl
    from kbound_edge import metrics as M

    metrics = C.load_json(C.resolve(cfg["paths"]["heldout_metrics"]))
    held = read_jsonl(C.resolve(cfg["paths"]["heldout_log"]))

    shadow_path = C.resolve(cfg["paths"]["shadow_log"])
    shadow = read_jsonl(shadow_path) if os.path.exists(shadow_path) else []
    shadow_counts = dict(Counter(r["decision"] for r in shadow)) if shadow else {}

    km = metrics["kga_full_metrics"]
    L = []
    L.append("# K-Bound Edge -- synthetic validation report")
    L.append("")
    L.append("> **SYNTHETIC DATA ONLY.** Every number below comes from generated frames "
             "with a fabricated 4-class structure. This report proves the *code runs end "
             "to end and the decision logic exercises all branches* -- it is **NOT an "
             "empirical result**. Real numbers require real recorded clips.")
    L.append("")
    L.append("## Protocol")
    L.append("")
    L.append(f"- protocol: `{cfg['protocol']}`")
    L.append(f"- model: MobileNetV3-Small + {cfg['num_classes']}-class head "
             f"(`model_version={metrics['model_version']}`)")
    L.append(f"- config_hash: `{metrics['config_hash']}`")
    L.append(f"- window size: {cfg['window_size']} frames; image size: {cfg['image_size']}")
    L.append(f"- certificate: split-conformal, alpha={metrics['alpha']}, eps={metrics['eps']:.4f}")
    L.append("")
    L.append("## Held-out replay (Tier-0/1)")
    L.append("")
    L.append(f"- windows: {metrics['n_windows']}")
    L.append(f"- KGA decisions: {metrics['decision_counts']}")
    allthree = set(metrics["decision_counts"]).issuperset({"adapt", "freeze", "abstain"})
    L.append(f"- adapt/freeze/abstain all present: **{allthree}**")
    L.append(f"- KGA-full mean regret: {km['mean_regret']:.4f}")
    L.append(f"- KGA-full false-adapt (uncond): {km['false_adapt_uncond']:.4f}; "
             f"(cond): {km['false_adapt_cond']:.4f}")
    L.append(f"- latency: mean {km.get('latency_ms_mean', 0):.1f} ms, "
             f"p95 {km.get('latency_ms_p95', 0):.1f} ms")
    L.append("")
    L.append("## Policy comparison (6 policies)")
    L.append("")
    L.append("```")
    L.append(M.format_comparison_table(metrics["policy_comparison"]))
    L.append("```")
    L.append("")
    L.append("Realised benefit semantics: adapt -> B, freeze/abstain -> 0; "
             "regret = max(B,0) - realised. Lower regret and lower false-adapt are better.")
    L.append("")
    L.append("## Tier-2 shadow")
    L.append("")
    if shadow:
        L.append(f"- shadow windows logged: {len(shadow)}")
        L.append(f"- shadow decision counts: {shadow_counts}")
        L.append(f"- source: `{shadow_path}`")
        L.append("- frozen model was the official output for every window; the candidate ran "
                 "in shadow (logged, never emitted).")
    else:
        L.append("- (no shadow log found -- run `07_shadow_live.py`).")
    L.append("")
    L.append("## Logs")
    L.append("")
    L.append(f"- held-out windows JSONL: `{cfg['paths']['heldout_log']}`")
    L.append(f"- shadow windows JSONL: `{cfg['paths']['shadow_log']}`")
    L.append("")
    L.append("Each JSONL record carries: schema_version, timestamp, window_id, model_version, "
             "config_hash, decision, bhat, eps, lower, upper, reason, latency_ms, evidence{14}.")
    L.append("")

    out = C.resolve(cfg["paths"]["report"])
    C.ensure_parent(out)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[08] report written -> {out}")
    print(f"[08] held-out decisions {metrics['decision_counts']}; all three present={allthree}")


if __name__ == "__main__":
    main()
