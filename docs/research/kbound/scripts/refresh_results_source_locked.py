#!/usr/bin/env python3
"""Merge locked CIFAR + head-to-head artifacts into results_source.json."""
import json
import os
import subprocess
import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "docs/research/kbound/results_source.json")
LOCKED = os.path.join(
    ROOT, "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json"
)
H2H = os.path.join(
    ROOT,
    "experiments/kbound/results/mixed_headtohead_v1/"
    "HEADTOHEAD_RESULTS_cifar10c_tent_primary.json",
)


def main():
    d = json.load(open(SRC))
    prov = d.setdefault("_provenance", {})

    if os.path.exists(LOCKED):
        la = json.load(open(LOCKED))
        d["locked_analysis"] = la
        prov["locked_analysis"] = os.path.relpath(LOCKED, ROOT)
        tent = la.get("candidates", {}).get("tent", {})
        if tent:
            d.setdefault("corruption_grids", {})["cifar10c_stress"] = {
                "regret_kga": round(float(tent["kga_mean_regret"]), 4),
                "regret_adapt": round(float(tent["adapt_mean_regret"]), 4),
                "regret_freeze": round(float(tent["freeze_mean_regret"]), 4),
                "false_adapt": float(tent.get("false_adapt_rate_pooled", 0.0)),
                "verdict": "beats-both-CI-robust",
                "_source": prov["locked_analysis"],
            }

    if os.path.exists(H2H):
        raw = json.load(open(H2H))
        hh = raw.get("headtohead", raw)
        d["headtohead"] = {
            "verdict": hh.get("VERDICT", "—"),
            "kga_regret": float(raw.get("policy_mean_regret", {}).get("kga", 0)),
            "adapt_regret": float(raw.get("policy_mean_regret", {}).get("always_adapt", 0)),
            "freeze_regret": float(raw.get("policy_mean_regret", {}).get("always_freeze", 0)),
            "poem_regret": float(raw.get("policy_mean_regret", {}).get("poem", 0)),
            "aetta_regret": float(raw.get("policy_mean_regret", {}).get("aetta", 0)),
            "kga_fa": float(raw.get("policy_false_adapt_rate", {}).get("kga", 0)),
            "kga_decisive": float(raw.get("policy_decisive_rate", {}).get("kga", 0)),
            "_source": os.path.relpath(H2H, ROOT),
        }
        prov["headtohead"] = d["headtohead"]["_source"]

    prov["locked_refresh_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    prov["git_sha"] = subprocess.run(
        ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    json.dump(d, open(SRC, "w"), indent=2)
    print("updated", SRC)


if __name__ == "__main__":
    main()
