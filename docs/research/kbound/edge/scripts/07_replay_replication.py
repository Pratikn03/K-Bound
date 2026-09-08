#!/usr/bin/env python3
"""07 -- replay the replication stream (Phone B) through the full Tier-0/1 chain.

Runs the ONLINE decision chain on each replication window (S09+S10), logging one
JSONL record per window, then OFFLINE measures the true benefit B per window and
reports the metric suite.
"""

import argparse
import os
import sys
from collections import Counter

import _common as C
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_real_phone_v1.yaml")
    ap.add_argument("--calib-config", default="edge_calibration_v1.yaml")
    C.add_deployment_arguments(ap)
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    cal = C.load_config(args.calib_config)

    from kbound_edge import metrics as M
    from kbound_edge import replay as RP
    from kbound_edge.logging import WindowLogger, config_hash
    from kbound_edge.tent_adapter import EpisodicTentAdapter

    is_real = cfg.get("protocol", "edge_label_inspection_v1") == "edge_real_phone_v1"
    if not is_real:
        raise SystemExit("[07] Replication replay is only supported/valid in real protocol mode.")

    f0, version = C.load_trusted_f0(cfg, args.expected_frozen_sha256)
    adapter = EpisodicTentAdapter(f0, lr=cfg["adapter"]["lr"], steps=cfg["adapter"]["steps"],
                                  device=cfg.get("device", "cpu"))
    est = C.load_deployment_gate(cfg, args, version)
    eps = est.eps
    metadata = est.estimator.decision_metadata if est.estimator is not None else None
    conf_tau = metadata.conf_tau if metadata is not None else cal["policies"]["conf_tau"]
    entropy_tau = metadata.entropy_tau if metadata is not None else cal["policies"]["entropy_tau"]

    # Load real replication NPZs (S09 and S10)
    from kbound_edge.real_dataset import load_window

    windows_dir = C.resolve(cfg["paths"]["windows_dir"])
    split_dir = os.path.join(windows_dir, "replication")
    files = sorted([f for f in os.listdir(split_dir) if not f.startswith(".") and f.endswith(".npz") and (f.startswith("S09_") or f.startswith("S10_"))])

    payloads = []
    offlines = []
    for fname in files:
        p_load, off_load = load_window(os.path.join(split_dir, fname))
        payloads.append(p_load)
        offlines.append(off_load)

    windows = payloads
    true_labels = [o["labels"] for o in offlines]

    log_path = C.resolve(cfg["paths"]["replication_log"])
    chash = config_hash(C.clean_config(cfg))
    with WindowLogger(log_path, model_version=version, config_hash=chash) as logger:
        res = RP.replay_windows(
            windows, f0, adapter, est, eps, logger=logger,
            image_size=cfg["image_size"], collect_policies=True,
            conf_tau=conf_tau, entropy_tau=entropy_tau,
        )

    # OFFLINE: true benefit per window
    trueB = []
    for labels, o in zip(true_labels, res["outcomes"]):
        froz = float((o.p0.argmax(1) == labels).mean())
        cand = float((o.pa.argmax(1) == labels).mean())
        trueB.append(cand - froz)
    trueB = np.asarray(trueB)

    decs = res["decisions"]
    counts = dict(Counter(decs))
    kga_metrics = M.evaluate(decs, trueB, res["latencies_ms"])
    comparison = M.policy_comparison(res["policy_decisions"], trueB, res["latencies_ms"])

    metrics_payload = {
        "schema_version": "kbound-edge-v2",
        "unavailable_windows": res["unavailable_windows"], "gate_records": res["gate_records"],
        "comparison_scope": "fixed policies are non-certified baselines",
        "model_version": version, "config_hash": chash, "eps": est.eps, "alpha": cfg["alpha"],
        "n_windows": len(decs), "decision_counts": counts,
        "kga_full_metrics": kga_metrics, "policy_comparison": comparison,
        "log_path": log_path,
    }

    # Load metadata dict mapping window_id to metadata dict
    from kbound_edge.real_manifest import expected_windows
    win_meta_map = {}
    for s_id in ["S09", "S10"]:
        for w in expected_windows(cfg, s_id):
            w_copy = dict(w)
            w_copy["session_id"] = s_id
            win_meta_map[w["window_id"]] = w_copy

    window_metadata = [win_meta_map[w["window_id"]] for w in payloads]

    # Calculate bootstrap metrics
    bootstrap_results = M.bootstrap_real_metrics(
        outcomes=res["outcomes"],
        true_labels=true_labels,
        window_metadata=window_metadata,
        policy_decisions=res["policy_decisions"],
        latencies_ms=res["latencies_ms"],
        seed=cfg["seed"],
    )
    metrics_payload["bootstrap_results"] = bootstrap_results

    results_dir = os.path.normpath(os.path.join(C.EDGE_ROOT, cfg["paths"]["results_dir"]))
    metrics_out_path = os.path.join(results_dir, "replication_metrics.json")

    C.save_json(metrics_out_path, metrics_payload)

    print(f"[07] replayed {len(decs)} windows -> {log_path}")
    print(f"[07] KGA decisions: {counts}")
    present = set(decs)
    print(f"[07] adapt/freeze/abstain all present: {present.issuperset({'adapt','freeze','abstain'})}")
    print(f"[07] KGA-full: regret={kga_metrics['mean_regret']:.4f} "
          f"false_adapt_uncond={kga_metrics['false_adapt_uncond']:.4f} "
          f"latency_mean={kga_metrics.get('latency_ms_mean',0):.1f}ms")
    print("\n[07] policy comparison:")
    print(M.format_comparison_table(comparison))


if __name__ == "__main__":
    main()
