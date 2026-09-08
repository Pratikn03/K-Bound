#!/usr/bin/env python3
"""06 -- replay the held-out synthetic stream through the full Tier-0/1 chain.

Runs the ONLINE decision chain on each held-out window (tensors only -- no
labels), logging one JSONL record per window, then OFFLINE measures the true
benefit B per window and reports the metric suite + the 6-policy ablation
(always-freeze, always-adapt, confidence-gate, entropy-gate, KGA-no-radius,
KGA-full).
"""

import argparse
from collections import Counter

import _common as C
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    ap.add_argument("--calib-config", default="edge_calibration_v1.yaml")
    C.add_deployment_arguments(ap)
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    cal = C.load_config(args.calib_config)

    from kbound_edge import metrics as M
    from kbound_edge import replay as RP
    from kbound_edge.dataset import build_conditions
    from kbound_edge.logging import WindowLogger, config_hash
    from kbound_edge.tent_adapter import EpisodicTentAdapter

    is_real = cfg.get("protocol", "edge_label_inspection_v1") == "edge_real_phone_v1"

    f0, version = C.load_trusted_f0(cfg, args.expected_frozen_sha256)
    adapter = EpisodicTentAdapter(f0, lr=cfg["adapter"]["lr"], steps=cfg["adapter"]["steps"],
                                  device=cfg.get("device", "cpu"))
    est = C.load_deployment_gate(cfg, args, version)
    eps = est.eps
    if est.estimator is not None:
        conf_tau = est.estimator.decision_metadata.conf_tau
        entropy_tau = est.estimator.decision_metadata.entropy_tau
    else:
        conf_tau = cal["policies"]["conf_tau"]
        entropy_tau = cal["policies"]["entropy_tau"]

    if not is_real:
        conds = build_conditions(
            C.plan_tuples(cfg["heldout_plan"]),
            n_frames=cfg["window_size"], image_size=cfg["image_size"],
            seed=9000, n_classes=cfg["num_classes"], prefix="held",
        )
        windows = [c.tensor() for c in conds]   # ONLINE payload: tensors, no labels
        true_labels = [c.labels for c in conds]
    else:
        import os

        from kbound_edge.real_dataset import load_window

        windows_dir = C.resolve(cfg["paths"]["windows_dir"])
        split_dir = os.path.join(windows_dir, "heldout")
        files = sorted([f for f in os.listdir(split_dir) if not f.startswith(".") and f.endswith(".npz") and (f.startswith("S07_") or f.startswith("S08_"))])

        payloads = []
        offlines = []
        for fname in files:
            p_load, off_load = load_window(os.path.join(split_dir, fname))
            payloads.append(p_load)
            offlines.append(off_load)

        windows = payloads
        true_labels = [o["labels"] for o in offlines]

    log_path = C.resolve(cfg["paths"]["heldout_log"])
    chash = config_hash(C.clean_config(cfg))
    with WindowLogger(log_path, model_version=version, config_hash=chash) as logger:
        res = RP.replay_windows(
            windows, f0, adapter, est, eps, logger=logger,
            image_size=cfg["image_size"], collect_policies=True,
            conf_tau=conf_tau, entropy_tau=entropy_tau,
        )

    # OFFLINE: true benefit per window (uses labels held outside the online path)
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

    if is_real:
        from kbound_edge.real_manifest import expected_windows
        win_meta_map = {}
        for s_id in ["S07", "S08"]:
            for w in expected_windows(cfg, s_id):
                w_copy = dict(w)
                w_copy["session_id"] = s_id
                win_meta_map[w["window_id"]] = w_copy

        window_metadata = [win_meta_map[w["window_id"]] for w in payloads]

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
        metrics_out_path = os.path.join(results_dir, "heldout_metrics.json")
    else:
        metrics_out_path = C.resolve(cfg["paths"]["heldout_metrics"])

    C.save_json(metrics_out_path, metrics_payload)

    print(f"[06] replayed {len(decs)} windows -> {log_path}")
    print(f"[06] KGA decisions: {counts}")
    present = set(decs)
    print(f"[06] adapt/freeze/abstain all present: {present.issuperset({'adapt','freeze','abstain'})}")
    print(f"[06] KGA-full: regret={kga_metrics['mean_regret']:.4f} "
          f"false_adapt_uncond={kga_metrics['false_adapt_uncond']:.4f} "
          f"latency_mean={kga_metrics.get('latency_ms_mean',0):.1f}ms")
    print("\n[06] policy comparison:")
    print(M.format_comparison_table(comparison))


if __name__ == "__main__":
    main()
