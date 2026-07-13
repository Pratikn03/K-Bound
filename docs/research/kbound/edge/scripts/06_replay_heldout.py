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
import numpy as np

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    ap.add_argument("--calib-config", default="edge_calibration_v1.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    cal = C.load_config(args.calib_config)

    from kbound_edge.tent_adapter import EpisodicTentAdapter
    from kbound_edge.benefit_estimator import EdgeBenefitEstimator
    from kbound_edge.dataset import build_conditions
    from kbound_edge.logging import WindowLogger, config_hash
    from kbound_edge import replay as RP
    from kbound_edge import metrics as M

    is_real = cfg.get("protocol", "edge_label_inspection_v1") == "edge_real_phone_v1"

    f0, version = C.load_f0(cfg)
    adapter = EpisodicTentAdapter(f0, lr=cfg["adapter"]["lr"], steps=cfg["adapter"]["steps"],
                                  device=cfg.get("device", "cpu"))
    est = EdgeBenefitEstimator.load(C.resolve(cfg["paths"]["kga_edge"]))

    kga_meta_path = cfg["paths"].get("kga_edge_meta", "artifacts_real/calibration/kga_edge_meta.json" if is_real else "artifacts_synth/kga_edge_meta.json")
    meta = C.load_json(C.resolve(kga_meta_path))
    eps = float(meta["eps"])

    if is_real:
        conf_tau = meta["policies"]["conf_tau"]
        entropy_tau = meta["policies"]["entropy_tau"]
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
        from kbound_edge.real_dataset import load_window
        import os

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
        "model_version": version, "config_hash": chash, "eps": eps, "alpha": cfg["alpha"],
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
