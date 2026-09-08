#!/usr/bin/env python3
"""12 -- run real-camera runtime profiling and generate runtime_profile.json."""

import argparse
import os
import sys

import _common as C

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

def main():
    from kbound_edge.profiling import profile_runtime
    from kbound_edge.real_dataset import load_window
    from kbound_edge.tent_adapter import EpisodicTentAdapter

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_real_phone_v1.yaml")
    C.add_deployment_arguments(ap)
    args = ap.parse_args()

    cfg = C.load_config(args.config)

    is_real = cfg.get("protocol", "edge_label_inspection_v1") == "edge_real_phone_v1"
    if not is_real:
        raise SystemExit("[12] Profiling script is only valid for physical real protocol mode.")

    f0, version = C.load_trusted_f0(cfg, args.expected_frozen_sha256)
    adapter = EpisodicTentAdapter(f0, lr=cfg["adapter"]["lr"], steps=cfg["adapter"]["steps"],
                                  device=cfg.get("device", "cpu"))
    est = C.load_deployment_gate(cfg, args, version)
    eps = est.eps

    # Load 15 windows from calibration_conformal for profiling
    windows_dir = C.resolve(cfg["paths"]["windows_dir"])
    split_dir = os.path.join(windows_dir, "calibration_conformal")
    files = sorted([f for f in os.listdir(split_dir) if not f.startswith(".") and f.endswith(".npz")])[:15]
    if not files:
        raise SystemExit(f"[12] No conformal windows found under: {split_dir}")

    payloads = []
    for fname in files:
        p_load, _ = load_window(os.path.join(split_dir, fname))
        payloads.append(p_load)

    print(f"[12] Running profiling on {len(payloads)} windows...")
    profile_summary = profile_runtime(
        f0=f0,
        adapter=adapter,
        estimator=est,
        eps=eps,
        windows=payloads,
        image_size=cfg["image_size"],
        device=cfg.get("device", "cpu"),
        warmup=5,  # discard first 5 windows as warmup
    )

    # Save to runtime_profile.json
    results_dir = os.path.normpath(os.path.join(C.EDGE_ROOT, cfg["paths"]["results_dir"]))
    out_path = os.path.join(results_dir, "runtime_profile.json")
    C.save_json(out_path, profile_summary)

    print(f"[12] Wrote profiling results to: {out_path}")
    print("\nProfiling Results:")
    print("-" * 80)
    print(f"{'Component':<35} | {'Mean (ms)':<10} | {'p95 (ms)':<10}")
    print("-" * 80)
    for stage in ["frozen_inference", "tent_update", "candidate_inference", "evidence", "gate", "end_to_end", "capture_preprocess"]:
        stats = profile_summary[stage]
        if stats["mean_ms"] is None:
            print(f"{stage:<35} | unavailable / retain frozen")
        else:
            print(f"{stage:<35} | {stats['mean_ms']:<10.2f} | {stats['p95_ms']:<10.2f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
