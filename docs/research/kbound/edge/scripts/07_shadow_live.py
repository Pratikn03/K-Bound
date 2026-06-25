#!/usr/bin/env python3
"""07 -- Tier-2 LIVE shadow mode (frozen = official output, candidate in shadow).

Drives a frame source window-by-window.  For every window the FROZEN model's
prediction is the official output; the episodic Tent candidate + kga_decide
verdict run in shadow and are only logged.  With source.kind=fake this runs the
exact loop against a synthetic FakeVideoCapture -- no camera required.  With
source.kind=opencv it drives a real webcam / phone-as-camera / video file.
"""

import argparse

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    ap.add_argument("--shadow-config", default="edge_shadow_v1.yaml")
    ap.add_argument("--camera", type=int, default=None, help="override: use OpenCV camera index")
    # --- live-demo dashboard options -----------------------------------------
    ap.add_argument("--view", choices=["console", "window"], default="console",
                    help="console = headless status lines (default); "
                         "window = live OpenCV overlay window (a watchable demo)")
    ap.add_argument("--record", metavar="PATH", default=None,
                    help="save the annotated stream to this mp4 (forces rendering)")
    ap.add_argument("--sample-dir", metavar="DIR", default=None,
                    help="save a few annotated PNG frames here (forces rendering)")
    ap.add_argument("--fps", type=float, default=8.0, help="fps for --record / window")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="stop after N windows (handy for a bounded demo / dry run)")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    sh = C.load_config(args.shadow_config)

    from kbound_edge.tent_adapter import EpisodicTentAdapter
    from kbound_edge.benefit_estimator import EdgeBenefitEstimator
    from kbound_edge.logging import WindowLogger, config_hash
    from kbound_edge.shadow_runtime import ShadowController
    from kbound_edge.dashboard import build_dashboard

    f0, version = C.load_f0(cfg)
    adapter = EpisodicTentAdapter(f0, lr=cfg["adapter"]["lr"], steps=cfg["adapter"]["steps"],
                                  device=cfg.get("device", "cpu"))
    est = EdgeBenefitEstimator.load(C.resolve(cfg["paths"]["kga_edge"]))
    is_real = cfg.get("protocol", "edge_label_inspection_v1") == "edge_real_phone_v1"
    meta_path = cfg["paths"].get("kga_edge_meta", "artifacts_real/calibration/kga_edge_meta.json" if is_real else "artifacts_synth/kga_edge_meta.json")
    eps = float(C.load_json(C.resolve(meta_path))["eps"])

    src_cfg = dict(sh["source"])
    if args.camera is not None:
        src_cfg["kind"] = "opencv"
        src_cfg["camera_index"] = args.camera

    max_frames = None
    if src_cfg["kind"] == "fake":
        from kbound_edge.capture import FakeVideoCapture, ListFrameSource
        from kbound_edge.dataset import build_conditions

        conds = build_conditions(
            C.plan_tuples(src_cfg["regime_plan"]),
            n_frames=src_cfg["n_frames_per_condition"], image_size=cfg["image_size"],
            seed=src_cfg["seed"], n_classes=cfg["num_classes"], prefix="shadow",
        )
        frames = [fr for c in conds for fr in c.frames]
        source = FakeVideoCapture(source=ListFrameSource(frames, image_size=cfg["image_size"]))
        print(f"[07] FAKE source: {len(frames)} frames across {len(conds)} conditions (no camera)")
    elif src_cfg["kind"] == "opencv":
        from kbound_edge.capture import open_opencv_source

        target = src_cfg.get("video_path") or src_cfg.get("camera_index", 0)
        source = open_opencv_source(target)
        max_frames = src_cfg.get("max_frames")
        print(f"[07] OPENCV source: '{target}' (max_frames={max_frames})")
    else:
        raise SystemExit(f"[07] unknown source.kind={src_cfg['kind']!r}")

    log_path = C.resolve(cfg["paths"].get("shadow_log", "artifacts_real/logs/shadow_live.jsonl" if is_real else "artifacts_synth/shadow_live.jsonl"))
    logger = WindowLogger(log_path, model_version=version, config_hash=config_hash(C.clean_config(cfg)))
    class_names = cfg.get("class_names")
    dash = build_dashboard(
        view=args.view, record_path=args.record, sample_dir=args.sample_dir,
        class_names=class_names, every=sh.get("log_every", 1), fps=args.fps,
    )
    if args.view == "window":
        print("[07] live window open -- press 'q' in the window to stop early")
    ctrl = ShadowController(
        f0, adapter, est, eps,
        window_size=sh["window_size"], image_size=cfg["image_size"],
        logger=logger, dashboard=dash, max_windows=args.max_windows,
    )
    summary = ctrl.run(source, max_frames=max_frames)
    logger.close()

    print(f"\n[07] shadow summary: {summary}")
    print(f"[07] {dash.summary_line()}  -> {log_path}")
    if args.record:
        print(f"[07] recorded annotated stream -> {args.record}")
    if args.sample_dir:
        print(f"[07] saved annotated sample frames -> {args.sample_dir}")
    budget = sh.get("latency_budget_ms")
    if budget and summary["mean_latency_ms"] > budget:
        print(f"[07] WARNING: mean latency {summary['mean_latency_ms']:.1f}ms exceeds budget {budget}ms")


if __name__ == "__main__":
    main()
