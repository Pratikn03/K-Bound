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


class LoopingFrameSource:
    """Wraps a FrameSource and loops it infinitely when it reaches the end."""

    def __init__(self, source, is_fake: bool = False) -> None:
        self.source = source
        self.is_fake = is_fake

    def read(self):
        ok, frame = self.source.read()
        if not ok or frame is None:
            if self.is_fake:
                self.source._src.reset()
            else:
                import cv2
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.source.read()
        return ok, frame

    def release(self) -> None:
        self.source.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    ap.add_argument("--shadow-config", default="edge_shadow_v1.yaml")
    ap.add_argument("--camera", default=None,
                    help="OpenCV camera index (int), or 'auto' to pick the live feed "
                         "(iPhone Continuity Camera is often index 1)")
    ap.add_argument("--video", default=None, help="override: use OpenCV video path (e.g. pilot video file)")
    ap.add_argument("--loop", action="store_true", help="loop the video / simulated stream infinitely")
    ap.add_argument("--eps", type=float, default=None, help="override: K-Bound conformal safety radius")
    ap.add_argument("--demo", action="store_true",
                    help="use the synthetic KGA calibrator (varied ADAPT/FREEZE/ABSTAIN) until "
                         "real S03–S06 calibration is captured")
    ap.add_argument("--kga-edge", default=None, metavar="PATH",
                    help="override path to kga_edge.joblib benefit estimator")
    ap.add_argument("--kga-meta", default=None, metavar="PATH",
                    help="override path to kga_edge_meta.json (eps + policies)")
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
    is_real = cfg.get("protocol", "edge_label_inspection_v1") == "edge_real_phone_v1"
    default_meta = (
        "artifacts_real/calibration/kga_edge_meta.json" if is_real
        else "artifacts_synth/kga_edge_meta.json"
    )
    meta_path = args.kga_meta or cfg["paths"].get("kga_edge_meta", default_meta)
    kga_path = args.kga_edge or cfg["paths"]["kga_edge"]
    meta = C.load_json(C.resolve(meta_path))
    use_demo_calibrator = bool(args.demo)
    if not use_demo_calibrator and is_real and C.is_placeholder_kga_meta(meta):
        if args.camera is not None or args.video is not None:
            use_demo_calibrator = True
            print("[07] NOTE: real-phone calibrator is still a placeholder (eps=0, n_fit<=20).")
            print("[07]       Auto-switching to synthetic KGA calibrator for a watchable live demo.")
            print("[07]       After S03–S06 capture + pipeline 03–05, decisions will use real calibration.")
    if use_demo_calibrator and not args.demo:
        meta_path = "artifacts_synth/kga_edge_meta.json"
        kga_path = "artifacts_synth/kga_edge.joblib"
        meta = C.load_json(C.resolve(meta_path))
    elif args.demo:
        meta_path = args.kga_meta or "artifacts_synth/kga_edge_meta.json"
        kga_path = args.kga_edge or "artifacts_synth/kga_edge.joblib"
        meta = C.load_json(C.resolve(meta_path))
        print("[07] DEMO mode: synthetic KGA calibrator (not certified for physical-phone protocol)")
    est = EdgeBenefitEstimator.load(C.resolve(kga_path))
    eps = float(meta["eps"])
    if args.eps is not None:
        eps = args.eps
        print(f"[07] safety radius override: eps={eps:.4f}")
    elif use_demo_calibrator and (args.camera is not None or args.video is not None):
        demo_eps = 0.12
        if eps > demo_eps:
            print(f"[07] demo: capping eps {eps:.4f} -> {demo_eps:.4f} so live webcam verdicts are visible")
            eps = demo_eps
    else:
        print(f"[07] calibrator: {C.resolve(kga_path)}  eps={eps:.4f}  (meta: {C.resolve(meta_path)})")

    src_cfg = dict(sh["source"])
    if args.camera is not None:
        src_cfg["kind"] = "opencv"
        src_cfg["video_path"] = None
        cam = str(args.camera).strip().lower()
        if cam == "auto":
            from kbound_edge.capture import pick_live_camera_index

            idx = pick_live_camera_index()
            src_cfg["camera_index"] = idx
            print(f"[07] auto-selected camera index {idx} (highest motion — use index 1 for iPhone if wrong)")
        else:
            src_cfg["camera_index"] = int(cam)
    if args.video is not None:
        src_cfg["kind"] = "opencv"
        src_cfg["video_path"] = args.video

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
        from kbound_edge.capture import CameraOpenError, open_opencv_source

        target = src_cfg.get("video_path") or src_cfg.get("camera_index", 0)
        try:
            source = open_opencv_source(target)
        except CameraOpenError as exc:
            raise SystemExit(f"[07] {exc}") from exc
        max_frames = src_cfg.get("max_frames")
        print(f"[07] OPENCV source: '{target}' (max_frames={max_frames})")
    else:
        raise SystemExit(f"[07] unknown source.kind={src_cfg['kind']!r}")

    log_path = C.resolve(cfg["paths"].get("shadow_log", "artifacts_real/logs/shadow_live.jsonl" if is_real else "artifacts_synth/shadow_live.jsonl"))
    logger = WindowLogger(log_path, model_version=version, config_hash=config_hash(C.clean_config(cfg)))
    class_names = cfg.get("class_names") or cfg.get("classes")
    dash = build_dashboard(
        view=args.view, record_path=args.record, sample_dir=args.sample_dir,
        class_names=class_names, every=sh.get("log_every", 1), fps=args.fps,
    )
    if args.view == "window":
        print("[07] live window open -- press 'q' in the window to stop early")
    if args.loop:
        source = LoopingFrameSource(source, is_fake=(src_cfg["kind"] == "fake"))
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
