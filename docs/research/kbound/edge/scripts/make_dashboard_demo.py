#!/usr/bin/env python3
"""make_dashboard_demo -- render a sample of the live dashboard with NO model.

This proves the dashboard *UI* (overlays + colour-coded decisions + stats panel)
renders end to end, using:

* real synthetic camera frames (:class:`kbound_edge.capture.SyntheticFrameSource`),
* the real certified gate (:func:`kbound_edge.policy.kga_decide`) to produce
  genuine ADAPT / FREEZE / ABSTAIN verdicts from chosen ``(B^, eps)`` pairs, and
* the real renderer (:func:`kbound_edge.dashboard.annotate_frame` /
  :class:`~kbound_edge.dashboard.VisualDashboard`).

It needs only numpy + OpenCV (no torch, no camera), so it is the fastest way to
eyeball the demo and it is what ``tests/test_dashboard_render.py`` and CI use to
confirm the render path. The decisions here are *illustrative* (hand-picked to
exercise all three branches) -- NOT a model result. For a real run with the
frozen model in the loop, use ``scripts/07_shadow_live.py --record out.mp4``.

Outputs (default ``edge/artifacts/``):
    dashboard_demo.mp4              annotated stream
    dashboard_demo_montage.png      one still showing adapt / freeze / abstain
    frames/window_XXXX_<dec>.png    a few annotated stills
    dashboard_demo.gif              (if imageio or PIL is available)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

# Run from edge/scripts or edge/: make the package importable either way.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
import sys
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from kbound_edge.capture import SyntheticFrameSource          # noqa: E402
from kbound_edge.policy import kga_decide, Decision           # noqa: E402
from kbound_edge.dashboard import annotate_frame, VisualDashboard, DecisionStats  # noqa: E402


@dataclass
class DemoOutcome:
    """Minimal duck-typed stand-in for replay.WindowOutcome (no torch needed)."""

    window_id: int
    decision: Decision
    p0: np.ndarray
    pa: np.ndarray
    latency_ms: float
    evidence: dict = field(default_factory=dict)
    upd_norm: float = 0.0

    @property
    def frozen_pred(self) -> List[int]:
        return self.p0.argmax(1).tolist()

    @property
    def candidate_pred(self) -> List[int]:
        return self.pa.argmax(1).tolist()


def _softmax_rows(n: int, cls: int, conf: float, n_classes: int, rng) -> np.ndarray:
    """Build an (n, C) softmax whose mean argmax is ``cls`` at roughly ``conf``."""
    logits = rng.normal(0.0, 0.2, size=(n, n_classes))
    logits[:, cls] += float(np.log(max(conf, 1e-3) / (1 - min(conf, 0.999)) + 3.0)) + 2.0
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def build_demo_windows(n_windows: int, n_classes: int = 4, frames_per: int = 8,
                       eps: float = 0.08, seed: int = 7):
    """A scripted mix that exercises adapt / freeze / abstain with varied frames."""
    rng = np.random.default_rng(seed)
    # (regime label for the synthetic frame, B^, candidate behaviour)
    plan = [
        ("clean", +0.32, "better"),    # adapt: benefit certified positive
        ("clean", +0.27, "better"),    # adapt
        ("bright", +0.02, "tie"),      # abstain: interval spans 0
        ("noisy", -0.30, "worse"),     # freeze: harm certified
        ("noisy", +0.21, "better"),    # adapt
        ("bright", -0.04, "tie"),      # abstain
        ("dark", -0.26, "worse"),      # freeze
        ("clean", +0.18, "better"),    # adapt
    ]
    shifts = {
        "clean": dict(brightness=0.0, contrast=1.0, noise=0.05, blur=0),
        "bright": dict(brightness=0.25, contrast=1.1, noise=0.05, blur=0),
        "dark": dict(brightness=-0.22, contrast=0.9, noise=0.06, blur=0),
        "noisy": dict(brightness=0.0, contrast=1.0, noise=0.22, blur=3),
    }
    windows = []
    for w in range(n_windows):
        regime, bhat, behav = plan[w % len(plan)]
        src = SyntheticFrameSource(
            num_frames=frames_per, n_classes=n_classes, image_size=64,
            seed=int(rng.integers(0, 1_000_000)), **shifts[regime],
        )
        frames = src.frames
        true_cls = int(np.bincount(src.labels, minlength=n_classes).argmax())

        # frozen (official) softmax: confident on the dominant class
        p0 = _softmax_rows(frames_per, true_cls, 0.72, n_classes, rng)
        # candidate (shadow) softmax depends on intended behaviour
        if behav == "better":
            pa = _softmax_rows(frames_per, true_cls, 0.88, n_classes, rng)
        elif behav == "worse":
            wrong = (true_cls + 1) % n_classes
            pa = _softmax_rows(frames_per, wrong, 0.66, n_classes, rng)
        else:  # tie
            pa = _softmax_rows(frames_per, true_cls, 0.70, n_classes, rng)

        decision = kga_decide(bhat, eps)
        lat = float(rng.uniform(9.0, 24.0))
        windows.append((frames[-1], DemoOutcome(w, decision, p0, pa, lat)))
    return windows


def _make_gif(frames_bgr, path, fps):
    rgb = [f[:, :, ::-1] for f in frames_bgr]
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, rgb, duration=1.0 / max(fps, 1e-3))
        return True
    except Exception:
        pass
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in rgb]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(fps, 1e-3)), loop=0)
        return True
    except Exception:
        return False


def _montage(by_decision, path):
    import cv2
    order = [d for d in ("adapt", "freeze", "abstain") if d in by_decision]
    if not order:
        return False
    tiles = [by_decision[d] for d in order]
    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    cv2.imwrite(path, np.hstack(tiles))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(_HERE, "..", "artifacts")))
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    out = args.out_dir
    frames_dir = os.path.join(out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    mp4_path = os.path.join(out, "dashboard_demo.mp4")

    windows = build_demo_windows(args.n_windows)

    dash = VisualDashboard(
        show=False, record_path=mp4_path, sample_dir=frames_dir,
        max_samples=args.n_windows, fps=args.fps,
        class_names=["good", "smudge", "wrong-stock", "misaligned"],
    )
    annotated_all = []
    first_by_decision = {}
    for frame, outcome in windows:
        ann = dash.update(outcome, frame=frame)
        annotated_all.append(ann.copy())
        first_by_decision.setdefault(outcome.decision.decision, ann.copy())
    dash.close()

    montage_path = os.path.join(out, "dashboard_demo_montage.png")
    _montage(first_by_decision, montage_path)

    gif_ok = False
    if not args.no_gif:
        gif_ok = _make_gif(annotated_all, os.path.join(out, "dashboard_demo.gif"), args.fps)

    print(f"[demo] {dash.summary_line()}")
    print(f"[demo] mp4     -> {mp4_path}")
    print(f"[demo] montage -> {montage_path}")
    print(f"[demo] frames  -> {frames_dir}/ ({dash._n_saved} stills)")
    if not args.no_gif:
        print(f"[demo] gif     -> {'written' if gif_ok else 'skipped (install imageio or pillow)'}")


if __name__ == "__main__":
    main()
