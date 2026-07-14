"""kbound_edge.dashboard -- live views of the shadow stream.

Two dashboards share one stats core:

* :class:`LiveDashboard`   -- headless console (one status line per window). No
  GUI dependency; safe on a headless edge box.
* :class:`VisualDashboard` -- a watchable OpenCV window with real-time overlays:
  the FROZEN model's predicted class + confidence (the **OFFICIAL** output that a
  downstream consumer would actually act on), the candidate's predicted class
  (**SHADOW**, never emitted), the KGA verdict (**ADAPT / FREEZE / ABSTAIN**)
  with the benefit estimate ``B^`` and its certified ``[lower, upper]`` interval,
  per-window latency, and a running stats panel (adapt/freeze/abstain counts,
  adapt-rate, abstain-rate, a false-adapt counter, and mean latency). The
  decision is colour-coded so it is glanceable in a demo
  (green = adapt-certified, red = freeze/harmful, grey = abstain). It can also
  record the annotated stream to mp4 (``--record``).

Both dashboards expose ``update(outcome, frame=None) -> str`` and an optional
``close()``, so either can be handed to
:class:`kbound_edge.shadow_runtime.ShadowController` via ``dashboard=``.

The rendering itself is a **pure function**, :func:`annotate_frame`, that depends
only on numpy + OpenCV and a duck-typed ``outcome`` (anything carrying
``window_id``, ``latency_ms``, ``decision`` and either ``p0``/``pa`` softmaxes or
``frozen_pred``/``candidate_pred`` lists). That keeps it unit-testable and
demo-able with no model, no camera, and no torch -- see
``scripts/make_dashboard_demo.py`` and ``tests/test_dashboard_render.py``.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Glanceable colour code (OpenCV uses BGR).
# ---------------------------------------------------------------------------
#: green = adapt-certified, red = freeze/harmful, grey = abstain.
DECISION_COLORS_BGR = {
    "adapt": (90, 200, 90),
    "freeze": (60, 60, 225),
    "abstain": (165, 165, 165),
}
DECISION_LABELS = {"adapt": "ADAPT", "freeze": "FREEZE", "abstain": "ABSTAIN"}
_SYMBOL = {"adapt": "A", "freeze": "F", "abstain": "-"}

_WHITE = (240, 240, 240)
_DIM = (170, 170, 170)
_PANEL_BG = (38, 36, 34)
_OFFICIAL_ACCENT = (60, 200, 240)   # amber/cyan -- marks the OFFICIAL output
_SHADOW_ACCENT = (200, 160, 120)    # muted blue -- marks the SHADOW candidate


# ---------------------------------------------------------------------------
# Shared running statistics.
# ---------------------------------------------------------------------------
@dataclass
class DecisionStats:
    """Running tally of window outcomes, shared by both dashboards.

    ``false_adapt`` is left at 0 on the online path (no labels are ever seen
    live). It is here so an OFFLINE label-join step can populate it later via
    :meth:`note_false_adapt`; the demo surfaces it as "offline" until then.
    """

    counts: Counter = field(default_factory=Counter)
    n: int = 0
    latency_sum: float = 0.0
    latency_last: float = 0.0
    false_adapt: int = 0

    def update(self, decision: str, latency_ms: float) -> None:
        self.counts[decision] += 1
        self.n += 1
        self.latency_last = float(latency_ms)
        self.latency_sum += float(latency_ms)

    def note_false_adapt(self, k: int = 1) -> None:
        """Record ``k`` false-adapts (only meaningful after an offline label join)."""
        self.false_adapt += int(k)

    def rate(self, name: str) -> float:
        return self.counts.get(name, 0) / self.n if self.n else 0.0

    @property
    def adapt_rate(self) -> float:
        return self.rate("adapt")

    @property
    def abstain_rate(self) -> float:
        return self.rate("abstain")

    @property
    def freeze_rate(self) -> float:
        return self.rate("freeze")

    @property
    def mean_latency(self) -> float:
        return self.latency_sum / self.n if self.n else 0.0

    def as_dict(self) -> dict:
        return {
            "windows": self.n,
            "counts": dict(self.counts),
            "adapt_rate": self.adapt_rate,
            "abstain_rate": self.abstain_rate,
            "freeze_rate": self.freeze_rate,
            "false_adapt": self.false_adapt,
            "mean_latency_ms": self.mean_latency,
        }


# ---------------------------------------------------------------------------
# Helpers to read class + confidence from an outcome.
# ---------------------------------------------------------------------------
def _class_conf(prob: Any) -> Tuple[int, float]:
    """Window-level (class, confidence) from a per-frame softmax (N,C) or (C,)."""
    p = np.asarray(prob, dtype=float)
    if p.size == 0:
        return -1, float("nan")
    if p.ndim == 1:
        p = p[None, :]
    m = p.mean(axis=0)
    return int(m.argmax()), float(m.max())


def _frozen_class_conf(outcome: Any) -> Tuple[int, float]:
    p0 = getattr(outcome, "p0", None)
    if p0 is not None:
        return _class_conf(p0)
    preds = getattr(outcome, "frozen_pred", None)
    if preds:
        return int(Counter(preds).most_common(1)[0][0]), float("nan")
    return -1, float("nan")


def _candidate_class_conf(outcome: Any) -> Tuple[int, float]:
    pa = getattr(outcome, "pa", None)
    if pa is not None:
        return _class_conf(pa)
    preds = getattr(outcome, "candidate_pred", None)
    if preds:
        return int(Counter(preds).most_common(1)[0][0]), float("nan")
    return -1, float("nan")


def _class_name(idx: int, class_names: Optional[Sequence[str]]) -> str:
    if idx < 0:
        return "n/a"
    if class_names and 0 <= idx < len(class_names):
        return str(class_names[idx])
    return f"cls {idx}"


def _to_bgr_uint8(frame: Any) -> np.ndarray:
    """Coerce an arbitrary frame to an HxWx3 uint8 BGR image."""
    img = np.asarray(frame)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------------------
# The pure renderer.
# ---------------------------------------------------------------------------
def annotate_frame(
    frame: Any,
    outcome: Any,
    stats: Optional[DecisionStats] = None,
    class_names: Optional[Sequence[str]] = None,
    frame_px: int = 360,
    panel_w: int = 420,
    stats_h: int = 104,
) -> np.ndarray:
    """Render a single annotated demo frame (BGR uint8) for ``outcome``.

    Pure: depends only on numpy + OpenCV and the duck-typed ``outcome``. ``frame``
    is the (small) camera frame for this window; it is upscaled into the left tile
    and given a decision-coloured border. The right panel shows the OFFICIAL
    frozen output, the SHADOW candidate, and the KGA verdict; the bottom bar shows
    running stats (pass ``stats`` to populate it).
    """
    import cv2

    d = outcome.decision
    decision = getattr(d, "decision", "abstain")
    color = DECISION_COLORS_BGR.get(decision, _DIM)

    # --- left tile: the camera frame, upscaled, with a coloured border --------
    img = _to_bgr_uint8(frame)
    tile = cv2.resize(img, (frame_px, frame_px), interpolation=cv2.INTER_NEAREST)

    W = frame_px + panel_w
    H = frame_px + stats_h
    canvas = np.full((H, W, 3), _PANEL_BG, dtype=np.uint8)
    canvas[0:frame_px, 0:frame_px] = tile
    cv2.rectangle(canvas, (3, 3), (frame_px - 4, frame_px - 4), color, 6)
    cv2.putText(canvas, "CAMERA", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

    def text(s, org, scale=0.52, col=_WHITE, thick=1):
        cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

    px = frame_px + 16          # panel left edge
    wid = getattr(outcome, "window_id", 0)
    lat = float(getattr(outcome, "latency_ms", 0.0))

    text("K-BOUND EDGE  -  LIVE SHADOW DEMO", (px, 28), 0.56, _WHITE, 2)
    text(f"window {wid:04d}        {lat:6.1f} ms", (px, 52), 0.48, _DIM, 1)

    # --- OFFICIAL frozen output ----------------------------------------------
    fcls, fconf = _frozen_class_conf(outcome)
    cv2.circle(canvas, (px + 6, 78), 5, _OFFICIAL_ACCENT, -1)
    text("OFFICIAL  (frozen model -> acted on)", (px + 18, 82), 0.46, _OFFICIAL_ACCENT, 1)
    conf_s = "n/a" if fconf != fconf else f"{fconf*100:4.1f}%"
    text(f"{_class_name(fcls, class_names)}   conf {conf_s}", (px + 18, 108), 0.62, _WHITE, 2)

    # --- SHADOW candidate -----------------------------------------------------
    ccls, cconf = _candidate_class_conf(outcome)
    changed = (ccls != fcls) and ccls >= 0 and fcls >= 0
    cv2.circle(canvas, (px + 6, 138), 5, _SHADOW_ACCENT, -1)
    text("SHADOW  (candidate -> not emitted)", (px + 18, 142), 0.46, _SHADOW_ACCENT, 1)
    cconf_s = "n/a" if cconf != cconf else f"{cconf*100:4.1f}%"
    tail = "  (would change class)" if changed else ""
    text(f"{_class_name(ccls, class_names)}   conf {cconf_s}{tail}", (px + 18, 168), 0.56, _DIM, 1)

    # --- KGA verdict (colour-coded banner) -----------------------------------
    by0 = 188
    cv2.rectangle(canvas, (px - 4, by0), (W - 8, by0 + 84), color, -1)
    label = DECISION_LABELS.get(decision, decision.upper())
    text(f"KGA: {label}", (px + 6, by0 + 32), 0.92, (20, 20, 20), 2)
    bhat = float(getattr(d, "bhat", 0.0))
    eps = float(getattr(d, "eps", 0.0))
    lo = float(getattr(d, "lower", bhat - eps))
    hi = float(getattr(d, "upper", bhat + eps))
    text(f"B^ = {bhat:+.3f}   eps = {eps:.3f}", (px + 6, by0 + 58), 0.5, (20, 20, 20), 1)
    text(f"[{lo:+.3f}, {hi:+.3f}]", (px + 6, by0 + 78), 0.5, (20, 20, 20), 1)

    # one-line reason (wrapped to two short lines if long)
    reason = str(getattr(d, "reason", "")).strip()
    if reason:
        for i, chunk in enumerate(_wrap(reason, 52)[:2]):
            text(chunk, (px, by0 + 104 + 18 * i), 0.4, _DIM, 1)

    # --- bottom stats bar -----------------------------------------------------
    sy = frame_px
    cv2.line(canvas, (0, sy), (W, sy), (70, 70, 70), 1)
    st = stats or DecisionStats()
    na = st.counts.get("adapt", 0)
    nf = st.counts.get("freeze", 0)
    nb = st.counts.get("abstain", 0)
    text(
        f"windows {st.n}    "
        f"A {na}  F {nf}  -{'-'} {nb}",
        (12, sy + 26), 0.5, _WHITE, 1,
    )
    # coloured legend chips
    _chip(canvas, (12, sy + 40), DECISION_COLORS_BGR["adapt"], "adapt", text)
    _chip(canvas, (120, sy + 40), DECISION_COLORS_BGR["freeze"], "freeze", text)
    _chip(canvas, (235, sy + 40), DECISION_COLORS_BGR["abstain"], "abstain", text)
    text(
        f"adapt-rate {st.adapt_rate*100:4.1f}%   abstain-rate {st.abstain_rate*100:4.1f}%",
        (12, sy + 78), 0.5, _WHITE, 1,
    )
    fa = "0 (offline)" if st.false_adapt == 0 else str(st.false_adapt)
    text(
        f"false-adapt {fa}    latency {st.latency_last:5.1f} / {st.mean_latency:5.1f} ms",
        (12, sy + 96), 0.48, _DIM, 1,
    )
    return canvas


def _chip(canvas, org, color, label, text_fn):
    import cv2
    x, y = org
    cv2.rectangle(canvas, (x, y - 11), (x + 14, y + 1), color, -1)
    text_fn(label, (x + 20, y), 0.44, _DIM, 1)


def _wrap(s: str, width: int) -> List[str]:
    out, line = [], ""
    for word in s.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


# ---------------------------------------------------------------------------
# Console dashboard (headless).
# ---------------------------------------------------------------------------
class LiveDashboard:
    """Accumulates window outcomes and prints a compact status line each window."""

    def __init__(self, every: int = 1, stream=None, quiet: bool = False) -> None:
        self.every = max(1, int(every))
        self.quiet = quiet
        self._stream = stream
        self.stats = DecisionStats()

    # back-compat alias used by older callers / tests
    @property
    def _counts(self) -> Counter:
        return self.stats.counts

    @property
    def _n(self) -> int:
        return self.stats.n

    def update(self, outcome, frame=None) -> str:
        """Record one window outcome and (optionally) print a status line."""
        d = outcome.decision
        self.stats.update(d.decision, getattr(outcome, "latency_ms", 0.0))
        line = self.render(outcome)
        if not self.quiet and (self.stats.n % self.every == 0):
            print(line, file=self._stream)
        return line

    def render(self, outcome) -> str:
        d = outcome.decision
        sym = _SYMBOL.get(d.decision, "?")
        return (
            f"[w{outcome.window_id:04d}] {sym} {d.decision:7s} "
            f"B^={d.bhat:+.3f} eps={d.eps:.3f} "
            f"[{d.lower:+.3f},{d.upper:+.3f}] "
            f"{outcome.latency_ms:6.1f}ms | "
            f"A:{self.stats.adapt_rate:.2f} F:{self.stats.freeze_rate:.2f} "
            f"-:{self.stats.abstain_rate:.2f}"
        )

    def rates(self) -> dict:
        if self.stats.n == 0:
            return {}
        return {k: v / self.stats.n for k, v in self.stats.counts.items()}

    def summary_line(self) -> str:
        return (
            f"windows={self.stats.n} decisions={dict(self.stats.counts)} "
            f"mean_latency={self.stats.mean_latency:.1f}ms"
        )

    def close(self) -> None:  # symmetry with VisualDashboard
        pass


# ---------------------------------------------------------------------------
# Visual dashboard (OpenCV window + optional mp4 record).
# ---------------------------------------------------------------------------
class VisualDashboard:
    """Watchable OpenCV dashboard: live overlays + running stats, optional record.

    Parameters
    ----------
    window_name : str
    show : bool, default True
        Open a live ``cv2.imshow`` window. Set ``False`` for headless/CI; the
        renderer still runs (so recording / frame-saving work).
    record_path : str, optional
        If given, write the annotated stream to this mp4.
    sample_dir : str, optional
        If given, save up to ``max_samples`` annotated PNG frames here.
    class_names : sequence of str, optional
    fps : float, default 8
    delay_ms : int, default 1
        ``cv2.waitKey`` delay (also lets you quit the window with 'q').
    """

    def __init__(
        self,
        window_name: str = "K-Bound Edge - live shadow",
        show: bool = True,
        record_path: Optional[str] = None,
        sample_dir: Optional[str] = None,
        max_samples: int = 8,
        class_names: Optional[Sequence[str]] = None,
        fps: float = 8.0,
        delay_ms: int = 1,
        every: int = 1,
    ) -> None:
        self.window_name = window_name
        self.show = bool(show)
        self.record_path = record_path
        self.sample_dir = sample_dir
        self.max_samples = int(max_samples)
        self.class_names = class_names
        self.fps = float(fps)
        self.delay_ms = int(delay_ms)
        self.every = max(1, int(every))

        self.stats = DecisionStats()
        self._writer = None
        self._win_ready = False
        self._n_saved = 0
        self.last_frame: Optional[np.ndarray] = None
        self.stopped = False

        if self.sample_dir:
            os.makedirs(self.sample_dir, exist_ok=True)

    # -- the ShadowController hook --------------------------------------------
    def update(self, outcome, frame=None) -> np.ndarray:
        """Render (and show/record) one window. Returns the annotated frame."""
        self.stats.update(outcome.decision.decision, getattr(outcome, "latency_ms", 0.0))
        if frame is None:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
        annotated = annotate_frame(frame, outcome, self.stats, class_names=self.class_names)
        self.last_frame = annotated

        if self.stats.n % self.every == 0:
            if self.record_path:
                self._write(annotated)
            if self.sample_dir and self._n_saved < self.max_samples:
                self._save_sample(annotated, outcome)
            if self.show:
                self._imshow(annotated)
        return annotated

    # -- output sinks ---------------------------------------------------------
    def _write(self, annotated) -> None:
        import cv2
        if self._writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs(os.path.dirname(os.path.abspath(self.record_path)), exist_ok=True)
            self._writer = cv2.VideoWriter(self.record_path, fourcc, self.fps, (w, h))
        self._writer.write(annotated)

    def _save_sample(self, annotated, outcome) -> None:
        import cv2
        name = f"window_{getattr(outcome, 'window_id', self._n_saved):04d}_{outcome.decision.decision}.png"
        cv2.imwrite(os.path.join(self.sample_dir, name), annotated)
        self._n_saved += 1

    def _imshow(self, annotated) -> None:
        try:
            import cv2
            if not self._win_ready:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._win_ready = True
            cv2.imshow(self.window_name, annotated)
            if (cv2.waitKey(self.delay_ms) & 0xFF) == ord("q"):
                self.stopped = True
        except Exception as exc:  # headless / no GUI backend -> degrade gracefully
            self.show = False
            print(f"[dashboard] live window unavailable ({exc}); continuing headless")

    # -- summary / teardown ---------------------------------------------------
    def summary_line(self) -> str:
        s = self.stats
        return (
            f"windows={s.n} decisions={dict(s.counts)} "
            f"adapt_rate={s.adapt_rate:.2f} abstain_rate={s.abstain_rate:.2f} "
            f"false_adapt={s.false_adapt} mean_latency={s.mean_latency:.1f}ms"
        )

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
        if self._win_ready:
            try:
                import cv2
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self._win_ready = False


def build_dashboard(
    view: str = "console",
    record_path: Optional[str] = None,
    sample_dir: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    every: int = 1,
    fps: float = 8.0,
):
    """Factory: 'console' -> :class:`LiveDashboard`; 'window' -> :class:`VisualDashboard`.

    A ``record_path`` or ``sample_dir`` forces the visual dashboard (rendering is
    required to produce them), running headless when ``view != 'window'``.
    """
    want_visual = view == "window" or record_path is not None or sample_dir is not None
    if not want_visual:
        return LiveDashboard(every=every)
    return VisualDashboard(
        show=(view == "window"),
        record_path=record_path,
        sample_dir=sample_dir,
        class_names=class_names,
        every=every,
        fps=fps,
    )
