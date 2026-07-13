"""kbound_edge.shadow_runtime -- live, window-by-window ShadowController (Tier 2).

Shadow mode is the SAFE way to run adaptation against a live camera:

* The FROZEN model's prediction is the OFFICIAL output for every window -- it is
  what a downstream consumer would actually act on.
* The episodic Tent candidate runs in SHADOW: its prediction and the
  ``kga_decide`` verdict are computed and LOGGED, but never emitted as the
  official output.  Nothing the candidate does can affect production behaviour;
  the logs let you later measure how often KGA *would* have safely adapted.

The controller drives any :class:`kbound_edge.capture.FrameSource` -- a real
``cv2.VideoCapture`` (webcam / phone-as-camera) or the
:class:`~kbound_edge.capture.FakeVideoCapture`, so the exact loop can be
exercised with no hardware (see ``scripts/07_shadow_live.py`` and
``tests/`` Tier-2 dry check).

No labels are ever read on this path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kbound_edge.replay import run_window, WindowOutcome


@dataclass
class ShadowController:
    """Live shadow runner: frozen = official, candidate = shadow-only.

    Parameters
    ----------
    f0, adapter, estimator, eps
        The frozen model, episodic Tent adapter, fitted benefit estimator and
        conformal radius.
    window_size : int
        Number of frames per decision window.
    image_size : int, default=64
    logger : WindowLogger, optional
        If given, one JSONL record is written per window.
    dashboard : object with ``.update(outcome)``, optional
    max_windows : int, optional
        Stop after this many windows (useful for dry runs / tests).
    """

    f0: Any
    adapter: Any
    estimator: Any
    eps: float
    window_size: int = 16
    image_size: int = 64
    logger: Any = None
    dashboard: Any = None
    max_windows: Optional[int] = None

    _buffer: List[Any] = field(default_factory=list, init=False)
    _wid: int = field(default=0, init=False)
    official_outputs: List[List[int]] = field(default_factory=list, init=False)
    shadow_decisions: List[str] = field(default_factory=list, init=False)
    outcomes: List[WindowOutcome] = field(default_factory=list, init=False)

    # -- frame ingestion -------------------------------------------------------
    def push_frame(self, frame) -> Optional[WindowOutcome]:
        """Add one frame; if a window is complete, process it and return outcome."""
        self._buffer.append(frame)
        if len(self._buffer) >= self.window_size:
            return self._flush_window()
        return None

    def _flush_window(self) -> WindowOutcome:
        frames = self._buffer[: self.window_size]
        self._buffer = self._buffer[self.window_size :]

        outcome = run_window(
            self._wid, frames, self.f0, self.adapter, self.estimator, self.eps,
            image_size=self.image_size,
        )

        # OFFICIAL output is ALWAYS the frozen model prediction.
        official = outcome.frozen_pred
        self.official_outputs.append(official)
        # The candidate verdict is SHADOW-only -- recorded, never emitted.
        self.shadow_decisions.append(outcome.decision.decision)
        self.outcomes.append(outcome)

        if self.logger is not None:
            self.logger.log(
                window_id=self._wid,
                decision=outcome.decision.as_dict(),
                evidence=outcome.evidence,
                latency_ms=outcome.latency_ms,
                frozen_pred=official,
                extra={
                    "shadow_candidate_pred": outcome.candidate_pred,
                    "upd_norm": outcome.upd_norm,
                    "mode": "shadow",
                },
            )
        if self.dashboard is not None:
            # Hand the visual dashboard a representative frame for this window;
            # fall back for dashboards whose update() takes only the outcome.
            frame = frames[-1] if frames else None
            try:
                self.dashboard.update(outcome, frame=frame)
            except TypeError:
                self.dashboard.update(outcome)

        self._wid += 1
        return outcome

    # -- driving a source ------------------------------------------------------
    def run(self, source, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Pull frames from a FrameSource (or cv2.VideoCapture-like) until done.

        Stops when the source is exhausted, ``max_windows`` windows have been
        processed, or ``max_frames`` frames have been read.
        """
        n_frames = 0
        while True:
            if max_frames is not None and n_frames >= max_frames:
                break
            ok, frame = source.read()
            if not ok or frame is None:
                break
            n_frames += 1
            self.push_frame(frame)
            if self.max_windows is not None and self._wid >= self.max_windows:
                break
            # let a visual dashboard request an early stop (e.g. user pressed 'q')
            if getattr(self.dashboard, "stopped", False):
                break
        try:
            source.release()
        except Exception:
            pass
        # flush/close any visual dashboard (releases the mp4 writer, closes window)
        if self.dashboard is not None and hasattr(self.dashboard, "close"):
            try:
                self.dashboard.close()
            except Exception:
                pass
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        from collections import Counter

        counts = Counter(self.shadow_decisions)
        return {
            "n_windows": self._wid,
            "n_official_outputs": len(self.official_outputs),
            "shadow_decision_counts": dict(counts),
            "mean_latency_ms": (
                sum(o.latency_ms for o in self.outcomes) / len(self.outcomes)
                if self.outcomes else 0.0
            ),
        }
