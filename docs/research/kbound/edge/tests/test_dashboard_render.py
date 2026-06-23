"""test_dashboard_render -- smoke tests for the live-demo dashboard render path.

Covers the pure renderer (:func:`annotate_frame`), the shared
:class:`DecisionStats`, both dashboards' ``update()`` contract, colour-coding of
the three verdicts, and the headless mp4 / PNG sinks of
:class:`VisualDashboard`. No model, no camera, no torch -- the dashboard render
path is exercised with synthetic frames and real ``kga_decide`` verdicts, so this
runs in the normal unit-test pass.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # the visual dashboard needs OpenCV

from kbound_edge.policy import kga_decide
from kbound_edge.dashboard import (
    annotate_frame,
    DecisionStats,
    LiveDashboard,
    VisualDashboard,
    build_dashboard,
    DECISION_COLORS_BGR,
)


# --- minimal duck-typed outcome (avoids importing replay -> torch) -----------
class _OC:
    def __init__(self, wid, decision, p0, pa, latency=12.0):
        self.window_id = wid
        self.decision = decision
        self.p0 = p0
        self.pa = pa
        self.latency_ms = latency
        self.evidence = {}

    @property
    def frozen_pred(self):
        return self.p0.argmax(1).tolist()

    @property
    def candidate_pred(self):
        return self.pa.argmax(1).tolist()


def _probs(n, cls, conf, C=4):
    p = np.full((n, C), (1.0 - conf) / (C - 1), dtype=float)
    p[:, cls] = conf
    return p


def _outcome(bhat, eps=0.08, frozen_cls=0, cand_cls=0, wid=0):
    return _OC(wid, kga_decide(bhat, eps),
               _probs(8, frozen_cls, 0.75), _probs(8, cand_cls, 0.85))


def _frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
def test_annotate_frame_shape_and_dtype():
    out = annotate_frame(_frame(), _outcome(+0.3), DecisionStats())
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3
    # canvas is larger than the camera tile (panel to the right, stats below)
    assert out.shape[0] > 360 and out.shape[1] > 360
    assert int(out.sum()) > 0  # something was actually drawn


def test_grayscale_and_float_frames_are_coerced():
    gray = np.zeros((64, 64), dtype=np.uint8)
    flt = np.ones((64, 64, 3), dtype=np.float32) * 300.0  # out of range on purpose
    for f in (gray, flt):
        out = annotate_frame(f, _outcome(0.0), DecisionStats())
        assert out.dtype == np.uint8 and out.ndim == 3


@pytest.mark.parametrize("bhat,eps,expected", [
    (+0.40, 0.08, "adapt"),
    (-0.40, 0.08, "freeze"),
    (+0.02, 0.20, "abstain"),
])
def test_decision_colour_coding(bhat, eps, expected):
    oc = _outcome(bhat, eps)
    assert oc.decision.decision == expected
    out = annotate_frame(_frame(), oc, DecisionStats())
    # probe a clear point inside the colour-coded verdict banner (defaults:
    # frame_px=360, panel_w=420 -> W=780; banner spans y in [188, 272]).
    probe = out[196, 760].astype(int)
    assert np.allclose(probe, np.array(DECISION_COLORS_BGR[expected]), atol=6)


def test_three_verdicts_render_differently():
    s = DecisionStats()
    imgs = [annotate_frame(_frame(), _outcome(b, e), s)
            for b, e in [(+0.4, 0.08), (-0.4, 0.08), (+0.02, 0.2)]]
    assert not np.array_equal(imgs[0], imgs[1])
    assert not np.array_equal(imgs[1], imgs[2])


def test_decision_stats_accumulate():
    s = DecisionStats()
    for d, lat in [("adapt", 10.0), ("adapt", 20.0), ("freeze", 30.0), ("abstain", 40.0)]:
        s.update(d, lat)
    assert s.n == 4
    assert s.counts["adapt"] == 2
    assert s.adapt_rate == pytest.approx(0.5)
    assert s.abstain_rate == pytest.approx(0.25)
    assert s.mean_latency == pytest.approx(25.0)
    assert s.latency_last == pytest.approx(40.0)
    s.note_false_adapt(2)
    assert s.false_adapt == 2


def test_live_dashboard_update_contract():
    dash = LiveDashboard(quiet=True)
    line = dash.update(_outcome(+0.3), frame=_frame())   # frame accepted + ignored
    assert "adapt" in line
    dash.update(_outcome(-0.3))                            # frame optional
    assert dash.stats.n == 2
    assert "windows=2" in dash.summary_line()
    assert set(dash.rates()) <= {"adapt", "freeze", "abstain"}


def test_visual_dashboard_records_mp4(tmp_path):
    mp4 = tmp_path / "demo.mp4"
    dash = VisualDashboard(show=False, record_path=str(mp4))
    for w, (b, e) in enumerate([(+0.4, 0.08), (-0.4, 0.08), (+0.02, 0.2), (+0.3, 0.08)]):
        dash.update(_outcome(b, e, wid=w), frame=_frame())
    dash.close()
    if not mp4.exists() or mp4.stat().st_size == 0:
        pytest.skip("no mp4 codec available in this environment")
    assert mp4.stat().st_size > 0


def test_visual_dashboard_saves_sample_frames(tmp_path):
    sdir = tmp_path / "frames"
    dash = VisualDashboard(show=False, sample_dir=str(sdir), max_samples=3)
    for w, (b, e) in enumerate([(+0.4, 0.08), (-0.4, 0.08), (+0.02, 0.2), (+0.3, 0.08)]):
        dash.update(_outcome(b, e, wid=w), frame=_frame())
    dash.close()
    pngs = list(sdir.glob("*.png"))
    assert 1 <= len(pngs) <= 3       # capped at max_samples
    assert all(p.stat().st_size > 0 for p in pngs)


def test_build_dashboard_factory():
    assert isinstance(build_dashboard("console"), LiveDashboard)
    assert isinstance(build_dashboard("window"), VisualDashboard)
    # a record/sample sink forces the visual dashboard even in console view
    assert isinstance(build_dashboard("console", record_path="x.mp4"), VisualDashboard)
