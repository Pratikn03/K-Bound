# K-Bound Edge — safe test-time adaptation for camera inspection (Tiers 0–2)

A small, additive runtime layer that puts the **K-Bound certificate** in front of
test-time adaptation for a camera-based label-inspection task. The frozen model
keeps producing the official prediction; an episodic [TENT](https://arxiv.org/abs/2006.10726)
candidate is proposed per window; and the **reused** K-Bound gate
(`kbound.certificate.decide`) decides — with a conformal benefit interval —
whether to **adapt**, **freeze**, or **abstain**.

> ⚠️ **This codebase currently ships validated on SYNTHETIC data only.** The
> generated clips prove the *code runs end to end and the decision logic exercises
> every branch*. **They are NOT an empirical result.** Real adapt/freeze/abstain
> numbers — and any claim about whether adaptation helps *your* line — require
> **real recorded clips** (see *What you do first*).

This layer does **not** fork or re-derive the certificate. The decision rule, the
label-free evidence vector, and the benefit router are imported verbatim from the
published `kbound` reproduction package via `kbound_edge._bridge`. Nothing in the
K-Bound paper, its scripts, the `kga/` package, the `KBOUND_*` locks, or any
results is modified.

---

## The three tiers

| Tier | What it does | Entry point |
|------|--------------|-------------|
| **Tier 0** | **Offline replay** of a recorded clip through the *full* decision chain — frozen prediction → episodic TENT candidate → 14 label-free evidence features → `kga_decide` → JSONL log. No camera, no estimator required beyond a fitted one. | `scripts/06_replay_heldout.py` |
| **Tier 1** | Same chain, now **certified**: a `HistGradientBoostingRegressor` benefit estimator (fit on the calibration-fit split) plus a **split-conformal radius** (from the calibration-conformal split, α = 0.10) turn each window's predicted benefit into an interval and a three-way decision. | `scripts/04`, `05`, `06` |
| **Tier 2** | **Live shadow mode.** The controller drives a real camera (or a `FakeVideoCapture`) window by window. The **frozen model is the official output**; the candidate + KGA verdict run **in shadow** — logged, never emitted. Lets you measure how often KGA *would* have safely adapted, with zero production risk. | `scripts/07_shadow_live.py` |

---

## Layout

```
edge/
  src/kbound_edge/
    _bridge.py          reuse point: imports decide / conformal_radius / evidence_vector / BenefitRouter from kbound
    capture.py          FrameSource: OpenCV camera, SyntheticFrameSource, FakeVideoCapture, ListFrameSource
    dataset.py          preprocessing, windowing, synthetic conditions (regime x diversity)
    model.py            MobileNetV3-Small + 4-class head; train + BN-recalibration; state-dict hash (model_version)
    tent_adapter.py     EpisodicTentAdapter — deepcopy f0, BN-affine only, 1 Adam step, NEVER mutates f0
    evidence.py         the 14 label-free features (11 paper features + 3 edge features), fixed schema
    benefit_estimator.py HistGradientBoostingRegressor wrapper (fit / predict / joblib save-load)
    conformal.py        split-conformal residual radius (conservative rank, alpha=0.10)
    policy.py           kga_decide -> adapt/freeze/abstain + lower/upper + reason; the 6 comparison policies
    metrics.py          regret, false-adapt (uncond/cond), adapt/abstain rate, latency  (OFFLINE)
    logging.py          JSONL window logger + the no-live-labels guard
    replay.py           offline recorded-stream runner (the online window engine)
    shadow_runtime.py   live ShadowController (frozen = official, candidate in shadow)
    dashboard.py        live views: headless console + watchable OpenCV overlay
                        (annotate_frame / VisualDashboard) with colour-coded
                        verdicts, a stats panel, and mp4 --record
  scripts/              01_capture_source .. 08_make_report, make_dashboard_demo (+ _common.py)
  configs/              edge_label_inspection_v1.yaml, edge_calibration_v1.yaml, edge_shadow_v1.yaml
  tests/                test_features, test_conformal, test_policy, test_no_live_labels,
                        test_candidate_isolation, test_log_integrity, test_dashboard_render
  README.md
```

---

## Environment

Runs in the existing `~/.venv_wilds` (torch + torchvision already present).
Extra deps: `opencv-python`, `scikit-learn`, `joblib` (and `pyyaml`).

```bash
source ~/.venv_wilds/bin/activate
pip install opencv-python scikit-learn joblib pyyaml      # torch/torchvision already present
```

The edge package finds `kbound` automatically: it is already importable in
`~/.venv_wilds`; otherwise `_bridge.py` adds the sibling `kbound_pkg/` to the path
(it never modifies it).

---

## Exact run order (synthetic, works out of the box)

From `docs/research/kbound/edge/`:

```bash
python scripts/01_capture_source.py            # generate the synthetic source clip (.npz)
python scripts/02_build_manifest.py            # write the run manifest
python scripts/03_train_source_model.py        # train f0 (MobileNetV3) + BN-recalibrate -> f0.pt
python scripts/04_generate_calibration_pairs.py# per-condition label-free Z + offline true benefit B
python scripts/05_fit_kga_edge.py              # fit benefit estimator + split-conformal eps (alpha=0.10)
python scripts/06_replay_heldout.py            # Tier-0/1: replay held-out stream -> JSONL log + metrics + 6-policy table
python scripts/07_shadow_live.py               # Tier-2: shadow run (default: FakeVideoCapture, NO camera)
python scripts/08_make_report.py               # assemble artifacts_synth/REPORT.md
```

Run the unit tests any time:

```bash
python -m pytest tests/ -v
```

All artifacts land in `edge/artifacts_synth/` (gitignored).

---

## Live-demo dashboard (Tier 2)

`07_shadow_live.py` can render a **watchable** view of the shadow stream. Every
window shows, side by side:

- the **CAMERA** frame (decision-coloured border);
- **OFFICIAL** — the *frozen* model's predicted class + confidence. This is the
  output a downstream consumer actually acts on;
- **SHADOW** — the candidate's predicted class + confidence, flagged *"not
  emitted"* (and *"would change class"* when it disagrees with the official one);
- **KGA verdict** — `ADAPT` / `FREEZE` / `ABSTAIN`, colour-coded
  (**green** = adapt-certified, **red** = freeze/harmful, **grey** = abstain),
  with the benefit estimate `B^`, the radius `eps`, and the certified
  `[lower, upper]` interval, plus the human-readable reason;
- per-window **latency**; and a running **stats panel**: adapt / freeze / abstain
  counts, adapt-rate, abstain-rate, a **false-adapt** counter (stays `0 (offline)`
  on the live path — there are no labels online; an offline label-join fills it),
  and current / mean latency.

```bash
# Headless status lines (default; safe on a headless box):
python scripts/07_shadow_live.py

# Watchable live OpenCV window (press 'q' to stop early):
python scripts/07_shadow_live.py --view window

# Record the annotated stream to mp4 (for a demo video; runs headless too):
python scripts/07_shadow_live.py --record artifacts/shadow_demo.mp4

# Real phone camera (Tier 2) with a live window + recording:
python scripts/07_shadow_live.py --camera 0 --view window --record artifacts/phone_demo.mp4

# Save a handful of annotated still frames instead of / as well as mp4:
python scripts/07_shadow_live.py --sample-dir artifacts/frames --max-windows 16
```

Flags: `--view {console,window}`, `--record PATH`, `--sample-dir DIR`, `--fps`,
`--max-windows N`. Any of `--record` / `--sample-dir` forces rendering even in
console view, so you can capture a demo on a headless box. If no GUI backend is
available, `--view window` degrades gracefully to headless and says so.

**Eyeball it with no model / no camera.** `scripts/make_dashboard_demo.py` drives
the *real* renderer with synthetic frames and genuine `kga_decide` verdicts (needs
only `numpy` + `opencv-python`, no torch) and writes a sample to `edge/artifacts/`:

```bash
python scripts/make_dashboard_demo.py     # -> artifacts/dashboard_demo.{mp4,gif}, _montage.png, frames/
```

The render path is covered by `tests/test_dashboard_render.py` (skips if OpenCV
isn't installed). `edge/artifacts/` is gitignored — regenerate any time.

---

## What you do first (for a REAL result)

Synthetic data only proves the pipeline runs. To get a real result on your line:

1. **Record a clean SOURCE clip** of correctly-presented labels (all 4 classes,
   good lighting). In `configs/edge_label_inspection_v1.yaml` set
   `source.kind: opencv`, `image_size: 224`, and either `source.camera_index` or
   `source.video_path`. Provide ground-truth labels for the source frames (a
   short manual labelling pass) before `03`.
2. **Record SHIFTED clips** (glare, motion blur, a different label stock, a new
   camera angle) and a few **degenerate** ones (static / single-label runs).
   Point the calibration/held-out plans at real clips.
3. Re-run `01 → 08`. Now the metrics in `REPORT.md` are real.

Until then, treat every number as a code smoke-test, not evidence.

### Phone as the camera (Tier 2)

The OpenCV source takes any device index or video path, so a phone makes a great
inspection camera:

- **iPhone — Continuity Camera (macOS):** with the iPhone nearby (same Apple ID,
  Wi-Fi + Bluetooth on, iPhone mounted/stationary), macOS exposes it as a normal
  webcam. It usually appears at OpenCV index `0` or `1`:
  ```bash
  python scripts/07_shadow_live.py --camera 0      # try 0, then 1
  ```
- **Android — Camo or EpocCam:** install the app on the phone and the companion
  driver on the Mac; it registers a virtual webcam. Use its index the same way
  (`--camera 0/1`). EpocCam (Elgato) and Camo (Reincubate) both work; pick
  whichever your phone is happiest with.

To sanity-check which index is your phone, list devices or just try `--camera 0`,
`--camera 1`, … and watch the dashboard.

---

## The guarantees (enforced by tests)

- **Candidate isolation** — `EpisodicTentAdapter` deep-copies f0 and updates only
  the copy's BatchNorm-affine params for one Adam step. f0's parameter+buffer hash
  is bit-identical before and after. *(test_candidate_isolation)*
- **No live labels** — ground-truth labels are unreachable from the online path
  (capture → adapt → evidence → decision → log). The runtime and the logger reject
  any label-like key. *(test_no_live_labels)*
- **Split-conformal** — the estimator is fit on calibration-fit only; residuals
  come from calibration-conformal only; `eps` is the conservative residual order
  statistic at α = 0.10. *(test_conformal)*
- **Decision rule** — `lower > 0 → adapt`, `upper < 0 → freeze`, interval spans
  0 → `abstain`, reusing `kbound.certificate.decide`. *(test_policy)*
- **Stable evidence schema** — 14 features, first 11 identical to the paper.
  *(test_features)*
- **Auditable logs** — every window record carries model_version, config_hash,
  decision, and latency. *(test_log_integrity)*

The 6-policy comparison (`always_freeze`, `always_adapt`, `confidence_gate`,
`entropy_gate`, `kga_no_radius`, `kga_full`) is in `policy.py` and reported by
`06` / `08`.

---

## Why a BN-recalibration step in `03`?

A freshly trained small net can fit perfectly in train mode (BatchNorm *batch*
stats) yet collapse to chance in eval mode because its *running* stats are poorly
estimated — the very BN mismatch TENT targets. `03` runs a short no-grad
recalibration so the frozen model is a competent eval-mode baseline; only then is
"does adaptation help?" a meaningful question.
