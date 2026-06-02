# Real-IAD as the Natural-Degradation Transfer Dataset — Audit (2026-06-02)

Goal (user): use **Real-IAD D3** as the transfer benchmark instead of Eyecandies,
because Real-IAD is *real, naturally degraded* multimodal data (vs Eyecandies'
calibration-shift failure). This audit reports what the data actually supports
after a full diagnosis. Bottom line: Real-IAD is the right *target*, but it
currently yields an **honest detector-limited negative**, not a transfer win —
and the cause differs between the two Real-IAD configurations present in the repo.

## Two distinct Real-IAD tracks (do not conflate)

| Track | Data root | Modalities | Per-view AUROC (test) | Root cause of weakness |
|---|---|---|---|---|
| **D13 multi-view** (`realiad_256_c1_c2`) | `data/raw/realiad` (4 GB) | rgb_c1, rgb_c2 | ~0.69–0.73 | **clip-saturation bug — FIXED** |
| **D15/D16 headroom** (`realiad_d3`) | `data/raw/realiad_d3` (259 GB) | rgb, ps, xyz | 0.48–0.55 pooled | **genuinely near-chance detectors** |

The archived `OFFICIAL_FAIL` (`realiad_d3_headroom_audit_result.json`) is on the
**multimodal D15/D16 track**.

## Finding 1 — D13 multi-view had a real normalization bug (now fixed)

`prepare_realiad_positive_transfer.py::_score_features` min-max-clipped the
per-view kNN distance to `[0,1]`:
`norm = clip((raw - p5)/(p95 - p5), 0, 1)`. This pinned **55%** of per-view
scores (17,732 / 32,152) to a constant `1.0`, and **41%** of CW scores to `1.0`,
collapsing CW via massive ties. Replaced with the established monotone
**z-sigmoid** centered on the train-normal distance distribution
(`1/(1+exp(-(raw-mu)/sd))`), which preserves within-category ranking and aligns
categories for honest pooling.

Effect (test): per-view saturation 55% → 16%; CW saturation 41% → 8%; **pooled
CW AUROC 0.698 → 0.727**; within-category **CW 0.753 ≈ SAR 0.752** (now tied).
This track is *clean* multi-camera RGB (not degraded), so by **T9** the gate
cannot beat CW here — and indeed CW ties the strong baseline. Consistent, not a
transfer win.

## Finding 2 — the multimodal D15/D16 FAIL is genuine, not a bug

The headroom inputs use a non-saturating `raw/(raw+scale)` normalization (0
samples pinned at 1.0), so the clip bug does **not** apply. The fail is driven by
the upstream detectors themselves:

- Pooled test AUROC: **ps 0.549, rgb 0.517, xyz 0.483** (xyz below chance).
- Within-category AUROC is higher (ps 0.630, rgb 0.604, xyz 0.554) but
  **inconsistent in direction across the 19 categories** (flip-corrected mean
  ps 0.685), so it does not survive pooling.
- Per-category z-sigmoid recalibration does **not** rescue pooling
  (ps 0.549→0.540, rgb 0.517→0.538, xyz 0.483→0.476): the signal isn't there
  consistently to be aligned.
- The routing signal `quality_reliability` barely separates the classes
  (normal 0.381 vs anomaly 0.412), capping any reliability-gated routing.

With near-chance per-modality scores there is **no signal for any fusion rule to
exploit** — gate, CW, or the candidate-vs-CW comparison alike. SAR looks strong
(~0.84) only because it is a *supervised* model over the full per-view embeddings
trained on validation labels, while CW uses the near-chance per-modality score;
the −0.29 candidate-vs-SAR gap is therefore expected and not informative about
the gate mechanism.

Root cause: the **lightweight handcrafted feature extractor** (color/texture
statistics on 96×96 crops) is insufficient for Real-IAD-3D, especially the
point-cloud (`xyz`) modality. This is the recurring bottleneck of the whole
program (near-chance upstream detectors), now on Real-IAD-3D.

## Verdict and decision (D17)

- **Adopt Real-IAD D3 as the natural-degradation transfer target** replacing the
  Eyecandies study role (Eyecandies stays permanently recorded as FAILED, D1).
  Real-IAD *is* the better natural multimodal stress source.
- **Current status: honest detector-limited NEGATIVE.**
  `gate_e_positive_transfer_confirmed` remains **FALSE**;
  `gate_s_natural_degradation_confirmed` remains **FALSE**. The archived
  `OFFICIAL_FAIL` stands; the holdout is now opened, so any further run is
  **development** until a fresh re-seal.
- **What a positive natural-degradation result requires (future work):**
  informative per-modality detectors (deep/patch features for RGB; a proper
  point-cloud detector for xyz) so per-modality AUROC clears chance pooled, plus
  a reliability signal that separates the classes. Even then, **T9** governs:
  the gate can only win in the genuine stress regime where degradation pulls CW
  below the Neyman–Pearson ceiling and opens recoverable headroom.

## Integrity notes

- The z-sigmoid fix makes the CW comparator *fairer/stronger* (de-saturated),
  i.e. harder for our own method — a correction, not goalpost-moving.
- No test labels were used for method or category selection; selection remains
  validation-only per D16.

## Artifacts
- Fix: `src/scripts/scenario_c/prepare_realiad_positive_transfer.py::_score_features`
- D13 inputs regenerated: `experiments/fusion/realiad_256_c1_c2_d13_{inputs,sar_scores,cw_scores}.csv`
- Archived multimodal fail: `experiments/fusion/realiad_d3_headroom_audit_result.json`
- Decision: D17 in `research_lock/DECISIONS_v1.md`
