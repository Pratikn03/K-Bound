# Positive-Transfer Claim — Independent Audit (2026-06-01)

A parallel workstream produced `run_positive_transfer_confirmatory.py`,
`residual_stack` cross-modal fusion, a new MulSen-AD dataset, and a
`positive_transfer_development_report.json` reporting RGA beating CW and SAR on
3D-ADAM. This document is the honest audit of that claim, covering **(B)** a
leakage/validity check of the 3D-ADAM result and **(A)** a fix-and-retry of the
near-chance MulSen confirmation.

The report itself is marked `status: DEVELOPMENT_ONLY`, `cannot_set_gate_e: true`.
This audit explains *why that is the correct status* and surfaces one bug.

## B — 3D-ADAM positive transfer does NOT survive audit

**No code leakage.** `residual_stack` is selected on validation labels only
(`used_test_labels=False`), and test scoring applies the frozen validation
coefficients via `_linear_logistic_score` — verified by reading
`positive_transfer.py`. The candidate test AUROC (0.9304) reproduces exactly in
an independent re-run.

**But the "win vs CW" used a mislabeled comparator.** The report's
`vs_cw.comparator_auc = 0.9123` is **exactly RGB-only AUROC (0.9123)**, not a
confidence-weighted mean. The true parameter-free baselines are:

| Baseline | Test AUROC |
|---|---|
| RGB-only | 0.9123  ← report called this "CW" |
| depth/IR-only | 0.9283 |
| **plain mean 0.5(rgb+depth)** | **0.9349** |
| confidence-weighted mean | 0.9353 |
| residual_stack (candidate) | 0.9304 |

Against the **actual** confidence-weighted / plain-mean baseline (0.9349),
residual_stack scores **0.9304 → Δ = −0.0044, 95% CI [−0.0126, +0.0037]** — it
**loses / ties, CI crosses zero.** The reported "+0.018 vs CW" was
residual_stack beating *RGB-alone dressed up as CW*.

**Verdict B: the 3D-ADAM positive-transfer win is not real.** It is an artifact
of comparing against a weaker (RGB-only) baseline. The honest result is
unchanged from before: on clean 3D-ADAM transfer, a simple average is the best
parameter-free method and no cross-modal rule beats it.

### Bug to fix
`run_positive_transfer_confirmatory.py` accepts a `--cw-scores` file and, when
given an RGB-only score file, labels it "CW". The comparator must be the genuine
confidence-weighted (or plain-mean) fusion of the two modalities, never a single
modality. Recommend: compute CW internally from (rgb, depth) and forbid an
external single-modality file as the "CW" comparator.

## A — MulSen confirmation: a real bug fixed, but transfer still fails honestly

**Bug found and fixed (same class as the MVTec depth-codec bug).** MulSen test
scores were a **constant 1.0** (AUROC 0.500) because `patchcore_knn_score` used
min-max-clip normalisation against the 95th train percentile, saturating every
out-of-range test score. Validation AUROC was fine (rgb 0.93, IR 0.90),
confirming the detector itself works. Fixed by switching to a **monotonic
z-sigmoid** normalisation (preserves ranking; same fix applied to MVTec v3).

**After the fix, test scores vary (nunique 276/278) — but test AUROC stays
near chance:**

| Split | RGB | Infrared |
|---|---|---|
| validation | 0.930 | 0.896 |
| **test (held-out categories)** | **0.519** | **0.541** |

Per-category test AUROC is wildly scattered (0.04–0.95; several *inverted*
below 0.5), e.g. RGB screen 0.041, IR plastic_cylinder 0.132. This is a genuine
**cross-category one-class generalisation gap**: a memory bank built on 8 train
categories does not describe the 7 held-out test categories. It is **not** a
normalisation artifact and is **not** fixable by re-normalising — it is the
fundamental hardness of cross-category transfer for a kNN-to-memory-bank
detector.

**Verdict A: MulSen does NOT provide a positive-transfer confirmation.** The
detector cannot generalise to the unseen categories, so no fusion rule on top of
it can produce a meaningful transfer win. (The normalisation fix is still
correct and is retained for any in-distribution use.)

## Combined honest verdict

- **3D-ADAM:** no real win (comparator was mislabeled RGB-only).
- **MulSen:** detector fails to transfer across categories (real OOD gap).
- **Gate E remains FAIL.** The positive-transfer claim does not hold under audit.

This is consistent with the prior diagnosis: positive *clean* transfer needs a
dataset whose modalities stay genuinely complementary on the held-out test fold
AND a detector that generalises to the held-out categories. Neither 3D-ADAM
(redundant clean modalities) nor MulSen (cross-category OOD) satisfies both.

**The standing honest result is unchanged:** reliability gating is a
stress-regime mechanism (in-domain Gate D/T5 win + degradation recovery); clean
held-out transfer superiority over the strongest parameter-free baseline is
**not** established. No goalpost-moving; D7 stands.

## Artifacts
- `experiments/fusion/m2_external_mulsen_v2fix_inputs.csv` (normalisation-fixed)
- `elara_master_c/audits/positive_transfer_development_report.json` (DEVELOPMENT_ONLY, as the runner correctly marked it)
- normalisation fix: `src/uais/fusion/attention/m3dm_features.py::patchcore_knn_score`
