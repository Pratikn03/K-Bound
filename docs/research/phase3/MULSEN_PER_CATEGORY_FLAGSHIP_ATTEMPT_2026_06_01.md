# MulSen Per-Category Flagship Attempt (2026-06-01)

Attempt to achieve official positive transfer (Gate E) on MulSen-AD (RGB +
infrared, genuinely complementary modalities) by fixing the cross-category
generalization failure. Outcome: **the detector fix is a major real success;
Gate E still does not pass — for a newly-understood, deeper reason (ceiling).**

## What was fixed (genuine engineering win)

Two real bugs in the MulSen pipeline:

1. **Flat train-good discovery.** MulSen stores normal images directly under
   `<cat>/RGB/train/*.png` (not under a `good/` subdir). The builder only looked
   for a `good/` subdir, so **every test category's own normals were never
   loaded** -> the test categories had no reference bank.
2. **Pooled vs per-category memory bank.** Scores were computed against ONE
   pooled bank of all train categories. A screw scored against a bank of buttons
   is meaningless. Added `patchcore_knn_score_per_category` +
   `--per-category-bank`: each category is scored against its OWN train-good
   normals with per-category z-sigmoid normalisation.

**Result — the detector now generalizes across held-out categories:**

| Modality | pooled bank (before) | per-category bank (after) |
|---|---|---|
| test RGB | 0.52 (chance) | **0.980** |
| test infrared | 0.54 (chance) | **0.979** |

Per-category test AUROC 0.94-1.00 across all 7 held-out categories. This is a
real, reusable fix for cross-category one-class transfer.

## The complementarity precondition IS satisfied (first time)

On clean MulSen test, the mean (0.9985) **beats both single modalities**
(RGB 0.980, IR 0.979) by +0.018, with cross-modal correlation +0.57 -- i.e.
RGB and infrared genuinely catch *different* anomalies. This is the
complementary structure 3D-ADAM lacked (3D-ADAM clean modalities were redundant,
corr +0.78, mean did not exceed the best single).

## But Gate E STILL does not pass — the ceiling effect

| Comparison | Result |
|---|---|
| residual_stack (val-selected) vs CW(mean) | Δ = **−0.0002**, CI [−0.0004, +0.0000] -> does NOT beat CW |

**Why:** the confidence-weighted mean already reaches **0.9985** -- essentially
perfect. The maximum possible gain over the mean is **0.0015**. There is no
headroom for any fusion method to win. CW captured the complementarity so
completely that nothing can beat it by a significant margin.

## The unified structural lesson (this is the real finding)

Clean-transfer Gate E is **unwinnable on both datasets, for two different
reasons that reduce to one principle:**

- **3D-ADAM:** modalities REDUNDANT on clean -> mean is optimal -> no gate win.
- **MulSen:** modalities COMPLEMENTARY but mean already ~1.0 (ceiling) -> no
  headroom -> no gate win.

**Principle:** on CLEAN data, a simple average is either optimal (redundant) or
near-ceiling (complementary-but-easy). A reliability gate can only help when
there is (a) genuine complementarity AND (b) enough difficulty that the mean is
far from ceiling AND (c) a reliability signal to exploit. Clean held-out
transfer rarely provides all three -- which is *why* the gate's value is the
STRESS regime, where degradation creates the difficulty and the reliability
signal simultaneously.

This is a stronger, more general version of the earlier diagnosis, now confirmed
on a complementary dataset. It is a publishable scientific finding in itself:
**reliability-gated fusion cannot beat averaging on clean transfer when the
average is already near-ceiling, regardless of modality complementarity.**

## Verdict

- **Detector fix: SUCCESS** (cross-category transfer 0.52 -> 0.98, reusable).
- **Gate E / flagship: STILL NOT ACHIEVED** -- ceiling effect, not a fixable bug.
- **Honest:** flagship via clean-transfer Gate E is not achievable on the
  available datasets. The standing result (stress-regime mechanism) is
  reinforced, and now backed by a complementary-dataset confirmation that the
  clean-transfer ceiling -- not redundancy alone -- is the fundamental barrier.

## Artifacts
- `experiments/fusion/m2_external_mulsen_percat_inputs.csv` (per-category, fixed)
- `src/uais/fusion/attention/m3dm_features.py::patchcore_knn_score_per_category`
- `src/scripts/prepare_mulsen_fusion_benchmark.py` (flat-good discovery + flag)
