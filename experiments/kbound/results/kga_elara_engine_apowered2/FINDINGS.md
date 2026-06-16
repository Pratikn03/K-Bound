# KGA over the REAL ELARA engine (A-POWERED-2 held-out category): HONEST NULL

*Cache: `experiments/phase2/predictions/A-POWERED-2__MVTec_3D-AD__PatchCore_held-out_category`.
12 real engine methods (incl. cross-attention `craf_attention`/`static_attention`, meta-router
`rga_*`, late/early fusion, RF, + tent/eata/sar/ttt score adapters) x validation+test x 30 seeds
(42-71). Design pre-registered in the script docstring; α=0.10. Run 2026-06-15.*

## Headline
This is the genuinely untapped lever (the full cross-attention engine, not the reliability proxy),
and run honestly it returns a **NULL, not a beats-both**. KGA **abstains on all 30 seeds**
(zero false-adapt). The reason is a real, verified phenomenon.

## The verified finding: total generalization collapse on the held-out category
| signal | value |
|---|---|
| validation AUROC (all 12 methods) | 0.76 – 0.998 (excellent) |
| **test AUROC (all 12 methods)** | **0.47 – 0.53 (chance)** |
| random_forest test AUROC across 10 seeds | 0.528, 0.520, 0.508, 0.509, 0.494, 0.517, 0.508, 0.512, 0.509, 0.501 |
| mean raw_score by label (test) | normal 0.882 vs anomaly 0.888 (barely separable) |

Verified **not** a label bug: test labels align perfectly with sample IDs (label 0 ⇔ `..._good_...`,
1339 normals; label 1 ⇔ defect types e.g. `..._color_...`, 342 anomalies). The collapse is real: an
engine trained on other MVTec-3D categories does **not** transfer to the held-out category (foam) —
near-perfect on validation, chance on test.

## KGA's behavior (correct)
With every candidate at chance, the per-sample placement benefits between any two candidates bracket
zero, so the certificate cannot certify any benefit and **abstains on 30/30 seeds** — zero
false-adapt. This is exactly the intended safety behavior under a total-collapse / weak-detectability
shift; there is simply no transferable signal to certify.

## Verdict
- **Not a multimodal beats-both.** The full engine does not rescue the result; it confirms the
  abstain-under-collapse behavior on the hardest real multimodal shift (held-out category).
- **Does not raise the ceiling.** Consistent with Camelyon17, ImageNet-R, iWildCam, Office-Home, and
  the D25 micro-probe negative: natural-shift / unseen-category regimes sit in the
  label-free-unidentifiable zone the impossibility theorem predicts.
- **Paper consequence:** no new claim. The existing multimodal section already states fusion ties /
  the certificate abstains; this is corroboration, not an upgrade.

## Data-quality caveat (logged)
`confidence_weighted_mean` **test** `raw_score` is a degenerate constant (1.0) — a serialization
artifact for that one method (its validation is fine). The other 11 methods have valid (non-constant)
test scores and all still collapse to chance, so the conclusion is robust to this artifact.
