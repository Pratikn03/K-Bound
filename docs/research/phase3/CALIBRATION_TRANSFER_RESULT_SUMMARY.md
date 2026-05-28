# Calibration-Transfer Result Summary

Status: completed (based on existing one-time Phase-2 v3 execution + refreshed inference)

## Primary outcome

- Family decision: FAMILY_D_V3_NOT_CONFIRMED
- D-EYE-1 decision: NOT_CONFIRMED
- D-EYE-2 decision: NOT_CONFIRMED

## Key metrics

- Clean false-fire rate on validation: see per-seed calibration logs under experiments/phase2/family_d/
- D-EYE-1 Delta AUC (RGA - static) with CI: -0.0010, 95% CI [-0.0114, +0.0092]
- D-EYE-2 Delta AUC (RGA - static) with CI: -0.0109, 95% CI [-0.0254, +0.0034]

## Mandatory calibration provenance disclosure

1. Data used to fit KS/calibration reference:
   - Train split for initial reliability estimator fit; validation split used for KS reference re-fit in v3 protocol.
2. Confirmation that only normal validation data were used:
   - Yes. Calibration reference and threshold selection policy are validation-only and normal-only by contract.
3. Confirmation that no test scores/labels/distributions influenced selection:
   - Confirmed by selection policy and archived selection flag requirement: selection_used_test_metrics = False.
4. Confirmation that calibration and threshold policy were frozen before outcomes were read:
   - Yes. Frozen artifacts and hashes recorded in manifests before evaluation outputs.

## Validity classification

- Held-out confirmatory transfer evidence: not obtained
- Exploratory post-test repair evidence: not indicated by current manifests; negative held-out outcome retained

## Claim ceiling

- Allowed claim text: Held-out confirmation was not obtained for evaluated endpoints; negative results are retained.
- Forbidden claims checked: ELARA universal/SOTA/deployment-safe and related over-claims remain disallowed.