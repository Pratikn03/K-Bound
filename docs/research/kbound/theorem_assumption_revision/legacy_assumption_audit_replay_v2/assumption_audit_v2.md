# Corrected legacy assumption diagnostic replay

This posthoc correction preserves the original v1 artifacts. Their guarantee and ADAPT recommendations are not valid scientific authority.

- **benign_transfer**: `not_falsified` → action `abstain`; guarantee `unresolved`
- **evidence_support_shift**: `not_falsified` → action `abstain`; guarantee `unresolved`
- **residual_drift**: `warning` → action `abstain`; guarantee `does_not_apply`
- **concept_shift_witness**: `not_falsified` → action `abstain`; guarantee `unresolved`
- **mild_helpful_shift**: `not_falsified` → action `abstain`; guarantee `unresolved`
- **low_margin_shift**: `not_falsified` → action `abstain`; guarantee `unresolved`

## Scope limitations

- The historical concept_shift_witness passes the same shifted Z as evidence_support_shift; it does not construct opposite-benefit matched-evidence worlds.
- The mild_helpful_shift and low_margin_shift labels are not supported by benefit inputs in this generator; no benefit or margin test was performed.
- The feature-range diagnostic does not warn on the recorded one-feature shift; failure to detect that shift is retained, not converted into a pass of the intended safeguard.
- Residual drift uses simulated residual arrays, not a new measured target study.
- No-warning diagnostics leave exchangeability, risk alignment and coverage unresolved.
