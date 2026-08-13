# Pre-registration: a track where always-adapt visibly loses and KGA does not

**Status: pre-registered, not yet run.** Written before any run so that the analysis
cannot be chosen after seeing the outcome (A6). Power analysis:
`research_lock/CONDITION_UNIT_POWER_v1.json`.

## Why the existing tracks cannot supply this

At the corruption-family unit -- the defensible one under (A5) -- only Tent clears:

| track | effect | sd across conditions | d | n needed | n available |
|---|---|---|---|---|---|
| CIFAR-10-C Tent | −0.00635 | 0.00478 | 1.33 | 5 | 6 | 
| CIFAR-10-C EATA | −0.00200 | 0.00329 | 0.61 | 22 | 6 |
| ImageNet-C SAR | −0.02651 | 0.13030 | 0.20 | 190 | 27 |

EATA needs 22 independent families against a 15-family ceiling; ImageNet-C SAR needs
190. Neither is recoverable by more seeds or more compute, and expanding the SAR grid
from 3 families to all 15 would not close a 190-family requirement. **Both should move
permanently to no-harm rather than being left as pending.**

## The design constraint, which is the non-obvious part

The obstruction is not effect size. SAR's mean gap is 4.2x Tent's; its across-condition
standard deviation is 27.3x Tent's. Writing the ratio sd/|effect|: Tent 0.75, EATA 1.65,
SAR 4.91.

So: **visible harm is concentrated harm, and concentration is exactly what destroys
power at the condition unit.** A track where always-adapt collapses on three corruptions
and is fine on the rest cannot demonstrate beats-both at the correct resampling unit, no
matter how large the collapse. This is a real tension between the result being
*striking* and the result being *significant at the honest unit*.

The requirement is therefore harm that is large **and uniform**: on a 15-condition
benchmark, `d >= 0.72`, i.e. across-condition sd at most 1.38x the mean gap.

Note also that harmful-*dominated* regimes are the wrong target. There KGA correctly
freezes and merely ties always-freeze, so beats-both is unreachable and the guarantee
goes untested -- exactly what ImageNet-C Tent shows at a 56% harmful base rate with zero
adapt decisions.

## Proposed regime: label shift as a uniform harm mechanism

Entropy minimisation reinforces the majority class under label-shifted batches. This is
a documented TTA failure mode, motivated independently of K-Bound, and -- unlike
corruption-specific collapse -- it acts on every corruption family alike. That is what
makes it a candidate for large-and-uniform harm.

Construction: resample target batches from the existing CIFAR-10-C per-cell data at a
declared class-imbalance ratio, applied identically to all 15 corruption families. No new
model training; the adapters and evidence map are unchanged and frozen.

## Declared in advance

- **Unit of inference:** corruption family. n = 15.
- **Calibration:** leave-one-corruption-out. Not leave-one-cell-out.
- **alpha:** 0.10. Estimator, evidence map, radius construction and decision rule frozen
  from the existing protocol; no refitting on the new regime.
- **Imbalance ratio:** declared once, before any run, and not swept.
- **Primary endpoint:** paired bootstrap on the gap vs always-adapt AND vs always-freeze,
  resampled at the corruption family, both 95% CIs excluding zero.
- **Secondary:** FA_u, adapt rate, decision coverage, regret-to-oracle.
- **Failure is reported.** If either CI includes zero, the track is reported as no-harm
  or null and does not become a beats-both claim. If KGA abstains on most conditions,
  that is reported as guarantee-untested, not as a safety success.
- **No post hoc subsetting.** No selection of corruption families, severities, or seeds
  after seeing outcomes. The pre-registered n is 15 and all 15 are reported.

## What a negative outcome would mean

If harm cannot be made both large and uniform, that is itself the finding, and it belongs
in the paper: it would say the regimes where adaptation is dangerous are precisely the
regimes where its danger is condition-specific, and that condition-specific danger cannot
be certified at the condition unit with any realistic number of conditions. That is a
sharper limitation than the paper currently states, and it is worth stating.
