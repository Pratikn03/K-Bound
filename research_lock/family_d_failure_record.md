# family_d_failure_record.md — Sealed Failed Transfer Evidence

This is the permanent, honest record of the Eyecandies held-out transfer
**failure**. It is sealed so that no future experiment can quietly convert it
into a "confirmed transfer" by re-tuning on the same test set.

## What was attempted

- **Goal**: confirm that ELARA's reliability-aware fusion benefit transfers to
  an unseen naturally paired RGB+depth dataset (Eyecandies) under a
  validation-only-calibrated one-class multimodal protocol.
- **Authoritative report**: `docs/research/phase2/FAMILY_D_V3_INFERENCE_REPORT.md`
- **Final decision**: `docs/research/phase2/FAMILY_D_V3_FINAL_DECISION_AUDITED.md`

## Result (NOT CONFIRMED)

| Cell | ΔAUC (ensemble) | bootstrap 95% CI | paired-t p | clean false-fire (budget ≤0.010) |
| --- | --- | --- | --- | --- |
| D-EYE-1 | −0.0010 | [−0.0114, 0.0092] | 0.3632 | 0.000 |
| D-EYE-2 | −0.0109 | [−0.0254, 0.0034] | 0.4468 | 0.000 |

The clean false-fire budget was met, but the paired tests were non-significant
and the bootstrap CIs include zero. **Transfer is not confirmed.**

## Diagnosed root cause (frozen)

Calibration transfer under score-distribution shift: a reliability gate
calibrated on the source validation distribution does not automatically preserve
its performance benefit on Eyecandies' shifted score distribution. This is the
central unresolved limitation and the motivation for the Phase-3 calibration-
transfer theorem (claim `C_P5_THEORY_TRANSFER`) and the gate decision rule
(`docs/research/phase3/GATE_DECISION_RULE.md`).

## Sealing policy

- **Default (Policy A)**: Eyecandies stays a **sealed FAILED external test**. It
  must not be used for model selection or gate tuning. Honest failed-transfer
  evidence remains in the paper.
- **Alternative (Policy B)**: reclassify Eyecandies as **development** to study
  why transfer failed — but then a **new untouched** naturally paired RGB+depth
  dataset must be acquired for the final transfer claim, and the paper must
  explicitly disclose that Eyecandies changed from confirmation to development
  after the initial failed result.

Decision is **D1** in `SCENARIO_C_CLAIM_CONTRACT.md` and is currently **Policy A
(sealed FAILED)** until the user ratifies otherwise.

## Prohibition

Re-tuning any gate on the D-EYE-1 / D-EYE-2 test partitions and reporting the
outcome as held-out confirmation is **prohibited** and invalidates the result.
