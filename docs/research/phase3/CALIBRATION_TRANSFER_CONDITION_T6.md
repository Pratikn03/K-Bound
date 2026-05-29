# T6 — Calibration-Transfer Condition for Reliability-Aware Switching

Status: design + controlled study. **NEW EXPLORATORY** (development/synthetic
only; not a confirmatory claim and does not touch any sealed/final test set).
Code: `src/scripts/run_calibration_transfer_study.py`
Result: `output/phase3/calibration_transfer_study.json`
Test: `tests/test_calibration_transfer_study.py`

## 1. The question (why Family D failed)

Family D (Eyecandies) satisfied the clean false-fire budget yet did **not**
confirm transfer (D-EYE-1 Δ −0.0010 p=0.3632; D-EYE-2 Δ −0.0109 p=0.4468). The
locked diagnosis is *calibration transfer under score-distribution shift*: a
reliability gate calibrated on the source validation distribution may not stay
valid on a target with a shifted score distribution. T6 asks:

> Under what score-distribution shift does a validation-calibrated reliability
> estimate remain valid enough to still improve fusion on a new environment —
> and how do we detect, without target labels, when it does not?

## 2. Controlled study

A synthetic source domain (D=4) is used to fit the `ReliabilityEstimator`
(freezing the per-domain KS reference + ECE). The target collapses one domain
into a **confident-but-label-uncorrelated** stream (so the per-sample sharpness
term cannot catch it — only the KS-vs-source-reference term can) and then shifts
all domains by a transfer offset `tau`. We compare static mean fusion to
reliability-weighted fusion and record ΔAUC = AUC(gated) − AUC(static).

### Result (seed 0)

| tau | ΔAUC | regime | drift_coherence | target divergence | source certified |
| --- | --- | --- | --- | --- | --- |
| 0.0 | +0.0308 | HELP | 1.000 | 0.030 | False |
| 0.5 | −0.0395 | HURT | 1.000 | 0.159 | False |
| 1.0 | −0.0260 | HURT | 1.000 | 0.308 | False |
| 1.5 | −0.0105 | HURT | 1.000 | 0.458 | False |
| 2.0 | −0.0015 | NEUTRAL | 1.000 | 0.596 | False |
| 3.0 | −0.0000 | NEUTRAL | 1.000 | 0.770 | False |
| 4.0 | +0.0020 | NEUTRAL | 1.000 | 0.898 | False |
| 6.0 | +0.0033 | NEUTRAL | 1.000 | 0.988 | False |

The gate **helps** only at near-zero divergence; under modest drift it **hurts**
(reproducing Family D), then washes out to neutral once the signal is destroyed.

## 3. The condition (T6)

Let `R_d` be the frozen source reference distribution for domain `d` and `Q_d`
the target score distribution. Let `KS(R_d, Q_d)` be the two-sample KS distance.
Define the **target reference divergence**
`Δ_ref = mean_{d healthy} KS(R_d, Q_d)`.

**T6 (operational form).** The validation-calibrated gate's benefit transfers
(ΔAUC ≥ 0) only while `Δ_ref ≤ δ*`. Beyond `δ*` the KS term can no longer
discriminate the corrupted domain from healthy-but-drifted domains, the
reliability ranking inverts, and the gate becomes harmful. In the study,
`δ* ≈ 0.094` separates every HELP row from every HURT row.

**Assumptions.** (i) The corruption is only detectable through drift (not
sharpness/ECE); (ii) the transfer map is distribution-level (not adversarial);
(iii) healthy and corrupted domains share the same reference family.
**Failure case (assumptions violated).** If the corruption is also detectable by
sharpness/ECE, the gate stays robust to much larger shifts (a separate run with
a noise-style corruption shows ΔAUC preserved under monotone shift); T6 then
under-states robustness. T6 is therefore a *sufficient-caution* condition.

## 4. Honest limitation of the existing source-side signals

The two signals ELARA already has are **computed without any target
information** and are **constant across the shift**:

- `drift_coherence` = 1.000 for every `tau` (does not separate HELP from HURT).
- source-validation `bounded_switching_certificate` = `certified: False` for
  every `tau` (blind to the target).

So neither the coherence rule nor the source certificate can detect a
calibration-transfer failure. This is the key corrective finding: **a
source-only certificate is not a transfer-safety certificate.**

## 5. The T6 abstention rule (label-free)

Because `Δ_ref` needs only target scores and the frozen source reference (no
target labels), it is computable at deployment time. Rule:

```
if Δ_ref(target) > δ*:   abstain  -> fall back to static fusion
else:                    allow the reliability switch
```

In the study this abstention threshold covers **all** HURT rows, converting the
harmful region into a safe static fallback. This is the concrete deployment
mechanism for Pillar P6 (auditability) and the Phase-9 safe-failure path, and it
gives the Phase-3 gate (R4/R5 abstention candidates) a principled trigger.

## 6. How to promote this beyond exploratory

Per `research_lock/`, this synthetic result is `NEW EXPLORATORY`. To become
evidence it must be reproduced:
1. on the **reclassified Eyecandies development set** (decision D1 / Policy B):
   show `Δ_ref` separates the cells where the gate helps vs hurts;
2. then validated on the **new untouched RGB+depth transfer dataset** (decision
   D3) under the frozen protocol, with `δ*` selected on development only.

It must NOT be tuned on the sealed Eyecandies test partitions.
