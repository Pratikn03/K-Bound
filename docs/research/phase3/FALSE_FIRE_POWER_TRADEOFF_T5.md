# False-Fire vs Detection-Power Trade-off (Theorem T5)

**Status:** NEW EXPLORATORY (development/synthetic only; not confirmatory; touches
no sealed or final test set).

**Script:** `src/scripts/run_false_fire_power_study.py`
**Test:** `tests/test_false_fire_power_study.py`
**Result:** `output/phase8/false_fire_power_study.json`

## 1. Why this study

Every other study in this program selects the gate threshold `tau` as a quantile
of clean validation reliability tied to a **false-fire budget** `b` (T3
partial-domain, P6 temporal monitoring). T5 is the theorem that *justifies that
rule* and quantifies its cost. It is the **operating characteristic** of the
reliability gate — the analogue of an ROC curve for "should I switch?".

The master plan lists T5 (false-fire / power trade-off) as **Needed**: "Bound
clean switching cost versus degradation detection power." This study supplies the
controlled, label-free characterisation.

## 2. Claim (operational form)

Let the per-sample gate fire when mean reliability `r_i < tau`. Define

```
FFR(tau) = P(fire | clean)      -- clean false-fire (the COST)
TFR(tau) = P(fire | degraded)   -- detection power   (the POWER)
```

Both are CDFs of the per-sample mean reliability, so:

1. **FFR(tau) and TFR(tau) are monotone non-decreasing in tau.** You cannot raise
   detection power without also raising clean false-fire.
2. **TFR(tau) ≥ FFR(tau) for all tau iff reliability separates clean from
   degraded** (equivalently the detector ROC-AUC of `-r` against the
   clean/degraded label is > 0.5). The gap between the two curves *is* the
   usable power.
3. **Budget calibration.** Choosing `tau*(b)` as the `b`-quantile of clean
   validation reliability bounds the out-of-sample clean false-fire at ≈ `b`.
4. **Power ceiling.** Relaxing `b` buys strictly more detection power
   (`TFR(tau*(b))` is non-decreasing in `b`).
5. **Benefit is not free.** At very tight budgets the gate fires on too few
   degraded samples to overcome its clean false-fire cost, so net ΔAUC is ≈ 0 or
   slightly negative; net benefit becomes positive only once the budget admits
   enough detection power. This is the trade-off.

The per-sample reliability is computed with `PerSampleReliabilityEstimator` (the
deployment-grade variant); the batch-level estimator is intentionally **not** used
here because it returns a constant reliability per batch (detector AUC = 0.5,
nothing to trade off — the ISSUE-2 degeneracy).

## 3. Setup

- D = 6 domains, k = 3 collapsed (confident-but-uncorrelated) under degradation.
- Test stream: 50% clean / 50% degraded, anomaly labels preserved.
- `tau` swept over the quantiles of clean reliability for the continuous ROC.
- Budgets `b ∈ {0.005, 0.01, 0.02, 0.05, 0.10}` for the budget→power/benefit curve,
  with `tau*(b)` selected on a **disjoint** clean calibration split.

## 4. Result (seed 0)

```
detector ROC-AUC (clean vs degraded) = 0.828

  budget     tau*  test_FFR  test_TFR   a_stat    a_act     dAUC
   0.005    0.558     0.001     0.013   0.8922   0.8922  -0.0000
   0.010    0.568     0.006     0.036   0.8922   0.8921  -0.0001
   0.020    0.578     0.020     0.081   0.8922   0.8920  -0.0002
   0.050    0.592     0.042     0.222   0.8922   0.8925   0.0003
   0.100    0.604     0.089     0.417   0.8922   0.8935   0.0013

FFR monotone in tau: True   TFR monotone in tau: True   power >= cost: True
detection power monotone in budget: True   max |FFR - budget| = 0.011
dAUC tightest budget = -0.0000   loosest budget = +0.0013
```

Reading:

- **Detector AUC 0.828** — per-sample reliability genuinely separates clean from
  degraded samples (the batch-level estimator gives 0.5).
- **The budget is honoured out-of-sample**: declared `b` vs realised clean
  false-fire matches to within 0.011 across the whole sweep.
- **Power is bought with budget**: detection rises 0.013 → 0.417 as the budget
  relaxes 0.5% → 10%, while clean false-fire stays at/under budget.
- **Benefit is bounded by the budget**: at 0.5–2% budgets ΔAUC is ≈ 0 / slightly
  negative (the gate fires on too few degraded samples to pay for its clean
  false-fires); by 10% budget ΔAUC is positive (+0.0013). The benefit is small
  here by construction — the point of T5 is the *shape* of the trade-off, not the
  magnitude on any one synthetic stream.

## 5. Consequences for the program

- The budget-quantile `tau` rule used in T3 and P6 is **the right rule**: it is
  the only one that controls clean false-fire out-of-sample, and it sits on the
  achievable cost/power frontier.
- "Lower `tau` is safer" is made precise: safety (low FFR) and power (high TFR)
  are co-monotone, so the operating point must be chosen by the **declared
  budget**, never by maximising degraded AUC (which would silently inflate clean
  false-fire).
- It explains why a 1% budget can look "neutral" on mild degradations: the
  achievable benefit at 1% is power-limited. Larger gains require either a higher
  budget (worse clean behaviour) or a more separable reliability signal (higher
  detector AUC), not a different threshold at the same budget.

## 6. Pass criteria (locked by the test)

1. FFR(tau) and TFR(tau) are monotone non-decreasing in `tau`.
2. The detector separates clean from degraded (ROC-AUC > 0.5) and TFR ≥ FFR
   everywhere.
3. Out-of-sample clean false-fire tracks the declared budget (max error small).
4. Detection power at `tau*(b)` is monotone non-decreasing in `b`.
5. Net benefit is non-positive-bounded at the tightest budget and positive at the
   loosest (the trade-off is exhibited).

## 7. Scope and limits

Synthetic and exploratory. It characterises the *mechanism* of the cost/power
trade-off and validates the threshold-selection rule; it does not claim a
specific operating point or effect size for any real dataset. Magnitudes depend
on the separability of the upstream reliability signal and the degradation
severity.
