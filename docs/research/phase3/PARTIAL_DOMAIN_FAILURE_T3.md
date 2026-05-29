# T3 — Partial (k-of-D) Domain-Failure Boundary

Status: controlled study. **NEW EXPLORATORY** (development/synthetic only; not a
confirmatory claim; touches no sealed/final test set).
Code: `src/scripts/run_partial_domain_failure_study.py`
Result: `output/phase4/partial_domain_failure_study.json`
Test: `tests/test_partial_domain_failure_study.py`

## 1. The gap this addresses

ELARA's strongest base-RGA evidence (Family B) is **coherent ALL-domain
collapse**. Scenario C requires moving beyond that to realistic **partial**
failures (1-of-D, 2-of-D, ...). The locked concern is **T3 (mean-gate
dilution)**: a single failed domain among D may not drag the batch mean
reliability below the gate threshold, so the hard gate never fires and the
isolated failure is silently fused in. The RGA-v2 sensitive gates (G1/G2/G3)
tried to fix this and instead exploded clean false-fire (FFR 1.000), so they
were rejected.

## 2. Study

D=8 synthetic domains. The estimator is fit on a clean **fit** split (KS
reference + ECE); the gate threshold tau is selected on a **separate clean
calibration split** to a 1% clean false-fire budget (validation-only; no
test-driven tuning). For each k we corrupt k domains into label-uncorrelated
noise and compare:

- `static` — unweighted mean fusion;
- `soft_rga` — per-domain reliability-weighted fusion (always applied);
- `hard_gate_g0` — fire (route a sample to the reliability path) only if its
  mean reliability < tau.

### Result (seed 0, tau selected on calibration ≈ 0.564, clean false-fire 0.000)

| k failed | static AUC | Δ soft vs static | Δ hard vs static | gate fire rate |
| --- | --- | --- | --- | --- |
| 0 (clean) | 0.9985 | +0.0001 | 0.0000 | 0.000 |
| 1 | 0.9967 | +0.0011 | 0.0000 | 0.000 |
| 2 | 0.9792 | **+0.0120** | 0.0000 | 0.000 |
| 3 | 0.9534 | +0.0179 | +0.0179 | 1.000 |
| 4 | 0.8762 | +0.0325 | +0.0325 | 1.000 |
| 5 | 0.8083 | +0.0263 | +0.0263 | 1.000 |
| 6 | 0.7344 | +0.0831 | +0.0831 | 1.000 |
| 7 | 0.5877 | +0.0659 | +0.0659 | 1.000 |
| 8 (all) | 0.4942 | 0.0000 | 0.0000 | 1.000 |

## 3. Findings

1. **Clean quietness holds**: at k=0 the gate does not fire (false-fire 0.000)
   and neither policy moves AUC — base RGA is quiet on clean evidence (P1).
2. **The hard mean gate is diluted at low k (T3 confirmed)**: at k=2 the hard
   gate does NOT fire (fire rate 0.000) and gives **zero** benefit, while
   per-domain soft weighting already recovers **+0.012**. The hard gate only
   begins to help at k=3, once enough domains fail to drag the mean below tau.
3. **Partial-failure handling is possible without sensitive gates**: per-domain
   soft reliability weighting handles partial failures (benefit from k=2)
   without the clean false-fire explosion that sank the RGA-v2 G1/G2/G3 gates.
4. **All-domain collapse is the easy-but-empty extreme**: at k=8 there is no
   signal left, so both policies are neutral — large-k "success" is not
   evidence of partial-failure competence; the interesting regime is small k.

## 4. Implication for the gate design (Phase 3)

The mean batch gate is the wrong instrument for partial failures; the benefit
comes from **per-domain reliability weighting** (and, by extension, the
`PerSampleReliabilityEstimator`). The recommended final gate should apply
reliability **per domain** rather than as a single batch fire/no-fire decision,
while keeping the validation-selected false-fire budget that the soft policy
respects here (clean false-fire 0.000).

## 5. Honest limitations / promotion path

- Synthetic, development-only (`NEW EXPLORATORY`). The corruption is a detectable
  noise collapse; confident-but-uncorrelated corruption (see
  `CALIBRATION_TRANSFER_CONDITION_T6.md`) is a harder, separate regime.
- The exact k-boundary (soft helps at k=2, hard at k=3) is a property of D=8 and
  the synthetic separability; the qualitative ordering (soft helps no later than
  hard; hard diluted at low k) is the transferable claim.
- To promote: reproduce the k-of-D sweep on the real MVTec 3D-AD paired data with
  strong upstream experts (Phase 2), under the frozen protocol, comparing
  per-domain weighting vs the batch gate vs the strongest baseline.
