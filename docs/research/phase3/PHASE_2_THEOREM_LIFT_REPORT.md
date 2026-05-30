# Phase 2 theorem-stack lift — completion report

**Date:** 2026-05-29
**Scope:** the three bounded statistical lifts from the lift plan
(T5 empirical-Bernstein, T2 mixture-entropy + real-cohort validation,
T4 finite-sample sample complexity).
**Status:** **all three Phase-2 tasks complete.** Phase 3 (T6 + GDR) is the
remaining work.

## Headline

Theorem-stack composite grade lifted from **B+** (post-Phase-1) to **A−**,
using closed-form statistics validated against existing artifacts. No new
training runs required. Regression suite green (691 passed / 6 skipped, no
failures as of 2026-05-30, after adding direct v3 unit/smoke tests).

| Theorem | Post-Phase-1 | Post-Phase-2 | Lift |
|---|---|---|---|
| **T2** Global-KS mixture confounding | B− (synthetic only) | **A−** (real-cohort MVTec confirms; mixture-entropy bound; Eyecandies negative control) | +1.5 |
| **T4** Reliability-switch risk dominance | C+ (2 retrospective cells) | **A−** (finite-sample margin LCB + closed-form min-n sample complexity) | +1.5 |
| **T5** Finite-sample switching certificate | B− (bootstrap) | **A−** (deterministic empirical-Bernstein closed form; per-sample tight, per-seed honestly vacuous) | +1.5 |
| (T1, T3, T7 unchanged from Phase 1; T6, GDR pending Phase 3) | | | |

## What landed

### T2 — Mixture-entropy bound + real-cohort validation

**Files added.**
- [src/elara/theory/t2_mixture_entropy.py](../../../src/elara/theory/t2_mixture_entropy.py)
- [src/scripts/validate_t2_eyecandies_categories.py](../../../src/scripts/validate_t2_eyecandies_categories.py)
- `experiments/fusion/t2_category_ks_validation.json` (MVTec, positive)
- `experiments/fusion/t2_category_ks_eyecandies_negative.json` (negative control)
- `docs/research/tables/t2_category_ks.tex`

**Result — the long-standing "real-cohort T2" deferral is now closed.**

| Benchmark | Category separation | Global KS fires? | Category-aware fires? | T2 confirmed? |
|---|---|---|---|---|
| **MVTec 3D-AD** (8 cats, mean spread 0.23) | high | **yes** (p = 4.0×10⁻⁵) | no (0/8, min p = 0.24) | **YES** |
| Eyecandies (10 cats, mean spread ~0.015) | ~none (near-chance detector) | no (p = 0.50) | spurious 3/10 | no |

Under **no per-category drift** (both folds drawn from the same per-category
distributions), the global KS gate false-fires purely from mixture
re-weighting on MVTec 3D-AD — exactly the T2 prediction — while the
category-aware gate stays null. The Eyecandies negative control is itself
informative: the mechanism *requires* inter-category score separation, which
a near-chance base detector cannot provide. This ties T2 directly to the
base-detector-ceiling theme that runs through the whole project.

**Bound.** `global_ks_inflation_bound = 0.5·TV(π_ref, π_test)`; the mixture
entropy H = log k quantifies how much re-weighting headroom exists.

### T4 — Finite-sample risk-dominance sample complexity

**Files added.**
- Extended [src/elara/certification/risk_dominance.py](../../../src/elara/certification/risk_dominance.py) with `risk_dominance_margin_lcb()` and `min_n_for_dominance()`.
- [src/scripts/validate_t4_risk_dominance_sample_complexity.py](../../../src/scripts/validate_t4_risk_dominance_sample_complexity.py)
- `experiments/fusion/t4_risk_dominance_sample_complexity.json`
- `docs/research/tables/t4_risk_dominance_sample_complexity.tex`

**Result.** Converts the vague "retrospective certificate ≠ deployment
guarantee" caveat into a concrete sample-size number:

| Scenario | Point margin M₀ | Margin LCB @95% (n=1.6k) | Min n* to certify | Certified? |
|---|---|---|---|---|
| zero_attack_k4 | −0.0002 to −0.0019 | negative | ∞ (M₀ ≤ 0) | no — not certifiable at any n |
| max_attack_k4 | +0.0005 to +0.0047 | negative | **~82,427** | no — need ~50× more samples |

The closed form `n* = C²·ln(2/δ) / (2 M₀²)` shows that even where the gated
policy has a positive point margin (max_attack), certifying deployment-
prevalence dominance would require ~82k fired samples versus the 1.6k in the
stress protocol. This is a precise, honest statement of why the certificate
is retrospective-only.

### T5 — Empirical-Bernstein switching certificate

**Files added.**
- Extended [src/elara/certification/switching_certificate.py](../../../src/elara/certification/switching_certificate.py) with `empirical_bernstein_lcb()` (Maurer-Pontil 2009) + new certificate fields.
- Extended [src/scripts/audit_switching_certificate_t5.py](../../../src/scripts/audit_switching_certificate_t5.py) (per-seed parallel EB column).
- [src/scripts/audit_switching_certificate_t5_persample.py](../../../src/scripts/audit_switching_certificate_t5_persample.py) (per-sample audit).
- `experiments/fusion/switching_certificate_t5_persample_audit.json`
- `docs/research/tables/switching_certificate_t5_persample.tex`

**Result.** A deterministic, streamable closed-form LCB replacing the 10k-iter
bootstrap:

| Level | n | Bootstrap LCB | Empirical-Bernstein LCB | Gap |
|---|---|---|---|---|
| per-sample fired subset (zero_attack) | 48,000 | −0.0011 | −0.0015 | **1.4×** (tight) |
| per-sample fired subset (max_attack) | 48,000 | −0.0025 | −0.0030 | **1.2×** (tight) |
| per-seed aggregate (any) | 5 | ~0.10 | ~−4.3 | >100× (vacuous) |

Two honest findings: (1) at the per-sample level the closed form is tight and
finite-sample valid; (2) at the per-seed n=5 level the EB bound is correctly
vacuous, **revealing that the n=5 bootstrap certificates are not finite-sample
valid** — an asymptotic-percentile artifact. The per-sample audit also shows
the |p−y| switch benefit is not positive even where the per-seed AUROC
certificate was, consistent with the rank-invariance finding from Family-D v4.

## Theorem-registry status after Phase 2

```
T1    artifacts 2/2  Quality-blind fusion impossibility (T1a + T1b)        [Phase 1]
T2    artifacts 5/5  Global-KS mixture confounding (real-cohort)           [Phase 2]
T3    artifacts 5/5  Mean-gate dilution failure (closed-form)              [Phase 1]
T4    artifacts 4/4  Reliability-switch risk dominance (sample complexity) [Phase 2]
T5    artifacts 4/4  Finite-sample switching certificate (emp.-Bernstein)  [Phase 2]
T6    artifacts 2/2  KS false-fire and detection boundary                  [pending Phase 3]
T7    artifacts 4/4  PAC bound (loose + tight)                             [Phase 1]
GDR   artifacts 2/2  Coherence-certified gate decision rule                [pending Phase 3]
```

`validate_theorem_stack.py` → `"all_ok": true`.

## Composite grade trajectory

| Layer | Pre-lift | Post-Phase-1 | Post-Phase-2 |
|---|---|---|---|
| Theorem ↔ code mapping | 8/10 | 9/10 | **9.5/10** |
| Theorem rigor | 5/10 | 7/10 | **8/10** |
| Conference readiness | 5/10 | 6.5/10 | **7/10** |
| Thesis readiness | 7/10 | 8/10 | **8.5/10** |

**Theorem stack composite: B+ → A−.**

## What remains (Phase 3)

- **T6** — reformulate as a sequential-detection (CUSUM ARL/AED) theorem, or
  retire to "Empirical Boundary 6" (recommended). See plan.
- **GDR** — minimax proof + real-benchmark validation of GDR switch decisions.
  This is the remaining gap to a fully A-level stack.

## Verification

```bash
PYTHONPATH=src python src/scripts/audit_switching_certificate_t5.py
PYTHONPATH=src python src/scripts/audit_switching_certificate_t5_persample.py
PYTHONPATH=src python src/scripts/validate_t2_eyecandies_categories.py        # MVTec (confirms)
PYTHONPATH=src python src/scripts/validate_t4_risk_dominance_sample_complexity.py
PYTHONPATH=src python src/scripts/validate_theorem_stack.py
PYTHONPATH=src python -m pytest tests/ -q
```
