# Theorem-stack lift — Phase 2 and Phase 3 follow-up plan

**Phase 1 completed 2026-05-29** in the session that wrote this document. See
`PHASE_1_THEOREM_LIFT_REPORT.md` for the closed Phase-1 deliverables.

This plan covers Phase 2 (~2–3 days) and Phase 3 (~1+ week, possibly with
external data) — the work needed to lift the remaining theorems to A-level.
Each item is scoped as **a self-contained task** that the next session can
pick up cold.

---

## Phase 2 — bounded statistical lifts (3 tasks)

### Task P2.T2 — Mixture-entropy bound + real-Eyecandies-category KS validation

**Goal.** Lift T2 from "synthetic mixture confounding validated" to
"real-cohort cohort-mixture confounding measured, with a closed-form
false-fire bound as a function of mixture entropy".

**Files to produce.**
- `src/elara/theory/t2_mixture_entropy.py` — new module:
  - `false_fire_bound(entropy: float, n: int, alpha: float) -> float`
    closed-form for global-KS under k-component mixture.
  - `category_aware_ks_fire_rate(scores, categories, ref_scores) -> float`
    per-category KS variant.
- `src/scripts/validate_t2_eyecandies_categories.py` — runs per-category
  KS on the 10 Eyecandies categories (already in `experiments/fusion/
  eyecandies_inputs.csv`), produces:
  - JSON: per-category KS p-values
  - LaTeX: `docs/research/tables/t2_eyecandies_category_ks.tex`
- Update `theorem_registry.py` T2 entry.

**Math sketch.** Under a k-component mixture with weights pi_1, ..., pi_k
where each component has its own score distribution, the global KS test
falsely rejects with probability ≥ f(H, n) where H = -sum pi_i log pi_i
is the mixture entropy. The looser closed form: for large n, the
expected KS statistic of the mixture vs a single reference component
is O(sqrt(H * log(n)/n)) above the iid prediction.

**Validation target.** Eyecandies has 10 categories; run KS per category
vs the global reference. Predict per-category fire rate from category-
specific KS distance + the entropy of the test fold's category mixture.

**Effort.** ~4-6 hours.

### Task P2.T5 — Empirical-Bernstein replacement for the switching certificate

**Goal.** Replace the 10K-iter paired bootstrap LCB in T5 with a
closed-form **empirical-Bernstein** lower bound. Tighter when the fired-
subset paired-benefit variance is small. Streamable.

**Files to produce.**
- `src/elara/certification/switching_certificate.py` — add
  `bounded_switching_certificate_empirical_bernstein(...)` function.
- `src/scripts/audit_switching_certificate_t5.py` — add a parallel
  empirical-Bernstein LCB column alongside the bootstrap LCB.

**Math.** For paired benefits X_i ∈ [0, R] with mean mu and variance
sigma^2, the empirical-Bernstein lower bound is
  LCB = mu - sqrt(2 sigma^2 log(2/delta) / n) - 7 R log(2/delta) / (3(n-1))
which is tighter than Hoeffding when sigma^2 << R^2.

**Effort.** ~3-4 hours.

### Task P2.T4 — Finite-sample risk-dominance sample complexity

**Goal.** Lift T4 from "indifference-prevalence point estimate on 2
cells" to "with prob 1-delta, ELARA risk-dominates static if observed
paired benefit > c·sqrt(log(1/delta)/n)".

**Files to produce.**
- `src/elara/certification/risk_dominance.py` — add
  `risk_dominance_sample_complexity_lcb(...)` and
  `min_n_for_dominance(delta_target, alpha, sigma_paired) -> int`.

**Math.** Standard concentration; Hoeffding gives n ≥ 2 log(2/delta) /
Delta^2 where Delta is the practical-effect threshold. Empirical-
Bernstein variant for tighter result.

**Effort.** ~6-8 hours.

---

## Phase 3 — structural rework (2 large items)

### Task P3.T6 — Reformulate T6 as a sequential-detection theorem OR retire

**Decision required.** Pick one:

(a) **Reformulate.** Recast T6 as a CUSUM-style sequential-detection
    ARL/AED trade-off. The result has the form:
       "given drift rate r and reference noise sigma^2, CUSUM with
        threshold h achieves average detection delay AED(h, r) <=
        h/(r - log(1+r/sigma^2)) at the cost of average run length
        ARL(h) >= exp(h - log(1+h)) under H0."
    Requires CUSUM implementation, calibration on real drift streams.
    ~12+ hours.

(b) **Retire.** Move the existing locked-grid power sweep from "Theorem
    6" to "Empirical Boundary 6" in the paper. Replace the T6 slot with
    a different theorem candidate (e.g., a tighter false-fire bound for
    category-aware KS, building on Task P2.T2).
    ~2 hours documentation, with a separate decision on replacement.

**Recommendation.** Retire unless (a) is on the critical path for a
specific reviewer concern. The current paper section will read more
honestly as "Empirical Boundary 6" because that's what the locked-grid
sweep actually is.

### Task P3.GDR — Minimax proof + real-benchmark GDR prediction validation

**Goal.** Lift GDR from "B+ math, C empirical" to "B+ math + B+
empirical" by:
1. Proving GDR's switching policy attains the *minimax* risk over the
   union of (coherent shift, heterogeneous mixture) operator families.
2. Demonstrating on a real benchmark (e.g., the Master-C M2_external
   3D-ADAM cell) that GDR's switch *decisions* correlate with held-out
   improvement — i.e., when GDR says "switch", risk drops; when it says
   "keep static", static-only would have done as well or better.

**Files to produce.**
- `src/elara/theory/gdr_minimax.py` — minimax proof + numerical
  validation on synthetic operator family.
- `src/scripts/audit_gdr_real_benchmark.py` — load M2_external archives
  (already in `elara_master_c/predictions/confirmation/`), apply GDR
  per-sample, measure correlation between switch decision and the
  per-sample loss difference between RGA and static.
- LaTeX: `docs/research/tables/gdr_real_benchmark_validation.tex`.

**Effort.** Math ~6 hours. Empirical ~1-2 days. May need a held-out
benchmark we can re-train on if the current archives don't carry the
per-sample reliability signals GDR needs.

---

## Phase 2/3 execution order (recommended)

1. **P2.T5 empirical-Bernstein** (least dependent, biggest tightening)
2. **P2.T2 Eyecandies category KS** (uses already-on-disk data; closes
   the long-standing "real-cohort T2" deferral)
3. **P2.T4 sample complexity** (small but worth doing for completeness)
4. **P3.T6 decision** (retire vs reformulate; pick before P3.GDR so the
   theorem stack reaches a stable shape)
5. **P3.GDR empirical validation** (the headline novelty piece)

Total estimated effort: **~3 days for Phase 2** + **1 week for Phase 3**
under sustained focus.

---

## Out of scope for this plan

- Re-running headline experiments under the new theorems (the closed
  forms validate against existing data; no new training runs required).
- Modifying the v3 frozen Family-D contract (those locks remain).
- Manuscript edits (separate editorial pass, already planned in
  `/Users/pratik_n/.claude/plans/what-is-best-nasme-streamed-kernighan.md`).

When Phase 2/3 lands, re-run:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/validate_theorem_stack.py
PYTHONPATH=src .venv/bin/pytest tests/ -q
```
