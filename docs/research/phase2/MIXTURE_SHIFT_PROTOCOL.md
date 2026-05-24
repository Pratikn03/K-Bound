# ELARA Mixture-Shift Protocol Lock

**Status:** locked before B-MECH-3S execution.

ELARA-Bench-LA has a natural 4-way `domain` variable (fraud, cyber, shoppers, news) but no separate per-sample category/cohort variable that would support a general category-mixture confounding test. Deriving an arbitrary category (e.g. label parity bands, random groups) would risk fitting the experiment to its outcome and would not provide evidence for the original theorem.

## 1. Disposition of the original mixture-shift experiment

The original category/cohort-mixture confounding theorem experiment is marked:

> `DEFERRED_PENDING_NATURAL_CATEGORY_METADATA`

This experiment requires a benchmark that ships with a natural,
domain-orthogonal category/cohort variable. ELARA-Bench-LA does not
provide one, and synthesising one here would be unsound.

## 2. Executable substitute

The current executable experiment is renamed:

> **B-MECH-3S — Exploratory Domain-Composition Shift False-Fire Audit**

- `category_column = domain`
- Mixture-shift sampler in `src/elara/family_b/mixture_shift.py` is invoked with `target_proportions` over the four ELARA-Bench-LA domains.
- The within-category KS invariance check at the sampler level holds the within-domain score distribution constant.
- Two references are compared: global KS (existing `ReliabilityEstimator`) vs domain-aware reference (specialised `ReliabilityEstimator` instances per domain, or `CategoryAwareReliabilityEstimator` when applicable).

## 3. Allowed conclusion (verbatim)

> "This exploratory audit evaluates whether gate firing changes under
> controlled shifts in evidence-domain composition while within-domain
> score distributions are held invariant."

## 4. Forbidden conclusion (verbatim)

> "This experiment closes the general category/cohort-mixture confounding theorem."

The B-MECH-3S result is **exploratory** and may **not** be used for:

- RGA-v2 promotion (C1..C6 contract);
- universal / cross-domain superiority claims;
- general distribution-shift safety claims;
- arguments that ELARA gates are deployment-safe in novel cohort regimes.

## 5. Operational invariants

- Driver: `src/scripts/run_phase2_mixture_shift.py --experiment-id B-MECH-3`.
- The output report MUST carry the label `B-MECH-3S — Exploratory Domain-Composition Shift False-Fire Audit` at the top.
- Decision labels are restricted to:
  - `DOMAIN_COMPOSITION_FALSE_FIRE_REDUCED`
  - `DOMAIN_COMPOSITION_FALSE_FIRE_NOT_REDUCED`
  - `INCONCLUSIVE`
- No `confirmatory` language. No `deployment-safe` language.

## 6. Provenance

This protocol lock is committed before B-MECH-3S execution per the Phase 2.2B.exec spec Step 1. The protocol-lock commit hash will be recorded in the B-MECH-3S report under "Provenance".
