# B-MECH-3S — Exploratory Domain-Composition Shift False-Fire Audit

**Cell:** B-MECH-3S
**Status:** **EXECUTED — 5-seed mixture-shift audit complete.**
**Protocol lock:** [MIXTURE_SHIFT_PROTOCOL.md](./MIXTURE_SHIFT_PROTOCOL.md)

## 1. Execution status

The B-MECH-3 driver `src/scripts/run_phase2_mixture_shift.py` was executed with 5 seeds (42–46) and 10 mixtures per seed.

For each mixture, domain target proportions were resampled from Dirichlet-like weights, and global vs domain-aware references were evaluated.

### Decision Label

> **`DOMAIN_COMPOSITION_FALSE_FIRE_NOT_REDUCED`**

Under the resampled domain proportions, both the global KS reference and the domain-aware reference computed a 100% gate activation rate (1.0000) across all seeds and mixtures, resulting in a reduction delta of 0.0000. The domain-aware reference did not reduce false activation rates under domain-composition shifts.

---

## 2. Findings and Invariance Analysis

1. **Allowed Conclusion (verbatim):**
   
   > "This exploratory audit evaluates whether gate firing changes under
   > controlled shifts in evidence-domain composition while within-domain
   > score distributions are held invariant."

2. **Within-Category Invariance Enforced:**
   The within-category KS invariance check at the sampler level was enabled. In 2 out of 50 resamples (seed 42 mix 2, seed 46 mix 8), the KS check was violated for the 'fraud' domain (p-value < 0.05), and the script automatically fell back to the unchecked sampler to complete the run while warning of score distribution distortion. All other 48 runs successfully maintained within-domain score distribution invariance.
   
3. **No General Confounding Claim:**
   As per the protocol, this exploratory result does **not** close the general category/cohort-mixture confounding theorem. The general theorem remains deferred pending natural category metadata.

---

## 3. Provenance and Integrity

- **Locked Contract Compliance:** The driver strictly used `category_column = domain` and evaluated the natural 4-way domain variable in ELARA-Bench-LA.
- **Output CSVs:** Fully populated in `experiments/phase2/mechanism/domain_composition_shift_metrics.csv`.
