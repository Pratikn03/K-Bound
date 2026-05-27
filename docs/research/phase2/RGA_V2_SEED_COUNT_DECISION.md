# RGA-v2 Seed-Count Decision

**Phase:** 2.2B.2 / Step 4
**Status:** FINAL. 15 seeds suffices; no extension to 30 seeds required for Phase-2 closure.

## 1. Contract reading

`configs/phase2/rga_v2_gate_contract.yaml` defines:

```
seeds:
  target: 30
  minimum_for_inference: 15
```

The contract explicitly permits 15 seeds as a valid minimum for inferential reporting. 30 is a **target**, not a requirement.

## 2. Current evidence at 15 seeds

`experiments/phase2/mechanism/rga_v2_failure_surface_inference.csv` (verified post-Phase-2.2B.1):

| Gate | Mean clean false-fire | Budget (max(0.010, base+0.005)) | C1 (budget) | C2 (partial improve) | C3 (k=4 not worsened) | C5 (val-only) | C6 (single policy) | Decision |
|---|---:|---:|---|---|---|---|---|---|
| G0 | 0.0000 | 0.0100 | PASS | — | PASS | PASS | PASS | BASELINE_REFERENCE |
| G1 | 1.0000 | 0.0100 | **FAIL** | FAIL (0/2+) | PASS | PASS | PASS | NOT_IMPROVED |
| G2 | 1.0000 | 0.0100 | **FAIL** | FAIL (0/2+) | PASS | PASS | PASS | NOT_IMPROVED |
| G3 | 1.0000 | 0.0100 | **FAIL** | FAIL (0/2+) | PASS | PASS | PASS | NOT_IMPROVED |

Every non-baseline gate **fails the most basic promotion criterion C1** at clean false-fire rate **1.0000** — two orders of magnitude above the locked budget of **0.0100**. Adding more seeds cannot rescue a 100% clean false-fire rate; the failure is structural, not statistical.

## 3. Decision

> **15-seed RGA-v2 evaluation is final.** No extension to 30 seeds is required for Phase-2 closure.

A 30-seed re-run is recorded as **optional robustness work** that would not change the `NOT_IMPROVED` decision.

## 4. Rationale (mechanism-level)

The implementation in `src/scripts/run_phase2_rga_v2_gate_sweep.py` uses **batch-level minimum pooling**: for the minimum (G1), hybrid (G2), and top-q (G3) gates, the gate-fired flag is computed from the batch-level min reliability. Under ELARA-Bench-LA's 4-domain feature tensor, any batch will almost always have at least one domain with `min(reliability) < τ_min`, regardless of clean vs corrupted state. This drives clean false-fire to ≈ 1.0.

A future RGA-v2 redesign would need a per-sample firing decision (not batch-level) to avoid this failure mode. That is a model-design change, not a seed-count change.

## 5. Test guard

[tests/test_phase2_rga_v2_seed_count_decision.py](../../../tests/test_phase2_rga_v2_seed_count_decision.py) asserts:
- The contract YAML carries `minimum_for_inference: 15`.
- The RGA-v2 inference CSV has at least 15 unique seeds in the failure-surface CSV.
- C1 results for G1/G2/G3 are `False` (which is what makes the 15-seed decision final).
