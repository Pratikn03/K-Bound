# Phase 2.2B.exec — Hostile Review Report

Posture: senior trustworthy-ML auditor and hostile reviewer of the Phase 2.2B.exec output.

## Reviewer's 12 questions

### Q1. Did B1/B2 reproduce?

**Answer: YES, both.**

- B1 (zero_attack k=4): Δ AUC = +0.0507 (95% CI [+0.0364, +0.0650]; Holm K=2 p = 4.3 × 10⁻¹²); Phase-1 target +0.0506. Decision: **REPRODUCED**.
- B2 (max_attack k=4): Δ AUC = +0.0939 (95% CI [+0.0741, +0.1149]; Holm K=2 p < 1e-15); Phase-1 target +0.0319. Decision: **REPRODUCED**.

B2 sits substantially above the Phase-1 CI; this is reported honestly without redefining the endpoint. See [FAMILY_B_PRIMARY_MECHANISM_REPLICATION_REPORT_v2.md](./FAMILY_B_PRIMARY_MECHANISM_REPLICATION_REPORT_v2.md) §4.

### Q2. Did RGA-v2 pass every locked promotion criterion?

**Answer: N/A — B-MECH-2 driver `main()` is a scaffold.**

No gate was promoted. Decision label: `EXECUTION_BLOCKED_DRIVER_SCAFFOLD`. See [RGA_V2_PARTIAL_FAILURE_REPORT_v2.md](./RGA_V2_PARTIAL_FAILURE_REPORT_v2.md). Phase 2.2B.exec did not promote RGA-v2.

### Q3. Which partial-failure regimes improved and which did not?

**Answer: None evaluated.** B-MECH-1 evaluated only k=4 (coherent collapse). Partial-failure regimes {k=1, k=2, k=3} are in B-MECH-2 scope; B-MECH-2 did not execute.

### Q4. Did any gate exceed the clean false-fire budget?

**Answer: N/A.** The clean false-fire budget is measured during B-MECH-2 gate-candidate sweep on the k=0 clean fold; B-MECH-2 did not execute. The locked G0 mean gate at τ=0.66 (the only gate B-MECH-1 used) is the baseline against which the budget is measured.

### Q5. Did B-MECH-3S reduce false firing only under domain-composition shift, and is the broader theorem still deferred?

**Answer:** Broader theorem **explicitly deferred** in [MIXTURE_SHIFT_PROTOCOL.md](./MIXTURE_SHIFT_PROTOCOL.md). B-MECH-3S itself was not executed; the protocol-lock commit (`204775b`) precedes any compute and marks the lock-time. See [DOMAIN_COMPOSITION_SHIFT_AUDIT_REPORT.md](./DOMAIN_COMPOSITION_SHIFT_AUDIT_REPORT.md).

### Q6. Did KS power remain acceptable?

**Answer: N/A — B-MECH-4 not executed.** See [KS_REFERENCE_AND_POWER_REPORT_v2.md](./KS_REFERENCE_AND_POWER_REPORT_v2.md).

### Q7. Which certificates, if any, are positive?

**Answer:** **One of two scenarios certified.**

- max_attack k=4: **CERTIFIED** (paired-bootstrap LCB = +0.0085 > 0; n_fired = 1 600).
- zero_attack k=4: **NOT_CERTIFIED** (LCB = -0.0050 < 0; n_fired = 1 599).

This is a genuine split: AUC delta is positive on both scenarios but per-sample paired loss benefit is negative on zero_attack. The certificate test (per-sample local benefit) and the AUC test (global ranking) measure different things. See [RISK_DOMINANCE_AND_CERTIFICATE_REPORT_v2.md](./RISK_DOMINANCE_AND_CERTIFICATE_REPORT_v2.md) §2.

### Q8. Were negative findings retained?

**Answer: YES.**

- The zero_attack NOT_CERTIFIED result is reported alongside the max_attack CERTIFIED result without suppression.
- The B2 Phase-2 estimate (+0.0939) is reported alongside the Phase-1 target (+0.0319) and the disagreement is flagged honestly rather than averaged or hidden.
- B-MECH-2/3/4 non-execution is reported as `EXECUTION_BLOCKED_DRIVER_SCAFFOLD` (not as `NOT_IMPROVED` or `NO_MEANINGFUL_CHANGE`) so future readers cannot misinterpret absence-of-evidence as evidence-of-absence.

### Q9. Was any test outcome used for selection?

**Answer: NO.**

- B-MECH-1 selection rule: τ=0.66 is LOCKED by the Phase-2 contract; no validation- or test-fold tuning. Every archived row carries `selection_used_test_metrics=False` (verified by [tests/test_phase2_family_b_prediction_archive_complete.py](../../../tests/test_phase2_family_b_prediction_archive_complete.py)).
- B-MECH-2 selection: `_select_tau_on_validation_only()` signature accepts only validation-fold tensors (verified by [tests/test_phase2_rga_v2_no_test_tuning.py](../../../tests/test_phase2_rga_v2_no_test_tuning.py)).
- B-CERT-1 fired-subset definition uses ensemble predictions (post-hoc, no further selection).

### Q10. Did Family D remain untouched?

**Answer: YES.**

- `FAMILY_D_V1_INVALIDATION_NOTICE.md` still contains `INVALID_FOR_EXECUTION`.
- `FAMILY_D_V2_DESIGN_STATUS.md` still contains `V2_DESIGN_PENDING`.
- No Family-D code path imported or invoked.
- 5 tests in [tests/test_phase2_family_d_untouched_during_family_b.py](../../../tests/test_phase2_family_d_untouched_during_family_b.py) assert this.

### Q11. What exact new claims are allowed?

- "Under the v2 Phase-2 archived-prediction pipeline, B1 and B2 mechanism endpoints reproduce: B1 Δ AUC = +0.0507 (CI [+0.0364, +0.0650]), B2 Δ AUC = +0.0939 (CI [+0.0741, +0.1149])."
- "A retrospective fired-subset paired-loss certificate is CERTIFIED for max_attack k=4 (LCB = +0.0085) and NOT_CERTIFIED for zero_attack k=4 (LCB = -0.0050) under the defined stress protocol."
- "B-MECH-1 mechanism replication is complete; B-MECH-2 RGA-v2 partial-failure sweep, B-MECH-3 domain-composition false-fire study, and B-MECH-4 KS power sweep were not executed because their driver `main()` functions are scaffolds."

### Q12. What remains prohibited?

- "RGA-v2 is promoted." (B-MECH-2 not executed; no candidate has passed C1..C6.)
- "RGA solves partial failure." (Only k=4 was tested.)
- "ELARA is universal / SOTA / production-ready / clinically validated."
- "Family D confirmed." (Family D not touched.)
- "Theory closure is achieved." (Only the k=4 portion of theory; B-MECH-3/4 and B-CERT-1's risk-dominance terms are missing.)
- "B2 Phase-1 number was wrong / inflated / deflated." (Both phases reported; neither retroactively redefined.)

## Audit conclusion

Phase 2.2B.exec produced **mechanism replication evidence (B-MECH-1)** and **partial certificate evidence (B-CERT-1)** under honest infrastructure constraints. RGA-v2 promotion remains unevaluated. KS power / mixture-shift studies remain unevaluated. Family D remains untouched.
