# Phase 1 Remaining Open Gaps

**Status:** Phase 1 closed for the audited reanalysis of inspected results. Phase 1 does **not** produce confirmatory or pre-registered results; that is by design and is locked by Phase 0.6 AR-11.

The following gaps remain for future phases. None is a P0 blocker for the audited-reanalysis manuscript; all are documented so the reader knows what is and is not supported.

---

## 1. Raw per-seed test predictions are not archived in the legacy JSONs

**Impact:** the policy-preferred audited inferential rule (seed-averaged ensemble DeLong + paired sample bootstrap CI over test samples) cannot be computed today. Phase 1.D falls back to the single-representative-seed DeLong using seed 42; other seeds are reported descriptively (mean ± SD) only.

**Mitigation:** the Phase 1.F runner patch removed the polarity flip but did not (yet) add prediction archiving. The next runner patch must persist `static_probs`, `craf_probs`, `rga_meta_router_probs`, `rga_boosted_fusion_probs`, plus every baseline's `*_probs` array per seed in the result JSON. The audited statistics emit script (Phase 1.D) is structured to consume those arrays when they appear.

**Open re-run:** a 30-seed re-run of the four Family A confirmatory cells (A2, A3, A5, A7) plus one 5-seed re-run of A8 will fully unblock the ensemble DeLong / paired sample bootstrap. The corrected Phase 1.D table will then carry an `ensemble_audited_DeLong_p` field per cell with a paired-bootstrap CI. Until that re-run, the audited inferential summary remains the single-representative-seed DeLong with the explicit "ensemble pending" label.

## 2. Family D future locked confirmatory replication is not run

**By design** (Phase 0.6). Family D requires test partitions that have not yet been inspected:
- D1: MVTec 3D-AD SP on a newly created untouched locked test split.
- D2: MVTec LOCO-AD SP on a newly created untouched locked test split.
- D3: UNSW-NB15 on a newly defined temporally / attack-held-out locked evaluation.
- D4: Any newly added naturally paired `independent_modalities` dataset.

**Open:** these are out of scope for Phase 1. They are the only family that can support the words "confirmatory" and "pre-registered" in the paper, and they require fresh data acquisition or split design that has not been touched.

## 3. Mechanism endpoint (Family B) inferential summary

Phase 1.D's locked audited statistics emit script reports Family A cells only. Family B endpoints (B1 zero-attack and B2 max-attack at locked $\tau{=}0.66$) are described in the paper's mechanism section using the already-shipped per-seed delta values, but a corresponding `experiments/audit/audited_family_B_inference.csv` was not produced (the runner does not currently archive per-seed predictions for the all-domain coherent-collapse attack predictions either).

**Mitigation:** the same runner patch in §1 above will, in a future revision, support a Family B audited DeLong table parallel to the Family A table. Until then, the mechanism numbers retain their per-seed mean ± 95% CI as reported in the existing JSONs.

## 4. Canonical PR-AUC / ECE / Brier remain blocked

**By design** (Phase 1.A audit verdict METRICS_VALID_BUT_MISINTERPRETED). The canonical PR/ECE/Brier values are the canonical test-fold's anomaly prevalence reflected through degenerate constant predictors. The paper and thesis report **only ROC-AUC** for canonical cells; PR/ECE/Brier are omitted with an explanatory caption.

**Open:** any future canonical-protocol method that produces a non-degenerate predictor (e.g., training the supervised fusion head with synthetic anomaly augmentation under a strictly pre-declared protocol) can be reported with PR-AUC / ECE / Brier if a follow-up label/metric semantics audit verifies the predictor is non-degenerate. The current paper does not need this.

## 5. Pre-declared comparator registry for Family D

The Phase 0.6 `HEADLINE_METHOD_POLICY.md` §4 holds a placeholder Family D comparator registry. Each row says **TBD** because the corresponding test partition has not been locked. Filling these rows is the first step of any future Family D study; doing so today would violate Phase 0.6 AR-16 (no reuse of inspected test partitions for confirmatory claims).

## 6. Open user decisions (still deferred from Phase 0.6 §11)

- **Causal-attribution rename.** Phase 1.G chose "Model-Response Sensitivity to Per-Domain Reliability" (the locked default). Alternative names listed in Phase 0.6 §11 remain on the menu if the user wants to swap.
- **Pairing-strength prose policy.** Per-cell reiteration vs one global paragraph. Phase 1.G used the global-paragraph approach by reference; per-cell reiteration is the documented alternative.
- **Family D seed-ensemble option** (deployed seed-ensemble vs single frozen seed). Deferred until Family D cells exist.

## 7. Real3D-AD 30-seed re-run

Real3D-AD is currently Family C exploratory (5 seeds). To upgrade to Family A audited primary or Family D confirmatory, a 30-seed re-run on the PCA shape + depth descriptor is required, with the split frozen before evaluation. The current Phase 1 manuscript treats Real3D-AD strictly as exploratory; no superiority claim is made.

## 8. Healthcare replay remains local retrospective

By design, per Phase 0.6 AR. Any prospective clinical-deployment claim is out of scope.

---

## Summary

Phase 1 closes the empirical-validity defects (B / C / D / E / F / Fisher) and produces a corrected audited-reanalysis manuscript. The remaining gaps are explicitly forward-looking: a runner patch to archive predictions, Family D future replication, an optional Real3D 30-seed upgrade, and the small set of deferred user decisions from Phase 0.6. None of these blocks the locked audited-reanalysis manuscript that Phase 1 produces.
