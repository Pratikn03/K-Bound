# Phase 1.1 — Real3D-AD Resolution

**Decision:** Real3D-AD remains **Family C exploratory** in Phase 1.1. Exactly one main-document row is permitted: the **validation-frozen RGA+ row** (router head, test ROC-AUC = 0.534, vs validation-frozen comparator TTT 0.537, Δ = −0.003). The previously-shown test-max router/boost rows (0.534 / 0.566) and the "Tent" / "best non-router" comparator framings are removed from the main document.

---

## 1. Evidence table

| run_id | artifact_path | descriptor_name | selection_policy | seed_count | rga_plus_head | comparator | rga_plus_auc | comparator_auc | delta_auc | analysis_family | valid_for_main_table | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real3d_valfrozen_audited | `experiments/audit/rga_plus_validation_frozen_selection.csv` + `experiments/audit/audited_comparator_selection.csv` | `PCA shape + depth supervised` (confirmed via `configs/attention_real3d_supervised_paired.yaml` and `src/uais/data/real3d_fpfh.py`) | validation-frozen RGA+ head (router) AND validation-frozen comparator (TTT) | 5 seeds [42..46] | **router** | **TTT** | **0.534** | **0.537** | **−0.003** | C exploratory | **YES (primary main-table row)** | Locked under the Phase 0.6 / Phase 1.B+1.C selection policies |
| real3d_testmax_boost_legacy | `experiments/fusion/real3d_supervised_paired_results.json` (clean_metric_summary.rga_boosted_fusion.roc_auc.mean) | same | TEST-MAX boost head | 5 seeds | boost | (post-hoc test winner: Tent 0.561) | 0.566 | 0.561 | +0.005 | C exploratory | **NO (legacy; deprecated by Phase 1.B AR-1)** | Selected by reading test winner; Rule 4 violation |
| real3d_switching_certificate | `experiments/fusion/switching_certificate_t5_audit.json` | same | boost head paired-bootstrap LCB | 5 seeds | boost | static-attention reference | 0.566 | 0.526 | +0.0374 LCB | B / C descriptive | descriptive only — switching-certificate audit (not a Real3D superiority claim) | This is the T5 LCB audit utility; it is NOT a Family A inferential comparison and may keep its existing "certified" wording so long as it is in the theorem-validation block, not the main result table |

## 2. Descriptor naming

The implementation in `src/uais/data/real3d_fpfh.py` (now using PCA shape statistics + pairwise-angle histogram + radial moments) and the config `configs/attention_real3d_supervised_paired.yaml` confirm that the displayed descriptor name is **"PCA shape + depth supervised"**. The legacy label "FPFH+depth" is stale and must not appear in any final manuscript table or caption.

## 3. Required main-document state

- **Master comparison table** (`tables/milestone2_cross_benchmark.tex`): Real3D row uses the validation-frozen row (router 0.534 vs TTT 0.537, Δ −0.003). Already correct.
- **`tables/rga_plus_ablation.tex`**: regenerate this table to use:
  - the validation-frozen RGA+ head (one column, not two);
  - the validation-frozen primary comparator (per `experiments/audit/audited_comparator_selection.csv`);
  - column header "Audited Δ vs validation-frozen comparator" (not "Best non-router");
  - the Real3D row showing router 0.534 vs TTT 0.537, Δ −0.003.
  Alternatively, retire this table altogether if its descriptive component-attribution role no longer justifies its existence.
- **Switching-certificate table** (`tables/switching_certificate_t5.tex`): may keep its existing row (the LCB audit is descriptive evidence for T5; not a superiority claim).
- **Real3D prose paragraph** in §sec:cross-benchmark-master: must say "Family C exploratory; Δ = −0.003 against the validation-frozen comparator TTT; no superiority claim".

## 4. Permitted Real3D claim

> "Real3D-AD is classified as Family C exploratory. In the audited reanalysis the validation-frozen RGA+ head is router (test ROC-AUC = 0.534) and the validation-frozen primary comparator is TTT (0.537); the descriptive delta is −0.003. No confirmatory superiority claim is made on Real3D-AD."

## 5. Forbidden Real3D claim

- "Real3D is therefore no longer the negative cell"
- "RGA+ boost reaches 0.5656, which is now above the strongest non-RGA baseline"
- "Real3D-AD descriptor upgrade closes the gap"
- "Real3D-AD: the boosted variant is the dominant contributor (0.566 vs the router's 0.534)" — this uses test-max framing

If any of these strings appear in the regenerated final PDFs, Phase 1.1 fails its hostile-review check.

## 6. Test gate

`tests/test_real3d_single_policy_valid_row.py` must fail if:

- the main-body manuscript prose shows a positive Real3D Δ value vs any non-RGA baseline;
- the main-body `rga_plus_ablation.tex` table shows both router 0.534 AND boost 0.566 as if both were headline values;
- any caption references "Tent" or "FPFH+depth" for Real3D.
