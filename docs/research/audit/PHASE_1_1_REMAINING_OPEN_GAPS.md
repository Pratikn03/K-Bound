# Phase 1.1 — Remaining Open Gaps

Phase 1.1 closes every contradiction identified in
`PHASE_1_1_CONTRADICTION_LEDGER.csv`. The following gaps are
explicitly carried forward to Phase 2 and are NOT P0 blockers for the
audited-reanalysis manuscripts.

| # | Gap | Phase-2 action |
|---|---|---|
| 1 | Raw per-seed test predictions are not archived in the legacy JSONs. | Runner patch + 30-seed re-run of Family A confirmatory cells (A2, A3, A5, A7, A8) to enable seed-averaged ensemble DeLong + paired sample bootstrap CI. Replaces the current single-representative-seed limitation. |
| 2 | Family D future-locked confirmatory replication is not yet executed. | Set up a fresh untouched test partition for at least one of D1 / D2 / D3 / D4 (see `EXPERIMENT_REGISTRY.csv`). Only Family D may support the words *confirmatory* and *pre-registered*. |
| 3 | Real3D-AD is on 5 seeds; classified as Family C exploratory. | Optional 30-seed re-run to enable Family A inclusion if the descriptor upgrade reproduces the +0.005 Δ. |
| 4 | Family B mechanism endpoint audited inference is reported descriptively (paper §sec:cross-benchmark-master) but no parallel CSV / Holm K=2 confirmatory table is yet produced. | Runner patch (same as #1) enables a parallel Family-B inferential table. |
| 5 | Healthcare replay remains local retrospective only. | By design — Phase 0.6 AR-16. |

None of the above blocks the corrected audited-reanalysis manuscript that Phase 1.1 produces.
