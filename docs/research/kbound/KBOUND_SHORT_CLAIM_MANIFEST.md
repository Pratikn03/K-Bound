# K-Bound Short Paper Claim-to-Artifact Manifest

Date: 2026-08-11

| Claim | Type | Exact requirement | Evidence location | Caveat |
|---|---|---|---|---|
| Interior matched-evidence obstruction | theorem | Declared target class contains evidence-identical worlds with opposite strict benefits | `kbound_short_body.tex`; Lean algebra map in `formal/KBound/TheoremMap.lean` | Deployment-law construction has external assumptions |
| Closed-band abstention | theorem | Sound three-way decision over the declared drift class | `kbound_short_body.tex`; theory appendix | Boundary zero-benefit case uses strict-decision semantics |
| Strict-commitment frontier | theorem | Declared valid bound `|gamma| <= beta` | Theory sections and claim accounting | Real-data KGA does not numerically apply this rule |
| Marginal false-adapt certificate | theorem | `P(|Delta_hat-Delta| <= epsilon) >= 1-alpha` | Certificate theorem and Lean implication | Controls `FA_u`, not `FA_c`; coverage justification remains protocol-specific |
| CIFAR-10-C mixed-regime win | empirical | Locked exact-rank stress protocol and declared clustering unit | Generated result manifest and main tables | Tent is cluster-robust; EATA adapt-side CI includes zero |
| ImageNet-C authoritative panel | empirical | 27 cells x 5 seeds, exact-LOO replay | `reconciled_panels_v1/canonical_panel_results.json` | SAR pooled point win is not CI- or seed-robust |
| Office-Home no-harm | diagnostic | Source-record transfer replay | Canonical panel, 35 test records | Ties freeze, zero adapts, A7 open |
| iWildCam no-harm | diagnostic | Source-record transfer replay | Canonical panel, 72 test records | Ties freeze, zero adapts, one test seed, A7 open |
| PACS negative result | diagnostic | Three-seed aggregate matches archived summaries | Canonical panel PACS section | Gate replay unavailable from archived fields |
| ImageNet-R negative result | diagnostic | 4 seeds x 10 backbones x 12 conditions, exact-LOO replay | Canonical panel and per-backbone table | Worse than adapt on 8/10 backbones |
| POEM/AETTA comparison | empirical port | Protocol-matched ports and disclosed implementation status | Main head-to-head table and baseline-faithfulness table | Not an official implementation claim |
| Constructed heterogeneous mixture | pending | Replay from reconciled component records | Manuscript TODO and remaining-work file | Historical result withdrawn |
| Universal improvement | not claimed | Would require dominance across deployment laws | Claim-accounting table | Impossible in general |
| Single-dataset natural beats-both | not claimed | Held-out, source-replayable, CI-robust win at declared unit | Result audit | No current natural track meets this bar |
| Real-camera validation | pending | Locked calibration/test sessions and fresh physical logs | Edge preregistration and blank templates | Templates are not evidence |

## Canonical generated chain

1. `scripts/reconcile_result_panels.py`
2. `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`
3. `scripts/sync_reconciled_panels.py`
4. `docs/research/kbound/paper/generated/kbound_result_manifest.json`
5. `docs/research/kbound/RESULT_MANIFEST.json`
6. `docs/research/kbound/paper/generated/kbound_numbers.tex`
7. `kbound_short_final_draft.pdf`

