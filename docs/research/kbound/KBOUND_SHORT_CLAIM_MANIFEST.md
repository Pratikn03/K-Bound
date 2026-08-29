# K-Bound Claim-To-Artifact Manifest

Date: 2026-08-27

| Claim | Type | Exact requirement | Evidence location | Caveat |
|---|---|---|---|---|
| Interior matched-evidence impossibility | theorem | `beta>0`, `|M|<beta`, and a rich class containing the constructed label kernels | `paper/sections/theory_core_main.tex`; `formal/KBound/Frontier.lean` | Opposite nonzero signs are interior-only |
| Closed-band abstention | theorem | Strict directional soundness over the rich declared class | same theory source and Lean frontier declarations | Boundary may pair zero with one strict sign |
| Exact strict-commitment frontier | theorem | Valid declared bound `|gamma|<=beta`, fixed augmented evidence, and class richness for necessity | Theorem in shared source; `distributional_frontier_maximal` | No robust commitment if no credible beta is supplied |
| Marginal FA_u certificate | theorem | `P(|Delta_hat-Delta|<=epsilon)>=1-alpha` at the declared inference unit | `paper/sections/theory_certificate.tex`; Lean measure containment | Does not control FA_c |
| Population-to-KGA bridge | theorem/method | Shared target `Delta`; separately justified operational class and coverage | `paper/sections/theory_algorithm_bridge.tex`; `frontier_kga_bridge_v1/bridge_results.json` | Controlled algebraic check only; KGA does not estimate M, gamma, or beta |
| CIFAR-10-C Tent routing | empirical | Five run seeds conditional on one archived checkpoint, controlled grid, candidate-specific calibration | canonical JSON, current-policy family-sensitivity artifact, and compact Tables 4-5 | Current exact-rank point win; retrospective ordinary intervals are positive over six families, but preregistered six-comparison Holm fails at 0.05 ($p=0.09375$ for both contrasts); no cluster-robust or confirmatory win |
| CIFAR-10-C EATA routing | empirical | Same controlled protocol | same current artifacts and compact Tables 4-5 | Current exact-rank point win; adapt-side family interval crosses zero and preregistered six-comparison Holm fails |
| CIFAR-10-C SAR negative | empirical | Completed five-seed exact-rank rebuild | canonical JSON | Zero false adapt but loses to always-adapt |
| ImageNet-C authoritative panel | empirical | 27 conditions per seed, five seeds, exact LOO replay | canonical JSON and ImageNet-C tables | Candidate dependent; SAR point-only |
| Office-Home primary/replication | empirical/diagnostic | Archived split and source-hashed records replayed under the release-locked numerical runtime | canonical JSON | Primary ties freeze with zero adapts; replication has a tiny point edge whose test-stream CI includes zero |
| Office-Home five-checkpoint candidate audit | invalidated diagnostic | Five distinct checkpoint hashes; post-hoc fixed-candidate accuracy comparison only | `KBOUND_OFFICEHOME_INDEPENDENT_CHECKPOINT_AUDIT_2026-08-27.md`; data-quality audit | Checkpoint identity passed, but the multicandidate route is invalid: binary-theory mismatch, sign/unbounded estimator, inadequate exact-rank calibration, duplicate candidate predictions, and invalid JSON |
| iWildCam numerical row | withheld | Pinned rerun using the official WILDS label-present macro-F1 contract and a sealed population manifest | archived canonical records plus data-quality audit | Stored sklearn macro-F1 values are not promoted; diagnostic recomputation gives KGA/adapt/freeze regret 0.005511/0.074502/0.005511, but runtime and population differ |
| Historical iWildCam beats-both flag | diagnostic/superseded | Reconcile archived radius and metric contract | canonical JSON `historical_reconciliation` and hashed historical artifact | Both the old narrow win and cross-fitted replay use the wrong archived metric; neither is promoted |
| Camelyon17 OOD one-sided diagnostic | diagnostic | Archived opened OOD evaluation row | canonical JSON | Already opened, all cells helpful, and always-adapt is oracle-equivalent; not prospective or untouched evidence |
| Camelyon17 B-v2 SAR | diagnostic | Three within-seed grids | canonical JSON | Not an untouched hospital-domain win |
| RxRx1 freeze behavior | empirical | Three source checkpoints in supporting artifacts; canonical displayed panel | canonical JSON and repetition table | Endpoint no-harm, zero adapt exposure |
| PACS negative | diagnostic | Aggregate domain-seed arithmetic | canonical JSON | Promoted-panel replay incomplete; one-domain smoke validates serialization only |
| ImageNet-R negative | diagnostic | Four seeds, ten backbones, 12 conditions each | canonical JSON and per-backbone table | Architecture panel, not one deployed policy |
| CIFAR-10.1 negative | diagnostic | Locked 48-cell replay | canonical JSON | Ties freeze; no adapts |
| POEM/AETTA comparison | historical empirical port | Protocol-matched archived port with fixed thresholds and an earlier KGA policy | archived head-to-head artifact; compact historical-context paragraph | Not official implementations; current-policy recomputation pending; confidence intervals are unadjusted and Holm applies only to archived p-values |
| Constructed heterogeneous mixture | pending | Replay from reconciled components under one prospective protocol | historical artifacts only | Not promoted as a natural win |
| Universal improvement | not claimed | Would require broad held-out dominance | limitations and claim ledger | Contradicted by negative rows |
| Single-dataset natural CI-robust beats-both | not claimed | Untouched natural environments and robust inference | remaining-work protocol | No current track meets the bar |
| Real-camera validation | pending | Fresh locked physical sessions and publication gate | `edge/` protocol and runbook | Templates and demos are not evidence |
| Unopened natural target | pending | A target with no prior result inspection and prospective seal | `natural_target_provenance_v1/NATURAL_TARGET_PROVENANCE_AUDIT.json` | No verified unopened target currently exists |
| Exact split-conformal confirmation | pending | Sealed disjoint fit/calibration/test unit manifest and one-pass execution | `research_lock/KBOUND_EXACT_CONFIRMATION_UNSEALED_v1.json` | Draft manifest only; no result claimed |
| Official baseline provenance | diagnostic | Pinned clean upstream source and complete successful native logs | `official_repro_v1/OFFICIAL_BASELINE_AUDIT.json` | AETTA and POEM remain protocol-matched ports |

Candidate-protocol disclosure: the gradient-based natural-shift adapters use standard
transductive TTA. Episodic candidates update on the unlabeled evaluation batch; online candidates
update on the separate auxiliary stream but predict with evaluation-batch BatchNorm statistics.
Recorded stream/evaluation disjointness therefore applies to the auxiliary stream identities, not
to a claim of inductive candidate evaluation. Target labels remain unavailable until offline
scoring.

## Canonical Generation Chain

1. `scripts/reconcile_result_panels.py`
2. `experiments/kbound/results/reconciled_panels_v1/source_manifest.json`
3. `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`
4. `experiments/kbound/results/reconciled_panels_v1/canonical_panel_table.tex`
5. `docs/research/kbound/scripts/analyze_current_policy_cluster_inference.py`
6. `experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json`
7. `scripts/sync_reconciled_panels.py`
8. `docs/research/kbound/paper/generated/current_policy_family_sensitivity.tex`
9. `docs/research/kbound/scripts/make_tables.py`
10. `docs/research/kbound/paper/generated/kbound_numbers.tex`
11. `docs/research/kbound/scripts/plot_canonical_decision_frontier.py`
12. `docs/research/kbound/kbound_submission.tex` and `docs/research/kbound/kbound_tmlr.tex`
13. maintained compact and synchronized long PDFs

Both maintained PDF drivers consume the same `kbound_submission_body.tex`. The stale
`kbound_short_body.tex` and `kbound_short_appendix.tex` are excluded from the current generation
chain.

`src/scripts/validate_manuscript_claims.py` enforces required distinctions and rejects known stale
counts and overclaim phrases in live LaTeX.
