# Phase 1.I — Final Hostile Review Report

**Reviewer role:** read-only senior empirical-ML reproducibility engineer + skeptical conference reviewer.
**Inputs:** Phase 1.A–1.H artifacts plus regenerated PDFs.
**Scope:** verify that Phase 1 has actually repaired the P0 defects identified in Phase 0 / 0.5 / 0.6 and that no new defects were introduced.

---

## 1. Eighteen review questions

| # | Question | Verdict | Evidence |
|---|---|---|---|
| 1 | Does any result in the abstract lack a source artifact? | **No** | The abstract's 0.739 / 0.735 / 0.919 / 6.7e-6 numbers all trace to `experiments/audit/audited_ensemble_inference_results.csv` row A2 / A8, which in turn reads from `experiments/fusion/mvtec3d_patchcore_supervised_paired_results.json` and `experiments/fusion/unsw_paired_results.json`. |
| 2 | Does any prose number disagree with a generated table? | **No** | The validator (`src/scripts/validate_manuscript_claims.py`) returns 0 violations. Spot-check: paper §sec:cross-benchmark-master quotes 0.7390 / 0.7354 / 0.7175 / 0.7260 / 0.8662 / 0.8548 / 0.9892 / 0.9889, all of which match `tables/milestone2_cross_benchmark.tex` to 3 decimals. |
| 3 | Does any paper number disagree with the thesis for the same cell? | **No (within audited cells)** | Both manuscripts reference the same metrics manifest (`docs/research/metrics_manifest.json`); the master comparison table is rendered once and inputted by both documents via `\input{tables/milestone2_cross_benchmark.tex}`. |
| 4 | Does any RGA+ headline still involve test-set selection? | **No** | `src/scripts/emit_milestone2_cross_benchmark.py` no longer contains `max(rga_router_test, rga_boost_test)`. `tests/test_no_test_selected_rga_plus.py` passes. The RGA+ headline reads from `experiments/audit/rga_plus_validation_frozen_selection.csv`. |
| 5 | Does any comparator selection still depend on the test winner? | **No** | `src/scripts/select_audited_validation_frozen_comparator.py` ranks baselines by seed-mean validation ROC-AUC. `tests/test_no_test_selected_comparator.py` passes; the master comparison reads from `experiments/audit/audited_comparator_selection.csv`. |
| 6 | Does any canonical metric remain reported without verified semantics? | **No** | The paper and thesis report **only ROC-AUC** for canonical cells; PR-AUC / ECE / Brier are omitted. The canonical reframing is explicit in the abstract and in the master comparison caption. The Phase 1.A audit recorded the verdict `METRICS_VALID_BUT_MISINTERPRETED` in `experiments/audit/canonical_label_semantics.json`. |
| 7 | Does any polarity correction alter primary predictions? | **No** | `run_breakthrough_experiment.py` has the four `1.0 - *_probs` lines removed and a Phase-1.F lock comment in their place. `tests/test_primary_metrics_do_not_apply_polarity_flip.py` passes. The polarity probe still runs and logs, but does not modify primary predictions. |
| 8 | Does any p-value rely on Fisher-combined dependent seeds? | **No** | `_fisher_combine` exists in `emit_milestone2_cross_benchmark.py` as a definition but is never invoked from the active emit path. `tests/test_no_fisher_seed_combination.py` passes. The audited p-values use single-representative-seed DeLong (seed 42). |
| 9 | Does any ensemble audited inference pretend to describe a single model? | **No** | The `audited_ensemble_inference_results.csv` rows carry `analysis_label = "single_representative_seed_DeLong (seed 42) — ensemble DeLong + paired sample bootstrap pending raw-prediction archive"` and the manuscript labels every quoted p-value as "single-representative-seed DeLong" or "audited reanalysis". `tests/test_ensemble_inference_label.py` passes. |
| 10 | Does any existing inspected result use "confirmatory" or "preregistered" language? | **No** | Validator finds 0 instances of "pre-registered" or "confirmatory" applied to existing cells. `tests/test_no_retroactive_confirmatory_language.py` passes. Family D (future replication) is the only place those words are reserved. |
| 11 | Are Family A, B, C and future Family D roles correctly separated? | **Yes** | `experiments/audit/statistical_family_registry.csv` has 11 rows: 8 Family A (5 audited-primary + 3 protocol-diagnostic), 0 Family B (mechanism endpoints described in the paper but no inferential CSV row yet — see open gap §3), 3 Family C exploratory. Family D rows exist in `EXPERIMENT_REGISTRY.csv` but are TBD. `tests/test_analysis_family_partition.py` passes. |
| 12 | Are proxy-derived views prevented from supporting independent-modality universal claims? | **Yes** | `EXPERIMENT_REGISTRY.csv` records `pairing_strength` per cell; VisA (derived_view_proxy) and MVTec LOCO (derived_view_proxy) are flagged. The manuscript prose explicitly states derived-view proxies are not equivalent to independent modalities. |
| 13 | Is any exploratory result used as evidence of superiority? | **No** | Real3D-AD (C1) is labelled exploratory; its sign-flipped corrected delta ($-0.003$) is reported as descriptive. VisA noise-floor (C2) and UNSW held-out (C3) are similarly descriptive only. The paper's "Real3D-AD descriptor upgrade" prose has been refrained from "Real3D-AD is no longer a negative cell" claim. |
| 14 | Does any ATE/SCM/causal language overclaim the model-response analysis? | **No** | §sec:causal-attribution is renamed "Model-Response Sensitivity to Per-Domain Reliability"; "do(...)" notation removed; "Structural Causal Model" / "interventional ATE" removed. 5 orphan bibitems (Pearl, Hernán, Imbens, VanderWeele, Peters) removed from the bibliography. |
| 15 | Does any healthcare replay language imply deployment or clinical validation? | **No** | The thesis explicitly labels healthcare replay as "local retrospective replay" (§healthcare) and the appendix's "deployment-grade calibration monitor" wording was reworded to "streaming calibration monitor". |
| 16 | Are all baseline methods shown in results also documented in methods/configs? | **Partial** | EATA and SAR are now mentioned in the abstract baseline roster and are documented in the paper's earlier methods section. The runner's `baselines.py` ships all 8 baselines (RF, MLP, LFE, Conf-mean, Tent, TTT, EATA, SAR) and the master comparison table uses all 8. |
| 17 | Are all table captions accurate about K values and analysis status? | **Yes** | The master comparison caption explicitly states "Holm-Bonferroni correction within Family A audited-primary cells only (K=5: A2, A3, A5, A7, A8)". `tests/test_holm_family_size_matches_registry.py` passes. |
| 18 | Are all PDFs reproducible from the documented commands? | **Yes** | `./scripts/rebuild_paper.sh` reproduces both PDFs from the regenerated tables; 0 LaTeX errors, 0 undefined refs, bibliography hygiene 187/187 and 21/21 clean. The reproduction commands are in `docs/research/audit/PHASE_1_REPRODUCTION_COMMANDS.md`. |

---

## 2. Severity classification of remaining issues

### P0 blockers remaining: **0**

The five Phase-0/0.6 P0 blockers (B test-oracle RGA+; C contradictory p-values; D mis-stated multiplicity; E canonical semantics; Fisher independence) are all resolved by Phase 1.A–1.H. The Phase 1.A audit verdict (METRICS_VALID_BUT_MISINTERPRETED) closes E without requiring a re-run.

### P1 major issues remaining: **2 (deferred, not blockers for audited reanalysis)**

1. **Raw per-seed test predictions are not archived in the legacy JSONs.** The policy-preferred ensemble DeLong + paired sample bootstrap CI cannot be computed today. The single-representative-seed DeLong is used as the audited inferential statistic; the manuscript labels this explicitly. Deferred to a future runner patch + re-run (Open gap §1 in `PHASE_1_REMAINING_OPEN_GAPS.md`).
2. **Family B audited inference CSV is not produced.** B1/B2 mechanism endpoints are reported in the paper from existing per-seed deltas; a parallel audited CSV will require the same runner patch as P1 #1. Deferred.

### P2 minor issues remaining: **3 (deferred, by design)**

1. **Family D future locked confirmatory replication is empty.** By Phase 0.6 design — Family D requires untouched test partitions.
2. **Real3D-AD is on 5 seeds only.** Marked exploratory under Phase 1.D / 1.E classification. A 30-seed re-run would unblock Family A inclusion if desired.
3. **Three Phase 0.6 user decisions remain deferred** (causal-attribution rename: locked default chosen; pairing-strength prose: global-paragraph approach used; Family D seed-ensemble: deferred until Family D rows exist).

---

## 3. Verified repaired items (Phase 1)

- ✅ RGA+ test-set oracle selection (Rule 4) removed in `emit_milestone2_cross_benchmark.py`.
- ✅ Best-non-router test-winner comparator selection removed; replaced by validation-frozen comparator.
- ✅ Fisher combination of dependent-seed DeLong p-values removed from primary reporting.
- ✅ Multiplicity-family caption mismatch corrected (K=5 Family A confirmatory).
- ✅ Canonical PR/ECE/Brier semantics audit completed with verdict.
- ✅ Polarity flip removed from primary metric path.
- ✅ Causal/SCM language reframed as model-response sensitivity.
- ✅ Real3D "FPFH+depth" label refreshed to "PCA shape + depth".
- ✅ FGSM/PGD table header disambiguated with multicolumn group label.
- ✅ Thesis clean-adaptation-rate prose updated to 0.000.
- ✅ "deployment-grade" / "SOTA" / "state of the art" / "universally superior" / "production-ready" prose removed or reworded.

---

## 4. Corrected headline results summary (before → after Phase 1)

| Cell | Before (test-max RGA+ vs test-winner-best) | After (val-frozen RGA+ vs val-frozen comparator) | Direction |
|---|---|---|---|
| A2 MVTec 3D-AD PatchCore SP | RGA+ 0.739 (router) vs SAR 0.735, p_raw≥0.99 | RGA+ 0.739 (router) vs SAR 0.735, p_Holm 0.919 | unchanged; statement softened |
| A3 MVTec 3D-AD held-out | RGA+ 0.517 (boost) vs TTT 0.516 | RGA+ 0.509 (router) vs Tent 0.503, p_Holm 0.202 | corrected lower; n.s. |
| **A5 MVTec LOCO-AD SP** | **RGA+ 0.734 (boost) vs Tent 0.726, p_Holm 1.5e-5** | **RGA+ 0.718 (router) vs Tent 0.726, Δ=-0.008, p_Holm 0.378 (n.s.)** | **sign flipped + lost significance** |
| A7 VisA SP | RGA+ 0.866 (boost) vs RF 0.855, p_Holm 1.8e-4 | RGA+ 0.866 (boost) vs RF 0.855, p_Holm 0.496 (n.s.) | unchanged Δ but lost significance under K=5 single-seed DeLong |
| A8 UNSW-NB15 SP | RGA+ 0.989 (router) vs RF 0.989, p_Holm 5.7e-12 | RGA+ 0.989 (router) vs RF 0.989, p_Holm 6.7e-6 | retained significance; near-zero effect |
| C1 Real3D-AD | RGA+ 0.566 (boost) vs Tent 0.561 | RGA+ 0.534 (router) vs TTT 0.537, Δ=-0.003 | exploratory; sign flipped |

**Per Rule 2, the lower / sign-flipped corrected numbers stand.** The audited reanalysis headline finding is that only UNSW-NB15 retains Holm-corrected significance under the locked policy, and at near-zero practical effect.

---

## 5. Claims removed or weakened

- "RGA+ beats every baseline" — removed (AR-12).
- "best non-router baseline is X" used inferentially — removed (AR-2).
- "deployment-grade sanity check" — reframed as "validation-only score-orientation diagnostic" (AR-13).
- "interventional ATE under SCM" — reframed as model-response sensitivity (Issue H).
- "Fisher-combined DeLong" — replaced by single-representative-seed DeLong + Holm K=5 (AR-3).
- "9 evaluated cells" / "9-test Holm" — replaced by "Family A K=5" (AR-7).
- "max(router, boost)" caption — replaced by "validation-frozen choice" (AR-1).
- MVTec LOCO SP Holm-significance claim — reported as n.s. under the corrected analysis.
- VisA SP Holm-significance claim — reported as n.s. under the corrected analysis.
- "RGA+ leads on Real3D" — removed; Real3D is Family C exploratory with corrected delta = −0.003.

---

## 6. Experiments requiring future re-run

- **Runner patch + re-run** to archive per-seed test predictions and produce ensemble-DeLong + paired-sample-bootstrap CI (Open gap §1 + §3).
- **Real3D-AD 30-seed re-run** if Family A inclusion is wanted (Open gap §7).
- **Family D fresh-partition runs** for any confirmatory replication claim (Open gap §2).

---

## 7. Final recommendation

| Option | Verdict |
|---|---|
| **(a) Phase 1 Failed — blockers remain** | No. All P0 blockers are repaired. |
| **(b) Phase 1 Passed for Audited Manuscript Repair** | **Yes. This is the active recommendation.** The audited reanalysis manuscript is internally consistent, free of test-set oracle selection, free of Fisher-combined dependent-seed inference, free of canonical-PR/ECE/Brier overclaims, and free of "confirmatory" / "pre-registered" framing on inspected results. |
| **(c) Ready to begin Phase 2 theory/robustness closure** | Yes, with the open gaps in §6 explicitly carried forward. Phase 2 is the right next step for prediction-archive + ensemble-DeLong + Family D / Real3D 30-seed upgrades. |
| **(d) Ready only for internal/thesis presentation, not external submission** | Acceptable interim posture. Until Family D evidence exists, the manuscript can be presented as an audited reanalysis of inspected results and as a methodology demonstration; it is not yet a confirmatory cross-domain superiority paper. |

**Recommendation:** (b) **Phase 1 Passed for Audited Manuscript Repair**, with (c) as the explicit forward path.

The audited-reanalysis manuscript is the strongest defensible form of the existing inspected results. External submission as an audited-reanalysis paper is permitted; external submission as a confirmatory cross-domain superiority paper is not (and requires Family D).
