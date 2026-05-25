# Phase 1.1 — Final Hostile Review Report (Step 15)

**Method:** read-only audit after all Phase 1.1 edits, regeneration, builds, text scans, and visual audits complete. Inspects the verified PDFs, the source `.tex` files, the generated tables, the metrics manifest, the validator output, and the test suite.

**Repo state:** branch `fix/elara-phase1-1-pdf-source-consistency`; build outputs match unique verified PDFs by SHA256.

---

## 1. Review questions

| # | Question | Verdict | Evidence |
|---|---|---|---|
| 1 | Does the abstract use exactly the same primary result regime as the main tables? | **Yes** | Paper + thesis abstracts cite the locked PRIMARY (+0.0506 / +0.0319 from k-of-D k=4 mean-gate). Phase 1.1 primary-run-resolution doc confirms. |
| 2 | Are hard-mode and original ELARA-Bench-LA results clearly separated? | **Yes** | `PHASE_1_1_PRIMARY_RUN_RESOLUTION.md` declares Run A PRIMARY (+0.0506 / +0.0319) and Run B SECONDARY (+0.0367 / +0.0538) — never mixed in a single sentence or table cell. |
| 3 | Are canonical one-class degenerate metrics removed from promoted tables/figures? | **Yes** | `phase1_1_canonical_cleanup.py` rewrites 8 canonical asset files (4 tables + 2 figures across 2 canonical runs). PDF text scan: 0 instances of `0.7835`. |
| 4 | Does the paper display any old `best non-router` or oracle-selection language? | **No** | PDF text scan: 0 hits on `best non-router`, `MAX(router`, `max(router`, `RGA+ Δ vs best`. Source: 0 hits in tables. |
| 5 | Does any displayed result use Fisher combination? | **No** | PDF text scan: 0 hits on `Fisher-combined`, `p (DeLong, Fisher)`. Single-representative-seed DeLong used. |
| 6 | Is Family A wrongly called confirmatory anywhere? | **No** | PDF text scan: 0 hits on `Family A confirmatory`. Phase 1.1 enforces `Family A audited-primary K=5`. |
| 7 | Does any UNSW sentence overclaim broad generalization? | **No** | PDF text scan: 0 hits on `prove the cross-benchmark`, `beats every non-ELARA`, `without losing the cross-domain generalization property`. UNSW section rewritten in both manuscripts. |
| 8 | Is Real3D resolved and correctly bounded? | **Yes** | `PHASE_1_1_REAL3D_RESOLUTION.md` locks it as Family C exploratory; paper Real3D paragraph rewritten to descriptive Δ=−0.003 with no "no longer negative" claim. |
| 9 | Does any causal/SCM/ATE language remain in reported findings? | **No** | PDF text scan: 0 hits on `Causal Reliability Attribution`, `Causal Inference for Reliability`, `Structural Causal Model`, `interventional ATE`, `Average Treatment Effect`. Renamed section: "Model-Response Sensitivity to Per-Domain Reliability". |
| 10 | Does any polarity sentence claim primary prediction flipping or deployment-grade status? | **No** | PDF text scan: 0 hits on `deployment-grade`, `deployment-time sanity check`. Phase 1.F code lock retained. |
| 11 | Does the thesis state the same audited policy as the paper? | **Yes** | New thesis subsection §sec:thesis-audited-policy "Locked Audited-Reanalysis Policy and Future Replication Boundary" added with all 9 required policy items. |
| 12 | Are EATA/SAR defined if reported? | **Yes** | Paper §I.B + §Background define EATA / SAR; thesis Test-Time Adaptation Baselines section now has an "EATA and SAR" paragraph with citations to `niu2022efficient` + `yang2023sar`. |
| 13 | Does the fixed-seed statistical limitation remain transparent? | **Yes** | Paper §Internal Validity + master-comparison caption explicitly state "fixed representative-seed audited inferential summary; not independent confirmatory replication". Thesis audited-policy subsection echoes the same. |
| 14 | Do source, generated tables, extracted PDF text and rendered PDF pages agree? | **Yes** | `validate_phase1_1_pdf_claims.py` returns 0 violations. 383 tests pass. |
| 15 | Are the final uniquely named PDFs the versions that should be uploaded for review? | **Yes** | `PAPER_DRAFT_PHASE1_1_VERIFIED.pdf` and `THESIS_CHAPTER_PHASE1_1_VERIFIED.pdf` are SHA256-identical to the standard `_v1.pdf` outputs. |

---

## 2. Severity summary

- **P0 source/PDF inconsistency remaining:** **0**.
- **P1 major issues remaining:** carried-forward Phase-2 work (raw-prediction archiving for ensemble DeLong; Family D fresh-partition replication; optional Real3D 30-seed upgrade). Documented in `PHASE_1_1_REMAINING_OPEN_GAPS.md`.
- **P2 minor issues:** none blocking.

## 3. Corrected headline numbers (after Phase 1.1)

| Cell | Validation-frozen RGA+ head | Validation-frozen comparator | Δ AUC | $p$ (rep seed) | $p_{\mathrm{Holm}}$ (A K=5) |
|---|---|---|---|---|---|
| A2 MVTec 3D-AD PatchCore SP | router 0.739 | SAR 0.735 | +0.004 | 0.919 | 0.919 (n.s.) |
| A3 MVTec 3D-AD PatchCore held-out | router 0.509 | Tent 0.503 | +0.006 | 0.050 | 0.202 (n.s.) |
| A5 MVTec LOCO-AD PatchCore SP | router 0.718 | Tent 0.726 | **−0.008** | 0.126 | 0.378 (n.s.) |
| A7 VisA RGB+edge SP | boost 0.866 | RF 0.855 | +0.011 | 0.248 | 0.496 (n.s.) |
| **A8 UNSW-NB15 flow/conn/context** | router 0.989 | RF 0.989 | **+0.000** | 1.3e−6 | **6.7e−6 (sig.)** |
| C1 Real3D-AD (Family C exploratory) | router 0.534 | TTT 0.537 | **−0.003** | 1.0 | — |
| B1 ELARA-Bench-LA zero-attack all-domain at τ=0.66 | RGA | static | **+0.0506** | (k-of-D k=4 mean-gate) | — |
| B2 ELARA-Bench-LA max-attack all-domain at τ=0.66 | RGA | static | **+0.0319** | (k-of-D k=4 mean-gate) | — |

## 4. Claims removed or weakened in Phase 1.1

- UNSW "prove the cross-benchmark", "beats every non-ELARA", "first naturally-paired benchmark where RGA's reliability-derived features turn a supervised fusion result into the top method": **REMOVED**.
- Real3D "no longer the negative cell", "boost reaches 0.5656 above the strongest non-router baseline": **REMOVED**.
- Canonical PR-AUC / ECE / Brier values (0.7835 etc.): **REMOVED** from all promoted tables and figures.
- Mixed regime ELARA-Bench-LA references: PRIMARY (k-of-D k=4 mean-gate) and SECONDARY (table_3 adversarial default-gate) explicitly separated.
- Causal/SCM/ATE language: **REMOVED** from results section; reframed as Model-Response Sensitivity.
- Deployment-grade polarity language: **REMOVED**.
- "Best non-router" / "Best ROC" column headers in `rga_plus_ablation.tex` and `mvtec3d_milestone1_comparison.tex`: **REMOVED**; replaced with validation-frozen comparator framing.
- "Family A confirmatory" / "pre-registered confirmatory" framing on existing cells: **REMOVED**; Family A is now audited-primary K=5.

## 5. Experiments requiring future re-run

- Runner patch + re-run to archive per-seed test predictions; enables seed-averaged ensemble DeLong + paired sample bootstrap CI (replaces the current single-representative-seed limitation).
- Family D future-locked confirmatory replication on previously-uninspected MVTec 3D / LOCO / UNSW partitions (or a new naturally-paired independent-modality dataset).
- Optional Real3D-AD 30-seed re-run if Family A audited-primary inclusion is wanted.

## 6. Final recommendation

The four allowed verdicts are:

- **FAIL — P0 source/PDF inconsistency remains.** — **NOT THE VERDICT.** Phase 1.1 has 0 P0 remaining; all forbidden-string scans pass; both PDFs build clean; 20 Phase 1.1 tests pass and the full suite is 383 / 2 skipped.
- **PASS FOR PHASE 1.1 — corrected audited manuscripts verified; Phase 2 may begin.** — **THIS IS THE VERDICT.** Phase 1.1 closes every contradiction the ledger identified. The verified PDFs are submission-safe under their declared scope (audited reanalysis only; not confirmatory).
- **PASS FOR INTERNAL PRESENTATION ONLY.** — A possible weaker reading, but Phase 1.1 has demonstrably strengthened the manuscripts to the point where workshop-style external review is acceptable provided no confirmatory language is added.
- **READY FOR EXTERNAL WORKSHOP REVIEW.** — Acceptable interpretation. The manuscripts are now consistent and conservative; they do not overstate their audited scope.

**Final rating: PASS FOR PHASE 1.1 — corrected audited manuscripts verified; Phase 2 may begin.**

The manuscripts are explicitly **not** ready for top-tier conference acceptance as confirmatory cross-domain superiority papers; they are ready as audited-reanalysis methodology + mechanism papers that motivate Family D future replication. External workshop submission under that scope is also acceptable.

Phase 2 (raw-prediction archive + ensemble DeLong + Family D study + Real3D 30-seed upgrade) is the right next step. Phase 2 is NOT begun in this task.
