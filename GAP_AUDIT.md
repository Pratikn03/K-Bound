# K-Bound — Gap Audit (Research Integrity + Experiment Coverage)

> **SUPERSEDED BY `docs/research/kbound/SUBMISSION_LEDGER.md` §3, §4 and §11.**
> Retained as a dated historical record of the 2026-06-14 audit. It is **not** a current statement
> of project status and must not be cited as one. Two specific reasons it cannot be re-verified
> today:
> - Its Part-B findings rest on `experiments/kbound/theory_validation/frontier_decisive/**`, all
>   17 files of which are now NUL-filled iCloud placeholders
>   (`docs/research/kbound/PLACEHOLDER_INVENTORY.md`, group F).
> - Its headline conclusion "**No fabrication was found**" was and remains true, but its clean bill
>   on calibration was scoped to the theory-validation suite. The separate defect found on
>   2026-07-26 — an in-sample conformal radius on five shipped experiment scripts — is outside
>   everything this document inspected. See `SUBMISSION_LEDGER.md §9`.
>
> Stamped 2026-07-26.

**Date:** 2026-06-14
**Scope requested:** (1) Research integrity — does every paper claim/theorem trace to a real validator script + result artifact, with no placeholders/fabrication? (2) Experiment coverage — are promised experiments actually run, with the claimed baselines/seeds/datasets?
**Method:** Deep *static* cross-check (no heavy execution). Paper of record = `tmp/pdfs/kbound_review/text/primary.txt` (1806 lines). Cross-referenced against `README.md`, `CODEBASE_MAP.md`, `docs/research/kbound/*`, `research_lock/`, `audits/`, `experiments/kbound/**`, `tests/**`. Roughly 25 result JSONs and all 7 main theorem validators were inspected; the four highest-stakes findings were independently re-verified against the files.

---

## ✅ Resolution log (2026-06-14) — see `INTEGRITY_FIXES.md`

- **A1 (Thm 3 α-mismatch) — RESOLVED.** Re-ran at α=0.05 → worst-case false-adapt **0.0316 ≤ 0.05**, persisted to `results_thm3_evalue_alpha005.json`. Paper digit `0.028` → `0.0316` (LaTeX edit still pending before next rebuild).
- **A2 (Thm 5 artifact) — RESOLVED.** Ran `val_thm5_multiclass.py` → `results_thm5_multiclass.json` (multiclass max-err 1.1e-16).
- **A3 (Thm 2 artifact) — RESOLVED.** Ran `val_thm2_regret.py` → `results_thm2_regret.json` (max gap 2.4e-17).
- **A5 (numbering) — RESOLVED.** Added `docs/research/kbound/THEOREM_NUMBERING_CROSSWALK.md`.
- **B1/B2/B3 (Camelyon17 SAR + win + serialization) — PARTIALLY RESOLVED.** τ\* recalibrated on the stored debug-scale grid (SAR present): recalibrated K-Bound beats both policies at false-adapt ≤ α (debug-scale). Full-scale re-run + serialized per-condition arrays + held-out τ\* still required. See `frontier_decisive/camelyon_recal/`.
- **B5 (stale README) — WITHDRAWN (false positive).** The current `README.md` is accurate (says "all experiments completed", lists ImageNet-C/Camelyon17/etc.); the original flag came from a stale read. The B5 entry below is retained for the record but does not reflect the current file.

---

## Executive summary

**No fabrication was found.** The theory-validation suite is genuine — real Monte-Carlo / exact-identity computations, fixed seeds throughout, honest "residual" disclosures, and a CI linter that actively blocks over-claim language. The promised core experiments (anomaly routing, regression, CIFAR-10-C in all three forms, online TTA, ImageNet-C including the SAR-collapse "win") are fully backed by artifacts whose numbers match the paper to the digit. The paper is also unusually candid about its own limitations.

The real gaps cluster into three buckets:

1. **A few claim ↔ artifact traceability mismatches** — most importantly one specific number (Thm 3's "0.028 ≤ 0.05") that the committed run does not reproduce, and two theorem validators that print their headline precision but never persist it.
2. **The natural-shift experiment tracks are thin** — Camelyon17 is missing its SAR baseline and its full-scale composition grid was never serialized (so the single most-wanted result, a *certified natural-shift policy win*, is not delivered); ImageNet-R / CIFAR-10.1 are single-seed.
3. **Internal status docs are stale** — the README and `EXECUTION_STATUS.md` under-claim relative to the finished paper (e.g. README still says "ImageNet-C pending"), and the paper's 5-theorem numbering doesn't match the repo's internal theorem ledger, which is a traceability hazard.

Severity legend: **CRITICAL** = blocks submission / integrity risk · **MAJOR** = a reviewer would flag it · **MINOR** = polish / hygiene. ✅ = independently re-verified against files during this audit.

---

## Part A — Research-integrity gaps

### A · CRITICAL
**None.** Every validator referenced for the 5 theorems exists and computes the claimed quantity; no `NotImplementedError`, no hardcoded "computed" numbers, no unseeded RNG in the theory suite.

### A · MAJOR

**A1 — Thm 3's cited number is not reproduced by the committed artifact. ✅**
Paper (`primary.txt:453`): *"e-process false-adapt 0.028 ≤ 0.05 (val_thm3_evalue.py)."*
Reality: `experiments/kbound/theory_validation/val_thm3_evalue.py` defaults `alpha=0.1`, and the only committed artifact `val_thm3_evalue_results.json` has `config.alpha = 0.1`, `verdict.worst_case_false_adapt_h0 = 0.0626` — i.e. the run demonstrates control at **0.0626 ≤ 0.10**, not **0.028 ≤ 0.05**. The string `0.028` never appears in this script's output anywhere in the repo (the `0.028` at `primary.txt:1226/1276/1287` is the unrelated SAR-online accuracy collapse). The *mechanism* (Ville anytime e-process) is sound; the *specific numeric pair the paper quotes* has no backing run.
**Fix:** either re-run at α=0.05 and commit the artifact, or correct the paper to the actual committed numbers (0.0626 ≤ 0.10).

**A2 — Thm 5 multiclass "to 10⁻¹⁶ over 4000 trials" has runnable code but no persisted artifact. ✅**
`val_thm5_multiclass.py` is **print-only** — it asserts `< 1e-9` and prints, but contains no `json.dump`/`open(...,'w')`/`write`/`savetxt`, and there is **no `results_thm5*.json`** in the validator directory (only the `.py` itself). The headline 10⁻¹⁶ figure exists solely as transient stdout.
This violates the paper's own standard ("every numeric claim traces to a script in the public repository") — the *code* traces, the *number* does not.
**Fix:** dump results to `results_thm5_multiclass.json` like its sibling validators.

### A · MINOR

**A3 — Thm 2 "Validated to 10⁻¹⁷" similarly persists nothing.** `val_thm2_regret.py` prints JSON to stdout but writes no file. It does `assert` loudly (so a passing run is meaningful), but the 10⁻¹⁷ value is not captured. Same fix as A2.

**A4 — Over-precise phrasing vs committed tolerance.** Thm 2/Thm 5 scripts check `< 1e-9`; the paper quotes `10⁻¹⁶`/`10⁻¹⁷` (`primary.txt:392,402,1696`). These are exact float64 algebraic identities so the claims are *plausibly* true, but the committed evidence only certifies `< 1e-9`. Either tighten the asserted tolerance or quote `< 10⁻⁹`.

**A5 — Theorem numbering is inconsistent across the project (traceability hazard).** The published paper uses a consolidated "exactly five theorems" scheme; `docs/research/kbound/THEOREM_CODE_STATUS.md` uses an older scheme (there, Thm 4 = covariate, Thm 5 = binary sign-of-difference). Published Thm 4 (one-bit dichotomy) maps to `theory_v2/`; published Thm 3 ↔ repo "T5 switching_certificate"; published Cor. 1 (covariate) ↔ repo "T4 risk_dominance". Separately, `tests/test_theorem_registry.py` and `tests/test_novel_theorem_bounds.py` operate on a *different* (LEGACY-PROJECT) theorem set and are **not** tests of the paper's five theorems — easy for an auditor to mistake. **Fix:** add a one-page numbering crosswalk and rename/annotate the registry tests.

**A6 — Two named label-free baselines are surrogates, not the real methods.** In the Thm 3 frontier discussion and Table X, **AETTA** and **agreement-on-the-line** appear as decision-style surrogates; grep of the decision-baselines JSON confirms no `aetta`/`agreement` keys are actually run. The paper says so explicitly, but a reader skimming the baseline list would over-count. (Also a coverage item — see B-minor.)

### A · Known-open (acknowledged in §VIII, not a defect)
**Conjecture 1** — the weakest falsifiable structural class under which the one-bit supplement suffices — is correctly left open (`primary.txt` §VIII). It is honestly scoped, not force-closed; listed here only for completeness.

---

## Part B — Experiment-coverage gaps

### B · CRITICAL
**None (no fabrication).** Every numeric result spot-checked traces to a real artifact; the isolated `smoke:true` toy runs are not cited by any table. The closest-to-critical item is a disclosed honesty issue, B3 below.

### B · MAJOR

**B1 — Camelyon17 is missing its SAR baseline. ✅**
Promised TTA candidate set is Tent / EATA / SAR. The headline Camelyon17 artifacts (`results/wilds/wilds_camelyon17_kga.json` and `results/camelyon17_fullscale_B_v1/wilds_camelyon17_kga.json`) contain **only `tent` and `eata`** — `sar` is absent. SAR *is* present on ImageNet-C, CIFAR-10.1, and RxRx1, so this is a Camelyon17-specific omission in the one benign-regime natural-shift table. **Fix:** run SAR on Camelyon17 or state its exclusion explicitly in the table caption.

**B2 / B3 — The natural-shift "policy win" is not delivered, and the stress-grid headline CI is weak — both because per-condition arrays were not serialized.**
- B2: `camelyon17_fullscale_B_v1/LOCKED_B_ANALYSIS.json` records that the pre-registered 72×6 composition grid "was NOT written to the new output"; only 5 per-seed aggregates survive. So ε-recalibration is only coarse, and the detectable-harm natural-shift win stays *latent / uncertified* (false-adapt 0.185 ≈ 1.9×α). The single most-wanted result — a *certified* natural-shift "beats both" — is therefore absent.
- B3: The marquee CIFAR-10-C **stress-grid** "beats both" claim cannot be given a proper per-condition CI because "its per-condition arrays were not serialized in the original runs" (`primary.txt:918–922`); the interval falls back to the weaker mixing bootstrap. The 5-seed `stress_grid_multiseed_v1/` re-run supports the point but the tight CI is still owed.
**Fix (covers both):** serialize per-condition arrays in the re-runs (already "scheduled" per the paper) and recompute paired per-condition CIs.

**B4 — Seed coverage on natural shifts is much thinner than on synthetic.** ImageNet-R and CIFAR-10.1 are **single-seed** ("light"/"quick" protocols); the Camelyon17 composition grid and RxRx1 are debug/light-scale (n_eval≈256 / light MPS). Only the synthetic CIFAR-10-C grid (5 seeds) and the anomaly mixed-regime (8 seeds) reach statistical depth. Each is labeled honestly, but the natural-shift evidence is statistically light. **Fix:** multi-seed re-runs (the paper already flags these as future work).

### B · MINOR

**B5 — README / EXECUTION_STATUS are stale and under-claim. ✅** `README.md:24` says "full ImageNet-C scale is pending" and `:22` advertises "9 real experiments"; the finished paper reports ImageNet-C complete (Tables IX/XI, backed by real JSONs) plus Camelyon17 / ImageNet-R / RxRx1 / CIFAR-10.1. `EXECUTION_STATUS.md` likewise lists CIFAR-10-C / multiclass-Thm5 as "needs GPU / pending" — superseded by the committed artifacts. A reviewer reading only these docs would undercount the work. **Fix:** refresh README status block and mark EXECUTION_STATUS/CHECKLIST as historical.

**B6 — "ImageNet-scale corruptions" is narrower than it sounds.** The headline ImageNet-C is the **noise corruptions only** (gaussian/shot/impulse × sev {1,3,5} = 36 cells), not the full 15-corruption ImageNet-C; ViT-B scale is explicitly future work. The text is technically accurate ("noise grid") but the abstract framing invites over-reading. **Fix:** say "ImageNet-C noise corruptions (36 cells)" in the abstract.

**B7 — Results directory clutter creates provenance ambiguity.** Many stale/smoke/internal siblings (`imagenetr_kbound_{smoke,debug_mps,1pct,full_mps,full_mps_internal,..._STALE33_backup}`, `imagenetc_{smoke,smoke01,sarfix_smoke,1pct,...}`, `wilds_{smoke,1pct,debug_mps}`). The canonical files carry `smoke:false`; no table cites a `smoke:true` file (good), but the naming invites mistakes. **Fix:** move non-canonical runs to an `archive/` and add a `RESULTS_INDEX.md` mapping each paper table → exact artifact path.

**B8 — Explicitly scoped-out future work (not defects, listed for a complete ledger):** CIFAR-100-C, ViT-B scale, WILDS iWildCam, TTT/SHOT baselines, and SAR's official gentler schedule (lr 2.5×10⁻⁴, layer4 frozen) — all named as future work in §VIII.

---

## Part C — What is solid (fair credit)

- **The theory validators are genuine and rigorous.** `val_thm1_lecam.py` (the strongest: closed-form + debiased-TV + MC `inf_g M` tracking `1−TV`, with `results_thm1_lecam.json` showing witness abstain=1.0), `val_frontier.py` (identity over 20 000 draws, `identity_violations:0`; frontier law over 5×10⁵), `val_thm2_regret.py` (exact identity + realized-loss MC + corollaries), `val_agl.py` (R²>0.9999). All seeded; integrity grep over the suite is clean.
- **The anomaly-routing core is fully backed and numbers match exactly** — mixed regime (freeze 0.586 / KGA 0.690 / oracle 0.719), 8-seed rigor (0.686±0.003; t=62.2, p=7.3×10⁻¹¹ vs freeze), regression covariate shift, clean Thm-1 witness, ablations, +62-task breadth.
- **CIFAR-10-C in all three promised forms** (65-cell per-corruption, 432-condition stress grid with 5 seeds, online non-stationary streams with both helpful + harm regimes).
- **ImageNet-C including the mechanism-faithful SAR re-run** — the central "harmful-candidate win survives" claim is backed (SAR harmful 0.444, KGA regret 0.0229, `beats_both=True`, matching Table XI).
- **Data is genuinely on disk** for every neural benchmark (CIFAR-10/-C/-10.1, ImageNet-C noise, ImageNet-R, RxRx1, Imagenette); reproduction scripts referenced in the paper all exist.
- **Honesty infrastructure exists:** `validate_manuscript_claims.py` is a real forbidden-token linter wired into the test suite, and tests like `test_gdr_binomial_is_honest_*` pin non-significant results so they can't be quietly upgraded.

---

## Part D — Prioritized action checklist

Do first (integrity, cheap, blocks a clean submission):

1. **A1** — Re-run `val_thm3_evalue.py` at α=0.05 and commit the artifact, *or* correct `primary.txt:453` to the committed (0.0626 ≤ 0.10). *(highest priority — concrete claim/evidence mismatch)*
2. **A2 + A3** — Make `val_thm5_multiclass.py` and `val_thm2_regret.py` persist `results_*.json`; re-state precision as the actually-certified tolerance (A4).
3. **B5** — Refresh README status block; mark EXECUTION_STATUS / CHECKLIST as historical.
4. **A5** — Add a theorem-numbering crosswalk; annotate the LEGACY-PROJECT registry tests so they aren't mistaken for K-Bound theorem tests.

Do next (coverage / reviewer-facing):

5. **B1** — Run SAR on Camelyon17 (or caption its exclusion).
6. **B2/B3** — Serialize per-condition arrays in the scheduled re-runs; recompute paired per-condition CIs for the stress grid and the Camelyon17 ε-recalibration; only then claim a certified natural-shift win.
7. **B4** — Multi-seed re-runs for ImageNet-R / CIFAR-10.1 (and full-scale RxRx1) before any are promoted from "scoping evidence" to "headline".
8. **B7** — Archive non-canonical runs; add `RESULTS_INDEX.md` (table → artifact path).

Polish / wording:

9. **B6** — Tighten abstract framing to "ImageNet-C noise corruptions (36 cells)".
10. **A6** — Flag AETTA / agreement-on-the-line as surrogates wherever the baseline list appears.

---

## Appendix — Audit method, coverage, and limitations

**Files examined (primary):** `primary.txt`; `README.md`; `docs/research/kbound/{THEOREM_CODE_STATUS,EXECUTION_STATUS,CHECKLIST_8PLUS_GAP_ANALYSIS}.md`; `research_lock/{README.md,M2_FINAL_AUDIT_PENDING_v1.yaml}`; `audits/training_truth_audit/{11_audit_summary.csv,12_training_critical_fixes.md}`; validators `experiments/kbound/theory_validation/val_*.py` + their `results_*.json`; `vendored_from_legacy-project/certification/switching_certificate.py`; ~25 experiment result JSONs across `experiments/kbound/results/**`; theorem/boundary tests under `tests/`.

**Independently re-verified during this audit (✅ items):** A1 (Thm 3 α/false-adapt), A2 (Thm 5 no artifact), B1 (Camelyon17 baselines), B5 (README text).

**Reported by sub-audit but not exhaustively re-run here (treat as high-confidence, not personally re-executed):** exact 10⁻¹⁷/10⁻¹⁶ magnitudes (no artifact persists them; tolerance is 1e-9), companion-appendix validators (`val_conj1_*`, `val_reach_unification.py`, etc.), and per-seed contents of multi-seed grids.

**Limitations of this audit:** static only — no validators or tests were executed (heavy ML deps + slow external drive); the 166-page `manuscript.txt` and several `docs/research/kbound/` inventories were grepped, not read line-by-line; newer in-progress run dirs (`*_protocol_c/d_*`, `imagenetc_official_sar_E_v1_s{0,1,2}`) appear to post-date the paper tables and were not fully mapped. Findings about *missing* artifacts mean "not committed to the repo as inspected on 2026-06-14", not "never computed".

---

## Resolution log (2026-06-19) — multi-agent loop

**Updated executive summary.** A four-agent gap-closure loop (cartographer → theory → experiment → integrity-verifier) re-walked every audit item. **No fabrication was found anywhere** — every theory validator was *executed this session* (not just inspected), all use fixed seeds with deterministic per-cell derivation, no hardcoded "computed" constants, and the new serialization module reshapes already-measured records only. **All research-integrity items (A1–A6) and all data-backed documentation items are now resolved or correctly reclassified.** The only work that genuinely remains is (i) the user's **Mac GPU re-runs** of two natural-shift tracks (ImageNet-R multi-seed; Camelyon17 `B_v2` full-scale completion) and (ii) one **honestly-open theory sub-conjecture** (the *unconditional* weakest one-bit class / general-position removal in Conjecture 1, which is explicitly left open in the paper, not force-closed).

**Verification method this loop:** the integrity-verifier *ran* all seven main-theorem validators + the CPU pipeline harness + the torch-free K-Bound certificate test suite in the sandbox, and read every NEW source file for anti-fabrication. Real headline numbers and exit codes are recorded inline below.

### Independent re-execution (real numbers, exit code 0 unless noted)
- `val_thm1_lecam.py` — Le Cam floor `1−TV` tracks empirical `inf_M` across the μ/n sweep (e.g. μ=0.25,n=64: floor 0.0455 vs inf_M 0.0452). PASS.
- `val_thm2_regret.py` — exact identity `max gap = 2.35e-17`; minimax ratio-to-floor 1.0004; `ALL CHECKS PASSED`.
- `val_thm2_lecam_finite_n.py` **(NEW)** — floor constant in n per c; Bayes (optimal) label-free rule **meets** the floor; brute-force panel over 5 label-free statistics × both polarities **never beats** the floor; Bretagnolle–Huber `(Λ/4)e^{-2c²} ≤ floor` for all cells; `|Δ|=1` exact. `ALL CHECKS PASSED`. Re-run is bit-identical (seed 20260619), confirming determinism.
- `val_thm3_evalue.py --alpha 0.05` — **worst-case anytime false-adapt = 0.0316 ≤ α=0.05** (verdict + `results_thm3_evalue_alpha005.json`). PASS.
- `val_thm5_multiclass.py` — multiclass identity `max err 1.11e-16`, sign agreement 4000/4000; regression identity `max err 5.6e-13`; covariate certificate correct, concept-shift certificate honestly fails. `ALL CHECKS PASS`.
- `val_conj1_genpos.py` **(NEW)** — general-position obstruction **CONFIRMED** (200,000 trials): `C_dom` strictly larger than `C_mono`, one bit certifies sign(Δ) on `C_dom` (error 0.0) but fails without domination (error 0.2509 ≈ ¼). Conclusion printed by the validator itself: **the unconditional weakest class remains OPEN**.
- `val_conj1_caltransfer.py` — gap-sweep commit/recovery behaviour and bracket-containment 1.0 as designed. PASS.
- `verify_runner_pipeline.py --smoke` — `ALL_ASSERTIONS_PASSED = True`; every metric explicitly labelled `SYNTHETIC` / `_pipeline_smoke_verify`; SAR column present; paired Holm-corrected CIs computable for both datasets. PASS.
- **Torch-free certificate tests:** `pytest test_kga_package.py test_holm_bonferroni.py test_phase2_certificate_boundary.py test_gate_p_and_scope_guard.py test_holm_family_size_matches_registry.py` → **54 passed, 2 skipped** (skips are missing optional `PAPER_DRAFT_v1.tex`/`THESIS_CHAPTER_v1.tex`). The K-Bound certificate logic — decision boundaries, e-value false-adapt-≤-α, non-identifiability witness, conformal radius — all pass. `test_audit_gate_decision_rule_e2e.py` and `test_gate_decision_rule.py` could **not** run (transitively `import torch` via `uais.fusion.attention`); SKIPPED per torch-free policy — environmental, not a logic regression.

### Item-by-item status

**A1 (Thm 3 α-mismatch) — RECLASSIFIED (the 2026-06-14 fix instruction was a MISREAD).**
The 2026-06-14 log said "paper digit `0.028` → `0.0316`". On direct inspection of `docs/research/kbound/kbound.tex`, the four `0.028`/`0.0283` occurrences are **Office-Home regret numbers** (L1155 `KGA=freeze 0.028/0.027`; L1181 `eata_online_mild: KGA/adapt/freeze regret 0.028/0.001/0.028`) and **anomaly-routing CI bounds** (L1407/L1417 `+0.0283 [0.020,0.037]`) — **none is the e-process false-adapt rate**, and `0.0316` appeared **nowhere** in the .tex. Editing those `0.028`s to `0.0316` would have *corrupted correct regret values* (a fabrication). **Action taken instead:** the verified e-value result (`worst-case anytime false-adapt 0.0316 ≤ α=0.05`, from `results_thm3_evalue_alpha005.json`) was added *conservatively and quantitatively* to the anytime e-process discussion in the companion-stack appendix (`kbound.tex`, App. "demoted theorem stack"), where it was previously only qualitative. The Thm-3 *mechanism* and *guarantee* were never in doubt; only the audit's pointer was wrong. **RESOLVED + reclassified.**

**A2 (Thm 5 artifact) — ALREADY-CLOSED.** `results_thm5_multiclass.json` present and re-confirmed by re-execution (max err 1.11e-16).

**A3 (Thm 2 artifact) — ALREADY-CLOSED.** `results_thm2_regret.json` present; re-execution gives max gap 2.35e-17.

**A4 (over-precise tolerance) — CLOSED-THIS-LOOP (theory agent).** Both `val_thm2_regret.py` (comment at the `exact_identity_holds` gate) and `val_thm5_multiclass.py` (two comments at the equality gates) now explicitly state that `1e-9` is a *deliberately loose* pass gate and the realized error sits at the float64 round-off floor (~2.35e-17 / ~1.1e-16). Asserts unchanged (still `< 1e-9`), so the gates keep BLAS/platform slack. Verified by direct read.

**A5 (theorem numbering crosswalk) — ALREADY-CLOSED, with a caveat.** `THEOREM_NUMBERING_CROSSWALK.md` was created 2026-06-14, and the paper now ships a consolidated "exactly five theorems" scheme with an in-paper consolidation table (`tab:consolidation`, App.) mapping each main theorem to its demoted results + validators. **Caveat (honest):** the standalone `docs/research/kbound/THEOREM_NUMBERING_CROSSWALK.md` shows as a *tracked deletion* (` D`) in the current working tree — it was removed as part of a broader `docs/research/kbound/*.md` cleanup, not lost to error. The crosswalk *content* survives in-paper; if a standalone file is still wanted it should be restored from git history. Not an integrity defect; flagged for the record.

**A6 (AETTA / agreement-on-the-line are surrogates) — UNCHANGED / already honest.** The paper labels these as decision-style surrogates explicitly; no run claims them as executed methods. No change needed.

**B1 (Camelyon17 missing SAR) — RECLASSIFIED (already done in the runner).** The experiment agent confirmed `run_camelyon17_kbound.py` **already runs SAR**; the SAR-less `results/wilds/wilds_camelyon17_kga.json` the 2026-06-14 audit flagged is a **stale older artifact**, not the current runner's behaviour. The CPU smoke harness shows the SAR column present for Camelyon17. No canonical result JSON was altered (mtimes preserved). **RESOLVED at the code level**; the SAR-bearing full-scale numbers come from the GPU re-run (see B2/B_v2).

**B2 / B3 (per-condition serialization + certified natural-shift win + tight stress-grid CI) — INFRASTRUCTURE CLOSED-THIS-LOOP; numbers NEED-MAC-GPU.** New torch-free `experiments/kbound/wilds/per_condition_serialize.py` emits the exact `per_condition_<dataset>_<method>_seed<S>.json` schema the locked stress-grid analysis consumes, and `experiments/kbound/wilds/multiseed_paired_ci.py` computes the paired, Holm-corrected per-condition CIs. `verify_runner_pipeline.py --smoke` proves the full serialize→aggregate→paired-CI pipeline end-to-end on synthetic data (labelled SYNTHETIC). The *plumbing* that was missing is now in place and tested; the *real* certified-win numbers still require the GPU runs.

**B4 (thin seed coverage on natural shifts) — PARTIALLY CLOSED (CIFAR-10.1) / NEEDS-MAC-GPU (ImageNet-R).**
- **CIFAR-10.1: CLOSED-THIS-LOOP (data-backed doc fix).** `experiments/kbound/results/cifar101_multiseed_v1/pooled_summary.json` confirmed: **seeds [0,1,2,3,4]**, 24 conditions/seed, completed 2026-06-16. The stale `kbound.tex` caption ("single quick pass… single-seed quick protocol; a multi-seed re-run is future work") was updated to cite the real 5-seed pooled numbers (harmful-condition rate 0.675±0.017 Tent / 0.500±0.059 EATA / 0.050±0.031 SAR; KGA regret-to-oracle 0.0024±0.0007 / 0.0035±0.0011 / 0.0045±0.0014). Single-pass table body left unchanged (distinct artifact).
- **ImageNet-R: OPEN / NEEDS-MAC-GPU.** Still single-split. `run_imagenetr_kbound.py` now has `--resume` + serialization wired; the multi-seed run itself is a GPU job.

**B5 (stale README) — ALREADY-CLOSED / WITHDRAWN (false positive, per 2026-06-14).** README is accurate. No change.

**B6 ("ImageNet-scale corruptions" over-reads) — CLOSED-THIS-LOOP (conservative wording).** The contributions bullet in `kbound.tex` (L160) read "ImageNet-scale corruptions"; softened to **"ImageNet-C noise corruptions (ResNet-50, 36 cells)"**, matching the paper's own table captions (which already say "ImageNet-C, ResNet-50… 36 cells, gaussian/shot/impulse × {1,3,5}") and the body text at L803.

**B7 (results-dir clutter) — UNCHANGED (hygiene, out of scope this loop).** Canonical files still carry `smoke:false`; no table cites a smoke file. Left as a non-integrity polish item.

**B8 (explicitly scoped-out future work) — UNCHANGED.** CIFAR-100-C, ViT-B, iWildCam, TTT/SHOT, SAR's official gentler schedule remain named future work.

### Theory targets (this loop)

**Thm 2 → STRENGTHENED.** New Proposition `prop:lecam-finite` (+ `cor:lecam-regret-floor`) with full proof in `docs/research/kbound/paper/sections/main_theory_5.tex` upgrades Thm 2 from an exact identity + asymptotic/plug-in minimax floor to a **genuine finite-n two-point Le Cam lower bound** on the expected regret of any label-free gate, with a closed-form Bretagnolle–Huber certificate `(Λ/4)e^{-2c²}>0`. Backed by NEW `val_thm2_lecam_finite_n.py` + `results_thm2_lecam_finite_n.json` (ALL CHECKS PASS, re-run deterministic).

**Conjecture 1 (general position) → remains OPEN, now SHARPENED.** The theory agent could **not** remove the general-position assumption and (correctly) did not claim to. `weakest_class.tex` now states plainly "the unconditional weakest class remains open" and "We leave this open and do not claim it," with an explicit two-equal-mass-region counterexample (`rmk:genpos` / `thm:cmono-weakest`). NEW `val_conj1_genpos.py` + `results_conj1_genpos.json` machine-check the obstruction. **Honest open status verified — not force-closed.**

### Loop-back decision
**No closeable-here item remains undone.** Every integrity item (A1–A6) and every data-backed documentation item (CIFAR-10.1 staleness, B6 wording, the A1 e-value addition) is resolved or correctly reclassified, and all theory work is either strengthened (Thm 2) or honestly left open (Conj 1 general position). The **only** remaining work is outside this sandbox: the user's **Mac GPU runs** — ImageNet-R multi-seed and the Camelyon17 `B_v2` full-scale completion (see `docs/research/kbound/RUN_ON_MAC.md` and `scripts/run_remaining_gpu_experiments.sh`) — plus the standing **Conjecture 1 general-position** sub-case, which is a research-open problem, not a documentation gap.
