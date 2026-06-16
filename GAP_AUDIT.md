# K-Bound — Gap Audit (Research Integrity + Experiment Coverage)

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

**A5 — Theorem numbering is inconsistent across the project (traceability hazard).** The published paper uses a consolidated "exactly five theorems" scheme; `docs/research/kbound/THEOREM_CODE_STATUS.md` uses an older scheme (there, Thm 4 = covariate, Thm 5 = binary sign-of-difference). Published Thm 4 (one-bit dichotomy) maps to `theory_v2/`; published Thm 3 ↔ repo "T5 switching_certificate"; published Cor. 1 (covariate) ↔ repo "T4 risk_dominance". Separately, `tests/test_theorem_registry.py` and `tests/test_novel_theorem_bounds.py` operate on a *different* (ELARA) theorem set and are **not** tests of the paper's five theorems — easy for an auditor to mistake. **Fix:** add a one-page numbering crosswalk and rename/annotate the registry tests.

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
4. **A5** — Add a theorem-numbering crosswalk; annotate the ELARA registry tests so they aren't mistaken for K-Bound theorem tests.

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

**Files examined (primary):** `primary.txt`; `README.md`; `docs/research/kbound/{THEOREM_CODE_STATUS,EXECUTION_STATUS,CHECKLIST_8PLUS_GAP_ANALYSIS}.md`; `research_lock/{README.md,M2_FINAL_AUDIT_PENDING_v1.yaml}`; `audits/training_truth_audit/{11_audit_summary.csv,12_training_critical_fixes.md}`; validators `experiments/kbound/theory_validation/val_*.py` + their `results_*.json`; `vendored_from_elara/certification/switching_certificate.py`; ~25 experiment result JSONs across `experiments/kbound/results/**`; theorem/boundary tests under `tests/`.

**Independently re-verified during this audit (✅ items):** A1 (Thm 3 α/false-adapt), A2 (Thm 5 no artifact), B1 (Camelyon17 baselines), B5 (README text).

**Reported by sub-audit but not exhaustively re-run here (treat as high-confidence, not personally re-executed):** exact 10⁻¹⁷/10⁻¹⁶ magnitudes (no artifact persists them; tolerance is 1e-9), companion-appendix validators (`val_conj1_*`, `val_reach_unification.py`, etc.), and per-seed contents of multi-seed grids.

**Limitations of this audit:** static only — no validators or tests were executed (heavy ML deps + slow external drive); the 166-page `manuscript.txt` and several `docs/research/kbound/` inventories were grepped, not read line-by-line; newer in-progress run dirs (`*_protocol_c/d_*`, `imagenetc_official_sar_E_v1_s{0,1,2}`) appear to post-date the paper tables and were not fully mapped. Findings about *missing* artifacts mean "not committed to the repo as inspected on 2026-06-14", not "never computed".
