# Phase 1.1.1 — Residual Caption / Policy Patch Report

**Status:** PASS. All three residual issues identified by independent render inspection of `PAPER_DRAFT_PHASE1_1_VERIFIED.pdf` are fixed.

**Branch:** `fix/elara-phase1-1-pdf-source-consistency` (continuation; Phase 1.1.1 is a small patch sub-stage).

---

## 1. Issue 1 — "Family A confirmatory" → "Family A audited-primary"

| Site | Before | After |
|---|---|---|
| `docs/research/PAPER_DRAFT_v1.tex:1574` | `Holm-corrected within the locked Family~A confirmatory set $K{=}5$` | `Holm-corrected within the locked Family~A audited-primary reanalysis family $K{=}5$` |

- Source forbidden-token sweep (POSITIVE noun-phrase pattern `Family A confirmatory (set|family|cells|K=…)`): **0 hits** in both paper and thesis sources.
- PDF text extraction (POSITIVE noun-phrase pattern): **0 hits** in `PAPER_DRAFT_PHASE1_1_1_VERIFIED.pdf` and `THESIS_CHAPTER_PHASE1_1_1_VERIFIED.pdf`.
- Defensive disclaimer "Family A is never called confirmatory" in the thesis audited-policy subsection is intentionally retained as a negative-context statement.

## 2. Issue 2 — Canonical MVTec figure caption claims PR-AUC display

| Site | Before | After |
|---|---|---|
| `docs/research/PAPER_DRAFT_v1.tex:910` (caption of `fig:mvtec-clean-benchmark`) | "Clean benchmark ROC-AUC and PR-AUC across score-level and attention fusion methods on naturally paired MVTec 3D-AD." | "Protocol-diagnostic ROC-AUC on naturally paired MVTec~3D-AD under the canonical one-class fusion protocol. PR-AUC, ECE, and Brier are omitted from promoted canonical results following the Phase~1.A audit ..." |
| `docs/research/PAPER_DRAFT_v1.tex:942` (caption of `tab:mvtec-clean-ci`) | "Naturally paired MVTec 3D-AD: clean ROC-AUC, PR-AUC, and F1 95\\% confidence intervals across configured seeds." | "Naturally paired MVTec~3D-AD canonical one-class: ROC-AUC mean and 95\\% confidence interval across configured seeds. PR-AUC, ECE, and Brier are omitted as protocol-diagnostic prevalence-valued artefacts (Phase~1.A audit)." |

Visual confirmation on rendered PNG of paper page 11 shows the ROC-AUC-only table with no PR-AUC values.

## 3. Issue 3 — Secondary hard-mode tables/figure not marked SECONDARY

| Site | Action |
|---|---|
| `docs/research/PAPER_DRAFT_v1.tex` Table XVII (`tab:adversarial-results`) caption | Prepended `\textbf{SECONDARY DESCRIPTIVE SURFACE:}` with explicit reference to PRIMARY B1/B2 values |
| `docs/research/PAPER_DRAFT_v1.tex` Table XVIII (`tab:tau-sweep`) caption | Prepended `\textbf{SECONDARY DESCRIPTIVE SURFACE:}` with PRIMARY cross-reference |
| `docs/research/PAPER_DRAFT_v1.tex` Figure 11 (`fig:adversarial-delta`) caption | Appended `\textbf{Secondary descriptive surface}` paragraph |
| `docs/research/PAPER_DRAFT_v1.tex` §sec:adversarial prose | Rewrote the paragraph that listed +0.0506/+0.0319 to clearly say those values are PRIMARY from the k-of-D evaluation, and the displayed +0.0367/+0.0538 are SECONDARY from the default-gate path |
| `docs/research/THESIS_CHAPTER_v1.tex` `tab:thesis-adversarial` caption | Same SECONDARY label added (shared generated asset causes the same visible issue) |
| `docs/research/THESIS_CHAPTER_v1.tex` `fig:thesis-adversarial` caption | Same SECONDARY label added |
| `docs/research/THESIS_CHAPTER_v1.tex` `tab:thesis-tau` caption | Same SECONDARY label added |

PDF text extraction:
- Paper: 2× `SECONDARY DESCRIPTIVE SURFACE` + 1× `Secondary descriptive` (3 total).
- Thesis: 2× `SECONDARY DESCRIPTIVE SURFACE` + 1× `Secondary descriptive` (3 total).
- PRIMARY `+0.0506` / `+0.0319` still appears in both manuscripts (8 / 8 in paper; 3 / 2 in thesis). The PRIMARY mechanism claim is preserved.

## 4. Test coverage

New file: `tests/test_phase1_1_1_residual_patch.py` with 8 tests covering:
- Source-level Family A confirmatory absence (paper + thesis).
- PDF-level Family A confirmatory absence (positive-noun pattern; allows defensive disclaimer).
- Canonical MVTec figure caption: no `benchmark ROC-AUC and PR-AUC` phrasing; must contain `Protocol-diagnostic`.
- `tab:adversarial-results`, `tab:tau-sweep`, `fig:adversarial-delta` captions must contain SECONDARY marker.
- Abstract and primary mechanism in both manuscripts must retain +0.0506 and +0.0319.

All 8 tests PASS. Full suite **391 passed, 2 skipped** (+8 over Phase 1.1's 383).

## 5. No regression

Phase 1.1 forbidden-string scan rerun: 0 forbidden hits on the original Phase 1.1 token list. The "benchmark ROC-AUC and PR-AUC" hits found in the paper (1) and thesis (1) are from the ELARA-Bench-LA figure caption (line 910 paper, line 779 thesis) — a non-canonical figure where ROC + PR display is valid. The canonical MVTec figure caption (line 910 paper, the one targeted by Issue 2) does not contain that phrase.

## 6. Verdict

All three Phase 1.1.1 residual issues are fixed. Phase 1.1 remains valid. Phase 2 not begun.
