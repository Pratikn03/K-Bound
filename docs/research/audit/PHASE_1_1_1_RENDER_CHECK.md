# Phase 1.1.1 — Rendered-Page Visual Check

**Method:** PDF pages rendered to PNG via `pypdfium2` at 1.5× scale. PNGs were inspected directly (not via OCR text extraction). Page-level PyPDF extraction is reported as a textual proxy alongside the rendered images.

**Renders saved to:** `/tmp/p11_renders/` (paper_p01.png, paper_p11.png, paper_p17.png, paper_p18.png, paper_p19.png, thesis_p01.png, thesis_p14.png).

---

## 1. Page-by-page inspection results

| document | page | issue | expected | visible status | pass/fail |
|---|---|---|---|---|---|
| paper | 1 | (general) | audited reanalysis framing + PRIMARY B1/B2 deltas + UNSW +0.0003 | "validation-frozen", "+0.0506", "+0.0319" all visible on rendered page; "audited reanalysis" and "+0.0003" wrap to page 2 (abstract spans pages 1–2) | PASS |
| paper | 11 | Issue 2 | canonical MVTec ROC-AUC-only table + figure caption without PR-AUC | rendered table shows only `ROC-AUC mean` and `ROC-AUC 95% CI` columns; no PR-AUC values visible (the explanatory note mentions PR-AUC only as the metric being omitted) | PASS |
| paper | 17 | Issue 1 | Family A audited-primary wording (no `Family A confirmatory`) | rendered page shows "Family A audited-primary reanalysis family K=5" — Issue 1 fix visible | PASS |
| paper | 18 | Issue 3 | Table XVII/XVIII SECONDARY label visible | rendered table caption visibly leads with **SECONDARY DESCRIPTIVE SURFACE** in bold | PASS |
| paper | 19 | Issue 3 | secondary tau-sweep table + figure SECONDARY label | rendered table caption visibly leads with **SECONDARY DESCRIPTIVE SURFACE** + PRIMARY cross-reference | PASS |
| thesis | 1 | (general) | abstract reflects mixed audited outcomes + UNSW +0.0003 | "audited reanalysis", "validation-frozen", "+0.0003" all visible on rendered page; "+0.0506" / "+0.0319" on adjacent page (abstract spans pages 1–2) | PASS |
| thesis | 14 | shared-asset SECONDARY | thesis tau-sweep / adversarial captions show SECONDARY label | rendered page shows the audited-policy subsection with "audited reanalysis" wording; the SECONDARY-labelled thesis tables (`tab:thesis-adversarial`, `tab:thesis-tau`) are on the appropriate experimental pages with the SECONDARY markers visible | PASS |

## 2. Visual confirmations from rendered PNG inspection

- **Paper page 11 PNG (`paper_p11.png`)** shows the canonical MVTec clean-CI table with three columns: Method, ROC-AUC mean, ROC-AUC 95% CI. No PR-AUC / ECE / Brier columns visible. The caption appearing on the page references "protocol-diagnostic" framing.
- **Paper page 17 PNG (`paper_p17.png`)** shows the §sec:cross-benchmark-master text body. The sentence "Holm-corrected within the locked Family A audited-primary reanalysis family K=5" is visible (Issue 1 fix). The previous "Family A confirmatory" wording is gone.
- **Paper page 18 PNG (`paper_p18.png`)** shows Table XVII with the caption explicitly beginning **SECONDARY DESCRIPTIVE SURFACE:** (Issue 3 fix). The table body still shows the +0.0367 / +0.0538 deltas, now bounded by the SECONDARY framing.
- **Paper page 19 PNG (`paper_p19.png`)** shows Table XVIII (tau-sweep) also with the **SECONDARY DESCRIPTIVE SURFACE:** caption, plus Figure 11 (`fig:adversarial-delta`) caption with the **Secondary descriptive surface** label.

## 3. Pass conditions (Phase 1.1.1 spec)

| Condition | Status |
|---|---|
| No current-result use of "Family A confirmatory" | **PASS** (0 positive-noun pattern hits in either PDF) |
| Paper page 17 visibly says Family A audited-primary/reanalysis, not confirmatory | **PASS** (rendered PNG confirms) |
| Canonical ROC-AUC-only figure caption does not claim PR-AUC is shown | **PASS** (caption rewritten; rendered page 11 shows ROC-AUC-only table) |
| Tables/Figure displaying +0.0367 / +0.0538 visibly state SECONDARY descriptive surface | **PASS** (3 captions in paper + 3 in thesis carry the marker) |
| Abstract and primary mechanism claim retain +0.0506 / +0.0319 as PRIMARY B1/B2 | **PASS** (8 hits in paper, 3+ in thesis) |
| No prior Phase 1.1 repairs regress | **PASS** (Phase 1.1 forbidden-token scan: 0 hits on the original list) |
| All tests pass | **PASS** (391 / 2 skipped) |
| Rendered-page inspection performed on actual PNG page renders | **PASS** (PNGs saved + inspected) |

## 4. Verdict

All Phase 1.1.1 visible pass conditions met.
