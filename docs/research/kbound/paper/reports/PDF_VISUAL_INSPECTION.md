# K-Bound final PDF visual inspection

## Verdict

**PASS — no blocking visual or structural defect remains in any maintained PDF.**

This report covers the release built from source snapshot `efecbafd74e192c1314ea81a0aa8868a8e784a94` (`efecbafd74e1`). Every one of the 186 final pages was rendered at 192 DPI. Every changed page was inspected at original render resolution; unchanged pages were accepted only after exact pixel identity to the immediately preceding fully inspected build. Contact sheets cover all final pages.

## Final artifact identity

| Role | File | Pages | SHA-256 | Result |
|---|---|---:|---|---|
| Short main paper | `kbound_short_main.pdf` | 20 | `8a60e37a3212236b2c2fce044e35eeeaab2e788c4fbbd2a3cc92463920b29dae` | PASS |
| Standalone supplement | `kbound_short_supplement.pdf` | 26 | `7549e8370ebb2035a5a6f08143d281f28d1f453d6d5818c20c5916dda27eea0a` | PASS |
| Maintained TMLR manuscript | `kbound_tmlr.pdf` | 43 | `a33dbbb403eedeb4df532bec1c697a99b660e8ef7c7e1e0f5fff1c303951ea62` | PASS |
| Full technical report | `kbound_full_report.pdf` | 97 | `9de2fac1405bb5f89077d2ae8fc9ccdbf636b068f9f964684f9edcc2a5b1c6d1` | PASS |

Release directory inspected:

`/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current`

## Reproducible inspection procedure

The definitive render command was:

```sh
python3 docs/research/kbound/scripts/render_pdf_pages.py \
  --pdf-root /tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current \
  --output-root /Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/kbound/paper/reports/final_renders \
  --dpi 192
```

It produced 186 valid PNGs, all `1632 x 2112` pixels:

- short main: 20/20
- supplement: 26/26
- TMLR: 43/43
- full report: 97/97

The definitive structural command was:

```sh
python3 docs/research/kbound/scripts/verify_pdf_structure.py \
  --pdf-root /tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current \
  --output-root /tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/paper/reports/final_structural_verification
```

Each PDF passed `pdfinfo`, `pdffonts`, `pdftotext -layout`, and a Ghostscript null-device parse. All pages are US Letter (`612 x 792 pt`), all fonts are embedded, text extraction is clean, the files are unencrypted, and Ghostscript reported no parse defect. The extracted text contains no unresolved-reference marker. The anonymous TMLR build contains no author name, institution, or email leak.

The baseline comparison command was:

```sh
python3 docs/research/kbound/scripts/compare_pdf_renders.py \
  --baseline-root /Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/kbound/paper/reports/baseline_renders \
  --current-root /Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/kbound/paper/reports/final_renders \
  --output-root /Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/kbound/paper/reports/render_diffs
```

Evidence locations:

- full-resolution renders: `paper/reports/final_renders/`
- final all-page contact sheets: `paper/reports/visual_inspection/final_efec/`
- structural report: `paper/reports/final_structural_verification/PDF_STRUCTURAL_VERIFICATION.md`
- baseline/current render comparison: `paper/reports/render_diffs/VISUAL_RENDER_COMPARISON.md`
- TMLR per-page metrics: `paper/reports/visual_inspection/final_efec/tmlr_page_diagnostics/`

## Immediate-predecessor regression

The final source change shortened one provenance sentence and updated displayed source identities. Pixel comparison against the fully audited `55a38d0aa647` build localized every rendered change:

| Role | Changed final pages | Verification |
|---|---|---|
| Short main | 1 | only the displayed source snapshot; page reinspected |
| Supplement | 1, 14 | source snapshot on p.1; shortened provenance sentence on p.14; both reinspected |
| TMLR | 32 | shortened provenance sentence only; p.32 reinspected; no repagination |
| Full report | 1, 82, 83 | source snapshot plus provenance reflow; pp.82–83 and adjacent pp.81/84 reinspected |

All other pages are pixel-identical to the immediately preceding fully inspected build. The final p.82 provenance paragraph now ends cleanly, and full-report p.83 begins with “Locked thresholds.” The former three-word widow is gone.

## Page-by-page ledger

Legend: `P` = pass; `P*` = pass with a nonblocking cosmetic note documented below. No page failed.

### Short main paper — 20 pages

```text
01:P  02:P  03:P  04:P  05:P  06:P  07:P  08:P  09:P  10:P  11:P  12:P
13:P  14:P  15:P  16:P  17:P  18:P  19:P  20:P*
```

Specific checks: p.1 identity block and abstract are intact; p.3/Figure 1 is legible; p.4 has two populated, balanced columns; p.17 no longer has a blank right column; p.20 is a clean final bibliography page with natural end-of-document whitespace.

### Standalone supplement — 26 pages

```text
01:P  02:P  03:P  04:P  05:P  06:P  07:P  08:P  09:P  10:P  11:P*  12:P
13:P  14:P  15:P  16:P  17:P  18:P  19:P  20:P  21:P  22:P  23:P  24:P
25:P  26:P*
```

All tables, captions, headings, and section transitions were checked in source order. The final p.14 provenance edit fits cleanly and does not move any following page.

### Maintained TMLR manuscript — 43 pages

```text
01:P  02:P  03:P  04:P  05:P  06:P  07:P  08:P  09:P  10:P  11:P  12:P
13:P*  14:P  15:P  16:P  17:P  18:P  19:P  20:P  21:P*  22:P  23:P  24:P
25:P  26:P  27:P  28:P  29:P  30:P*  31:P  32:P  33:P  34:P  35:P  36:P
37:P  38:P  39:P  40:P  41:P  42:P  43:P*
```

The authoritative blueprint problem areas pass: pp.2, 4, 27, 35, 38–39, and 41–43 have no broken paragraph, column imbalance, avoidable white gap, bad float, heading orphan, clipping, or malformed glyph. The former pp.44–45 do not exist in the compacted 43-page final pagination; the ending material is present and complete on p.43.

Named checks also pass: p.1 anonymity/abstract, p.3 Figure 1, p.4 notation, p.13 algorithm, p.16 CCT table placement, p.18 prospective-results table placement, pp.19–21 main references, p.23 Appendix Table 10, pp.32–35 release/CCT/schema tables, pp.37–40 deployment/protocol tables, and p.43 final audit table and ending.

### Full technical report — 97 pages

```text
01:P  02:P  03:P  04:P  05:P  06:P  07:P  08:P  09:P  10:P  11:P  12:P
13:P  14:P  15:P  16:P  17:P  18:P  19:P  20:P  21:P  22:P  23:P  24:P
25:P*  26:P  27:P  28:P  29:P  30:P  31:P*  32:P  33:P  34:P  35:P  36:P
37:P  38:P  39:P  40:P  41:P  42:P  43:P  44:P  45:P  46:P  47:P  48:P
49:P  50:P  51:P  52:P  53:P  54:P  55:P  56:P  57:P  58:P*  59:P*  60:P
61:P  62:P  63:P  64:P  65:P  66:P  67:P  68:P  69:P  70:P  71:P  72:P
73:P  74:P  75:P  76:P  77:P  78:P  79:P  80:P  81:P*  82:P  83:P  84:P
85:P  86:P  87:P  88:P  89:P  90:P  91:P  92:P  93:P  94:P  95:P  96:P
97:P
```

The final regression explicitly confirms Tables 15–16 on pp.69–70, Tables 18–19 on pp.74–75, release inventory Table 24 on p.82, Table 25 on p.83, Table 26 after its introduction on p.84, deployment Table 29 on p.88, future-protocol Table 30 on p.90, release-audit Table 32 on p.95, and complete references/final page on pp.95–97.

## Resolved float and ordering defects

The final PDFs resolve all previously observed blockers, including the original five supplement float defects:

1. Supplement Table 4 now follows the B.10/B.11 material and Section C introduction instead of jumping ahead.
2. Supplement Table 6 no longer interrupts a sentence.
3. Supplement Table 7 now follows its F.2 introduction and no longer interrupts prose.
4. Supplement Table 14 now follows the complete update-coordinate paragraph.
5. Supplement Table 17 now follows the complete boundary paragraph.

Additional resolved issues:

- Supplement Table 3 follows its Office-Home introduction.
- Supplement Table 9 remains with its F.3 material rather than crossing F.4/F.5.
- Supplement Table 13 no longer splits the common-schema discussion.
- TMLR Tables 6 and 7 now follow their Section 8.3 and 9.2 introductions.
- TMLR Appendix Table 20 no longer interrupts its evidence-schema sentence.
- Full-report Tables 15 and 16 follow their introductions and no longer interrupt B.3.
- Full-report Table 25 no longer overtakes the end of the provenance paragraph.
- Full-report Table 26 follows the paragraph that introduces it.
- The full-report p.83 three-word widow was removed without repaginating later material.
- Short-main pp.4 and 17 have no blank right column; p.20 is a valid final references tail.

## Cosmetic notes accepted as nonblocking

- Short main p.20: natural whitespace after the final references.
- Supplement p.11: lower-page whitespace caused by preserving Table 9 section order.
- Supplement p.26: sparse final bibliography tail.
- TMLR p.13: algorithm block creates unusual but intentional density.
- TMLR p.21: main-text bibliography ends before the appendix starts on a new page.
- TMLR p.30: table placement creates extra whitespace but preserves correct section order.
- TMLR p.43: natural final-page whitespace.
- Full report p.25: lower whitespace after Algorithm 1.
- Full report p.31: moderate lower whitespace before Table 7.
- Full report pp.58–59: the framed non-evidentiary provenance note splits across pages but remains readable and correctly ordered.
- Full report p.81: lower whitespace is forced by indivisible Table 24 beginning on p.82.

None of these notes hides content, breaks reading order, strands a heading, leaves a blank column, or creates clipping/overlap.

## Final publication-layout result

Across all four roles there is no clipping, overlap, blank or near-blank page, malformed glyph, identity leak in the anonymous manuscript, unresolved reference marker, broken paragraph, bad float order, caption separation, or blocking white-space defect. The final four-PDF family passes visual and structural publication QA.
