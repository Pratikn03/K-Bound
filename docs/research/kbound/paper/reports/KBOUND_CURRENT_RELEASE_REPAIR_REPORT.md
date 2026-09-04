# K-Bound current-release repair report

## 1. Executive summary

The four maintained K-Bound manuscript roles were reconciled to one evidence-safe source closure,
rebuilt from frozen numerical authorities, and checked for scientific wording, structural validity,
anonymity, reproducibility, and page layout. The release contains a 20-page double-column main
paper, a 26-page standalone supplement, a 43-page anonymous integrated TMLR manuscript, and a
97-page full technical report. No experiment, model, calibration rule, action, protocol status, or
canonical numerical result was changed in this repair.

## 2. Branch and worktree used

- Branch: `fix/kbound-current-release-sync-2026-09-03`
- Primary checkout: `/Users/pratik_n/Documents/AutoML_Flagship_V8`
- Isolated resident build mirror: `/tmp/kbound-r1-final.SZUjxm/worktree`

The resident mirror avoided reading partly dataless iCloud Git objects during builds and audits.
No reset, stash, clean, or broad checkout was used against the user's primary checkout.

## 3. Source snapshot commit

The final source snapshot is
`efecbafd74e192c1314ea81a0aa8868a8e784a94` (`efecbafd74e1`). It includes the evidence-safe
manuscript text, corrected theory notation, restored single-column tables, short-paper float repair,
and byte-reproducible PDF settings.

## 4. Artifact commit

Artifact commit: `ab8439204015ff4a666d670819a977f2587fc73c` (`ab8439204015`).

## 5. Source-to-output map

The authoritative role map is `paper/reports/KBOUND_SOURCE_OUTPUT_MAP.md`. Each output has a
different publication role and a distinct root driver; the four outputs share the same abstract,
main argument, theorem modules, generated numerical authorities, and bibliography where applicable.

## 6. Files modified

The material source edits are concentrated in:

- `kbound_abstract_core.tex`, `kbound_submission_body.tex`, and
  `kbound_submission_supplement.tex`;
- `paper/sections/theory_core_main.tex` and `paper/sections/theory_certificate.tex`;
- the five maintained PDF drivers and `scripts/build_pdfs.sh`;
- the current release identity, source-output map, dashboard snapshot, empirical-audit summary,
  generated display tables, and audit reports;
- focused tests and release-audit scripts needed to enforce the revised wording and build contract.

## 7. Files created

The current release directory contains four count-free PDFs and one checksum manifest. The final
structural-verification directory contains per-role PDF information, embedded-font inventories,
Ghostscript parse logs, extracted text, a JSON summary, and a Markdown report. This repair report
is also new.

## 8. Files archived

No provenance record was destroyed. Superseded publication roots remain quarantined under the
existing dated archives and `archive/superseded_do_not_cite/`; they are excluded from the current
release and must not be cited or submitted. No stale PDF is published under a current filename.

## 9. Exact CCT-20 changes

The prose now reports the complete record: 45 cells; 0/44/1 ADAPT/FREEZE/ABSTAIN actions;
1/0/44 helpful/tied/harmful effects; one false FREEZE among 44 FREEZE actions and 1/45 overall;
all 45 served predictions used the frozen model. The locked utility-preservation endpoint passes,
whereas strong success fails because ADAPT exposure is zero and there is no improvement over
always-freeze. The text calls this all-frozen fail-closed behavior, not selective routing.

## 10. Terminology changes

The manuscript consistently separates K-Bound's population target from KGA's measured-cell target.
Formal actions use ADAPT, FREEZE, and ABSTAIN. Empirical decisions are described as
interval-supported; utility preservation is not called universal safety; opened and dependent
diagnostics are not called prospective confirmation.

## 11. Development and calibration wording

The active CIFAR procedure is described using labeled development cells, disjoint
residual-calibration cells, and label-free evidence from each scored cell. It is not described as a
prospective held-out experiment. The one-checkpoint, five-seed, grid-composition, equal-cell-weight,
local-port, and composite BatchNorm/update details are explicit.

## 12. Certificate language

The theoretical coverage-to-action proposition remains conditional on a stated marginal-coverage
premise for a declared target and unit. The empirical method is called a benefit-interval gate.
Missing or unsupported assessment returns ABSTAIN; merely serving the frozen model is not presented
as a certified FREEZE decision or as proof that the frozen model is safe.

## 13. CCT interval-family clarification

The main text distinguishes the locked utility endpoint's nominal pointwise 95% intervals from the
strong-success audit's nominal 97.5% Bonferroni intervals and Holm-adjusted location sign-flip
values. It states explicitly that these are not alternate versions of one interval.

## 14. Release identity

- Release ID: `KBOUND-2026-09-03-R1`
- Source snapshot: `efecbafd74e1`
- Canonical panel SHA-256:
  `35d4c165843de1ece3cb35ffb4ac50dbfcbfa33b646754e2b6d133fbdfa78c6e`
- Source-manifest SHA-256:
  `03b1d2b1e9e5ed1cf835126f871ed83eb8497a1ac81727b3955af0d69eb89742`

Named artifacts display the named release identity. The anonymous TMLR manuscript suppresses the
author and source commit from visible text and PDF metadata.

## 15. Table labels and crosswalk

Stable semantic labels remain the source of truth across layouts; rendered table numbers differ by
role. The release crosswalk is `paper/RELEASE_TABLE_CROSSWALK.md`. A page-by-page audit detected and
repaired two `table*[H]` floats that had vanished in one-column drivers. The release-status inventory
and future natural-shift protocol now appear as Tables 10/16 in the supplement, 17/23 in TMLR, and
24/30 in the full report, with continuous numbering.

## 16. Legacy scan

The current-release audit checks the exact four-file inventory and scans for stale claims, promoted
historical comparison language, incorrect denominators, and identity-role mismatches. It passes.
POEM/AETTA numbers remain only as historical local-port diagnostics, and their stored `WIN` or
`kga_beats` labels are explicitly withdrawn from the current claim set.

## 17. Layout

The TMLR manuscript was inspected as a 43-page integrated paper rather than padded to exactly 45
pages. The two missing one-column tables were restored. In the short main paper, section-scoped
float barriers that stranded the right columns on former pages 4 and 17 were replaced by a single
pre-bibliography barrier. Wide tables in the supplement, TMLR manuscript, and full report were
placed after their introducing prose, and the full-report CCT provenance widow was removed.
Standard one-inch margins improve readability, the two-line benefit-set display avoids overflow,
and the terminal bibliography is balanced across both columns. No large negative spacing or forced
blank material was introduced.

## 18. Final filenames

- `kbound_short_main.pdf`
- `kbound_short_supplement.pdf`
- `kbound_tmlr.pdf`
- `kbound_full_report.pdf`

Only `kbound_tmlr.pdf` is intended for an anonymous TMLR review upload. The other three PDFs are
identified reading or archival copies.

## 19. Page counts

| Artifact | Pages |
|---|---:|
| Short main paper | 20 |
| Short supplement | 26 |
| Integrated TMLR manuscript | 43 |
| Full technical report | 97 |

## 20. SHA-256 digests

| Artifact | SHA-256 |
|---|---|
| `kbound_short_main.pdf` | `8a60e37a3212236b2c2fce044e35eeeaab2e788c4fbbd2a3cc92463920b29dae` |
| `kbound_short_supplement.pdf` | `7549e8370ebb2035a5a6f08143d281f28d1f453d6d5818c20c5916dda27eea0a` |
| `kbound_tmlr.pdf` | `a33dbbb403eedeb4df532bec1c697a99b660e8ef7c7e1e0f5fff1c303951ea62` |
| `kbound_full_report.pdf` | `9de2fac1405bb5f89077d2ae8fc9ccdbf636b068f9f964684f9edcc2a5b1c6d1` |

Two consecutive all-role builds produced these same four byte-level digests.

## 21. Build commands

The final build command was:

```bash
SOURCE_SNAPSHOT_COMMIT=efecbafd74e1 \
PYTHON=/tmp/kbound-release-venv2/bin/python \
BUILD_LONG_TMLR=1 BUILD_SHORT_MAIN=1 \
BUILD_SHORT_SUPPLEMENT=1 BUILD_FULL_REPORT=1 \
bash docs/research/kbound/scripts/build_pdfs.sh
```

The script fixes the release epoch and suppresses random PDF trailer/engine metadata so repeated
builds are byte-reproducible.

## 22. Tests and results

The focused manuscript and scientific matrix passed 251/251 logical tests. One direct run hit a
30-second `git ls-tree` timeout in the iCloud-backed object store; the same blocked node passed in a
resident temporary Git inventory view, giving the full 251/251 result. Build flags, toolchain lock,
and reproducibility passed 33/33 tests. The whole-pipeline outcome-invariance suite passed 11/11.
Shell syntax for `build_pdfs.sh` passed.

## 23. PDF structural checks

`verify_pdf_structure.py` passed all four roles. Ghostscript parsed every page cleanly; all fonts are
embedded; Poppler extracted text; every PDF is US Letter; and the reported page counts and hashes
match the checksum manifest. Details are in
`paper/reports/final_structural_verification/PDF_STRUCTURAL_VERIFICATION.md`.

## 24. TMLR anonymity

`audit_tmlr_anonymity.py` passed. The PDF Author, Subject, and Keywords fields are empty; there is no
XMP stream, attachment, JavaScript, author name, institution, email, personal repository URL, local
home path, acknowledgment, or author-contribution statement. The audit report is
`paper/reports/TMLR_ANONYMITY_AUDIT.md`.

## 25. Visual inspection

All four maintained PDFs passed the final visual inspection. Every one of the 186 pages was
rendered at 192 DPI. Pages changed from the immediately preceding fully inspected build were
rechecked at original render resolution; unchanged pages were accepted only after exact pixel
identity. The inspection found no clipping, overlap, blank or near-blank page, malformed glyph,
broken paragraph, bad float order, caption separation, or blocking whitespace defect. The complete
page ledger and the accepted cosmetic notes are recorded in
`paper/reports/PDF_VISUAL_INSPECTION.md`.

## 26. Warnings

The final four TeX logs contain no overfull boxes, undefined controls, unresolved references,
undefined citations, multiply defined labels, or rerun warnings. The only test anomaly was the
environmental iCloud Git-inventory timeout described in Section 22; it was independently rerun and
passed in a resident Git view.

## 27. Intentionally not changed

No new experiment was run and no model, checkpoint, adapter, seed family, train/calibration/test
split, feature schema, estimator, residual rule, action threshold, bootstrap, sign-flip test,
multiplicity family, or deployment policy was altered. Historical negative and invalid records were
retained with restricted status rather than rewritten as current evidence. The exact professor-
requested beta-proxy failure remains in main Limitations, despite the generic blueprint's suggestion
to move its numbers to an appendix, because the direct revision instruction is stronger and more
evidence-safe.

## 28. Final confirmation

All requested repairs completed and all required checks passed.
