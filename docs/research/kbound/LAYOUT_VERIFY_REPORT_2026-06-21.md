# K-Bound paper — layout verification + canonical-PDF rebuild (2026-06-21)

**Scope:** verify current state, then fix only what was still broken. **No result or verdict changed.**
Run on the host via desktop-commander (latexmk/pdflatex + ghostscript + ImageMagick; poppler absent).

- Canonical source: `docs/research/kbound/kbound.tex` (+ 5 `\input`s: `paper/sections/main_theory_5.tex`,
  `paper/appendix_cifar10c_cells.tex`, `kbound_frontier_appendix.tex`, `paper/sections/weakest_class.tex`,
  `paper/references_kbound_expanded.tex`)
- Canonical output: `docs/research/kbound/K-Bound_paper.pdf`
- Audit trail: `_audit_verify_20260621/ALL28_final.png` (28-page contact sheet), `_audit_verify_20260621/_verify_build.log`

## Headline

The previous (2026-06-20) pass had **already fixed the source** — but it never refreshed the canonical PDF.
`K-Bound_paper.pdf` on disk was a **stale older build** that still showed two half-empty pages, while the
current `kbound.tex` already compiles clean. The fix this session was therefore a **rebuild + promote**, not
a source edit: no `.tex` changes were required.

## Step 1 — Verified current state

| Check | Current `K-Bound_paper.pdf` (stale, as found) | Current source build of `kbound.tex` |
|---|---|---|
| Pages | 28 | 28 |
| Margin overflow (text past column/text width) | none on any page | none on any page |
| Overfull \hbox / \vbox (log) | n/a (old build) | **0 / 0** |
| Underfull \hbox | — | 0 |
| Undefined / multiply-defined refs | — | **0 / 0** |
| Broken-layout pages | **p14, p23 half-empty** (single-column, ~36–40% blank) | **none** |

- `md5(K-Bound_paper.pdf) != md5(kbound.pdf)` confirmed the canonical artifact was out of sync with source.
- Every page was rendered at 150 dpi and scanned programmatically (ink bounding box per page) **and** viewed.
- Dataset references in the build source (main + all 5 includes): **PovertyMap 0, ACDC 0, MVTec 0, Real-IAD 0,
  MulSen 0, 3D-ADAM 0, multimodal 0, target-label-light 0**. All 9 canonical present
  (CIFAR-10-C, ImageNet-C, Camelyon17, iWildCam, Office-Home, RxRx1, ImageNet-R, CIFAR-10.1, fMoW).
- The only non-canonical-dataset hits in the tree are in **backups** and the **separate `manuscript/` document**
  (ch01–ch10) — neither feeds `K-Bound_paper.pdf`, so they were left untouched.

## Step 2 — Fixes applied

- **Tables / overfull hboxes:** nothing to fix — the source build already has **0 overfull boxes** and every
  one of the 26 tables / 15 figures sits within its column / text width (verified visually across all table
  pages, incl. the full-length 65-row CIFAR-10-C appendix table on p24, which is centered and within margins).
- **Dataset trim:** nothing to remove — PovertyMap, the multimodal/target-label-light panel and its floats,
  MVTec-3D, and the duplicate consolidation table were **already removed in the prior pass**; verified 0 remain.
  ACDC was already absent.
- **Redundant per-protocol tables:** the one true duplicate (five-theorem consolidation table) was already
  removed previously; no further consolidation was warranted — the paper fits cleanly in 28 pages and forcing
  more merges would risk altering reported numbers, which is out of scope (integrity constraint).

## Step 3 — Recompile, re-render, confirm

1. Backed up source and stale artifact (see below).
2. `latexmk -pdf` → exit 0, `kbound.pdf` (28 pages), log: **0 overfull / 0 underfull / 0 ref errors**.
3. Promoted the clean build: `cp kbound.pdf K-Bound_paper.pdf` — **md5 now identical** (`3a5ff690…`).
4. Re-rendered the canonical `K-Bound_paper.pdf` at 150 dpi and re-scanned all 28 pages.

**Result — all 28 pages clean. BROKEN PAGES: NONE.** Right margin 95–103 px on every text page
(≈0.63–0.69 in @150 dpi); nothing touches any page edge.

### Pages broken-then-fixed this session

| Page | Before (stale canonical) | After (rebuilt canonical) |
|---|---|---|
| 14 | half-empty: content only in left column, ~595 px bottom blank | full two columns, normal fill |
| 23 | sparse: rightmost ink at col 626 (right half empty) | full width, rightmost ink col 1171 |

(No margin-overflow pages existed in the stale canonical PDF; the prior pass had already cleared overflow at the
source level. The remaining defect was purely the two stale half-empty pages, now resolved by the rebuild.)

## Before / after page count

- Prior pass (historical): 33 → 28.
- This session: 28 (stale, 2 broken pages) → **28 (0 broken pages)**.

## Backups created

- `kbound.tex.bak_verifyfix_20260621_073505` (pre-session source backup; source ultimately unchanged)
- `K-Bound_paper.pdf.bak_stale_20260621_073805` (the stale pre-rebuild canonical PDF)

## Verdicts — unchanged (Table I `tab:regime-summary`)

Wins = CIFAR-10-C, ImageNet-C, Office-Home (CI-robust), iWildCam (point-estimate).
Null = Camelyon17 (pooling artifact, withdrawn), RxRx1, ImageNet-R, CIFAR-10.1, fMoW. **Not re-litigated.**

## Rebuilt PDF

`docs/research/kbound/K-Bound_paper.pdf` — 28 pages, 1,921,097 bytes, 0 broken pages, 0 overfull boxes.
