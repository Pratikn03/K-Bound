# Editing pass, 2026-07-23 (Claude session)

> **KEEP THIS FILE. Annotated 2026-07-26.**
>
> This document is the evidence that the 2026-07-19 freeze recorded in `SUBMISSION_LEDGER.md` was
> invalid, and it is cited as such at `SUBMISSION_LEDGER.md §0`. Two of the 12 edits below change
> the compiled output — item 8 (5 policy row labels renamed in two tables) and item 9 (two
> citations plus two bibitems) — so the pinned PDF sha256
> `5b01e5e7...` cannot correspond to the `.tex` on disk. The page-count note at the bottom
> ("24 pp; your Mac build: 23 pp") contradicts the pinned "PDF pages: 23" independently.
>
> The rule "**simplify wording, never scope**" was followed and is not in dispute; the problem is
> not these edits, it is that the ledger continued to advertise a freeze after them. That is fixed:
> the ledger now reads **NOT FROZEN** and carries a dated re-freeze procedure instead of a stale
> hash.
>
> Two flags raised at the bottom of this file were correct and are now tracked:
> - the stale ImageNet-C row in `kbound.tex` Table 1 — recorded in `README.md` under Manuscript
>   Policy;
> - the iCloud placeholders — now a full census at `PLACEHOLDER_INVENTORY.md` (143 files, not just
>   the 24 figure PNGs noted here).
>
> One claim in this file needs a qualifier: "the compiled PDFs pass the forbidden-phrase greps from
> `claim_ledger.json`". They pass **after manual review of 7 hits**, every one of which is the
> paper correctly *denying* the forbidden phrase. The gate is a substring grep and fires on
> disclaimers; making it context-aware is `SUBMISSION_LEDGER.md §12.8`.

## What was changed — and what was not

Rule applied throughout: **simplify wording, never scope.** No number, theorem statement,
table value, tier label, or claim-controlled phrase was altered. All edits are prose-layer
only. Both papers compile clean (0 undefined refs/citations) and the compiled PDFs pass the
forbidden-phrase greps from `claim_ledger.json` (counts identical to the frozen originals;
the one "natural-shift win" hit in the long paper is the pre-existing future-work sentence).

## kbound_short.tex (submission core) — 12 edits

1. Abstract: split the two longest sentences (KGA definition; no-harm headline). No wording
   of claims changed — pure sentence surgery.
2. Intro ¶2: split the KGA definition sentence the same way.
3. Intro ¶3: split the "identifiable mixed regimes" sentence at the semicolon.
4. Contributions bullet 2: split per REVIEW_SHORT_PAPER_2026-07-04 P0 item.
5. §V System Overview: split the leave-one-out calibration mega-sentence; split the
   jackknife/jackknife+ mega-sentence.
6. §V Population-vs-empirical: split the ATC/AGL contrast sentence into three.
7. Conclusion ¶2: split the mixed-regimes sentence.
8. **K-Bound → KGA row labels** in Tables (CIFAR-10-C decisive; ImageNet-C faithful):
   5 policy rows renamed from "K-Bound" to "KGA" — the framework/method discipline item
   from your own adversarial review (framework = K-Bound, deployed rule = KGA).
9. **Two citations added** (both verified against arXiv before insertion):
   - `lim2026reset` — Lim, Hwang, Lee, "When and Where to Reset Matters for Long-Term
     Test-Time Adaptation," ICLR 2026, arXiv:2603.03796 → cited in §II Guarded/Monitored.
   - `sonoda2025lean` — Sonoda, Mizuno, Tsukamoto, Onda, "Lean Formalization of
     Generalization Error Bound...," arXiv:2503.19605 → cited in §Reproducibility next to
     the Lean artifact (review item 13: Lean prior art).
   Bibitems added to `paper/references_kbound_expanded.tex`.

## kbound.tex (long manuscript) — 7 edits

1. Abstract: split the exact-success-condition mega-sentence (70 words → three sentences).
2. Intro: split the identifiability-is-established sentence at the colon.
3. Limitations (1): removed a triple-em-dash pileup; split into two sentences.
4. Limitations (7): split the final compound sentence.
5. Discussion (ii): double-colon construction → em-dash + colon.

## Flags noticed during the pass (no action taken — your call)

- **Stale ImageNet-C row in the long paper.** `kbound.tex` Table 1 (regime summary) still
  carries the superseded single-seed values 0.0108/0.0625/0.0319 with the 2026-07-09
  footnote. The short paper, SUBMISSION_LEDGER, and the regenerated manifest use the
  pooled 5-seed exact-rank values 0.0264/0.0529/0.0319. Consistent with the README's note
  that kbound.tex predates current claim corrections — worth syncing next time you touch it.
- **iCloud placeholders.** 24 of 28 figure PNGs in `figures/` and `theory_v2/` are iCloud
  placeholders on this Mac (0 bytes readable locally). For this rebuild they were
  reconstructed losslessly from the compiled PDFs and verified (anchor dims + pixel-level
  page comparison of the worst-case page). If you want the true originals back on disk:
  right-click the `kbound` folder in Finder → Download Now.
- **Page-count drift.** This container's TeX Live 2023 renders the short paper at 24 pp
  (your Mac build: 23 pp) — font-metric drift only, no content difference. Your local
  `bash scripts/build_pdfs.sh` remains the canonical build.
- The `*_edited.pdf` files delivered alongside are container builds for preview; the
  committed `.tex` files are the source of truth. Canonical PDFs on disk were not touched.
