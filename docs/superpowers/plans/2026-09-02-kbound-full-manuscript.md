# Corrected Full K-Bound Manuscript Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a new authoritative full K-Bound manuscript from the maintained paper sources, with the beta-surrogate failure visible in the main limitations section and only audited long-form additions.

**Architecture:** A new single-column `kbound_full.tex` driver will reuse the maintained abstract, scientific body, bibliography, and supplement, then append one focused full-only supplement. The shared body receives the beta correction so every maintained layout carries the disclosure. The existing build script gains an opt-in `BUILD_FULL=1` path and publishes `kbound_full.pdf` without modifying archival sources.

**Tech Stack:** LaTeX/latexmk, Bash, Python/pytest, Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`)

**Spec:** `docs/superpowers/specs/2026-09-02-kbound-full-manuscript-design.md`

## Global Constraints

- Preserve `docs/research/kbound/kbound.tex` and historical PDFs as superseded archives; do not use them as build inputs or overwrite them.
- Keep `kbound_submission.tex` and `kbound_tmlr.tex` as maintained compact and anonymous submission drivers.
- Use `kbound_submission_body.tex`, `kbound_submission_supplement.tex`, current generated tables, claim ledgers, and canonical result artifacts as scientific authorities.
- Describe the tested beta route as a source-development benefit-scale surrogate, not a direct estimate of the theorem's disagreement-conditional calibration-residual bound.
- Do not pad the manuscript to a target page count.
- Do not promote withheld iWildCam rows, historical mixed aggregates, protocol-port superiority, confirmatory CIFAR claims, confidence-robust ImageNet-C SAR claims, one-bit theory closure, or universal safety claims.

---

### Task 1: Add a build-flag regression test

**Files:**

- Create: `tests/test_kbound_build_flags.py`

**Interfaces:**

- Consumes: `docs/research/kbound/scripts/build_pdfs.sh` as an executable command-line boundary.
- Produces: a pytest check that rejects an invalid `BUILD_FULL` value before dependency checks or manuscript generation.

- [ ] **Step 1: Write the failing behavioral test**

Run the real build script in a subprocess with `BUILD_FULL=2` and a minimal `PATH` so the current implementation fails at the unrelated `latexmk` dependency check. Assert the intended configuration error:

```python
def test_build_rejects_invalid_full_flag_before_dependency_checks() -> None:
    env = {"PATH": "/usr/bin:/bin", "BUILD_FULL": "2"}
    result = subprocess.run(
        ["/bin/bash", str(BUILD_SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "ERROR: BUILD_FULL must be 0 or 1" in result.stderr
```

- [ ] **Step 2: Run the tests and verify the new-artifact checks fail**

Run:

```bash
python -m pytest tests/test_kbound_build_flags.py -q
```

Expected: FAIL because the script reaches the unrelated missing-`latexmk` error instead of rejecting `BUILD_FULL=2`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_kbound_build_flags.py
git commit -m "test: reject invalid full manuscript build flag"
```

### Task 2: Move the beta-surrogate negative result into the main paper

**Files:**

- Modify: `docs/research/kbound/kbound_submission_body.tex` in `Limitations and Broader Impact`
- Modify: `docs/research/kbound/kbound_submission_supplement.tex` in `Withdrawn Development Proxy for the Population Budget`

**Interfaces:**

- Consumes: audited beta-sweep findings already summarized in the maintained supplement.
- Produces: a visible main-text disclosure and a nonduplicative supplement pointer.

- [ ] **Step 1: Add the explicit main-text paragraph**

After the paragraph explaining that the residual budget is external, insert a `\paragraph{The development-based surrogate failed.}` paragraph containing all of these facts:

```tex
We tested a source-development benefit-scale proxy as an operational surrogate for
$\beta$ rather than declaring the population budget directly; this proxy does not
estimate the theorem's disagreement-conditional calibration-residual bound. On
CIFAR-10-C, the surrogate is $1.4$--$50\times$ too small on real evaluation cells:
24--73\% of cells fall outside the corresponding declared class $\mathcal C_\beta$,
and commit error is 0.4--16.6\%. On ImageNet-C, 5 of 10 configurations return zero
commitments on all 405 evaluation cells. We therefore withdraw this surrogate
estimation route and declare $\beta$ from domain knowledge or an explicit transfer
assumption instead. The empirical KGA does not estimate or numerically use $\beta$.
```

- [ ] **Step 2: Remove duplicated numerical prose from the supplement**

Keep the supplement subsection and its explanation of why beta remains external, but replace the repeated number paragraph with a cross-reference to the new main-text paragraph and retain the sampling-argument discussion that follows.

- [ ] **Step 3: Run focused and existing claim checks**

Run:

```bash
python src/scripts/validate_manuscript_claims.py
```

Expected: the existing manuscript validator exits successfully. Confirm the new paragraph manually against the six audited numeric tokens before committing; human prose is not protected by a source-grep unit test.

- [ ] **Step 4: Commit the disclosure**

```bash
git add docs/research/kbound/kbound_submission_body.tex docs/research/kbound/kbound_submission_supplement.tex
git commit -m "docs: disclose failed beta surrogate in main paper"
```

### Task 3: Add the audited full-only companion material

**Files:**

- Create: `docs/research/kbound/kbound_full_supplement.tex`

**Interfaces:**

- Consumes: `experiments/kbound/results/frontier_kga_bridge_v1/bridge_results.json`, `experiments/kbound/results/controlled_multimodal_d33/results.json`, claim entries `KB-CLAIM-043` and `KB-CLAIM-027`, and `DOCS_INDEX.md`.
- Produces: two explicitly controlled, non-headline long-form sections.

- [ ] **Step 1: Add the population-to-KGA bridge section**

Write `\section{Controlled Population-to-KGA Bridge}` with a table reporting seven constructed examples and the audited aggregate of five action agreements and two disagreements. Explain that population decisions use `(M, beta)` while empirical KGA uses `(\widehat\Delta, \varepsilon)`; the examples compare interfaces and do not estimate beta or provide deployment evidence.

- [ ] **Step 2: Add the D33 controlled diagnostic section**

Write `\section{Controlled Two-View MNIST Diagnostic}` with a compact table containing:

```text
conditions = 130
ADAPT / FREEZE / ABSTAIN = 9 / 119 / 2
KGA / single-A / always-fuse accuracy = 85.6785% / 85.3554% / 58.3231%
observed false ADAPT = 0 of 9 ADAPT decisions
```

State directly that nine ADAPT decisions are too few to establish a small conditional error rate and that injected two-view corruption is not natural-shift evidence.

- [ ] **Step 3: Add provenance notes**

End each section with the exact authoritative artifact path and claim-ledger identifier. Do not reproduce unavailable per-row values from memory; if a row-level artifact cannot be read, report only the audited aggregate above.

- [ ] **Step 4: Run the manuscript validator**

Run:

```bash
python src/scripts/validate_manuscript_claims.py
```

Expected: the existing manuscript validator exits successfully.

- [ ] **Step 5: Commit the full-only supplement**

```bash
git add docs/research/kbound/kbound_full_supplement.tex
git commit -m "docs: add audited K-Bound full supplement"
```

### Task 4: Add the full driver and build integration

**Files:**

- Create: `docs/research/kbound/kbound_full.tex`
- Modify: `docs/research/kbound/scripts/build_pdfs.sh`
- Modify: `docs/research/kbound/kbound_repro/manuscript_sources.py`
- Modify: `docs/research/kbound/DOCS_INDEX.md`
- Modify: `tests/test_kbound_build_flags.py`

**Interfaces:**

- Consumes: the shared abstract/body/supplement, full-only supplement, bibliography, generated number macros, and existing `build_pdf()` helper.
- Produces: `docs/research/kbound/kbound_full.pdf` through `BUILD_FULL=1`, with the full source closure included in claim validation.

- [ ] **Step 1: Create the single-column full driver**

Base its package and theorem setup on `kbound_submission.tex`, but use one-column `article` at 11pt with one-inch letter-paper margins. Set the author to Pratik Niroula, set `\anonfalse`, identify the subtitle as `Full Companion Manuscript`, and input sources in this order:

```tex
\input{kbound_abstract}
\input{kbound_submission_body}
\input{paper/references_kbound_expanded}
\clearpage
\input{kbound_submission_supplement}
\input{kbound_full_supplement}
```

- [ ] **Step 2: Add the opt-in build flag**

In `scripts/build_pdfs.sh`, define and validate `BUILD_FULL` exactly as `0` or `1`; add `kbound_full.pdf`, `kbound_full.log`, and `kbound_full_build.log` to the `publish_derived` allowlist; call:

```bash
if [[ "$BUILD_FULL" == "1" ]]; then
  build_pdf kbound_full.tex kbound_full_build.log
fi
```

Then chmod and list the full PDF only inside the enabled branch.

- [ ] **Step 3: Register the full source closure**

Add `kbound_full.tex` to `ACTIVE_DRIVER_RELATIVE_PATHS` in
`kbound_repro/manuscript_sources.py`. Extend the behavioral test to call
`active_source_paths(ROOT)` and require both `kbound_full.tex` and
`kbound_full_supplement.tex` in the returned repository-relative paths.

- [ ] **Step 4: Update the research map**

Add a maintained-artifact row naming `kbound_full.tex`, `kbound_full_supplement.tex`, and `kbound_full.pdf`. Describe it as a synchronized full companion, not a replacement for the compact/TMLR review layouts.

- [ ] **Step 5: Run the complete behavioral test**

Run:

```bash
python -m pytest tests/test_kbound_build_flags.py -q
bash -n docs/research/kbound/scripts/build_pdfs.sh
```

Expected: all focused tests pass and Bash syntax validation exits successfully.

- [ ] **Step 6: Commit the build integration**

```bash
git add docs/research/kbound/kbound_full.tex docs/research/kbound/scripts/build_pdfs.sh docs/research/kbound/kbound_repro/manuscript_sources.py docs/research/kbound/DOCS_INDEX.md tests/test_kbound_build_flags.py
git commit -m "docs: add authoritative full K-Bound build"
```

### Task 5: Validate, build, and visually inspect the PDFs

**Files:**

- Generate: `docs/research/kbound/kbound_short_final_draft.pdf`
- Generate: `docs/research/kbound/kbound_full.pdf`
- Inspect: `docs/research/kbound/kbound_short_final_draft.log`
- Inspect: `docs/research/kbound/kbound_full.log`
- Generate temporarily: `tmp/pdfs/kbound-full-pages/*.png`

**Interfaces:**

- Consumes: completed TeX source closure and build integration.
- Produces: validated current compact and full PDFs plus a visual-inspection record in the final handoff.

- [ ] **Step 1: Run scientific and manuscript validators**

Run:

```bash
python docs/research/kbound/scripts/validate_canonical_release_data.py
python src/scripts/validate_manuscript_claims.py
python docs/research/kbound/scripts/build_current_policy_interval_diagnostics.py --check
python -m pytest tests/test_kbound_build_flags.py -q
```

Expected: every command exits successfully. If a pre-existing release validator fails, capture the exact check and do not characterize the manuscript as release-sealed.

- [ ] **Step 2: Build the compact and full manuscripts**

Run:

```bash
BUILD_FULL=1 bash docs/research/kbound/scripts/build_pdfs.sh
```

Expected: nonempty `kbound_short_final_draft.pdf` and `kbound_full.pdf`, both accepted by `pdfinfo`.

- [ ] **Step 3: Check PDF text and compiler diagnostics**

Run:

```bash
pdfinfo docs/research/kbound/kbound_short_final_draft.pdf
pdfinfo docs/research/kbound/kbound_full.pdf
pdftotext docs/research/kbound/kbound_full.pdf - | rg "1.4|24--73|0.4--16.6|405|Controlled Population-to-KGA Bridge|Controlled Two-View MNIST"
rg -n "Undefined|Citation.*undefined|Reference.*undefined|LaTeX Error|Overfull" docs/research/kbound/kbound_short_final_draft.log docs/research/kbound/kbound_full.log
```

Expected: both PDFs have page counts greater than zero; required disclosures/headings are extractable; no LaTeX errors or undefined references/citations. Review every overfull warning individually.

- [ ] **Step 4: Render every full-manuscript page**

Run:

```bash
mkdir -p tmp/pdfs/kbound-full-pages
pdftoppm -png -r 120 docs/research/kbound/kbound_full.pdf tmp/pdfs/kbound-full-pages/page
```

Inspect all rendered pages as a contact sheet, then inspect the title page, the Section 9 beta disclosure page, representative theorem/table pages, the transition into each supplement, both full-only sections, and the final page at full resolution. Reject clipping, overlap, broken glyphs, unreadable tables, or incorrect page transitions.

- [ ] **Step 5: Re-run the verification after any layout fix**

Repeat Steps 1--4 after the last source change so the inspected PNGs correspond exactly to the delivered PDF.

- [ ] **Step 6: Commit source changes only**

If generated PDFs are tracked by repository policy, stage only the two maintained outputs and their expected logs; otherwise leave derived artifacts uncommitted. Never stage `tmp/pdfs/`.

```bash
git diff --check
git status --short
```

Report the final PDF paths, page counts, validator outcomes, and any explicitly unresolved release-seal limitation.
