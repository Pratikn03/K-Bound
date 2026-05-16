# Fix Paper Claim Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove stale paper/thesis claims and make the manuscript accurately reflect the current evidence: MVTec is naturally paired but one-class for supervised fusion, RealFusion is stress-only label-aligned evidence, and RGA is a scoped diagnostic gate rather than a broad performance winner.

**Architecture:** Keep existing experiment code and JSON artifacts as the evidence base. Update the manuscript, thesis chapter, and README language so they distinguish official one-class MVTec, held-out-category MVTec, M3DM-style MVTec, RealFusion, and hard RealFusion. Add text checks that fail if obsolete claims such as “random forest leads at 0.959” or “all deltas are negative” reappear.

**Tech Stack:** LaTeX, Markdown, Python text tests, pytest, existing rebuild script.

---

### Task 1: Add Stale-Claim Regression Tests

**Files:**
- Modify: `tests/test_paper_asset_metadata.py`
- Check: `docs/research/PAPER_DRAFT_v1.tex`, `docs/research/THESIS_CHAPTER_v1.tex`, `README.md`

- [x] **Step 1: Write failing text-regression tests**

Add tests that reject obsolete MVTec claims and require the new scoped claim language.

- [x] **Step 2: Run tests to verify failure**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_paper_asset_metadata.py::test_manuscript_rejects_stale_mvtec_claims -q`

Expected before edits: fail on stale claims in manuscript/thesis/README.

- [x] **Step 3: Update prose**

Replace stale “RF dominates / RGA hurts” phrasing with the current disciplined evidence: one-class supervised fusion is near chance, held-out-category MVTec remains hard, M3DM-style variant is diagnostic, RealFusion is label-aligned stress evidence, and the final claim is scoped.

- [x] **Step 4: Verify**

Run targeted text tests, full pytest, Ruff correctness gate, and `./scripts/rebuild_paper.sh`.

### Task 2: Rebuild Evidence Artifacts

**Files:**
- Output: `output/pdf/PAPER_DRAFT_v1.pdf`
- Output: `output/pdf/THESIS_CHAPTER_v1.pdf`

- [x] **Step 1: Rebuild manuscript and thesis**

Run: `./scripts/rebuild_paper.sh`

Expected: paper and thesis compile without LaTeX errors.

- [x] **Step 2: Summarize rating**

Report the updated paper level and remaining blockers for main-conference quality.
