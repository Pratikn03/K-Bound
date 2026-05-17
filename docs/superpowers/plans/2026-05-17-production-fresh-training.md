# Production Fresh Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean the repository state and run a fresh, reproducible ELARA training/evidence pass that can become the basis for the next paper draft.

**Architecture:** Keep the paper evidence path centered on `src/scripts/run_union_research_system.py`. Treat legacy domain runners as continuity checks, paper MVTec/RealFusion/healthcare runs as evidence, and generated PDFs/tables/figures as rebuild artifacts.

**Tech Stack:** Python, PyTorch/scikit-learn, pytest, Ruff, LaTeX, shell wrappers.

---

### Task 1: Repository Hygiene

**Files:**
- Inspect: `.gitignore`
- Inspect: `src/scripts/run_union_research_system.py`

- [ ] **Step 1: Remove generated sidecar files**

Run:
```bash
find . -path './.git' -prune -o -path './.venv' -prune -o -name '._*' -type f -delete
```

Expected: no output.

- [ ] **Step 2: Confirm tracked tree state**

Run:
```bash
git status --short
```

Expected: only intentional source/doc changes, or no output.

### Task 2: Preflight Verification

**Files:**
- Test: `tests/test_union_research_system.py`
- Test: `tests/test_patchcore_score.py`
- Test: `tests/test_paper_asset_metadata.py`
- Test: `tests/test_healthcare_gap_closure.py`

- [ ] **Step 1: Run focused tests**

Run:
```bash
PYTHONPATH=src .venv/bin/pytest \
  tests/test_union_research_system.py \
  tests/test_patchcore_score.py \
  tests/test_paper_asset_metadata.py \
  tests/test_healthcare_gap_closure.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 2: Run correctness lint**

Run:
```bash
.venv/bin/ruff check --select=E9,F63,F7,F82 .
```

Expected: `All checks passed!`

### Task 3: Fresh Training Run

**Files:**
- Execute: `src/scripts/run_union_research_system.py`
- Config: `configs/attention_mvtec3d_patchcore.yaml`
- Output: `experiments/union_research_system/fresh_2026_05_17_summary.json`

- [ ] **Step 1: Run the fresh full pipeline**

Run:
```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_union_research_system.py \
  --mode full \
  --with-tests \
  --continue-on-error \
  --include-optional-nlp-vision \
  --summary experiments/union_research_system/fresh_2026_05_17_summary.json
```

Expected: legacy domain runs, MVTec standard/heldout/M3DM/PatchCore, RealFusion, healthcare audits, PDF rebuild, and verification steps complete.

### Task 4: Result Extraction

**Files:**
- Read: `experiments/fusion/mvtec3d_patchcore_results.json`
- Read: `experiments/fusion/mvtec3d_results.json`
- Read: `experiments/fusion/craf_real_results.json`
- Read: `experiments/fusion/healthcare_gap4_deployment_audit_validation.json`

- [ ] **Step 1: Extract headline metrics**

Run:
```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from pathlib import Path

for path in [
    "experiments/fusion/mvtec3d_results.json",
    "experiments/fusion/mvtec3d_heldout_results.json",
    "experiments/fusion/mvtec3d_m3dm_results.json",
    "experiments/fusion/mvtec3d_patchcore_results.json",
    "experiments/fusion/craf_real_results.json",
    "experiments/fusion/craf_real_results_hard.json",
]:
    data = json.loads(Path(path).read_text())
    rocs = {k: v["roc_auc"]["mean"] for k, v in data["clean_metric_summary"].items()}
    best = max(rocs.items(), key=lambda item: item[1])
    print(path, "best", best, "RGA", rocs.get("craf_attention"))
PY
```

Expected: concise benchmark summary for the new paper-writing baseline.

### Task 5: Final Clean State

**Files:**
- Inspect: repository root

- [ ] **Step 1: Remove sidecars again**

Run:
```bash
find . -path './.git' -prune -o -path './.venv' -prune -o -name '._*' -type f -delete
git diff --check
git status --short
```

Expected: no whitespace errors; only intentional changed files.
