# Phase 1.1 — Build and Hash Report

**Branch:** `fix/elara-phase1-1-pdf-source-consistency`
**Source commit before Phase 1.1:** `5d9cf46`
**Build timestamp:** 2026-05-23 (Phase 1.1 closure).

---

## 1. Source `.tex` and generated-table hashes (Phase 1.1 final)

| File | SHA256 |
|---|---|
| `docs/research/PAPER_DRAFT_v1.tex` | `64cd81bbbf5aa107be221bb023a97a6edba6b69ddc01099739a6dfc205d4c1dd` |
| `docs/research/THESIS_CHAPTER_v1.tex` | `a5a310c7a5e5ca788aa19f6ad70ae2b88d260eae81dcae1dd31134978cfb85d4` |
| `docs/research/tables/milestone2_cross_benchmark.tex` | `e5c0a676d7c8ecfc87f35421a001e641b489a1ae39e43a0c5823bb9238215464` |
| `docs/research/tables/rga_plus_ablation.tex` | `641e194db2c2ce38ba6bd2366cc91d1dbac41787f4abb4740e9f3d77e4b81816` |
| `docs/research/tables/mvtec3d_milestone1_comparison.tex` | `df3d5dc283c6008a016c2b4dbfbf9c584cf8212cefa3898828161ca79417898b` |
| `docs/research/tables/mvtec3d_patchcore_clean_ci_results.tex` | `2e9b26432d2d891015d0ba0fcabb653e514cffc18ad9673da4909074124b75ca` |
| `docs/research/tables/mvtec3d_patchcore_calibration_cda.tex` | `4a27a2a619a2bbeb37ef35d33dfbae238d76dfd7c3e1b4824c1c461ed3f0dd6e` |
| `docs/research/generated/elara_verified_metrics_macros.tex` | `b97566bbccb725a40b9039f8219e0dc0db56acfa43cd19b9c9d2757b63e39f65` |

## 2. PDF hashes (Phase 1.1 final)

| File | Pages | SHA256 |
|---|---|---|
| `output/pdf/PAPER_DRAFT_v1.pdf` | 35 | `4394833e14d10d70a445371181b4d78a035e4533c5582665f539dfa0e630e3ef` |
| `output/pdf/THESIS_CHAPTER_v1.pdf` | 40 | `77de50a99db57790e9cc13bf64d27cf570306fb63ae0e4e60b1dc2d0e917d585` |
| `output/pdf/PAPER_DRAFT_PHASE1_1_VERIFIED.pdf` | 35 | `4394833e14d10d70a445371181b4d78a035e4533c5582665f539dfa0e630e3ef` |
| `output/pdf/THESIS_CHAPTER_PHASE1_1_VERIFIED.pdf` | 40 | `77de50a99db57790e9cc13bf64d27cf570306fb63ae0e4e60b1dc2d0e917d585` |

**Pairwise hash check:**
- `PAPER_DRAFT_v1.pdf` == `PAPER_DRAFT_PHASE1_1_VERIFIED.pdf`: ✅ MATCH
- `THESIS_CHAPTER_v1.pdf` == `THESIS_CHAPTER_PHASE1_1_VERIFIED.pdf`: ✅ MATCH

## 3. Build pipeline (Phase 1.1)

`scripts/rebuild_paper.sh` now:
1. Regenerates the LA / MVTec / VisA / UNSW asset tables (existing behaviour).
2. Regenerates the master comparison table + RGA+ component ablation table + Real3D-affected tables via the Phase-1.B/C-aware emitters.
3. **NEW (Phase 1.1):** runs `src/scripts/phase1_1_canonical_cleanup.py` to strip degenerate canonical PR-AUC / ECE / Brier columns and regenerate canonical figures as ROC-AUC only.
4. **NEW (Phase 1.1):** rebuilds the metrics manifest + LaTeX macros.
5. Compiles paper and thesis PDFs from clean source.

## 4. LaTeX-build cleanliness

| Document | LaTeX errors | Undefined refs | Bibliography defined / cited / uncited / undefined |
|---|---|---|---|
| `PAPER_DRAFT_v1.pdf` | 0 | 0 | 187 / 187 / 0 / 0 |
| `THESIS_CHAPTER_v1.pdf` | 0 | 0 | 23 / 23 / 0 / 0 |

## 5. Reproduction command

```bash
./scripts/rebuild_paper.sh
PYTHONPATH=src .venv/bin/python src/scripts/validate_phase1_1_pdf_claims.py
PYTHONPATH=src:. .venv/bin/python -m pytest tests/ -q
cp output/pdf/PAPER_DRAFT_v1.pdf output/pdf/PAPER_DRAFT_PHASE1_1_VERIFIED.pdf
cp output/pdf/THESIS_CHAPTER_v1.pdf output/pdf/THESIS_CHAPTER_PHASE1_1_VERIFIED.pdf
```
