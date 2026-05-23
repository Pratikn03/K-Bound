# Phase 1.1 — Baseline Snapshot

**Branch:** `fix/elara-phase1-1-pdf-source-consistency` (created from `fix/elara-phase1-empirical-validity`).
**Pre-Phase-1.1 commit:** `5d9cf46` (Repair ELARA empirical validity).
**Snapshot timestamp:** 2026-05-23.

---

## 1. File hashes BEFORE Phase 1.1

| File | SHA256 |
|---|---|
| `docs/research/PAPER_DRAFT_v1.tex` | `9a0e12efe9a5098949165a8316787057894c05c55ba52e51ecc40d2427d7e7c5` |
| `docs/research/THESIS_CHAPTER_v1.tex` | `4ea6c9cea3d50c3a6a34bcbebc43dc913ea1bc24d2544cd6267d1736615d1e24` |
| `output/pdf/PAPER_DRAFT_v1.pdf` | `93a596c60a74eb9f5280538cfcd7748f1ca32085915788dbf346dcb5ce904e2e` |
| `output/pdf/THESIS_CHAPTER_v1.pdf` | `b7dd6b688baab7ef2c2580402e76fa69087b3323c0c9a0dc79045a8a65d8bd17` |
| `docs/research/metrics_manifest.json` | `b7bd85110102dee46bb31dbbde566d7aa1fa4072aac54df671e6db2c225f698c` |
| `docs/research/claim_matrix.csv` | `234ae5d4661ab4d282590a3f1e6362a5a0754685ee62ff4552bcdd06f8dd0190` |

## 2. PDF page counts BEFORE Phase 1.1

- `output/pdf/PAPER_DRAFT_v1.pdf`: 35 pages
- `output/pdf/THESIS_CHAPTER_v1.pdf`: 39 pages

## 3. Contradictions detected by extracted-text scan of CURRENT PDFs

### Paper PDF (`output/pdf/PAPER_DRAFT_v1.pdf`)

| Pattern | Hits | Location summary |
|---|---|---|
| `0.7835` | 90 | Canonical PR-AUC / ECE / Brier values visible in canonical MVTec / VisA / LOCO tables |
| `MAX(router, boost)` | 2 | Component-ablation prose at offsets 101041, 102437 (residual descriptive mention) |
| `best non-router` | 3 | Headers in `mvtec3d_milestone1_comparison.tex`, `rga_plus_ablation.tex` |
| `beats every non-ELARA` | 1 | UNSW section overclaim |
| `prove the cross-benchmark` | 1 | UNSW subsection opener |
| `0.0506 / 0.0319` | 9 / 10 | Original ELARA-Bench-LA mechanism deltas (abstract, intro, master) |
| `0.0367 / 0.0538` | 7 / 7 | Hard-mode ELARA-Bench-LA mechanism deltas (mechanism tables) |

### Thesis PDF (`output/pdf/THESIS_CHAPTER_v1.pdf`)

| Pattern | Hits | Location summary |
|---|---|---|
| `0.7835` | 68 | Same canonical degenerate metrics still visible |
| `0.0506 / 0.0319` | 4 / 5 | Original ELARA-Bench-LA deltas (mechanism prose) |
| `0.0367 / 0.0538` | 8 / 7 | Hard-mode ELARA-Bench-LA deltas (thesis tables) |
| `prove the cross-benchmark` | 1 | UNSW subsection opener |
| `beats every non-ELARA` | 1 | UNSW subsection |

### Cross-document contradictions

- Paper abstract states "canonical PR-AUC / ECE / Brier are omitted (Phase 1.A audit)" — but paper body still contains 90 visible 0.7835 values in canonical tables.
- Paper master comparison §sec:cross-benchmark-master uses validation-frozen language, but the **§sec:unsw-paired** subsection still contains "prove the cross-benchmark" and "router beats every non-ELARA" overclaim.
- Both manuscripts cite the *original* ELARA-Bench-LA mechanism deltas (+0.0506 / +0.0319) in the abstract but display *hard-mode* deltas (+0.0367 / +0.0538) in the rendered tables.

## 4. Pre-Phase-1.1 archive

Archive root: `docs/research/archive/pre_phase1_1/`

Contents:
- `PAPER_DRAFT_v1.before_p11.tex`
- `THESIS_CHAPTER_v1.before_p11.tex`
- `output/PAPER_DRAFT_v1.before_p11.pdf`
- `output/THESIS_CHAPTER_v1.before_p11.pdf`
- `metrics_manifest.before_p11.json`
- `claim_matrix.before_p11.csv`
- `tables/` (full copy of pre-Phase-1.1 generated tables)
- `figures/` (full copy of pre-Phase-1.1 generated figures)

## 5. PDFs being repaired are repository build outputs

Confirmed: `output/pdf/PAPER_DRAFT_v1.pdf` and `output/pdf/THESIS_CHAPTER_v1.pdf` were produced by `scripts/rebuild_paper.sh` at the end of Phase 1 (commit `5d9cf46`). They are the repo build outputs, not external stale copies.
