# Phase 1.1.1 — PDF Text Extraction Scan

**Inputs:**
- `output/pdf/PAPER_DRAFT_PHASE1_1_1_VERIFIED.pdf` (36 pp, SHA256 `fb06129f618a9d8bd32a7865dbab0fdc7c838fd93536cb6080f71497ce04f5c5`)
- `output/pdf/THESIS_CHAPTER_PHASE1_1_1_VERIFIED.pdf` (41 pp, SHA256 `81bc79c42b26844c88a10f6ba7d162fa18ccf6f4a379d3a584ea4fc06d2e36ab`)

**Method:** PyPDF text extraction → regex scan.

---

## 1. Phase-1.1 forbidden tokens (must remain 0)

| Pattern | Paper hits | Thesis hits |
|---|---|---|
| `max(router` | 0 | 0 |
| `MAX(router` | 0 | 0 |
| `best non-router` | 0 | 0 |
| `strongest non-router` | 0 | 0 |
| `Fisher-combined` | 0 | 0 |
| `p (DeLong, Fisher)` | 0 | 0 |
| `nine evaluated cells` | 0 | 0 |
| `9-test Holm` | 0 | 0 |
| `pre-registered confirmatory` | 0 | 0 |
| `beats every non-ELARA` | 0 | 0 |
| `prove the cross-benchmark` | 0 | 0 |
| `without losing the cross-domain generalization property` | 0 | 0 |
| `deployment-grade` | 0 | 0 |
| `deployment-time sanity check` | 0 | 0 |
| `Causal Reliability Attribution` | 0 | 0 |
| `Causal Inference for Reliability` | 0 | 0 |
| `Structural Causal Model` | 0 | 0 |
| `interventional ATE` | 0 | 0 |
| `Average Treatment Effect` | 0 | 0 |
| `universally superior` | 0 | 0 |
| `production-ready` | 0 | 0 |
| `SOTA` | 0 | 0 |
| `FPFH+depth` | 0 | 0 |
| `0.7835` | 0 | 0 |
| `Family A confirmatory (set|family|cells|reanalysis|K…)` | **0** | **0** |

## 2. Phase 1.1.1-specific positive checks

| Check | Paper hits | Thesis hits | Verdict |
|---|---|---|---|
| `Family A audited-primary` | 2 | 1 | PASS (Issue 1) |
| `Protocol-diagnostic` (canonical figure caption) | present | present | PASS (Issue 2) |
| `SECONDARY DESCRIPTIVE SURFACE` | 2 | 2 | PASS (Issue 3) |
| `Secondary descriptive` | 1 | 1 | PASS (Issue 3) |
| `+0.0506` (PRIMARY B1) | 8 | 3 | PASS — PRIMARY preserved |
| `+0.0319` (PRIMARY B2) | 8 | 2 | PASS — PRIMARY preserved |
| `+0.0367` (SECONDARY) | 8 | 8 | OK — only inside SECONDARY-labelled tables/figures |
| `+0.0538` (SECONDARY) | 8 | 7 | OK — only inside SECONDARY-labelled tables/figures |

## 3. ELARA-Bench-LA figure caption note

The phrase `benchmark ROC-AUC and PR-AUC` appears once in each PDF, in the ELARA-Bench-LA `elara_clean_benchmark.png` caption (paper line 910, thesis line 779). This is **NOT** a canonical-protocol figure: ELARA-Bench-LA is a label-aligned stress benchmark, not a one-class canonical run, and PR-AUC on it is interpretable. The Phase 1.A canonical-metric block applies only to canonical one-class cells (MVTec / LOCO / VisA canonical). Issue 2 explicitly targets the MVTec canonical figure (`fig:mvtec-clean-benchmark`), which has been fixed. Therefore this is documented as out-of-scope for Issue 2, not a failure.

## 4. Verdict

**PASS.** No Phase-1.1 forbidden token regressed; the three Phase-1.1.1 residual issues are fully resolved in both verified PDFs.
