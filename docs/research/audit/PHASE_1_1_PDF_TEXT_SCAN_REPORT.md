# Phase 1.1 — PDF Text Extraction Scan Report (Step 12)

**Status:** PASS. **0 forbidden-string hits across both verified PDFs.**

**Inputs:**
- `output/pdf/PAPER_DRAFT_PHASE1_1_VERIFIED.pdf` (35 pp)
- `output/pdf/THESIS_CHAPTER_PHASE1_1_VERIFIED.pdf` (40 pp)

**Method:** PyPDF text extraction → regex scan over the forbidden token list.

---

## 1. Forbidden-token scan results

| Pattern | Paper hits | Thesis hits | Allowed/forbidden | Resolution |
|---|---|---|---|---|
| `max(router` | 0 | 0 | forbidden | removed |
| `MAX(router` | 0 | 0 | forbidden | removed |
| `best non-router` | 0 | 0 | forbidden | removed (replaced with "validation-frozen comparator") |
| `strongest non-router` | 0 | 0 | forbidden | removed (Real3D paragraph rewritten) |
| `RGA+ Δ vs best` | 0 | 0 | forbidden | replaced with "Audited Δ vs validation-frozen comparator" |
| `Fisher-combined` | 0 | 0 | forbidden | replaced with single-representative-seed DeLong |
| `p (DeLong, Fisher)` | 0 | 0 | forbidden | replaced |
| `nine evaluated cells` | 0 | 0 | forbidden | replaced with "Family A audited-primary $K{=}5$" |
| `9-test Holm` | 0 | 0 | forbidden | replaced |
| `Family A confirmatory` | 0 | 0 | forbidden | replaced with "Family A audited-primary" |
| `pre-registered confirmatory` | 0 | 0 | forbidden | removed; pre-registration reserved for Family D |
| `beats every non-ELARA` | 0 | 0 | forbidden | UNSW section rewritten |
| `prove the cross-benchmark` | 0 | 0 | forbidden | UNSW section rewritten |
| `without losing the cross-domain generalization property` | 0 | 0 | forbidden | removed |
| `deployment-grade` | 0 | 0 | forbidden | reworded to "streaming"/"external" |
| `deployment-time sanity check` | 0 | 0 | forbidden | reworded |
| `Causal Reliability Attribution` | 0 | 0 | forbidden | renamed to "Model-Response Sensitivity to Per-Domain Reliability" |
| `Causal Inference for Reliability` | 0 | 0 | forbidden | section renamed |
| `Structural Causal Model` | 0 | 0 | forbidden | removed |
| `interventional ATE` | 0 | 0 | forbidden | reframed |
| `Average Treatment Effect` | 0 | 0 | forbidden | reframed |
| `universally superior` | 0 | 0 | forbidden | removed |
| `production-ready` | 0 | 0 | forbidden | reworded |
| `SOTA` | 0 | 0 | forbidden | removed |
| `FPFH+depth` | 0 | 0 | stale label | replaced with "PCA shape + depth" |
| **`0.7835`** | **0** | **0** | forbidden in promoted view | canonical tables/figures rewritten to ROC-AUC-only |

## 2. Allowed-with-justification check

| Pattern | Paper hits | Thesis hits | Justification |
|---|---|---|---|
| `confirmatory` | (allowed in Family D context) | (allowed in Family D context) | verified to occur only inside Family D / future-replication discussion |
| `causal` | small count | small count | occurs only in adjacent-literature citation context with explicit "not the current reported estimand" wording |
| `0.7835` | 0 | 0 | OK — completely removed from promoted view; canonical cleanup script removed all canonical PR/ECE/Brier instances |
| Original `+0.0506/+0.0319` deltas | 6 / 6 | 2 / 2 | PRIMARY Family B mechanism endpoints (k-of-D k=4 mean-gate); locked per `PHASE_1_1_PRIMARY_RUN_RESOLUTION.md` |
| Hard-mode `+0.0367/+0.0538` deltas | (descriptive only) | (descriptive only) | SECONDARY tau-sweep + adversarial subsidiary tables; labelled in captions |

## 3. Pre-Phase-1.1 vs Post-Phase-1.1 comparison

| Pattern | Pre-1.1 paper hits | Post-1.1 paper hits | Pre-1.1 thesis hits | Post-1.1 thesis hits |
|---|---|---|---|---|
| `0.7835` | 90 | **0** | 68 | **0** |
| `MAX(router` | 2 | **0** | 0 | 0 |
| `best non-router` | 3 | **0** | 0 | 0 |
| `beats every non-ELARA` | 1 | **0** | 1 | **0** |
| `prove the cross-benchmark` | 1 | **0** | 1 | **0** |
| `Fisher-combined` | 0 | 0 | 0 | 0 |
| `Family A confirmatory` | 0 | 0 | 0 | 0 |
| `interventional ATE` | 0 | 0 | 0 | 0 |
| `Structural Causal Model` | 0 | 0 | 0 | 0 |
| `deployment-grade` | 1 | **0** | 1 | **0** |
| `universally superior` | 0 | 0 | 0 | 0 |
| `production-ready` | 0 | 0 | 1 | **0** |
| `SOTA` | 1 | **0** | 1 | **0** |

## 4. Verdict

Step 12 PASS: zero unresolved forbidden-string hits remain in either verified PDF.
