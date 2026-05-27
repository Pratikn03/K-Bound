# Family-D v2 — Execution Blocked: Protocol Change Required

**Phase:** 2.2E (execution task) — BLOCKED branch.
**Status:** **`FAMILY_D_V2_EXECUTION_BLOCKED — EYECANDIES_SCORING_PIPELINE_NOT_IN_FROZEN_SCOPE`**
**Companion file:** [FAMILY_D_V2_EXECUTION_PRECHECK.md](./FAMILY_D_V2_EXECUTION_PRECHECK.md) (Stage 0 PASS, Stage 2 GAP).

## 1. Why this report exists

The Phase 2.2E spec explicitly authorises this report when execution cannot proceed under the unchanged frozen contract:

> "If any frozen artifact must be changed to make execution possible:
> STOP.
> Do not execute Family-D.
> Create:
> docs/research/phase2/FAMILY_D_V2_EXECUTION_BLOCKED_PROTOCOL_CHANGE_REQUIRED.md"

Stage 0 of the execution task (authorisation + integrity gate) **passed completely** — all 14 SHA256 anchors match, the independent sign-off is committed, no Family-D outcome exists. The block surfaces at **Stage 2** (implementation check).

## 2. The gap

The frozen Family-D v2 contract specifies:

- **Primary method:** `base_RGA` (reliability-aware gating).
- **Primary comparator:** `static_attention`.
- **Primary modalities:** RGB + depth.
- **Degradation operators:** D-EYE-1 / D-EYE-2 / D-EYE-3 operate at the **modality-score level** (set per-domain score to 0.0 on the gated batch).

`base_RGA` in this codebase consumes a fusion-input CSV with per-(sample, domain) columns: `sample_id, domain, label, score, confidence, embedding_0..N`. The `score` column is a **pre-computed one-class anomaly score per modality**, produced by an upstream image-scoring pipeline.

Eyecandies on disk contains **raw image data**: 6 RGB views + 1 depth map + 1 normal map per sample. **No pre-computed per-modality anomaly scores exist** at any stage of the frozen freeze.

Therefore: to feed Eyecandies into the frozen base-RGA pipeline, an upstream image → per-modality-score pipeline must be implemented. That pipeline is **not part of any frozen artifact**.

## 3. Why this is a protocol change, not pure implementation

Building the scoring pipeline requires committing to research-level choices that materially affect the per-sample anomaly score (and therefore the held-out ROC-AUC):

| Choice | Examples | Why it affects outcome |
|---|---|---|
| Feature backbone | ResNet-18 / 50 / 101, ViT, MobileNet, etc. | Different feature spaces produce different score distributions |
| Pretrained weights | ImageNet-1k vs 21k vs custom | Pretraining domain matters for transfer |
| Feature layer | layer2 / layer3 / penultimate | Different abstraction levels |
| Pooling | average-pool / adaptive-pool / patch-level | Spatial vs global summary |
| One-class scoring | PatchCore memory bank / Padim / k-NN | Distance metrics differ |
| Coreset subsampling | 1% / 10% / 100% of train | Memory bank density affects scoring |
| RGB view aggregation | average / max / single canonical view | Multi-view bias |
| Score normalisation | raw / z-score / min-max | Affects RGA gating threshold sensitivity |
| Depth pre-processing | range-normalise / log / raw | Affects depth feature scale |

The independent reviewer (commit `5679790`) reviewed the **frozen artifacts** — protocol YAML, operator spec, hypotheses CSV, selection policy. None of the above choices are recorded in any of those files. Making them now in this execution task would mean the executor (me) introduces parameters that materially affect the result without independent review.

That is the *de facto* protocol change the spec's BLOCKED branch is designed to catch.

## 4. What would close this gap properly

The honest engineering path forward is a Phase 2.2E.0 sub-task:

1. **Design the Eyecandies one-class scoring pipeline as a new frozen artifact**, e.g. `configs/phase2/family_d_v2_eyecandies_scoring_pipeline.yaml`. Specify every choice in §3 explicitly.

2. **Implement the pipeline in code**, with test guards that prove it doesn't read anomaly labels during scoring (only the train memory bank uses anomaly-free train images; only test labels are read at the final ROC-AUC computation step authorised by Stage 5 of the execution spec).

3. **Record its SHA256 in an updated partition manifest** (e.g. `FAMILY_D_PARTITION_MANIFEST_v3.json`) and **re-run independent pre-execution review** against the expanded frozen scope.

4. **Then execute Family-D v3** under the expanded freeze.

This adds one round-trip but preserves the held-out invariant. Without it, any result this task produces would be a hybrid of "independently reviewed protocol + executor-defined scoring pipeline" — which is exactly the contamination the held-out invariant is designed to prevent.

## 5. What this report does NOT authorise

- Executing Family-D anyway with executor-chosen scoring pipeline.
- Computing any Eyecandies test-set ROC-AUC / PR-AUC / F1 / ECE / Brier / Δ / p-value.
- Inspecting any Eyecandies anomaly mask.
- Editing the paper / thesis.
- Beginning Phase 3 / ELARA-Universal / ORIUS.

## 6. What this report DOES preserve

- Family-A K=5 results — unchanged.
- Family-B closure (`FAMILY_B_COMPLETE_WITH_NEGATIVE_RGA_V2_AND_BOUNDED_THEORY_EVIDENCE`) — unchanged.
- Family-D v1 (`INVALID_FOR_EXECUTION`) — unchanged.
- Family-D v2 freeze + independent sign-off — **valid and re-usable** for the future Phase 2.2E.0 + Phase 2.2E re-execution path.
- `test_evaluation_executed = false` everywhere — **preserved**.
- Eyecandies has not been spent as a confirmation dataset; the test split has not been read.

## 7. State of the held-out invariant

> **PRESERVED.** No test-fold information has crossed into the executor's decision-making in this task. The frozen contract remains a valid future authorisation pending the pipeline-spec extension above.

## 8. Recommended next user action

Either:

- **Path A** — open a Phase 2.2E.0 task that designs + freezes the Eyecandies scoring pipeline spec, re-runs independent review, and then executes Family-D under the expanded freeze; or
- **Path B** — formally close Phase 2 without Family-D confirmatory evidence by writing `PHASE_2_FORMAL_CLOSURE_WITHOUT_FAMILY_D.md`. This is allowed; it just bounds the manuscript story to "audited static-reference Family-A + bounded Family-B mechanism evidence" without held-out confirmation.

Both paths preserve scientific validity. Both leave Family-A and Family-B intact. Both keep Phase 3 explicitly forbidden until Phase 2 is closed under one of them.

## 9. Final decision (this Phase 2.2E task)

> **`FAMILY_D_V2_EXECUTION_BLOCKED — PROTOCOL_CHANGE_REQUIRED`**

No model was trained. No metric was computed. No anomaly mask was inspected. The frozen contract is unchanged. The independent sign-off remains in force for a future re-execution under an expanded freeze.
