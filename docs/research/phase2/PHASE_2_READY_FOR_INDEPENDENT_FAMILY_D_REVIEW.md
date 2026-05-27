# Phase 2 — Ready for Independent Family-D Review

**Status:** **`READY FOR INDEPENDENT FAMILY-D v2 REVIEW`**
**Phase:** 2.2D close.

## 1. Family-B closure (preserved)

Family-B work is complete and committed:

- Family-B evidence lock commit: **`2c780cf`** ("Complete ELARA Phase 2 Family-B evidence and lock bounded conclusions").
- Family-B closure decision: **`FAMILY_B_COMPLETE_WITH_NEGATIVE_RGA_V2_AND_BOUNDED_THEORY_EVIDENCE`** (see [PHASE_2_FAMILY_B_FINAL_DECISION.md](./PHASE_2_FAMILY_B_FINAL_DECISION.md)).
- Phase 2.2B.1 commit: `dbf8dca`; manifest hash-fix: `4993a14`.

## 2. Family-D v2 contract is now FROZEN

Locked artifacts (all with real content, no placeholders):

| File | Purpose | SHA256 (Phase-2.2C frozen design) |
|---|---|---|
| `configs/phase2/family_d_v2_eyecandies_protocol.yaml` | one-class multimodal degradation protocol | `104d90c6bab38671bb4dba15414a05ccebc890679cd681a5d46e06e7c8be4f15` |
| `docs/research/phase2/FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md` | D-EYE-1 / D-EYE-2 / D-EYE-3 operators | `e18bc05d12fb717b6b5dac738e41c022480374cefaa5556b5cfb48ce1e667f5d` |
| `docs/research/phase2/FAMILY_D_HYPOTHESES_v2.csv` | 2 primary + 1 secondary | `0361a960217f0b32f9a96eef9c261d47af2877a895cbb5e10a0115e8303ad8e2` |
| `docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md` | val-only selection + Holm K=2 | `65f81a240b41e54fd7dafdbdf045f65d5e2d5c06909f0e05d9a56e286712e60b` |
| `docs/research/phase2/FAMILY_D_PARTITION_MANIFEST_v2.json` | per-category real archive SHA256 + on-disk schema | (recorded inside the manifest) |
| `docs/research/phase2/FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md` | exact future commands (NOT RUN) | n/a |
| `docs/research/phase2/FAMILY_D_V2_PRE_TEST_HOSTILE_REVIEW_REPORT.md` | verdict `FAMILY_D_V2_VALID_FOR_PRE_TEST_FREEZE` | n/a |

## 3. Data state (Phase 2.2D)

- 10/10 Eyecandies category archives downloaded to `data/raw/eyecandies/_archives/` (≈ 27.2 GB total).
- Per-archive SHA256 recorded in [experiments/phase2/family_d/eyecandies_archive_sha256.txt](../../../experiments/phase2/family_d/eyecandies_archive_sha256.txt) and mirrored in `FAMILY_D_PARTITION_MANIFEST_v2.json`.
- On-disk schema verified per-(category, split): 1000 / 100 / 50 / 400 samples for train / val / test_public / test_private; every sample has paired RGB + depth + 6 RGB views + 1 normal map.
- Anomaly mask files counted but **never opened**:
  - train: 5 000 placeholder mask files per category (zero-anomaly by official spec; not used by base-RGA protocol).
  - val: 500 placeholder mask files per category.
  - test_public: 250 ground-truth mask files per category (NOT inspected before execution).
  - test_private: 0 public mask files.
- `test_evaluation_executed: false` asserted in manifest, protocol YAML, every hypotheses row.

## 4. No test outcomes accessed

- No model trained on Eyecandies.
- No anomaly mask opened.
- No test ROC-AUC / PR-AUC / F1 / ECE / Brier / delta / p-value computed.
- No method comparison performed.

## 5. Independent reviewer authorisation requirement

Family-D execution remains forbidden until the independent reviewer **(NOT the original Phase-2 agent)** completes review of the freeze artifacts and writes the sign-off file:

```
docs/research/phase2/FAMILY_D_V2_INDEPENDENT_REVIEW_SIGNOFF.md
```

The reviewer should verify:
1. Per-archive SHA256 in the partition manifest matches `eyecandies_archive_sha256.txt`.
2. Protocol YAML, operator spec, hypotheses CSV, selection policy SHA256 anchors match the manifest's recorded hashes.
3. `test_evaluation_executed: false` is preserved everywhere.
4. No file referenced in the freeze contains placeholders.
5. The claim ceiling matches the Phase-2 contract terminology (no universality / SOTA / deployment-safety claims).
6. The 15 questions in `FAMILY_D_V2_PRE_TEST_HOSTILE_REVIEW_REPORT.md` answer as documented.

## 6. Phase 2 status

> **`PHASE_2_READY_FOR_FAMILY_D_EXECUTION_AFTER_INDEPENDENT_REVIEW`**

- **Phase 2 completion (weighted):** ~95% (Family-D execution + post-test hostile review remains).
- **Phase 2 IS NOT complete.** A successful Family-D run would unlock the "held-out confirmatory evidence under the frozen protocol" claim; a negative outcome would unlock the bounded negative-evidence claim. Either way, the manuscript-update phase remains a separate task.

## 7. Mandatory invariants preserved

- Family-A K=5 results unchanged.
- Family-B closure unchanged.
- Family-D v1 still `INVALID_FOR_EXECUTION`.
- General category/cohort-mixture theorem still deferred.
- Permanent forbidden claim list preserved verbatim:
  - ELARA is universal.
  - RGA+ beats every strongest baseline.
  - Existing Family A is confirmatory.
  - Current work is SOTA.
  - Current work is production-ready or deployment-safe.
  - Retrospective certificate equals real-world safety certification.
  - Family D was executed (unless an authorised future run lands real artifacts).

## 8. Provenance

- Phase 2.2C lock commit: `407c090`.
- Phase 2.2D commit (downloading + hashing + schema + manifest write): see §1 of the upcoming commit message.
- All commits will be pushed to `origin/exp/elara-phase2-mechanism-and-replication`.
