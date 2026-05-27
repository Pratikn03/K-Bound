# Phase 2 — Family-D v2 Freeze Blocked Report

**Phase:** 2.2C (Step 11 blocked branch).
**Status:** **`FAMILY_D_V2_FREEZE_BLOCKED_AT_PARTITION_MANIFEST_ARCHIVE_SHA256`**
**Supersedes:** the Phase 2.2B.2 version of this file (which carried `V2_FREEZE_BLOCKED_PENDING_USER_RESEARCH_DECISIONS` — those user research decisions are now locked).

## 1. What is now resolved (no longer blocking)

All seven Phase-2.2B.2-era blockers are closed:

| Blocker (was) | Status (now) |
|---|---|
| 1. Choose Eyecandies protocol | **LOCKED** — validation-only degradation-calibrated one-class multimodal (D4) |
| 2. Specify synthetic-corruption operator | **LOCKED** — score-level evidence-degradation operators D-EYE-1 / D-EYE-2 / D-EYE-3 fully specified in [FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md](./FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md) |
| 3. Eyecandies download authorisation | **GRANTED** by Phase 2.2C spec for hash-only download |
| 4. Verify Eyecandies modality manifest | **DOCUMENTARY DONE**; on-disk verification deferred to download pass |
| 5. Identify additional untouched candidate | **DEFERRED** per spec D8 (not required for first freeze) |
| 6. Choose confirmation target | **LOCKED** — base RGA vs `static_attention` reference on RGB+depth (D6) |
| 7. Independent external review | NOT YET REQUESTED — appropriate, because the freeze itself cannot complete until the remaining blocker (§2) is closed |

## 2. The single remaining blocker (Phase 2.2C scope)

**Cannot create `FAMILY_D_PARTITION_MANIFEST_v2.json` without placeholders.**

Spec Step 5 forbids placeholder fields in the partition manifest (`TBD`, `TO_BE_FILLED`, `TO_BE_RECORDED`, `placeholder`, `unknown hash`, `planned later`). The manifest's required `archive_sha256` field per Eyecandies category requires a local download pass to compute, because the Eyecandies maintainers do **not** publish per-archive SHA256 hashes.

Phase 2.2C did NOT perform the download pass. Per [FAMILY_D_V2_RAW_DATA_ACCESS_LOG.md](./FAMILY_D_V2_RAW_DATA_ACCESS_LOG.md):

> "No download performed. No anomaly mask inspected. No test ROC-AUC / PR-AUC / F1 / ECE / Brier / delta / p-value computed."

Therefore the partition manifest is **deliberately not written** in Phase 2.2C.

## 3. What WAS produced (no placeholders) in Phase 2.2C

The following freeze-required artifacts are complete and present:

| File | Status | SHA256 |
|---|---|---|
| `configs/phase2/family_d_v2_eyecandies_protocol.yaml` | COMPLETE | `104d90c6bab38671bb4dba15414a05ccebc890679cd681a5d46e06e7c8be4f15` |
| `docs/research/phase2/FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md` | COMPLETE | `e18bc05d12fb717b6b5dac738e41c022480374cefaa5556b5cfb48ce1e667f5d` |
| `docs/research/phase2/FAMILY_D_HYPOTHESES_v2.csv` | COMPLETE | `0361a960217f0b32f9a96eef9c261d47af2877a895cbb5e10a0115e8303ad8e2` |
| `docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md` | COMPLETE | `65f81a240b41e54fd7dafdbdf045f65d5e2d5c06909f0e05d9a56e286712e60b` |
| `docs/research/phase2/FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md` | COMPLETE | (per file) |
| `docs/research/phase2/FAMILY_D_V2_DATA_PROVENANCE_AND_HASH_REPORT.md` | COMPLETE (provenance) — archive SHA256 entries explicitly deferred | (per file) |
| `docs/research/phase2/FAMILY_D_V2_SCHEMA_VERIFICATION_REPORT.md` | COMPLETE (documentary) — on-disk verification deferred | (per file) |
| `docs/research/phase2/FAMILY_D_V2_RAW_DATA_ACCESS_LOG.md` | COMPLETE (no-download access log) | (per file) |
| `docs/research/phase2/FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md` | COMPLETE with `STATUS: NOT RUN` marker | (per file) |
| `docs/research/phase2/FAMILY_D_V2_PRE_TEST_HOSTILE_REVIEW_REPORT.md` | COMPLETE — verdict `FAMILY_D_V2_FREEZE_BLOCKED` | (per file) |

Missing only: `FAMILY_D_PARTITION_MANIFEST_v2.json` (correctly withheld).

## 4. What the next task must do

The follow-up task (call it **Phase 2.2D** — "Family-D v2 archive hashing pass") must, in order:

1. Run Stage 1 of [FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md](./FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md): `eyec ec-get +o data/raw/eyecandies`. This downloads all 10 category zip archives.
2. Compute `shasum -a 256` per archive and record to `experiments/phase2/family_d/eyecandies_archive_sha256.txt`.
3. Run Stage 2 of the execution commands: schema verification (modality alignment, sample-ID uniqueness, no-anomaly-mask-in-train/val). **Do not inspect anomaly labels.**
4. Write `FAMILY_D_PARTITION_MANIFEST_v2.json` with all per-category archive SHA256 values populated.
5. Update [FAMILY_D_V2_DATA_PROVENANCE_AND_HASH_REPORT.md](./FAMILY_D_V2_DATA_PROVENANCE_AND_HASH_REPORT.md) §4 with the recorded hashes.
6. Update [FAMILY_D_V2_SCHEMA_VERIFICATION_REPORT.md](./FAMILY_D_V2_SCHEMA_VERIFICATION_REPORT.md) §2 with on-disk verification results.
7. Re-run [FAMILY_D_V2_PRE_TEST_HOSTILE_REVIEW_REPORT.md](./FAMILY_D_V2_PRE_TEST_HOSTILE_REVIEW_REPORT.md): re-answer Q11, Q12; if all 15 questions pass, the final verdict flips to `FAMILY_D_V2_VALID_FOR_PRE_TEST_FREEZE`.
8. Commit:
   ```
   git commit -m "Freeze ELARA Family D v2 Eyecandies contract before test evaluation"
   ```
9. Create `docs/research/phase2/PHASE_2_READY_FOR_INDEPENDENT_FAMILY_D_REVIEW.md`.
10. **STOP**. Independent external review required before any test execution.

## 5. Mandatory invariants preserved in Phase 2.2C

- v1 `INVALID_FOR_EXECUTION` (unchanged).
- v2 `V2_FREEZE_BLOCKED_AT_PARTITION_MANIFEST_ARCHIVE_SHA256` (refined from "user research decisions" because those decisions are now closed).
- `test_evaluation_executed = false` (asserted in protocol YAML, hypotheses CSV, raw-data access log).
- **No confirmatory evidence currently exists.**
- **Successful future Family-D execution may not retroactively convert Family-A into confirmatory evidence.**
- General category/cohort-mixture theorem remains deferred (Phase 2.2A protocol lock).
- No paper / thesis edits.
- No Phase 3 / ELARA-Universal / ORIUS work.

## 6. Phase-2 closure decision under Step 11 blocked branch (Phase 2.2C)

> **`FAMILY-D v2 FREEZE BLOCKED — PHASE 2 REMAINS AT FAMILY-B COMPLETION`**

Phase 2 cannot reach `READY FOR INDEPENDENT FAMILY-D v2 REVIEW` in Phase 2.2C without the archive hash recording pass.
