# Phase 2 — Family-D v2 Freeze Blocked Report

**Phase:** 2.2B.2 / Step 11 (blocked branch)
**Status:** **`FAMILY_D_V2_FREEZE_BLOCKED_PENDING_USER_RESEARCH_DECISIONS`**

The Phase 2.2B.2 spec's Step 11 blocked branch authorizes this report when the v2 freeze cannot complete without placeholders. This is that report.

## 1. What the v2 freeze requires (per spec Step 10E)

Six fully populated files with **no placeholders**:

1. `FAMILY_D_CONTRACT_v2_PRE_TEST_FREEZE.md`
2. `FAMILY_D_PARTITION_MANIFEST_v2.json`
3. `FAMILY_D_HYPOTHESES_v2.csv`
4. `FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md`
5. `FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md`
6. `FAMILY_D_V2_PRE_TEST_HOSTILE_REVIEW_REPORT.md`

Each must contain: exact dataset name and release/version; official source/reference; official modality proof; access/license status; partition hash or deterministic construction; permitted hashing procedure; model candidates; comparator; validation-only selection rule; primary metric; effect-size threshold; seed count; multiplicity family; archive schema; allowed positive claim; allowed negative claim; forbidden claims; `test_evaluation_executed=false`.

No field may be `TBD`, `TO_BE_FILLED`, `TO_BE_RECORDED`, placeholder, or pending hash after freeze.

## 2. Exact missing requirements (blockers)

| # | Requirement | Current state | Why I cannot resolve in this task |
|---:|---|---|---|
| 1 | Choose Eyecandies protocol: canonical one-class OR validation-only synthetic-corruption | open | Research decision belonging to the user / research team — both protocols are scientifically valid; the choice constrains downstream confirmation target |
| 2 | If synthetic-corruption: specify operator family, parameter set, seed, hash of generated synthetic validation labels | open | Requires user research input on corruption operator (e.g., random masking, score-noise injection, missing-domain failure) and reproducibility-grade specification |
| 3 | Download Eyecandies official release to compute archive SHA256 | open | Network access + storage authorization not granted in a closure task; would also require selecting an exact release tag |
| 4 | Verify Eyecandies official modality manifest (RGB + normal + depth + multi-view) against the exact release tag | open | Same as 3 |
| 5 | (Optional but spec-preferred) Identify a second untouched RGB+depth/normal/point-cloud candidate beyond Eyecandies, then verify items 1–4 for it | open | No qualifying candidate identified in the Phase 2.1 eligibility review; verification requires external research |
| 6 | Choose confirmation target compatible with chosen protocol (e.g., base RGA vs static under one-class, or supervised-head under synthetic-corruption) | open (downstream of 1) | Research decision |
| 7 | Independent external review of v2 freeze before any execution | open | Cannot be performed by an LLM agent; external reviewer required by spec |

## 3. What WAS achieved in this Phase 2.2B.2 toward v2

- Step 10A explicitly enumerated invalid candidates (VisA, MPDD-without-verification) in [FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md](./FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md).
- Step 10C narrowed Eyecandies down to **two** valid protocols (down from "supervised-paired" which would have been invalid).
- All earlier Phase-2 work that touches Family-D was verified to leave v1 `INVALID_FOR_EXECUTION` and v2 `V2_DESIGN_PENDING` (5-test invariant in [tests/test_phase2_family_d_untouched_during_family_b.py](../../../tests/test_phase2_family_d_untouched_during_family_b.py)).
- Phase 2.2B.2 itself touched no Family-D test partition and ran no Family-D model.

## 4. Mandatory negative claims preserved

- **No confirmatory evidence currently exists.**
- **Successful future Family-D execution may not retroactively convert Family-A into confirmatory evidence.**
- **The general category/cohort-mixture theorem remains deferred** (per `MIXTURE_SHIFT_PROTOCOL.md`).
- **`test_evaluation_executed = false`** (no v2 freeze file exists yet; no v2 test outcome has been read).

## 5. Required next user action

To unblock v2 freeze, the research team must, in order:

1. Decide between Eyecandies canonical one-class vs validation-only synthetic-corruption protocol.
2. If synthetic-corruption: fully specify the operator (family, parameters, seed) before any data inspection.
3. Authorize Eyecandies download for hash recording only (no model evaluation, no test-fold inspection).
4. (Optional) Identify and verify one additional untouched RGB+depth/normal/point-cloud candidate.
5. Pass the produced v2 freeze through independent external review.

Only then may a v2 contract be frozen with no placeholders.

## 6. Phase-2 closure decision under Step 11 blocked branch

> **`FAMILY-B COMPLETE; FAMILY-D v2 DESIGN BLOCKED`**

The Family-B work is complete and committed. Family-D v2 is design-blocked by user research decisions. Phase 2 cannot reach `READY FOR INDEPENDENT FAMILY-D v2 REVIEW` in this task without making unauthorized research decisions.
