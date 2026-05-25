# Family D v2 — Design Status

**Status:** `V2_DESIGN_PENDING`.

No v2 contract is frozen in Phase 2.1.
No v2 partition manifest, hypotheses file, statistical policy, or
execution-commands file is written under `_v2` suffixes in this task.
Writing any of them with placeholders is explicitly forbidden by the
Phase 2.1 contract-repair requirements.

## 1. What is missing to allow v2 to be frozen

The following items must each be closed in a follow-up review before
the v2 freeze:

| Item | Status |
|---|---|
| MPDD official modality verification | open — see [FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md](./FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md) §A |
| VisA removal from candidates | resolved — VisA is registry-locked in Family A; see §B |
| Eyecandies protocol selection (one-class vs synthetic-corruption) | open — see §C |
| If synthetic-corruption: operator fully specified, parameters frozen | open |
| At least one additional genuinely untouched RGB+depth/normal/point-cloud candidate | open — see §D |
| License / access status per retained candidate | open |
| Hash strategy that requires no edit to a frozen file after execution begins | open — see §2 below |
| Independent external review of the v2 design | open |

Until every row above is `resolved`, the v2 design is
`V2_DESIGN_PENDING`. No v2 freeze.

## 2. Hash strategy for frozen v2 manifest

The Phase-2.1 contract-repair task fixes the v1 placeholder failure
mode. The v2 freeze MUST avoid all of:

- `TO_BE_FILLED`
- `TO_BE_RECORDED`
- `TBD`
- empty string
- any other placeholder text

A valid v2 hash strategy must satisfy one of these two patterns:

1. **Pre-download eligibility contract + final pre-test freeze.**
   - A pre-download eligibility contract is written first; it records
     dataset names, official release tags, license/access, and intended
     protocols.
   - Raw archives may be downloaded **only to compute file hashes** —
     no label or test inspection.
   - A separate `FAMILY_D_CONTRACT_v2_PRE_TEST_FREEZE.md` is then
     written with the actual SHA256 values populated.
   - The freeze commit lands AFTER this freeze file is fully populated;
     it MUST NOT contain placeholders.

2. **Source-by-release-tag with deterministic mirror.**
   - The contract pins the official release tag (e.g. a GitHub
     release, a Zenodo DOI) and trusts the platform's content-addressed
     hash. The v2 manifest records the tag, not the per-file SHA256.
   - At execution time, the verifier asserts that the downloaded
     archive equals the platform-provided hash for that tag. No edit
     to the frozen manifest is required.

The v2 contract may use **either** pattern but must declare which one
explicitly. Phase 2.1 records this requirement; Phase 2.1 does **not**
choose the pattern, because the choice depends on which candidate
datasets clear the eligibility review.

## 3. Correct v2 claim boundary

When v2 is eventually frozen, executed, and reported, the only
permitted statement from a CONFIRMED v2 hypothesis is:

> "Held-out confirmatory evidence under a frozen Family-D v2 protocol."

It does **not**:

- Remove or modify the audited-reanalysis status of Family A.
- Support "universality", "SOTA", "deployment-ready",
  "clinically validated", or "broad cross-domain superiority" claims.
- Authorize Phase 3, ELARA-Universal, or ORIUS work.

## 4. Reproduction-of-status check

Anyone wishing to re-derive this `V2_DESIGN_PENDING` state may run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
    tests/test_family_d_v1_never_executable.py \
    tests/test_family_d_v2_no_placeholders_before_freeze.py \
    tests/test_family_d_no_previously_touched_dataset.py \
    tests/test_family_d_claim_boundary.py \
    -q
```

All five tests pass while the v2 design is correctly pending.
