# Journal revision progress

Plan: IMPLEMENTATION_PLAN.md. User approval covers implementation, verification, rebuilds and versioned GitHub publication.

## Baseline

- Verified all 49 editorial input hashes; preserved all 13 dirty editorial files and binary diff under output/journal_revision_20260922/baseline.
- Continuing in the existing linked APFS worktree on codex/kbound-journal-20260922; preceding branch/release preserved.
- Main source and experimental inputs have separate owners. No current scientific failures are silently treated as software defects.

## Preflight

| Tasks | Interface or overlap | Resolution |
|---|---|---|
| Theory / manuscript | theory audit and comparison consumed by integration | Theory worker never edits manuscript |
| Empirical / manuscript | generated report/rows consumed by supplement and register | Frozen inputs unchanged; root integrates summaries |
| Eligibility / manuscript | fresh-data decision determines new-study scope | No target access; blocked result is an allowed completion |
| All / release | sources and reports enter seal | All writers finish before commit and full verification |
| Theory | count and historical extension | Preserve 150 and scoped FAIL_EXPECTED distinction |
| Empirical | metrics and inference | Missing quantities null; dependence explicit |
| Eligibility | scientific success versus feasibility | No qualifying candidate means no new experiment |
| Integration | localization and adverse findings | Preserve completed editorial changes and all limitations |
| Release | evidence and artifact truth | Verify download identities after publishing |

Ruling: run independent workers concurrently only on disjoint outputs, with root-only integration and commits, following the active delegation instruction; this avoids conflicting manuscript edits. Cost if wrong: review/rework before sealing.

Ruling: retain execution evidence and QA records rather than delete them as disposable workflow scratch, because they are requested deliverables. Public bundles exclude machine-local scratch and secrets.

## Status

- Task 1: implemented; independent review found one minor literature attribution, corrected
- Task 2: implemented; independent arithmetic review passed, 70 frozen inputs unchanged
- Task 3: completed metadata decision; no candidate qualifies, new study blocked without target access
- Task 4: integrated; 30-claim register, historical figure label and supplement synthesis; independent review passed; population-transfer assumption entry corrected
- Task 5: source freeze and full verification next; 67-page compact and 75-page TMLR built, Word built; page inspection in progress

Baseline editorial guard failures were reproduced before changes. Protocol-order parsing and outdated narrative expectations were corrected with regression tests; scientific assertions were retained. Strict formal verification passed 150/150; the broader historical extension remains FAIL_EXPECTED. Final package records, rather than this pre-freeze status note, establish completion of verification and publication.
