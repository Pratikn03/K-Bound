# KBOUND PhD-level release cleanup plan

Date: 2026-08-27

> **Completed planning record.** The executed cleanup is documented in
> `KBOUND_RELEASE_CLEANUP_REPORT_2026-08-27.md`; current manuscript and evidence authority is listed
> in `DOCS_INDEX.md`. This file remains as the safety and verification plan, not as release status.

## Safety boundary

The repository began this pass with 188 modified, deleted, or untracked paths. Those pre-existing changes are treated as user-owned. This pass will not reset, checkout, overwrite, or bulk-delete them. Cleanup is limited to files whose generated or invalid status is demonstrated by the empirical audit, plus regenerable caches ignored by Git.

## Correctness gates

1. Route B must be invariant to spectral sign, bounded, finite, binary-task guarded, and candidate-diversity checked.
2. Resume state must match the full scientific configuration, split role, checkpoint hash, candidate set, and seed semantics.
3. Natural extraction must reject duplicate scientific keys, stale files, missing candidates, target-label calibration leakage, and false model-seed inference.
4. iWildCam must match the official WILDS macro-F1 contract.
5. ERROR, missing, or incomplete cells cannot be scored or published as freeze-equivalent successes.
6. Every promoted artifact must be strict JSON with explicit feasibility status rather than non-standard Infinity tokens.
7. Manuscript denominators, task validity, target-opening status, and inference units must match the archived data exactly.

## Artifact policy

- Raw experiment evidence is retained even when a derived claim is invalid.
- Invalid derived aggregates, tables, and figures are removed from the release tree and listed by path and SHA-256 in `audits/empirical_data_quality_2026_08_27/quarantine_manifest.json`.
- Recreated derived artifacts must be produced by the hardened pipeline in a fresh staging directory and pass provenance checks.
- Release checksums are regenerated only after code, data, manuscript, and PDF are frozen.

## Verification ladder

1. Focused regression tests for every repaired bug.
2. Relevant KBOUND unit and reconciliation tests.
3. Re-executed empirical data-quality notebook and zero critical pipeline failures on newly generated artifacts.
4. Clean LaTeX compilation of the submission target.
5. PDF page rendering and visual inspection for clipping, broken tables, unreadable text, missing references, and layout regressions.
6. Final file inventory, checksum seal, and a release report that distinguishes retained history, quarantined outputs, and current promotable evidence.
