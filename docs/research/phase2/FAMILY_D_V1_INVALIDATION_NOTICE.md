# Family D — v1 Invalidation Notice

**Status:** `INVALID_FOR_EXECUTION`. v1 is preserved as historical contract evidence only.

The v1 Family-D contract artefacts at
[FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md](./FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md),
[FAMILY_D_DATASET_INVENTORY.md](./FAMILY_D_DATASET_INVENTORY.md),
[FAMILY_D_HYPOTHESES.csv](./FAMILY_D_HYPOTHESES.csv),
[FAMILY_D_PARTITION_MANIFEST.json](./FAMILY_D_PARTITION_MANIFEST.json),
[FAMILY_D_SELECTION_AND_STATISTICAL_POLICY.md](./FAMILY_D_SELECTION_AND_STATISTICAL_POLICY.md),
and
[FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md](./FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md)
are retained byte-for-byte. They are **not** edited by Phase 2.1. No
content in those files has been modified.

## 1. Why v1 is invalid for execution

1. **Frozen file contains placeholders that would be edited at execution time.**
   [FAMILY_D_PARTITION_MANIFEST.json](./FAMILY_D_PARTITION_MANIFEST.json) carries:
   - `freeze_commit = "TO_BE_FILLED_BY_FREEZE_COMMIT"`
   - per-dataset `release_tag = "TO_BE_RECORDED_AT_DOWNLOAD_TIME"`
   - per-dataset `expected_sha256_of_archive = "TO_BE_RECORDED_AT_DOWNLOAD_TIME"`

   Filling these values after the freeze commit would mutate a frozen
   contract. The v1 file's own §5 ("pre-registration integrity contract")
   states that any post-freeze edit invalidates the freeze. Therefore
   v1 was unsound the moment it was frozen with placeholders.

2. **MPDD multimodal description is unverified.**
   v1 [FAMILY_D_DATASET_INVENTORY.md](./FAMILY_D_DATASET_INVENTORY.md)
   §D1 describes MPDD as "Modalities: RGB, 3D (depth via structured light)."
   This description was written without citing an official source that
   verifies paired depth/3D files in the exact intended release. Without
   that proof, MPDD cannot serve as an independent multimodal Family-D
   benchmark.

3. **VisA is incorrectly described as untouched.**
   v1 §D3 marks VisA as "never touched outside of citation-level reading."
   This is incorrect: VisA appears in the locked Phase-2 registry as
   `A-POWERED-4 — VisA, RGB+edge supervised-paired` (an inspected
   Family-A cell). VisA cannot simultaneously be an inspected Family-A
   cell and an untouched Family-D candidate.

4. **Eyecandies supervised-paired protocol is underspecified.**
   The official Eyecandies release exposes anomalous samples only in
   the test split — train and validation are anomaly-free. v1 does not
   define how a supervised-paired protocol obtains labelled anomalous
   validation evidence without reading official test outcomes. Any
   protocol that reads the official test labels for validation
   tuning would invalidate the held-out nature of the family.

5. **v1 claim boundary is wrong.**
   v1 [FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md](./FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md)
   §5 states that a CONFIRMED Family-D hypothesis would entitle
   "removal of the 'audited reanalysis' qualifier from the
   corresponding Family-A claim." This is incorrect: Family A is an
   audited-reanalysis family on inspected benchmarks; Family-D
   confirmation on a different held-out benchmark cannot retroactively
   change the inspection status of the Family-A cells. Family-D
   confirmation adds a held-out confirmatory statement; it does not
   delete the audited-reanalysis label on Family A.

## 2. What v1 may still be cited for

v1 may be cited as the **historical contract** that motivated the
Phase-2.1 repair. It must always be cited as `INVALID_FOR_EXECUTION`.
No Family-D test outcomes have been read under v1.

## 3. What v1 may NOT be used for

- No Family-D experiment may be executed under v1.
- No downstream report may treat v1 hypotheses (D-H1..D-H5) as
  pre-registered.
- No future text may state that v1 was unfrozen.
- v1 files must not be edited going forward; further repairs land in
  v2 sibling files.

## 4. Path forward

A new v2 design is required. The v2 design is `V2_DESIGN_PENDING`
pending the eligibility review at
[FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md](./FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md).
No v2 contract may be frozen until the eligibility review is closed and
all dataset / protocol / hash issues are resolved without placeholders.

## 5. Provenance

This invalidation notice is required by the Phase-2.1 contract-repair
task. It does not delete or alter any v1 artefact. It explicitly
overrides the executability of v1.
