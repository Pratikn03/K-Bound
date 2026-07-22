# External-storage and release policy

The Git release contains the evidence needed to audit claims without embedding redistributable datasets or large derived binaries.

## Tracked in Git

- claim ledger, evidence manifests, seals, checksums, and compact summaries;
- frozen configurations and exact reproduction commands;
- source, tests, Lean files, manuscripts, generated tables, and publication figures;
- small schema-validated per-condition JSON artifacts when redistribution is permitted.

## Stored externally

- raw datasets and licensed corpora;
- model checkpoints and large prediction arrays;
- caches, temporary files, raw console logs, and partial runs.

Every external dependency must have a stable environment-variable location, an official acquisition or regeneration procedure, and a checksum when licensing permits. `STORAGE_MANIFEST.json` is the machine-readable index. Release guards reject unapproved oversized files, accidental external data, and absolute machine-local paths.

The existence of an artifact is not sufficient evidence. A promoted result additionally requires a valid schema, the planned seed set, a declared protocol, claim linkage, and a passing evidence seal. Partial and quarantined artifacts remain excluded from manuscript claims.
