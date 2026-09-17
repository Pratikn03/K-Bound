# Superseded empirical authorities (2026-09-02)

This directory preserves 21 tracked files byte-for-byte from source commit
`660d893caede49c3b7daa8c18e43bb6cbbce5480`. They are historical evidence and not current authority
for empirical claims, executable protocols, or release generation.

The archive contains three deliberately separated classes:

- 13 completed or superseded protocol files that no longer define an active execution;
- 7 narrative findings files whose claims are superseded or not promotable under the canonical
  panel and current claim ledger; and
- 1 zero-byte JSON output retained as evidence of a failed historical output, not as data.

`MANIFEST.json` records each original path, archive path, category, byte count, SHA-256 digest,
reason, and replacement authority. Protocols without a direct current successor use a null
`replacement_authority` plus the explicit status `historical_only_no_current_successor`; the
unrelated root protocol registry is not presented as their successor. The original directory
structure is retained below `tree/`. Tests compare every archived file with its exact Git blob at
the source commit.

Runnable sources, stale lock/inventory surfaces, obsolete reproduction notebooks, and retired
showcase/readiness guides are recorded separately in `RETIRED_SURFACES_MANIFEST.json` and preserved
below `retired_tree/`. They are not members of the 21-file empirical-authority archive recorded by
`MANIFEST.json`.

Two syntactically malformed protocols intentionally remain at their original active paths because
current evidence and runtime receipts bind their exact bytes:

- `research_lock/RICH_EVIDENCE_CAMELYON_PROTOCOL_F_v1.yaml`
- `research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml`

Their fixed byte counts and SHA-256 digests are enforced by `tests/test_archived_authorities.py`.
Neither this archive nor a syntax tool may rewrite them.
