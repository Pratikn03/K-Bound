# Non-promotable K-Bound result snapshots

This directory preserves small result files that had been left under active
`experiments/kbound/results` paths even though they do not satisfy the release
evidence contract. Their bytes are preserved for auditability and are listed in
`MANIFEST.json` with SHA-256 hashes.

Nothing in this directory is an authoritative paper result. Release builders,
claim ledgers, and canonical-panel generators must not ingest these files.

The ImageNet-R snapshot is explicitly a smoke run: it used only 20 classes, 40
evaluation images, two batches, and a heuristic multicandidate route. The
iWildCam snapshot contains its own `BEATS-BOTH` string, but that embedded label
is not a release verdict. It is a one-checkpoint, one-seed result without a
canonical metric/provenance acceptance record.
