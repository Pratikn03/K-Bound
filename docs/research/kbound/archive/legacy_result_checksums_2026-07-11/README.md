# Superseded July 2026 result checksum snapshot

These two files are preserved byte-for-byte from the July 11, 2026 result-checksum workflow
introduced by commit `4bd5c1979b8dfd6f6581b5cac6582ffa2f767b79`. They are historical provenance only.
They are **not current release authority**, and the archived Python generator must not be run.

Use these maintained release controls instead:

- `docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt` is the current 24-entry byte seal.
- `docs/research/kbound/runbooks/release_candidate.sh checksums` is its only supported generator.

At the time this snapshot was retired, four of its fourteen entries were red against the current
tree:

| Historical path | July checksum status against the retirement tree |
|---|---|
| `experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json` | Changed after the July snapshot; its current bytes are sealed elsewhere. |
| `docs/research/kbound/results_source.json` | Regenerated as a canonical compatibility view. |
| `docs/research/kbound/paper/generated/kbound_numbers.tex` | Regenerated for the maintained compact and long papers. |
| `docs/research/kbound/percondition_bootstrap.json` | Intentionally deleted; its historical per-condition inference is superseded and must not be restored or promoted. |

The archived generator is unsafe for current use because it writes a partial checksum file before
exiting nonzero when an artifact is missing. Regenerating this snapshot, deleting its missing row,
or restoring `percondition_bootstrap.json` would falsely make a historical bundle appear current.
