# Legacy K-Bound publication surfaces (2026-09-02)

This directory is a byte-preserving archive of 107 superseded publication
surfaces that were still present outside an archive on 2026-09-02. None of
these files is current manuscript, claim, protocol, evidence, build, or release
authority. The original repository-relative layout is preserved below
`retired_tree/` so historical links remain unambiguous.

Current publication authority is limited to the two maintained TeX drivers:

- `docs/research/kbound/kbound_submission.tex`
- `docs/research/kbound/kbound_tmlr.tex`

Their live transitive TeX closure is computed by
`docs/research/kbound/kbound_repro/manuscript_sources.py`. Current claim and
release authority lives in `docs/research/kbound/claim_ledger.json`,
`docs/research/kbound/DOCS_INDEX.md`, and
`docs/research/kbound/runbooks/release_candidate.sh`.

## Inventory

| Category | Files | Bytes |
|---|---:|---:|
| `book_narrative` | 18 | 474,931 |
| `historical_docs` | 42 | 381,882 |
| `legacy_tex` | 11 | 510,466 |
| `obsolete_scripts` | 17 | 85,411 |
| `panel_narrative` | 17 | 671,153 |
| `stale_snapshots` | 2 | 4,546 |
| **Total** | **107** | **2,128,389** |

`MANIFEST.json` is the normative inventory. It records each original path,
archive path, byte count, SHA-256 digest, category, replacement authority, and
explicit false authority flags. It also records the corresponding blob digest
and size at base commit `660d893caede49c3b7daa8c18e43bb6cbbce5480`.

Three files had intentional, uncommitted working-tree changes when archived.
Their working-tree bytes—not the older base-commit blobs—were preserved:

- `docs/research/kbound/scripts/kbound_tour.py`
- `docs/research/kbound/scripts/reproduce_headlines.py`
- `docs/research/kbound/scripts/reproduce_submission.sh`

The manifest records both hashes and marks these entries with
`working_tree_diverged_from_base: true`.

## Scientific-content disposition

The retirement audit found no unique authoritative theorem, result, or claim that exists only in
these 107 files. All 24 `KB-CLAIM-*` identifiers mentioned by the retired tree remain represented
in `docs/research/kbound/claim_ledger.json`, including withdrawn, withheld, diagnostic, and
non-promotable entries.

The 28 theorem-like environments whose local labels do not occur verbatim in the maintained TeX
closure were also checked rather than inferred from label names. They are aliases or restatements
of retained results (for example, the multiclass decomposition and Aud-C/D/E/G), standard
background lemmas/definitions, or scoped extensions already indexed with qualifications in
`docs/research/kbound/audits/research_traceability.json` and backed by formal/theory sources that
remain outside this archive. The historical `def:detectable` threshold is a descriptive benchmark
classification, not a theorem or release claim.

In particular, old “beats both” prose is not revived by archival placement. Current permissions and
prohibitions are those in the claim ledger: the Camelyon headline and universal-improvement claim
are withdrawn, the cross-protocol aggregate is diagnostic and requires a reconciled rerun, and the
current CIFAR statement is limited to scoped point estimates plus explicitly retrospective
sensitivity analysis. `NUMBERS_PACK.md` is retired review prose; its underlying
`NUMBERS_PACK.json` and recomputation tree remain live as raw historical evidence.

## Deliberate exclusions

This archive does not contain unique formal derivations, theory-spine sources,
protocols, raw recomputation material, datasets, checkpoints, result evidence,
current So2Sat v2 work, the canonical three publication outputs, the release
runner, the anonymous-supplement builder, source/toolchain verifiers, or any
file in the maintained TeX closure. Those remain at their live paths.

Archived text and scripts are historical records only. Do not execute them or
cite their claims as current results. To reproduce the historical tree, check
out the recorded base commit; to inspect the exact pre-move working-tree bytes,
use `retired_tree/` and verify them against `MANIFEST.json`.
