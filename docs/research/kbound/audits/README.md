# Audit archive and current snapshots

Start with [the current research map](../DOCS_INDEX.md) and
[research_traceability.json](research_traceability.json), the full 2026-08-31 audit.
The latter separates checked proofs, unsupported historical claims, reproduced software defects,
test/environment failures, and evidence unavailable in the original working tree.

The three superseded July `MAIN_PAPER_*` Markdown process records were removed from the active
tree after a byte-verified recovery archive was created. Their original paths, hashes and
replacement authorities are in `research_traceability.json`; historical references can be resolved
through its cleanup receipt. No proof, result source or sealed protocol was removed.

Current audit surfaces are:

- `../KBOUND_SHORT_RESULT_AUDIT.md`
- `../KBOUND_SHORT_CLAIM_MANIFEST.md`
- `../claim_ledger.json` for reviewed wording/status, not an automatic theorem-proof verifier
- `phase1_provenance_2026_08_27/` for the canonical panel's point-in-time provenance
- `../paper/generated/cct20_release_manifest.json` plus its receipt for CCT-20
- `../../../../experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/`
  for the So2Sat development stop

The dated empirical-data-quality and Phase-1 provenance directories predate the CCT-20 and So2Sat
executions. Preserve them as immutable audit history; do not infer current project status from their
unopened-target statements.

The older revision hashes, formal receipts and release checksums remain point-in-time records.
The fresh full compiler inventory is broader than the 142-capstone registry but does not close
the historical sixth foundational layer. Neither a successful schema check nor a dirty-working-tree
hash record is a clean-source publication release.
