# Audit archive and current snapshots

The `MAIN_PAPER_*` files are July process records for the superseded
`kbound_short.tex` manuscript and are not current claim authority. They retain statements that were
later corrected, including iWildCam promotion, CIFAR confirmatory wording, older ImageNet-C cell
counts, and historical POEM/AETTA comparisons.

Current audit surfaces are:

- `../KBOUND_SHORT_RESULT_AUDIT.md`
- `../KBOUND_SHORT_CLAIM_MANIFEST.md`
- `../claim_ledger.json`
- `phase1_provenance_2026_08_27/` for the canonical panel's point-in-time provenance
- `../paper/generated/cct20_release_manifest.json` plus its receipt for CCT-20
- `../../../../experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/`
  for the So2Sat development stop

The dated empirical-data-quality and Phase-1 provenance directories predate the CCT-20 and So2Sat
executions. Preserve them as immutable audit history; do not infer current project status from their
unopened-target statements.
