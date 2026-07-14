# Compact Result Evidence

This tree contains the immutable, compact evidence used by the authoritative
short paper. It intentionally excludes raw datasets, model checkpoints, and
large exploratory run trees.

- Promoted claims retain their locked summaries and, where available,
  per-condition records.
- Diagnostic tracks retain enough evidence to verify the manuscript's negative
  or incomplete verdict.
- Camelyon17 multi-seed summaries are regenerated from the 12 files under
  `camelyon17_multiseed_v1/raw/`.
- `controller_cost_v1/` is an environment-specific controller-only
  microbenchmark; it is not end-to-end TTA latency.
- `gate_baselines_v1/gate_comparison.json` is the archived table artifact.
  `gate_comparison_exactrank.json` is the clean exact-rank recomputation.
- `imagenetc_seed0_v1/` contains the immutable 27-cell seed-0 records used
  only to bootstrap the clean ImageNet-C multi-seed completion. Its provenance
  file records the operating point and every imported hash.

The canonical index is
`docs/research/kbound/paper/generated/kbound_result_manifest.json`.
