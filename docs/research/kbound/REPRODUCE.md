# Reproducing the K-Bound Release

This release supports two different activities: auditing committed evidence and
rerunning models from raw datasets. The fast audit is self-contained. Full model
reruns require separately obtained datasets and checkpoints listed in
[`DATA.md`](../../../DATA.md).

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[research,test]"
```

Node.js 20 is required to rebuild the dashboard. A TeX Live installation with
`latexmk` is required to rebuild the PDF. Lean verification uses the pinned
toolchain in `formal/lean-toolchain`.

## Fast Release Audit

```bash
make verify-fast
```

This command runs the complete Python suite, regenerates the dashboard snapshot
from the canonical result manifest, and compiles the TypeScript dashboard. It
does not claim to retrain any model.

The exact test count may grow as integrity checks are added. Two display-dependent
tests are intentionally skipped when optional capabilities are unavailable.

## Paper Build

```bash
make paper
```

The authoritative entry point is `kbound_short.tex`. The build must produce a
21-page `kbound_short_final_draft.pdf` with no fatal error, undefined citation,
undefined reference, duplicate label, or missing figure.

Repeated headline values are generated from:

```text
paper/generated/kbound_result_manifest.json
    -> scripts/make_tables.py
    -> paper/generated/kbound_numbers.tex
    -> kbound_short.tex
```

## Dashboard Build

```bash
make dashboard
python3 -m http.server 8765 --directory docs/research/kbound
```

Open `http://127.0.0.1:8765/kbound_dashboard.html`. The snapshot is deterministic
and records the canonical manifest SHA-256. It reads only the promoted paper
manifest and active physical-study output tree.

## Lean Build

```bash
make formal
```

This kernel-checks the indexed strict-core theorem spine. The formal README and
paper disclose the foundational probability and deployment assumptions that
remain external.

## Physical Study

```bash
make physical-preflight
```

This gate is expected to fail before fresh S01-S10 physical sessions exist. It
checks protocol identity, physical provenance, unique clip hashes, chronology,
held-out and replication replays, and anti-leakage records. A passing integrity
gate is necessary but not sufficient for a positive empirical conclusion.

## Full Dataset Reruns

Raw benchmark reruns are intentionally separate from the artifact audit. The
maintained completion matrix is locked by
`research_lock/MULTISEED_COMPLETION_PROTOCOL_v1.json` and is executed by one
launcher. Historical repository paths are not runtime dependencies.

After T9 is mounted, run:

```bash
bash docs/research/kbound/scripts/kbtrain.sh preflight --device mps
bash docs/research/kbound/scripts/kbtrain.sh plan --device mps
bash docs/research/kbound/scripts/kbtrain.sh run --device mps --yes
bash docs/research/kbound/scripts/kbtrain.sh status
bash docs/research/kbound/scripts/kbtrain.sh analyze --device mps
```

Use `--device cuda` on a CUDA host. CPU execution must be requested explicitly
with `--device cpu`; the default path does not silently replace a missing GPU
with a multi-day CPU run. Dataset overrides are documented in
[`DATA.md`](../../../DATA.md).

The default queue has four tracks:

| Track | Analysis seeds | New model runs | Completion rule |
|---|---:|---:|---|
| CIFAR-10-C SAR | 0-4 | 0-4 | one SAR per-condition artifact per seed |
| ImageNet-C SAR, locked 27 cells | 0-4 | 1-4 | imported seed 0 plus four new artifacts |
| PACS | 0-2 | 0-2 | Tent, EATA, and SAR artifacts for every seed |
| ImageNet-R Protocol D | 0-3 | 3 | all ten backbone artifacts for every seed |

For each target seed, the common scorer uses the next seed for residual
calibration and every remaining seed for fitting. It validates identical
condition and evidence schemas, recomputes the exact finite-sample residual
order statistic, and joins target outcomes only for offline metrics. The final
JSON separates point beats-both, gain-CI beats-both, and a stricter CI-robust
criterion that also requires the hierarchical 95% upper bound on FA_u to be at
most alpha.

The run directory contains immutable command locks, logs, per-seed compact
records, uniform summaries, and optional ablations. It is ignored by Git. Do
not replace the canonical paper manifest until the new records pass lineage,
protocol-hash, and claim-scope review.

The current release scope is summarized in [`DATA.md`](../../../DATA.md) and
[`KBOUND_SHORT_RESULT_AUDIT.md`](KBOUND_SHORT_RESULT_AUDIT.md).
