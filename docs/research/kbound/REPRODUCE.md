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

Expected status for this release: 221 tests collected after the deterministic
dashboard test was added, with two intentional skips when optional display
capabilities are unavailable.

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

Raw benchmark reruns are intentionally not hidden inside the artifact audit.
Each track must use its lock in `research_lock/`, preserve the specified seeds
and split roles, and write a new immutable result directory. Do not replace the
canonical manifest until the new run passes the declared statistical and
lineage checks.

The current release scope is summarized in [`DATA.md`](../../../DATA.md) and
[`KBOUND_SHORT_RESULT_AUDIT.md`](KBOUND_SHORT_RESULT_AUDIT.md).
