# K-Bound Research Guide

This directory is the maintained research surface for K-Bound and KGA. It
contains the conference draft, formalization, canonical
result manifest, dashboard, and physical-camera validation package.

## Start Here

| Goal | Entry point |
|---|---|
| Read the claim-controlled short paper | [kbound_short_final_draft.pdf](kbound_short_final_draft.pdf) |
| Edit the short paper | [kbound_short.tex](kbound_short.tex) and [kbound_short_appendix.tex](kbound_short_appendix.tex) |
| Inspect every promoted number | [paper/generated/kbound_result_manifest.json](paper/generated/kbound_result_manifest.json) |
| Audit claim-to-artifact links | [KBOUND_SHORT_CLAIM_MANIFEST.md](KBOUND_SHORT_CLAIM_MANIFEST.md) |
| Reproduce the submission | [REVIEWER_REPRO_PACKET.md](REVIEWER_REPRO_PACKET.md) |
| Inspect theory-to-code mapping | [THEORY_TO_CODE_MAP.md](THEORY_TO_CODE_MAP.md) |
| Build the research dashboard | [dashboard/README.md](dashboard/README.md) |
| Start the physical study | [edge/PHYSICAL_STUDY_RUNBOOK.md](edge/PHYSICAL_STUDY_RUNBOOK.md) |
| Verify Lean files | [formal/README.md](formal/README.md) |

## Fixed Terminology

- **K-Bound**: population theory and the adapt/freeze/abstain framework.
- **KGA**: finite-sample empirical wrapper around a candidate adapter.
- **Population frontier**: M, gamma, and beta.
- **Empirical certificate**: Delta_hat and epsilon.
- **Abstain**: do not commit the update; continue prediction with the frozen fallback.

The population frontier and empirical certificate are related but distinct.
Real-data KGA does not receive beta, and empirical abstention does not by itself
prove structural non-identifiability.

## Evidence Tiers

### Promoted controlled results

- CIFAR-10-C Tent: five seeds, 432 cells per seed, archived CI beats-both.
- CIFAR-10-C EATA: five seeds, 432 cells per seed, archived CI beats-both.
- ImageNet-C SAR: 27 cells, seed 0, paired-bootstrap beats-both with a
  single-seed caveat.

### Natural no-harm results

Office-Home M v2 (reconciled v3), iWildCam H v2 (reconciled v3), Camelyon17 genuine OOD reconciliation,
and RxRx1 J. These are not described as clean single-dataset natural beats-both
wins.

### Diagnostic or incomplete tracks

- CIFAR-10.1 fails the declared transfer bar.
- ImageNet-R has three of four planned seeds and no stable CI-robust win.
- PACS has one of three planned seeds.
- The three-source OOF stream is researcher-constructed routing evidence, not
  unseen-domain transfer.

## Canonical Build

~~~bash
bash scripts/reproduce_submission.sh
bash scripts/build_dashboard.sh
~~~

Paper-only build:

~~~bash
bash scripts/build_pdfs.sh
~~~

The generated result manifest is authoritative for repeated headline values.
Each local source path in that manifest is checked by the release-integrity
tests. Historical notes and archived runs are provenance, not automatic evidence.

## Dashboard

~~~bash
bash scripts/build_dashboard.sh
python3 -m http.server 8765 --directory .
~~~

Open http://127.0.0.1:8765/kbound_dashboard.html.

The dashboard reads the canonical paper manifest and the active
experiments/kbound/results/edge_real_phone_v1 tree. It never reads
archive/legacy_elara.

## Physical Validation

The edge code is a maintained, tested module rather than an informal demo. The
publication workflow is:

1. Prepare the protocol lock and deterministic checklists.
2. Capture S01-S02 and pass the source-model quality gate.
3. Capture S03-S06 and seal development plus conformal calibration.
4. Open S07-S08 once for held-out Phone A evaluation.
5. Capture S09-S10 on Phone B for replication.
6. Run the strict anti-leakage and publication gates.
7. Export camera tables and refresh the dashboard.

Start with:

~~~bash
python edge/scripts/preflight_r2.py
~~~

Browser preview, simulation, pilot data, and mock captures are connectivity or
software tests only. They cannot satisfy the publication gate.

## Formalization

~~~bash
cd formal
bash build.sh
~~~

The theorem map reports exactly which Lean declarations correspond to paper
statements. Do not describe the repository as a full foundational Mathlib
development: several measure-theoretic and deployment assumptions remain
external.

## Manuscript Policy

Use the 21-page short draft as the claim-controlled source. The historical
59-page `kbound.tex` is excluded from this release because it predates the
current claim corrections.
A balanced version should retain:

- problem and validity boundary;
- three core theory results plus the multiclass bridge;
- KGA architecture and calibration protocol;
- controlled beats-both evidence;
- natural no-harm and negative evidence;
- concise limitations and reproducibility.

Keep extended minimax, one-bit, martingale, historical ELARA, and large
diagnostic ladders in the supplement unless a venue explicitly allows them.
The detailed keep/move policy is in [KBOUND_MANUSCRIPT_STRATEGY.md](KBOUND_MANUSCRIPT_STRATEGY.md).
