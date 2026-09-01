# K-Bound

K-Bound studies when available evidence can support adapting a model, retaining the frozen model,
or withholding a directional certificate. KGA implements the empirical interval decision rule.
The contribution is a scoped characterization and decision framework, not a promise of universal
accuracy improvement.

**Current research map:** [DOCS_INDEX.md](DOCS_INDEX.md). It links the actual findings to the
maintained manuscript, Lean statements, implementation, and evidence. The
[2026-08-31 full audit](audits/research_traceability.json) records coverage, counterexamples,
reproduced software failures, and the recoverable documentation cleanup.

## What is established—and what is not

- The maintained population theory characterizes the strict frontier for its declared admissible
  class under binary zero-one loss, or labels supported only on the two model predictions on
  their disagreement region. The population variables are `M, gamma, beta`; gamma is a calibration
  residual, not automatically distribution drift.
- Empirical KGA uses `Delta_hat, epsilon`, not a numerical beta input. A certificate about a
  measured batch outcome does not automatically protect population risk or repeated deployment.
- A fresh full Lean audit verified 238 authored theorem/lemma statements, including all 142
  registered capstones. This is a proof inventory, not a count of novel contributions. Five
  foundational layers are mechanized under explicit assumptions; the historical sixth
  one-bit/H/ratio-rate extension is not closed.
- Controlled CIFAR-10-C Tent/EATA point estimates favor routing against both fixed policies.
  The retrospective Holm adjustment over the six prospectively named candidate-by-baseline
  contrasts gives Tent adjusted p-values of 0.09375; this is not confirmatory.
  SAR favors always-adapt. Sign-flip inference needs joint invariance under each cluster sign
  flip; exchangeability alone does not establish that premise.
- CCT-20 completed a prospective safe-utility study: 44 FREEZE, zero ADAPT and one ABSTAIN.
  It ties always-freeze and avoids measured adaptation harm at the nominal bootstrap level.
  It does not establish selective-routing utility.
- So2Sat stopped at development with no feasible candidate, before gate calibration or target
  access. It has no target natural-shift score.
- The controlled two-view MNIST result and additional formal transfer/counterexample results
  are now indexed explicitly. They must not be relabeled as natural-shift evidence.
- No single natural dataset establishes confidence-supported selective routing against both fixed
  policies. The iWildCam numerical/action row is withheld; no completed physical-camera
  experiment is reported.

The paper, evidence, and root decision core are not invalidated by every historical draft defect.
Unsupported theorem-ledger closure claims have now been narrowed or withdrawn. Reproduced
wrapper-validation failures and the unfinished clean-source release gate still mean the repository
is **not yet a verified publication release**.

## Read and reproduce

| Purpose | Entry |
|---|---|
| Understand the research and remaining gaps | [Research map](DOCS_INDEX.md) |
| Read the compact manuscript | [Source](kbound_submission.tex), [PDF](kbound_short_final_draft.pdf), [Word](kbound_short_final_draft.docx) |
| Read the synchronized long companion | [Source](kbound_tmlr.tex), [PDF](kbound_tmlr.pdf) |
| Inspect the current claim wording | [claim_ledger.json](claim_ledger.json) |
| Inspect canonical empirical results | [Canonical panel](../../../experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json), [result audit](KBOUND_SHORT_RESULT_AUDIT.md) |
| Verify proofs and scope | [Formal README](formal/README.md) |
| Reproduce the publication artifacts | [REPRODUCE.md](REPRODUCE.md), [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md), [dataset acquisition](../../../DATA.md) |
| Explore evidence or prepare the prospective camera study | [Dashboard](dashboard/README.md), [camera runbook](edge/PHYSICAL_STUDY_RUNBOOK.md) |

The canonical panel binds 106 compact source artifacts. Later CCT-20 and So2Sat studies retain
separate receipt-linked authorities; they are not silently inserted into that older panel.
The [Phase-1 provenance audit](KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md) is a dated snapshot,
not proof that missing historical checkpoint or execution identities have been recovered.

## Implementation boundary

The root installation publishes only `kga` and `kga.*`; its command is `kga.cli:main`.
The maintained core and HTTP tests pass their focused checks. Score-only `proxy` mode is
diagnostic; `full` mode is a paired-benefit/external-estimate audit, not automatically the
schema-bound label-free estimator.

The full audit nevertheless reproduced problems in **shipped** surfaces:
`kga.assumptions`, `kga.integrations.elara`, `kga.integrations.claims`, and the shared
experiment shim. They can lose masked-input information, accept invalid certificate metadata,
or perform an ineffective protocol/schema check. See the
[exact failures and boundaries](DOCS_INDEX.md#current-software-and-release-gaps).
Do not infer deployment readiness from passing core tests.

The historical `kbound_pkg/kbound` implementation is excluded from the root installation.
Its heuristic gate and gradient-scaling optimizer are not the maintained certified contract.
A zero gradient does not prevent weight decay or optimizer momentum from moving parameters.
Historical edge paths that use this prototype need the same caution.

**Decision semantics:** ABSTAIN means no certified directional commitment while retaining the
frozen predictor. Certified FREEZE requires a valid negative interval; missing evidence is not
a certified FREEZE. Malformed HTTP requests are rejected. CCT-20 missing/nonfinite live features
produce ABSTAIN, while invalid sealed artifacts abort. So2Sat v1 aborts invalid/incomplete bundles
and has no operational ABSTAIN continuation; its target path remains disabled until the
city-versus-city/checkpoint action unit and fallback semantics are resolved under a new lock.

## Canonical build

From the repository root, in the pinned Python 3.12 research environment:

~~~bash
KBOUND_PYTHON=.venv/bin/python bash docs/research/kbound/runbooks/release_candidate.sh all
~~~

This workflow requires a reviewed clean source commit. It validates evidence, regenerates derived
surfaces, runs tests and Lean, builds both PDFs and the compact Word file, renders PDF pages, and
seals the release. It never launches training. Component checks and dirty-working-tree hashes are
not substitutes for this gate.

For a manuscript-only build after evidence validation:

~~~bash
BUILD_LONG_TMLR=1 BUILD_DOCX=1 bash docs/research/kbound/scripts/build_pdfs.sh
~~~

Maintained outputs are exactly `kbound_short_final_draft.pdf`, `kbound_tmlr.pdf`, and
`kbound_short_final_draft.docx`. The direct build defaults to the compact PDF unless the two
flags above are set. `BUILD_HISTORICAL_TMLR=1` remains a compatibility alias for the long build.
Historical compatibility PDFs and the old two-column diagnostic are not current deliveries.

## Manuscript policy

`kbound_submission.tex` and `kbound_tmlr.tex` consume the shared
`kbound_submission_body.tex` and synchronized supplement. Do not substitute
`kbound_short_body.tex` or `kbound_short_appendix.tex`. The actual dependency closure is computed
by `kbound_repro.manuscript_sources.active_source_paths()`, not by a static list of every TeX file.

Keep the main paper centered on the identification boundary, interval decision rule, controlled
evidence, scoped natural diagnostics, and limitations. Additional valid results should enter only
with their exact assumptions and evidence. Excluded drafts are not automatically valid simply
because a numerical validator or old status file says “closed.”

Eight obsolete process documents were recoverably removed during this audit. Unique derivations,
failed proof attempts, negative results, protocols and reproducibility/download scripts were kept.
The [audit-history index](audits/README.md) and [cleanup record](DOCS_INDEX.md#documentation-cleanup)
explain the retained history. No dataset or checkpoint was deleted.
