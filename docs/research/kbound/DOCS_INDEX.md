# K-Bound research map

**Research-map review 2026-09-03; audit-floor update 2026-09-05.** Start here for the maintained claims, their evidence, and their limits.
The full audit is [research_traceability.json](audits/research_traceability.json).
Older status documents and successful numerical validators do not establish theorem closure.

## What the research actually establishes

| Finding | Manuscript | Proof or implementation | Scope |
|---|---|---|---|
| Benefit is not identified from prediction evidence alone when matched-evidence worlds have opposite signs. | [Core theory](paper/sections/theory_core_main.tex) | [Impossibility](formal/KBound/Impossibility.lean), [measurable frontier](formal/KBound/Probability/MeasureFrontier.lean) | Explicit admissible target class; not a claim that every practical problem is unidentifiable. |
| The clipped identified benefit interval gives the strict ADAPT/FREEZE/ABSTAIN frontier. | [Core theory](paper/sections/theory_core_main.tex) | [Measurable target construction](formal/KBound/Probability/MeasureTarget.lean), [frontier](formal/KBound/Probability/MeasureFrontier.lean) | Binary zero-one loss (or labels supported only on the two predictions on their disagreement region), full declared correctness-field class, feasible margins, positive disagreement mass; equality supports no strict commitment. |
| A valid interval controls erroneous directional commitments. | [Certificate theorem](paper/sections/theory_certificate.tex) | [Measure certificate](formal/KBound/Probability/MeasureCertificate.lean), [exchangeable residual coverage](formal/KBound/Probability/MeasureConformal.lean), [current certificate](../../../kga/certificate.py) | Marginal, named-target coverage premise; neither conditional false-adapt control nor repeated-use protection follows automatically. |
| The audit floor explains why unrelated source labels cannot identify the target residual budget. | [Shared body](kbound_submission_body.tex) | [General randomized floor](formal/KBound/Probability/AuditFloor.lean), [full correctness-field radius](formal/KBound/Probability/AuditFloorCorrectness.lean) | Nonempty bounded residual fibre, measurable audit and common joint evidence/seed law. The full-class radius equality additionally needs feasible margins, positive disagreement mass and `0 <= beta <= 1/2`; no learned beta or empirical validity follows. |
| Additional probability foundations supply coverage, testing, concentration, and anytime ingredients. | [Formal-scope supplement](kbound_submission_supplement.tex) | [Formal package](formal/README.md) | Five foundational layers under explicit assumptions; the historical sixth one-bit/H/ratio-rate extension remains incomplete. |
| Orbit selection alone does not ensure a consistent sign on an evidence fibre. | [Supplement](kbound_submission_supplement.tex) | [Channel counterexample](formal/KBound/Probability/ChannelCounterexample.lean) | A genuine negative result. It must not be overwritten by older “one bit closes the theory” language. |

The population variables are `M, gamma, beta`; gamma is a calibration residual, not automatically
distribution drift. Empirical KGA uses `Delta_hat, epsilon`, not a numerical beta input. Covering
an observed batch outcome is different from covering population risk.

### Formal inventory: counts have different meanings

The historical full compiler audit checked **238 authored theorem/lemma statements**:
142 registered capstones, 11 further indexed results, and 85 unindexed support/results.
The complete compiled inventory has 632 declarations; generated auxiliaries, definitions, instances,
and projections are not additional independent research contributions. All compiled declarations
passed the axiom/safety audit, with only the standard `propext`, `Classical.choice`, and `Quot.sound`
axioms observed. Exact types, source hashes, and the compiler receipt are recorded in the audit bundle.

This was broader than the then-142-item release registry, but it does not close the historical sixth
foundational layer or establish that empirical calibration assumptions hold.

The 2026-09-05 audit-floor addition raises the current registry to **150**. Its
fresh direct pinned compile covers all **44 local modules**, and its actual axiom
audit covers the **150 registered plus 15 additional supporting/example declarations**.
Those are different scopes from the historical 238-statement census; they must not
be added together or treated as a new all-declaration census. Independent review
passed. This is not a standard Lake build or a final release receipt. See the
[new CM-04 verification index](../../../protocols/confirmatory_v2/FORMAL_VERIFICATION_20260905.json);
the older dated traceability audit is preserved unchanged.

## Results, including the work that was hard to find

| Evidence | Authority | Honest interpretation |
|---|---|---|
| CIFAR-10-C Tent/EATA/SAR | [Canonical panel](../../../experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json), [current-policy sensitivity](../../../experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json) | Tent/EATA have lower pooled point-estimate regret than both fixed policies; SAR favors always-adapt. Tent's retrospective six-contrast Holm-adjusted values are 0.140625 against always-adapt and 0.09375 against always-freeze. No confirmatory routing result. |
| Interval quality and action exposure | [Interval diagnostics](paper/generated/current_policy_interval_diagnostics.json) | 2,160 cells per candidate; retrospective inclusion, widths, and false-direction counts. Rank-based observed inclusion is not new independent coverage evidence. |
| Controlled two-view MNIST (D33) | [Results](../../../experiments/kbound/results/controlled_multimodal_d33/results.json), claim KB-CLAIM-027 | 130 conditions: 9 ADAPT, 119 FREEZE, 2 ABSTAIN. Recorded mean accuracy is 85.6785%, versus 85.3554% single-A and 58.3231% always-fuse. Zero observed false ADAPT among only nine ADAPT decisions does not establish a small conditional error rate. Controlled injected corruption, not a natural-shift result; absent from the maintained paper. |
| Population/empirical decision bridge | [Seven-example bridge](../../../experiments/kbound/results/frontier_kga_bridge_v1/bridge_results.json), claim KB-CLAIM-043 | Five agreements and two disagreements show why the two APIs are related but have different abstention sets. A controlled algebraic diagnostic, not a real-data beta estimator. |
| CCT-20 | [Receipt-linked result](paper/generated/cct20_release_manifest.json) | Cell-outcome-unopened, internally sealed execution, but not globally label-unopened or publicly preregistered: 0/44/1 ADAPT/FREEZE/ABSTAIN. All 45 predictions used the frozen model, but the sole helpful cell received FREEZE: one false FREEZE among 44 FREEZE actions (1/45 overall). Safe-utility only; no selective-routing or strong-success result; bootstrap levels are nominal. |
| So2Sat | [Development stop](../../../experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/README.md) | No feasible candidate; no target access and no target natural-shift score. |
| Other natural studies | [Result audit](KBOUND_SHORT_RESULT_AUDIT.md), [claim manifest](KBOUND_SHORT_CLAIM_MANIFEST.md) | Office-Home, Camelyon17 OOD, RxRx1, PACS, ImageNet-R and CIFAR-10.1 have differing diagnostic/retention roles. The iWildCam numerical/action row remains withheld; fMoW is not cleared and PovertyMap stopped before held-out evaluation. |

The historical `three_source_oof` block in
[the table manifest](paper/generated/kbound_result_manifest.json) is now generated from the live
KB-CLAIM-024 authority as a historical diagnostic: its numeric record and raw-evidence linkage are
preserved, the retired nine-track seal is archive-qualified, and current-policy, numeric-release,
and headline-promotion eligibility are all false. The compatibility results source is regenerated
from that table. A reconciled per-track rerun remains required; the maintained paper makes no
current CI-supported, natural-shift, or transfer claim from this aggregate.

### Useful theory outside the paper—not a queue of ready-made theorems

- The [joint wrong-direction bound](formal/KBound/Probability/MeasureConformal.lean) controls the
  union of the two errors at alpha in one covered experiment; the active statement gives the
  marginal bounds separately.
- [UnitMismatch](formal/KBound/UnitMismatch.lean) and [Stability](formal/KBound/Stability.lean)
  contain useful transfer/negative results. They do not prove that arbitrary LOO residuals are
  exchangeable.
- [JackknifePlus](formal/KBound/JackknifePlus.lean) proves counting ingredients, not the complete
  Jackknife+ coverage theorem.
- Episode budgets, regression brackets, Gaussian certification prices, and multiclass witnesses
  contain useful ideas, but their excluded drafts have assumption, constant, or statement defects.
  The audit records what can be retained and what needs a new proof. Do not import whole files.
- The [separate multiclass-vector track](../multiclass_vector_capacity/README.md) has 46 verified
  local named Lean proofs and 103 exact certificates. Its ledger explicitly forbids promotion into
  K-Bound: most of its proposed research program and novelty assessment remain open.

## Current software and release status

The fail-open cases previously listed here are now closed in the maintained surface. The public
decision helpers, experiment shim, KGA/ELARA integrations, promotion assessor, baseline converter,
release wrappers, checksum/source-seal readers, and So2Sat target boundary reject missing,
malformed, masked, non-finite, partial, duplicate-key, stale, or forged inputs. Insufficient
evidence remains `ABSTAIN` and retains the frozen model; it is not relabelled as certified
`FREEZE`. The dated machine-readable audit is
[`maintained_wrapper_fail_closed_audit_2026_09_02.json`](audits/maintained_wrapper_fail_closed_audit_2026_09_02.json),
with its human-readable companion
[`MAINTAINED_WRAPPER_FAIL_CLOSED_AUDIT_2026_09_02.md`](audits/MAINTAINED_WRAPPER_FAIL_CLOSED_AUDIT_2026_09_02.md).

This closure is scoped to the dynamically inventoried maintained entry points. Archived historical
writers remain non-executable evidence, and a newly tracked wrapper or validator fails repository
classification until it is reviewed. It also does not, by itself, certify the numerical truth of
the canonical empirical panel. The final clean-commit, full-repository execution receipt, rebuilt
publication artifacts, source seal, and independently verified checksums remain release gates.

## Maintained artifacts and authoritative inputs

| Read/build | Maintained path |
|---|---|
| Named compact main paper (20--28 page, double-column target; main argument and references only) | [kbound_short_main.tex](kbound_short_main.tex), [current PDF](release/current/kbound_short_main.pdf) |
| Named standalone supplement (Appendices A--N; independent of the main paper's page numbers and auxiliary file) | [kbound_short_supplement.tex](kbound_short_supplement.tex), [current PDF](release/current/kbound_short_supplement.pdf) |
| Anonymous integrated TMLR review manuscript | [kbound_tmlr.tex](kbound_tmlr.tex), [current PDF](release/current/kbound_tmlr.pdf) |
| Named full technical report (current theory, evidence, proof, and reproducibility record) | [kbound_full_report.tex](kbound_full_report.tex), [current PDF](release/current/kbound_full_report.pdf) |
| Nonrelease combined comparison artifact | [kbound_submission.tex](kbound_submission.tex), [local comparison build](kbound_short_final_draft.pdf) |
| Shared manuscript | [Body](kbound_submission_body.tex), [supplement](kbound_submission_supplement.tex); the recursive live closure has 30 TeX inputs |
| Current publication checksum manifest | [KBOUND_CURRENT_SHA256SUMS.txt](release/current/KBOUND_CURRENT_SHA256SUMS.txt); `release/current/` and `output/pdf/` contain the same four role-distinct current PDFs |
| Current claim wording | [claim_ledger.json](claim_ledger.json); theorem entries were reviewed against actual proofs in this audit |
| Canonical panel provenance | [source_manifest.json](../../../experiments/kbound/results/reconciled_panels_v1/source_manifest.json), [Phase-1 audit](KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md) |
| Current full audit | [research_traceability.json](audits/research_traceability.json) |
| Reproduction and publication checklist | [REPRODUCE.md](REPRODUCE.md), [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md), [data acquisition](../../../DATA.md) |
| Superseded empirical authorities and retired entry points (not current authority) | [2026-09-02 archive](archive/superseded_empirical_authorities_2026-09-02/README.md), [retired-surface guide](archive/superseded_empirical_authorities_2026-09-02/RETIRED_SURFACES_README.md), and their byte manifests |
| Legacy publication surfaces (not current authority) | [107-file archive](archive/legacy_publication_surfaces_2026-09-02/README.md) and its [byte manifest](archive/legacy_publication_surfaces_2026-09-02/MANIFEST.json) |
| Historical publication builds (not deliverables) | [2026-09-02 archive](archive/stale_publication_builds_2026-09-02/README.md) and its byte manifest |

This audit does not rebuild the PDFs/Word file or manufacture a clean-source release seal.
[The prior revision receipt](audits/revision_verification_2026_08_31.json) and
[KBOUND_RELEASE_SHA256SUMS.txt](KBOUND_RELEASE_SHA256SUMS.txt) are historical snapshots; they do not
attest the newly edited working tree. Final source freeze, clean-checkout verification and fresh
independently verified release checksums remain required. Full Git object integrity is still
unverified, not demonstrated corrupt.

From the repository root, these are separate checks:

~~~bash
python docs/research/kbound/scripts/validate_canonical_release_data.py
python src/scripts/validate_manuscript_claims.py
python docs/research/kbound/scripts/build_current_policy_interval_diagnostics.py --check
bash docs/research/kbound/formal/build.sh --json-out /tmp/kbound-formal-audit.json
~~~

The full publication workflow remains `bash docs/research/kbound/runbooks/release_candidate.sh all`
from a reviewed clean commit using the pinned research environment. It is not equivalent to the
component checks above and it does not launch training.

## Documentation cleanup

The 2026-09-02 authority cleanup also retired three duplicate or obsolete reproduction notebooks,
the historical G5/showcase/full-panel launchers, the unbannered WIN_HUNT v4/v5 copy-paste
runsheets, and the subjective 85+ readiness route. Their baseline bytes are in the
superseded-authority archive. Compatibility launchers now fail closed; the maintained notebook
entry point is `notebooks/00_KBound_Master_Guide.ipynb`, and release verification runs only through
`runbooks/release_candidate.sh all`.

Eight superseded process documents were removed from the active tree only after byte-for-byte
recovery verification. Their paths, SHA-256 hashes, reasons, replacement authorities, and recovery
archive name are in [the audit](audits/research_traceability.json). The recovery archive lives
outside this repository; dated historical references to removed paths can be resolved through that
receipt.

Removed: the old manuscript strategy, two completed/obsolete cleanup plans, three July
`MAIN_PAPER_*` review projections, the July nontraining claim matrix, and the old 10X rating gate.
No dataset, checkpoint, proof, TeX source, implementation, unique derivation, sealed protocol,
or unreadable cloud placeholder was deleted.

The same review moved 107 superseded paper drivers, body/appendix variants, book-narrative files,
status/plan/report documents, panel-review prose, obsolete scripts, and stale snapshots into the
[legacy-publication archive](archive/legacy_publication_surfaces_2026-09-02/README.md). The
[deterministic manifest](archive/legacy_publication_surfaces_2026-09-02/MANIFEST.json) records every
original path, replacement authority, byte count, and SHA-256 digest. These files remain readable
as history but cannot override this map, the claim ledger, either maintained driver, or the release
runner.

Retained live but **not current authority**: audit receipts, research locks, unique derivations and
failed attempts, raw recomputation evidence, formal sources, sealed protocols, and the
reproducibility/download material needed to obtain data. They were deliberately excluded from the
legacy-publication archive.

[Audit-history rules](audits/README.md) explain how dated records are kept separate from current
evidence. No old cleanup plan authorizes further deletion.
