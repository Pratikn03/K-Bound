# K-Bound research map

**Reviewed 2026-08-31.** Start here for the maintained claims, their evidence, and their limits.
The full audit is [research_traceability.json](audits/research_traceability.json).
Older status documents and successful numerical validators do not establish theorem closure.

## What the research actually establishes

| Finding | Manuscript | Proof or implementation | Scope |
|---|---|---|---|
| Benefit is not identified from prediction evidence alone when matched-evidence worlds have opposite signs. | [Core theory](paper/sections/theory_core_main.tex) | [Impossibility](formal/KBound/Impossibility.lean), [measurable frontier](formal/KBound/Probability/MeasureFrontier.lean) | Explicit admissible target class; not a claim that every practical problem is unidentifiable. |
| The clipped identified benefit interval gives the strict ADAPT/FREEZE/ABSTAIN frontier. | [Core theory](paper/sections/theory_core_main.tex) | [Measurable target construction](formal/KBound/Probability/MeasureTarget.lean), [frontier](formal/KBound/Probability/MeasureFrontier.lean) | Binary zero-one loss (or labels supported only on the two predictions on their disagreement region), full declared correctness-field class, feasible margins, positive disagreement mass; equality supports no strict commitment. |
| A valid interval controls erroneous directional commitments. | [Certificate theorem](paper/sections/theory_certificate.tex) | [Measure certificate](formal/KBound/Probability/MeasureCertificate.lean), [exchangeable residual coverage](formal/KBound/Probability/MeasureConformal.lean), [current certificate](../../../kga/certificate.py) | Marginal, named-target coverage premise; neither conditional false-adapt control nor repeated-use protection follows automatically. |
| The audit floor explains why unrelated source labels cannot identify the target residual budget. | [Shared body](kbound_submission_body.tex) | Pen-and-paper fibre argument in the body; finite validators are supporting examples. | This general audit-floor statement is not itself one of the registered Lean capstones. |
| Additional probability foundations supply coverage, testing, concentration, and anytime ingredients. | [Formal-scope supplement](kbound_submission_supplement.tex) | [Formal package](formal/README.md) | Five foundational layers under explicit assumptions; the historical sixth one-bit/H/ratio-rate extension remains incomplete. |
| Orbit selection alone does not ensure a consistent sign on an evidence fibre. | [Supplement](kbound_submission_supplement.tex) | [Channel counterexample](formal/KBound/Probability/ChannelCounterexample.lean) | A genuine negative result. It must not be overwritten by older “one bit closes the theory” language. |

The population variables are `M, gamma, beta`; gamma is a calibration residual, not automatically
distribution drift. Empirical KGA uses `Delta_hat, epsilon`, not a numerical beta input. Covering
an observed batch outcome is different from covering population risk.

### Formal inventory: counts have different meanings

A fresh full compiler audit checked **238 authored theorem/lemma statements**:
142 registered capstones, 11 further indexed results, and 85 unindexed support/results.
The complete compiled inventory has 632 declarations; generated auxiliaries, definitions, instances,
and projections are not additional independent research contributions. All compiled declarations
passed the axiom/safety audit, with only the standard `propext`, `Classical.choice`, and `Quot.sound`
axioms observed. Exact types, source hashes, and the compiler receipt are recorded in the audit bundle.

This is broader than the 142-item release registry, but it does not close the historical sixth
foundational layer or establish that empirical calibration assumptions hold.

## Results, including the work that was hard to find

| Evidence | Authority | Honest interpretation |
|---|---|---|
| CIFAR-10-C Tent/EATA/SAR | [Canonical panel](../../../experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json), [current-policy sensitivity](../../../experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json) | Tent/EATA have lower pooled point-estimate regret than both fixed policies; SAR favors always-adapt. Tent's retrospective six-contrast Holm-adjusted p-values are 0.09375. No confirmatory routing result. |
| Interval quality and action exposure | [Interval diagnostics](paper/generated/current_policy_interval_diagnostics.json) | 2,160 cells per candidate; retrospective inclusion, widths, and false-direction counts. Rank-based observed inclusion is not new independent coverage evidence. |
| Controlled two-view MNIST (D33) | [Results](../../../experiments/kbound/results/controlled_multimodal_d33/results.json), claim KB-CLAIM-027 | 130 conditions: 9 ADAPT, 119 FREEZE, 2 ABSTAIN. Recorded mean accuracy is 85.6785%, versus 85.3554% single-A and 58.3231% always-fuse. Zero observed false ADAPT among only nine ADAPT decisions does not establish a small conditional error rate. Controlled injected corruption, not a natural-shift result; absent from the maintained paper. |
| Population/empirical decision bridge | [Seven-example bridge](../../../experiments/kbound/results/frontier_kga_bridge_v1/bridge_results.json), claim KB-CLAIM-043 | Five agreements and two disagreements show why the two APIs are related but have different abstention sets. A controlled algebraic diagnostic, not a real-data beta estimator. |
| CCT-20 | [Receipt-linked result](paper/generated/cct20_release_manifest.json) | Prospective safe-utility endpoint: 44 FREEZE, zero ADAPT, one ABSTAIN; ties freeze and avoids measured adaptation harm. No selective-routing gain; bootstrap levels are nominal. |
| So2Sat | [Development stop](../../../experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/README.md) | No feasible candidate; no target access and no target natural-shift score. |
| Other natural studies | [Result audit](KBOUND_SHORT_RESULT_AUDIT.md), [claim manifest](KBOUND_SHORT_CLAIM_MANIFEST.md) | Office-Home, Camelyon17 OOD, RxRx1, PACS, ImageNet-R and CIFAR-10.1 have differing diagnostic/retention roles. The iWildCam numerical/action row remains withheld; fMoW is not cleared and PovertyMap stopped before held-out evaluation. |

The historical `three_source_oof` block in
[the table manifest](paper/generated/kbound_result_manifest.json) still carries a stale positive CI
verdict. KB-CLAIM-024 correctly treats this as a historical constructed aggregate requiring a
reconciled rerun. Do not promote that block; correcting its generator and adding a regression guard
remains open. The maintained paper does not claim that aggregate as a natural win.

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

## Current software and release gaps

The root distribution publishes `kga` and `kga.*`. The hardened core decision path and focused
HTTP checks pass, but this does **not** make all shipped wrappers safe:

1. [Experiment shim](scripts/kbound_decide.py): masking can be lost; errors can fall through to
   comparisons that certify infinite estimates or negative radii.
2. [Public assumption helpers](../../../kga/assumptions.py): masked calibration can produce a
   zero radius and a “certify” gate; empty calibration can raise a formatting exception.
3. [ELARA integration](../../../kga/integrations/elara.py): the artifact's own protocol/schema
   identity is supplied as its expected identity; masked inputs and non-JSON-safe infinity remain.
4. [Promotion assessor](../../../kga/integrations/claims.py): malformed Boolean, failure-list, and
   probability metadata can be accepted.
5. The historical [prototype](kbound_pkg/README.md) and edge adapter paths have further legacy
   contract gaps. Gradient scaling is not parameter-preserving FREEZE. Compatibility tests do not
   certify deployment safety.

These failures were reproduced with synthetic fixtures, without opening target data. They do not
by themselves show that the existing finite-valued canonical panel is numerically wrong. They are
release blockers until corrected and covered by regression tests.

The full isolated audit ran 130 test modules. It is not a clean-checkout release: missing data,
Git/cache context, stale assertions, and a native OpenMP problem are reported separately from
passing tests and reproduced software bugs. See the machine audit for exact counts and logs.

## Maintained artifacts and authoritative inputs

| Read/build | Maintained path |
|---|---|
| Compact submission | [kbound_submission.tex](kbound_submission.tex), [PDF](kbound_short_final_draft.pdf), [Word](kbound_short_final_draft.docx) |
| Synchronized long companion | [kbound_tmlr.tex](kbound_tmlr.tex), [PDF](kbound_tmlr.pdf) |
| Shared manuscript | [Body](kbound_submission_body.tex), [supplement](kbound_submission_supplement.tex); the recursive live closure has 23 TeX inputs |
| Current claim wording | [claim_ledger.json](claim_ledger.json); theorem entries were reviewed against actual proofs in this audit |
| Canonical panel provenance | [source_manifest.json](../../../experiments/kbound/results/reconciled_panels_v1/source_manifest.json), [Phase-1 audit](KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md) |
| Current full audit | [research_traceability.json](audits/research_traceability.json) |
| Reproduction and publication checklist | [REPRODUCE.md](REPRODUCE.md), [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md), [data acquisition](../../../DATA.md) |

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

Eight superseded process documents were removed from the active tree only after byte-for-byte
recovery verification. Their paths, SHA-256 hashes, reasons, replacement authorities, and recovery
archive name are in [the audit](audits/research_traceability.json). The recovery archive lives
outside this repository; dated historical references to removed paths can be resolved through that
receipt.

Removed: the old manuscript strategy, two completed/obsolete cleanup plans, three July
`MAIN_PAPER_*` review projections, the July nontraining claim matrix, and the old 10X rating gate.
No dataset, checkpoint, proof, TeX source, implementation, unique derivation, sealed protocol,
or unreadable cloud placeholder was deleted.

Retained but **not current authority**:

- `THEORY_TO_CODE_MAP.md`, `THEORY_100_PERCENT_CLOSURE_PLAN.md`, and
  `PROJECT_STATUS_AND_OPEN_PROBLEMS.md`: contain obsolete closure/status assertions; do not use
  them to override this map or the active theorem statements.
- `KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md`: executed in part and superseded as current
  status, not an active blanket work order.
- `KBOUND_REMAINING_TODOS.md`: preserves calibration/leakage correction history and still has
  incoming references.
- `reports/THEORY_AUDIT_FULL.md`: generated by a historical checker with stale fixed expectations;
  not a current proof-completeness gate.
- Audit receipts, research locks, unique failed attempts, reproducibility instructions, and
  download scripts still needed to obtain data: retained for scientific reproducibility.

[Audit-history rules](audits/README.md) explain how dated records are kept separate from current
evidence. No old cleanup plan authorizes further deletion.
