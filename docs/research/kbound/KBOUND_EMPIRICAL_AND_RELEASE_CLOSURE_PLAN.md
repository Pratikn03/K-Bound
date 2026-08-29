# K-Bound Empirical and Release Closure Plan

**Status:** ACTIVE PLAN -- no new empirical claim is created by this document

**Created:** 2026-08-21

**Last execution audit:** 2026-08-24

**Current evidence authority:**
`experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`

**Current manuscript authority:** `kbound_submission.tex` and `kbound_tmlr.tex`

## 1. Objective

Close the ten gaps that currently prevent K-Bound from being a strong, reproducible empirical
submission while preserving every negative result and every validity qualification.

The target is not to force a positive result. The target is a prospectively locked evaluation in
which a positive, null, abstention-heavy, or negative result is equally admissible. A natural
CI-robust beats-both result may be promoted only if it clears the predeclared gate below.

## 2. Definition of Done

The closure is complete only when all of the following are true:

1. One natural single-dataset test is prospectively locked and evaluated once on untouched target
   environments; its result is retained whether positive or negative.
2. The same KGA decision contract, metric definitions, record schema, and claim rules are used by
   every promoted track.
3. Every promoted multi-seed claim uses independently trained checkpoints, not repeated gate seeds
   applied to one checkpoint.
4. The estimator-fit, residual-calibration, and final-test roles are disjoint and recorded.
5. Inference is performed at a predeclared independent unit: model seed, environment, camera, or
   corruption family as appropriate.
6. A controlled split-conformal confirmation is separate from the historical leave-one-condition-
   out stress-grid result.
7. PACS can be replayed from per-cell records.
8. POEM and AETTA rows use pinned official code or remain labelled protocol-matched ports.
9. The physical-camera publication gate passes using real captures.
10. The optional population-frontier diagnostic accepts externally declared `beta`; empirical KGA
    continues to use `Delta_hat` and `epsilon` without claiming that either estimates `beta`.
11. Repository-wide test collection has zero errors under the declared Python version.
12. Canonical JSON, source manifest, tables, figures, PDFs, commit, environment, and checksums are
    frozen in one release.

Completion does **not** require a natural beats-both win. If the prospective test is negative, the
experiment is complete but the natural-win claim remains absent.

## 3. Common Experimental Contract

### 3.1 Locked roles

Every confirmatory track must declare these roles before test labels are opened:

| Role | Permitted use | Forbidden use |
|---|---|---|
| Source train | Train `f0` | KGA selection using target-test labels |
| Development | Select evidence, estimator class, adapter configuration | Final claim |
| Estimator fit | Fit `h_theta: Z -> Delta` | Radius fitting or final scoring |
| Residual calibration | Compute exact-rank `epsilon` | Estimator selection |
| Test | One final evaluation after freeze | Retuning, condition removal, threshold changes |
| Replication | Confirm frozen pipeline on new seed/device/environment | Replacing a failed primary result |

### 3.2 Uniform decision and metric semantics

Use one rule in every track:

```text
if Delta_hat - epsilon > 0: ADAPT
else if Delta_hat + epsilon < 0: FREEZE
else: ABSTAIN and serve the frozen prediction
```

Use these definitions everywhere:

```text
Delta   = risk_freeze - risk_adapt
FA_u    = P(ADAPT and Delta <= 0)
FA_c    = P(Delta <= 0 | ADAPT)
coverage = P(ADAPT or FREEZE)
regret  = oracle risk gap under the deployed action
```

Always-adapt and always-freeze have decision coverage one. `FA_c` is descriptive unless a new
theorem is supplied. Zero observed false adaptation with zero adapt exposure is not a powered
safety result.

### 3.3 Required record schema

Each run must write an immutable decision file before labels are joined and a separate offline
evaluation file after labels are revealed.

Required decision fields:

```text
run_id, protocol_id, protocol_sha256, git_sha, dataset_version, split_role,
unit_id, environment_id, model_seed, checkpoint_sha256, adapter, adapter_config_sha256,
estimator_config_sha256, calibration_pool_sha256, alpha, evidence_schema_version,
Delta_hat, epsilon, action, decision_timestamp_utc
```

Required offline-only fields:

```text
Delta, risk_freeze, risk_adapt, oracle_action, regret, false_adapt,
balanced_accuracy, macro_f1, evaluation_timestamp_utc
```

The live decision file must not contain `Delta`, oracle action, labels, or target-test losses.

### 3.4 Seed policy

- Promotion target: five independently trained model seeds, normally `0 1 2 3 4`.
- Minimum exploratory panel: three independent model seeds, labelled exploratory.
- Save training logs, best-checkpoint selection metric, checkpoint hash, initialization seed, data
  order seed, and adapter seed separately.
- A repeated KGA fit or batch permutation on one `f0` is not an independent model seed.
- Checkpoint hashes must be distinct unless two runs are explicitly deterministic replications.

### 3.5 Inference policy

For the two beats-both comparisons, define
`g_b = regret_KGA - regret_baseline_b`, where lower is better. A CI-robust win requires the upper
endpoint of the simultaneous 95% CI for both `g_adapt` and `g_freeze` to be below zero. Use paired
cluster bootstrap or a paired seed-level procedure at the predeclared independent unit, with Holm
correction over the two comparisons.

The confirmatory gate is:

- point regret lower than both fixed policies;
- simultaneous 95% intervals exclude zero in the favorable direction;
- observed `FA_u <= alpha` and a reported one-sided uncertainty bound;
- exact interval coverage reported at the same unit;
- strict decision coverage and adapt exposure reported;
- no test-selected hyperparameter, comparator, subgroup, seed, or condition.

Before sealing, set minimum exposure requirements using development data. The recommended default
is adapt rate at least 0.10 and strict decision coverage at least 0.30. If the sealed test does not
meet them, report the result as low-exposure safety evidence rather than a routing win.

## 4. Phase 0 -- Consolidate Before Any GPU Run

### Tasks

1. Create `research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml` containing the common contract,
   primary dataset, split identifiers, model seeds, adapters, estimator, alpha, inference unit,
   bootstrap method, exposure gate, and fail-honest branch.
2. Add a protocol validator that rejects missing roles, overlapping unit IDs, reused checkpoint
   hashes, test-selected configuration fields, and label-bearing decision logs.
3. Add one orchestration entry point:
   `docs/research/kbound/runbooks/run_submission_closure.sh` with `preflight`, `smoke`, `train`,
   `evaluate`, and `release` modes.
4. Make dataset roots environment-driven. Code remains internal; raw datasets remain on T9.
5. Make every new run write outside `reconciled_panels_v1/`. Canonical evidence changes only after
   validation and explicit promotion.
6. Add a dry-run that prints the complete grid, split roles, checkpoints, estimated cell count,
   and output directories without loading a GPU model.

### Existing scripts that must not be used as confirmatory launchers yet

- `run_multiseed_chain.sh` assumes an older T9 layout containing both results and data.
- `finish_empirical_training.sh` finalizes the obsolete 23-page driver and older result policy.
- `seal_nine_track_lock.py` still contains historical SAR-withheld and older verdict language.
- `run_natural_win_v1.sh` belongs to an already unblinded, completed negative protocol.
- `ITEM13_PREREG_natural_mixed_v1.yaml` is an unsealed draft. It must receive a genuinely fresh
  test partition or be retained as design history only.

### Exit gate

- Protocol validator passes.
- Dry-run inventory is reviewed and sealed before training.
- No target-test label has been read by the new analysis path.
- All output paths are new and empty.

## 5. Gap 1 -- Natural Single-Dataset CI-Robust Beats-Both

### Primary natural route

First create a target-provenance inventory for Office-Home, iWildCam, fMoW, and any other candidate
natural dataset. For every environment, record whether its labels, aggregate losses, or gate
outcomes have ever been inspected. Select the primary dataset only from genuinely unopened
environments, and seal their IDs before evaluation. Prefer an official hidden-label evaluation
server or a newly acquired cohort when available.

Already opened Office-Home/iWildCam test units cannot be made prospective by renaming or repartitioning
them. If no unopened natural environment exists, acquire a new natural dataset or cohort before
claiming this gap is closed. Do not substitute a pooled cross-dataset mixture.

### Replication route

Run Office-Home and iWildCam as independent-seed replications. They can strengthen stability and
failure analysis but are not called untouched prospective confirmations if their target domains or
cameras have already been inspected.

The physical package/label study is a separate prospective real-world validation in Section 12.
Its shifts are staged physical conditions, so it must not be relabelled as an observational natural
benchmark. It can provide a clean held-out real-world win without satisfying the natural-benchmark
claim by itself.

### Acceptance artifacts

```text
research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml
experiments/kbound/results/prospective_natural_v1/protocol_lock.json
experiments/kbound/results/prospective_natural_v1/decisions.jsonl
experiments/kbound/results/prospective_natural_v1/offline_evaluation.jsonl
experiments/kbound/results/prospective_natural_v1/paired_ci.json
experiments/kbound/results/prospective_natural_v1/VERDICT.md
```

### Fail-honest branch

If either simultaneous CI touches zero, `FA_u` fails, or exposure is insufficient, preserve the
run as a null/negative/low-exposure result. Do not change alpha, evidence, adapter, split, or
inference unit and rerun the same test.

## 6. Gap 2 -- Controlled CIFAR Is the Strongest Existing Result

This is a positioning gap, not a reason to hide CIFAR-10-C.

### Tasks

1. Keep CIFAR-10-C Tent as the controlled mixed-regime result.
2. Keep EATA point-only and SAR negative; do not pool candidates into a universal claim.
3. Add the prospective natural result beside CIFAR only after it clears its gate.
4. If the natural result fails, frame the contrast as controlled detectability versus natural weak
   evidence, not as a missing experiment.

### Exit gate

The abstract, result table, figure captions, and conclusion use exactly the verdict in the
canonical claim ledger.

## 7. Gap 3 -- Low Adaptation Exposure on Natural Tracks

### Tasks

1. Report adapt, freeze, abstain, harmful-cell, and helpful-cell counts for every natural track.
2. Add Wilson or Clopper-Pearson intervals for adapt exposure and `FA_c` where meaningful.
3. Add development-only power calculations for the planned test size.
4. Predeclare an exposure gate before test evaluation.
5. Diagnose abstention using five non-exclusive causes: structural ambiguity, finite sample size,
   estimator inadequacy, calibration transfer, and conservative radius.

### Exit gate

No table calls zero `FA_u` a strong safety result when the gate never adapts. Every no-harm row is
paired with its action counts and uncertainty.

## 8. Gap 4 -- Exact Split-Conformal Confirmation

The historical stress grid remains leave-one-condition-out cross-fitted empirical calibration. Do
not relabel it as exact split conformal.

### New confirmation branch

1. Define a randomized deployment-unit generator before sampling: corruption family, severity,
   composition, batch size, candidate, and seed are sampled from a declared distribution.
2. Generate disjoint estimator-fit, calibration, and test units from that generator.
3. Sort calibration residuals and use
   `k = ceil((n_cal + 1)(1-alpha))`; if `k > n_cal`, return an infinite radius and abstain.
4. Evaluate the frozen rule once on the test units.
5. Report this as a separate randomized-condition confirmation, not a replacement for the standard
   corruption-family panel.

### Exit gate

- Unit-generation seed and distribution are sealed.
- No calibration residual appears in estimator fitting or final scoring.
- Exact-rank implementation tests include feasible and infeasible rank cases.
- Paper distinguishes the exact confirmation from the historical LOO panel.

## 9. Gap 5 -- Office-Home and iWildCam Independent Seeds

### Tasks

1. Train or recover five recipe-identical, independently initialized source checkpoints per track.
2. Verify that the runner's `--seeds` option selects those checkpoints; it must not merely reseed
   batch order around one seed-0 model.
3. Freeze one adapter configuration per track from development data.
4. Score all seeds using the common record schema.
5. Aggregate at model-seed plus environment/camera level; do not treat batches as independent.

### Existing entry points to audit and then reuse

```bash
bash docs/research/kbound/scripts/run_multiseed.sh officehome
bash docs/research/kbound/scripts/run_multiseed.sh iwildcam
```

Do not run them as confirmatory jobs until Phase 0 verifies checkpoint resolution, data roots, split
roles, and output schema.

### Exit gate

Five distinct checkpoint hashes, five complete decision/evaluation files, no missing seed, and a
paired cluster-level analysis with the predeclared unit.

## 10. Gap 6 -- PACS Per-Cell Replay

### Tasks

1. Extend `pacs_vlcs_runner.py` to serialize one record per
   `(model_seed, source_domains, target_domain, adapter, condition)`.
2. Include `Delta`, `Delta_hat`, `epsilon`, residual-pool hash, action, losses, oracle action,
   evidence vector/version, and all config/checkpoint hashes.
3. Add a replay command that reconstructs every aggregate from only the per-cell file.
4. Add a test requiring reconstructed aggregates to match the displayed panel within a declared
   floating-point tolerance.
5. Rerun seeds whose required per-cell state was not archived. Do not reconstruct missing latent
   fields from aggregate summaries.

### Exit gate

`PACS_REPLAY_AUDIT.json` reports complete cells, zero duplicate IDs, zero missing fields, matching
aggregates, and a recorded reason for every excluded cell.

## 11. Gap 7 -- Official POEM and AETTA Comparisons

### Tasks

1. Pin official repository commit, license, dependency lock, and checkpoint hash.
2. Run the authors' entry point without changing method-specific requirements.
3. Define the protocol adapter that maps each method's output to update/rollback decisions.
4. Replay exactly the same predeclared stream used for KGA.
5. Export raw official logs and converted decisions; test the conversion on known examples.
6. Report method-native and protocol-adapted results separately when semantics differ.

### Existing starting point

```bash
bash docs/research/kbound/runbooks/run_item11_official_baselines.sh
```

The row may be labelled `official implementation under a protocol adapter` only after commit,
environment, native command, and conversion audit are all present. Otherwise retain
`protocol-matched port`.

### Exit gate

Pinned commits, successful native logs, conversion tests, full action files, and a head-to-head
analysis produced without target-test tuning.

## 12. Gap 8 -- Physical-Camera Evidence

Follow `edge/PHYSICAL_STUDY_RUNBOOK.md` exactly.

### Commands

```bash
source .venv/bin/activate
python docs/research/kbound/edge/scripts/00_prepare_real_protocol.py
python docs/research/kbound/edge/scripts/preflight_r2.py

EDGE_CAMERA=0 EDGE_PHONE_ID=phone_a \
  bash docs/research/kbound/edge/scripts/run_edge_source_gate.sh

# Capture S03-S10 according to the locked runbook, then:
bash docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh
```

### Exit gate

`experiments/kbound/results/edge_real_phone_v1/publication_gate.json` passes without bypass or mock
data. The paper table includes held-out Phone A, Phone B replication, balanced accuracy, macro-F1,
regret, `FA_u`, `FA_c`, action exposure, abstention, and end-to-end latency.

## 13. Gap 9 -- Operational Population-Frontier Companion

KGA must not be modified to pretend that `epsilon = beta` or `Delta_hat = M`.

### Tasks

1. Add an optional `frontier_action(M, beta)` component that requires both quantities explicitly
   and fails closed when no credible `beta` is supplied.
2. Add a declared-class sensitivity analysis over externally specified `beta`; label it as
   sensitivity analysis, not data-driven certification.
3. In controlled binary experiments where `M` is computable, record population-frontier action and
   KGA empirical-certificate action side by side.
4. Add tests proving that the KGA API does not accept `beta` and that the frontier API does not
   accept `epsilon` as a substitute.
5. Report agreement, disagreement, and abstention causes without claiming equality of the two
   abstention sets.

### Exit gate

The theory-to-code map has separate entries for population frontier and empirical KGA, plus one
controlled bridge experiment using their shared target `Delta`.

## 14. Gap 10 -- Repository-Wide CI

### Resolved collection blockers (2026-08-24)

- Orphaned tests for removed legacy study modules and utilities were archived with their claims.
- Torch-dependent imports are lazy, and Python 3.12 is the declared release runtime.
- The full maintained collection now succeeds; Python 3.14 is not a supported release runtime.

### Tasks

1. Declare Python 3.12 as the release environment; either constrain unsupported Python versions or
   make Torch-dependent imports lazy and tested.
2. For each missing module, restore a source-hashed implementation or archive both its test and
   claim. Do not add empty stubs solely to make collection green.
3. Split CI into core KGA, experiment/reconciliation, edge, dashboard, Lean, and PDF jobs.
4. Run collection as its own required job before tests.
5. Make the canonical-claim validator and result-manifest integrity tests required.
6. Add a clean-checkout release job using `runbooks/release_candidate.sh` after that script is
   updated to the current 20-page and 89-page drivers.

### Exit gate

```bash
python -m pytest --collect-only
python -m pytest
bash docs/research/kbound/formal/build.sh
bash docs/research/kbound/scripts/build_pdfs.sh
python src/scripts/validate_manuscript_claims.py
```

All commands exit zero in a clean checkout. Required jobs may not hide failures with `|| true`.

## 15. Execution Order

| Milestone | Work | GPU? | Promotion unlocked |
|---|---|---:|---|
| M0 | Protocol schema, validator, launcher audit, clean output roots | No | None |
| M1 | Repository CI collection and release-driver repair | No | Reproducible development |
| M2 | PACS per-cell serializer/replay | Possibly | Replayable PACS diagnostic |
| M3 | Office-Home/iWildCam independent source seeds | Yes | Multi-seed natural stability |
| M4 | Unopened-natural-environment audit, seal, and one-shot test | Yes | Prospective natural verdict |
| M5 | Official POEM/AETTA execution and conversion audit | Yes | Faithful baseline table |
| M6 | Physical S01-S06 development/calibration and seal | Yes | Permission to open held-out sessions |
| M7 | Physical S07-S10 one-shot evaluation | Yes | Prospective real-world physical verdict |
| M8 | Randomized-condition exact split-conformal confirmation | Yes | Finite-sample controlled confirmation |
| M9 | Population-frontier companion and bridge experiment | No/low | Stronger theory-to-code linkage |
| M10 | Reconcile, rebuild, visual QA, freeze, tag, archive | No | Submission release |

M7 must not run before M6 is sealed. Results may be inspected only at the analysis point declared in
the protocol. Failed milestones are recorded; they are not silently dropped.

## 16. Final Reconciliation and Paper Fold-In

After all completed runs:

1. Copy no result manually into LaTeX.
2. Validate each candidate artifact and add its hash to a new source manifest.
3. Promote only artifacts whose protocol and statistical gates pass.
4. Run `scripts/reconcile_result_panels.py` into a new versioned panel, never over an old release.
5. Regenerate numerical macros, tables, and figures from that panel.
6. Update the claim ledger first; manuscript prose consumes the ledger's verdict.
7. Rebuild both maintained PDFs and render every page.
8. Freeze commit, environment, protocol hashes, model hashes, result hashes, PDF hashes, and page
   counts together.

Required release artifacts:

```text
canonical_panel_results.json
source_manifest.json
claim_ledger.json
protocol_lock.json
environment_lock.json
model_manifest.json
test_report.txt
formal_audit_report.json
kbound_short_final_draft.pdf
kbound_tmlr.pdf
RELEASE_CHECKSUMS.sha256
```

## 17. Claim Decision Table

| Outcome | Allowed claim |
|---|---|
| Natural gate clears both simultaneous CIs and safety/exposure gates | Clean prospective natural CI-robust routing win |
| Beats both at a point but one CI touches zero | Point result; not CI-robust |
| Ties better fixed policy with meaningful exposure | No-harm/safety result with exposure qualification |
| Ties freeze with zero adapts | Conservative endpoint reproduction, not routing success |
| Loses to either fixed policy | Negative diagnostic |
| Integrity, leakage, or protocol gate fails | Invalid for evidence; preserve as engineering log only |

## 18. Non-Negotiable Prohibitions

- No changing the primary dataset after seeing its result.
- No pooled multi-dataset result presented as a single natural-dataset win.
- No post-test alpha, radius, evidence, adapter, seed, condition, or subgroup selection.
- No relabelling one checkpoint as multiple model seeds.
- No exact-conformal claim for leave-one-condition-out residual reuse.
- No `epsilon = beta`, `Delta_hat = M`, or claim that empirical abstention proves structural
  non-identifiability.
- No promotion of mock camera data, browser demos, pilots, or blank templates.
- No official-baseline label without official code provenance.
- No suppressing a negative result from a sealed protocol.
- No release while canonical documents disagree about the promoted verdict.

## 19. Immediate Next Actions

1. Repair the protocol/launcher layer in Phase 0 before starting another long run.
2. Verify or train independent Office-Home and iWildCam checkpoints.
3. Add PACS per-cell serialization and replay tests.
4. Prepare and seal physical sessions S01-S06, then run the source and calibration gates.
5. Resolve the official baseline environments in parallel with data collection.
6. Run S07-S10 once only after the physical calibration seal exists.
7. Execute the exact split-conformal controlled confirmation.
8. Close repository-wide CI and issue one source-hashed release.

The current honest manuscript remains the fallback release throughout this program. New evidence is
folded in only after its gate passes; otherwise the existing scoped claims remain unchanged.

## 20. Execution Status

This table records implementation and evidence separately. ``Software ready`` does not mean that a
GPU, official-code, prospective-target, or physical-camera result exists.

| Milestone | Software status | Evidence status | Auditable artifact |
|---|---|---|---|
| M0 protocol contract | Complete | Draft remains intentionally unsealed | `research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml` |
| M1 repository CI/release | Complete | Maintained suites pass under Python 3.12 | `tests/CI_COLLECTION_CLOSURE.md` |
| M2 PACS replay | Complete | Real MPS smoke replay passes; full five-seed replay pending | `smoke_pacs_replay_v2/PACS_REPLAY_AUDIT.json` |
| M3 independent natural seeds | Complete | Office-Home/iWildCam five-checkpoint GPU runs pending | `scripts/audit_independent_checkpoints.py` |
| M4 unopened natural target | Complete | No verified unopened target was found; prospective natural claim remains blocked | `natural_target_provenance_v1/NATURAL_TARGET_PROVENANCE_AUDIT.json` |
| M5 official baselines | Complete, fail closed | Both methods remain port-only; complete successful native exports are absent | `official_repro_v1/OFFICIAL_BASELINE_AUDIT.json` |
| M6 physical development | Preflight and reporting logic complete | S01--S06 have zero real captures and are unsealed | `edge_real_phone_v1/publication_gate.json` |
| M7 physical held-out | Complete fail-closed pipeline | S07--S10 cannot run before M6 is captured and sealed | `edge/PHYSICAL_STUDY_RUNBOOK.md` |
| M8 exact confirmation | Complete | Draft unit manifest generated; not sealed or executed | `research_lock/KBOUND_EXACT_CONFIRMATION_UNSEALED_v1.json` |
| M9 frontier companion | Complete | Controlled algebraic bridge only; no real-data beta estimate | `frontier_kga_bridge_v1/bridge_results.json` |
| M10 release | Complete in the current working tree | Tests, Lean, compilation, and all-page rendering pass; clean-checkout checksum freeze remains pending | `runbooks/release_candidate.sh` |

### External work that code cannot manufacture

1. Train five distinct Office-Home and iWildCam source checkpoints and retain their hashes.
2. Obtain a genuinely unopened natural environment or hidden-label evaluation target.
3. Run clean pinned official AETTA/POEM entry points and archive complete native logs.
4. Record physical sessions S01--S06, seal calibration, then record S07--S10 exactly once.
5. Review and seal the exact-confirmation manifest before generating its fit/calibration/test units.

Until those actions occur, the release must retain the current descriptive, no-harm, negative, and
pending labels. The implementation is ready to evaluate those claims; it is not evidence for them.
