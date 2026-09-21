# K-Bound Next-Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute the user's six approved research improvements and produce a tested, honestly evaluated next-phase package.

**Architecture:** Separate retrospective calibration, constructive theory, locked natural experiments, and deployment instrumentation. Integrate only validated outputs into a new manuscript and release.

**Tech Stack:** Existing pinned Python 3.12.13 runtime, NumPy/scikit-learn/SciPy, PyTorch/MPS, pytest, Docker, LaTeX.

**Spec:** docs/superpowers/specs/2026-09-21-kbound-next-phase-design.md

## Global Constraints

- Canonical immutable reference: /Users/pratik_n/Desktop/AutoML_Flagship_V8 2.
- Edit only the isolated next-phase worktree; never overwrite historical or sealed evidence.
- Use source/decision/output hashes and fresh output directories. Unknown provenance remains unknown.
- Freeze experimental choices before score outcomes; preserve negative results and stop rules.
- All saved-panel studies remain retrospective; no hidden target labels in model/threshold selection.
- Exact conformal rank exceeding calibration size gives infinity; finite cells do not imply independent environments.
- Physical-camera study remains future work. No forced scientific rating or unpublished priority claim.

## Review Focus

1. Scored-outcome perturbation must leave fitted predictions, thresholds, radii, and decision bytes unchanged.
2. Semantic cluster/model/environment overlap must be rejected or explicitly excluded by the split construction.
3. Ties, zero ADAPT exposure, empty feasible sets, missing classes, and insufficient calibration must fail honestly.
4. Tampered weights, expired calibration, nonfinite inputs, and repeated serving requests must not mutate the frozen model or accept an unsupported update.
5. Result tables must distinguish cell arithmetic, prospective protocol evidence, conditional mathematical guarantees, and measured deployment costs.

### Task 1: Matched calibration-value experiment

**Files:** Create kga/calibration_value.py, docs/research/kbound/scripts/run_calibration_value.py, tests/test_calibration_value.py, docs/research/kbound/next_phase/calibration_value_protocol.json.

**Interfaces:** Aggregate cells contain id, group, checkpoint, features, frozen_score, candidate_score and descriptive provenance. Decide consumes disjoint fit/tune/cal outcomes and outcome-free score records; score consumes sealed decisions plus separate scored outcomes. CLI provides prepare, decide, score and a safe orchestration command. Serialized infinity uses explicit status/null, never nonstandard JSON.

- [ ] Write tests first for whole-group exclusion, disjointness, insufficient-rank abstention, scored-outcome invariance, constant-radius rank equivalence, zero-exposure metrics, hash mismatch and occupied-output refusal.
- [ ] Observe failures using the pinned Python with -m pytest tests/test_calibration_value.py.
- [ ] Implement shared-predictor point/margin/exact-conformal/scaled-conformal/risk-selection arms, deterministic grouped splits and explicit interpretation flags.
- [ ] Use tests equivalent to assert rank_radius([.1]*8,.1) == inf; assert decisions(before_score_perturbation) == decisions(after_score_perturbation); assert matched_exposure_ids(point) == matched_exposure_ids(constant_radius).
- [ ] Freeze the protocol and authenticated input manifest before experimental execution. Run CIFAR, ImageNet noise and mixed-checkpoint hold-outs; save rejected/infeasible arms, all metrics, per-group outputs, descriptive cluster intervals and timings.
- [ ] Run targeted tests, review the implementation and result arithmetic, then commit only this task's files and bounded output summaries.

### Task 2: Constructive paired-benefit theory and optimization

**Files:** Create kga/paired_transport.py, tests/test_paired_transport.py, docs/research/kbound/next_phase/paired_transport_theory.md, docs/research/kbound/scripts/run_paired_transport_diagnostic.py.

**Interfaces:** A validated feasible-set specification supplies class-conditional pair bounds, target pair-frequency bounds, optional conditional-transport sensitivity and loss contrasts. The solver returns lower/upper paired benefit, feasibility/status and numerical witnesses; no result is a certificate without its declared assumptions.

- [ ] Write analytic two-class examples and infeasible/missing-class tests before implementation; test containment of a known data-generating joint distribution.
- [ ] Derive the LP and confidence construction explicitly; compare its scope with current label-shift confidence-region papers. Mark any unresolved proof as conjecture and exclude it from the manuscript.
- [ ] Implement finite-dimensional extrema and break-even sensitivity without altering existing population theorem semantics.
- [ ] Check exhaustive tiny cases, endpoint witnesses and synthetic hold-outs generated before fitting. Report scientific diagnostic outcomes even when vacuous.
- [ ] Obtain independent proof/code review and commit the scoped result and tests.

### Task 3: Lock and execute the natural-shift study

**Files:** New experiments/kbound/next_phase/natural_runner.py, tests/test_next_phase_natural.py, docs/research/kbound/next_phase/natural_protocol.json; reuse experiments/kbound/so2sat/prospective_v2.py and target-boundary interfaces without modifying historical authorities.

**Interfaces:** Preparation reads only source/checkpoint/access metadata. A sealed protocol binds development/calibration/target role IDs, adapters, model identities and acceptance criteria. Calibrate precedes any target access; target decision and outcome scoring are separate operations.

- [ ] Test that the target reader cannot run before a matching lock and eligible calibration result; tampered checkpoint/configuration or opened target status must be rejected.
- [ ] Authenticate materialized source checkpoints and metadata. Record exact earlier exposure; do not reconstruct missing provenance by assumption.
- [ ] Write and hash the specific So2Sat v2 protocol before calibration. Apply its existing city-level stop rules, or record why it cannot be faithfully executed before choosing a separately locked alternative.
- [ ] Execute development/eligible calibration, then permitted one-shot target decisions and scoring. Stop the target path honestly when eligibility fails.
- [ ] Save source tensors/model hashes, software versions, logs, decisions and outcomes with separate authority; test outcome perturbation invariance and reproduce report arithmetic.
- [ ] Review and commit the public protocol/runner/summary, excluding raw or restricted data and large checkpoints.

### Task 4: Deployment harness and cost measurements

**Files:** Create kga/deployment_audit.py, docs/research/kbound/scripts/run_deployment_audit.py, tests/test_deployment_audit.py.

**Interfaces:** A frozen contract binds model, adapter, evidence schema, estimator, calibration and validity lifetime. Assessment returns ADAPT/FREEZE/ABSTAIN with reason and immutable receipt. Benchmark stages distinguish real candidate execution from saved-feature replay.

- [ ] Write tests for tampered identities, nonfinite input, stale calibration, failed candidate generation, rollback invariance and repeat-request behavior; observe failure.
- [ ] Implement the local lifecycle and stage timer/memory recorder using existing gate/runtime interfaces. Do not create an unauthenticated external endpoint.
- [ ] Run fault injection and repeated local serving. Measure real candidate execution on development data where adapters/checkpoints are supported; record unavailable stages rather than inferring them.
- [ ] Run Docker/local checks and export median/tail latencies, warmups/repetitions, peak memory and measured scope.
- [ ] Review and commit implementation/tests and measured summary.

### Task 5: Integrate results and sharpen the manuscript

**Files:** Maintained shared manuscript/bibliography in this worktree, new generated next-phase tables/figures, docs/research/kbound/next_phase/RESULTS.md.

- [ ] Add verified closest-work comparisons and correct the audit's minor overstatements (set-theoretic decoder, residual terminology, traceability scope, historical table caveats).
- [ ] Present new proved/observed contributions with exact assumptions and provenance; separate exploratory results and failed attempts.
- [ ] Recompute every added table from saved new authorities and check arithmetic independently.
- [ ] Build full/main-only PDFs and Word; inspect updated pages and verify all final documents and source linkage.

### Task 6: Independent review and new release

- [ ] Review the entire diff, proofs, protocol, statistics, source privacy and final document claims using a fresh reviewer.
- [ ] Resolve material issues with focused failing tests and corrected proofs/claims. Run the repository's authorized source-bound verification scope and document exclusions.
- [ ] Create new seals/checksums/archive only after final source freeze; never reuse the earlier seal for changed source.
- [ ] Deliver documents, evidence, deployment receipts and an updated candid assessment. A negative study completes its protocol but does not close the scientific limitation.
