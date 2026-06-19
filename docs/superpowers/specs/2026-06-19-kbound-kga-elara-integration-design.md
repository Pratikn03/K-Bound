# K-Bound, KGA, and ELARA Research Integration Design

**Date:** 2026-06-19
**Status:** Approved design

## Objective

Build one reproducible research path in which ELARA-U proposes a multimodal
fusion candidate, KGA decides whether to deploy that candidate, and K-Bound
supplies the decision framework and certificate interpretation. The integration
must strengthen the empirical story without changing the existing headline
claim: KGA remains the only headline method and ELARA-U is a worked multimodal
instantiation.

The integration must not convert retrospective or target-label-light evidence
into a label-free claim. It must fail closed whenever the requested evaluation
mode lacks the information needed to make a valid decision.

## Current State and Integrity Finding

The repository already contains the main pieces:

- `kga/` is the canonical executable KGA certificate and decision package.
- `src/uais/elara_u/router.py` is the ELARA-U reliability-routing implementation.
- `src/uais/kbound/multimodal_guard.py` composes those systems.
- `experiments/kbound/kga_elara_demo.py` evaluates the composition on cached
  multimodal scores.
- `docs/research/kbound/kbound.tex` reports the composition as preliminary
  multimodal evidence.

The integration is fragmented across scripts, result formats, and paper text.
More importantly, the current multimodal guard computes placement benefits from
`y_test` even when no probe size is supplied. That computation is a
retrospective labeled audit, not a deployable label-free KGA decision. Existing
results may remain as retrospective evidence, but code, schema, and paper text
must not call this path label-free.

## Research Hierarchy

The paper and software will use one stable hierarchy:

1. **K-Bound** is the theory and decision framework.
2. **KGA** is the general adapt/freeze/abstain algorithm implementing K-Bound.
3. **ELARA-U** produces a reliability-routed multimodal fusion candidate.
4. **KGA-over-ELARA** decides between the validation-selected frozen candidate,
   the ELARA-U fused candidate, or abstention.

ELARA-U is not renamed as KGA, and its fusion gains are not automatically counted
as KGA beats-both wins.

## Architecture

The canonical data flow is:

```text
validation scores + unlabeled deployment scores
                    |
                    v
       ELARA-U reliability router
                    |
                    v
       frozen candidate vs fused candidate
                    |
                    v
     mode-specific KGA benefit certificate
                    |
                    v
          ADAPT / FREEZE / ABSTAIN
                    |
                    v
 versioned result + generated paper table + claim guard
```

The implementation will introduce a focused integration module rather than
merging the `kga`, `src/uais/elara_u`, and `src/uais/kbound` packages. The module
will import the canonical KGA and ELARA-U APIs and own only orchestration, mode
validation, result serialization, and claim classification.

## Evaluation Modes

Every run must declare exactly one mode. The output schema and paper table must
show that mode.

### Retrospective Audit

`retrospective_audit` may use all test labels to compute realized placement
benefits and evaluate what KGA would have done with those benefits. It is useful
for mechanism analysis and debugging. It is never eligible for a label-free,
deployment, sealed-holdout, or headline win claim.

### Target-Label-Light

`target_label_light` may use only a preregistered target probe of size `k`.
Probe selection, `k`, random seed, alpha, and benefit range must be fixed before
scoring. Non-probe labels may be used only after decisions are frozen, to evaluate
the resulting action. This mode is eligible only for a target-label-light claim.

### Label-Free

`label_free` accepts validation labels, validation predictions, and unlabeled
deployment predictions. It must not accept deployment labels. It requires a
frozen benefit estimator or certificate calibrator fitted on disjoint calibration
conditions. If that artifact is absent, incompatible, or not provenance-locked,
the runner exits without producing a decision. It must never fall back to
full-target placement benefits.

## Canonical Runner

The runner will be exposed through:

```bash
bash docs/research/kbound/scripts/kbtrain.sh kga-elara-integrated
```

It will also support a CPU-only dry run that validates configuration, cache
availability, output paths, and mode constraints without scoring:

```bash
bash docs/research/kbound/scripts/kbtrain.sh kga-elara-integrated-dry-run
```

The runner will consume a locked protocol file rather than embed dataset lists or
thresholds in Python. It will process every declared track, report unavailable or
invalid tracks explicitly, and never silently shrink the evaluation panel.

## Versioned Result Contract

The canonical output directory is:

```text
experiments/kbound/results/kga_elara_integrated_v1/
```

The runner will write:

- `results.json`: machine-readable per-track and aggregate results.
- `results_table.tex`: generated paper table derived only from `results.json`.
- `FINDINGS.md`: concise interpretation with eligible and ineligible claims.
- `run_manifest.json`: protocol hash, code revision, inputs, timestamps, and
  environment details.

`results.json` will include:

- schema and protocol versions;
- evaluation mode and claim tier;
- protocol path and content hash;
- input cache paths and hashes;
- frozen and fused candidate definitions;
- alpha, estimator identity, probe settings, and random seeds;
- per-track decisions and certificate bounds;
- always-freeze, always-fuse/adapt, KGA, and oracle metrics where evaluable;
- regret to oracle, coverage, false-adapt count and rate;
- uncertainty intervals and aggregation unit;
- missing-track and invalid-track records;
- machine-readable claim eligibility with reasons.

## Claim Guard

A deterministic claim guard will classify each run. It will not infer eligibility
from filenames or prose.

The integration may be promoted from preliminary/retrospective evidence only when
all of the following hold:

- evaluation is label-free with a frozen disjoint calibrator, or is explicitly
  reported as target-label-light;
- at least two held-out natural multimodal datasets are evaluated;
- the configuration was frozen before target scoring;
- at least three independent seeds or preregistered splits are reported;
- KGA regret is lower than both always-fuse/adapt and always-freeze;
- false-adapt is at most alpha;
- decision coverage is at least 20 percent;
- confidence intervals and declared strong baselines are complete;
- all required tracks are present and no integrity check failed.

The initial integration of already-opened caches is necessarily a retrospective
integration audit. It is not eligible for promotion merely because the software
pipeline passes.

## Paper Integration

The abstract, theorem statements, and current headline result table will remain
unchanged. The paper will:

- state the K-Bound -> KGA -> ELARA-U hierarchy once and consistently;
- replace ambiguous multimodal `label-free` wording with the declared mode;
- consume the generated integration table with `\input`;
- label the current result as a retrospective multimodal instantiation;
- list the exact promotion requirements for a future held-out result;
- keep target-label-light findings separate from the label-free core claim.

No ELARA-U fusion result will be described as a KGA beats-both result unless the
claim guard marks it eligible.

## Error Handling

The runner will stop with a nonzero exit code for:

- `label_free` mode supplied with deployment labels;
- `label_free` mode without a compatible frozen calibrator;
- a protocol hash mismatch;
- malformed cache arrays or inconsistent sample dimensions;
- an unknown evaluation mode;
- an attempt to overwrite a locked result with a different protocol.

Missing dataset caches will produce explicit failed track records and make the
aggregate run claim-ineligible. The dry run will report these conditions without
writing scored results.

## Test Strategy

Tests will cover:

1. The integrated guard maps certified positive, certified negative, and
   zero-bracketing benefits to adapt, freeze, and abstain.
2. Label-free inputs do not contain or consume target labels.
3. Permuting target labels cannot change a label-free decision.
4. Label-free mode fails closed without a frozen estimator.
5. Target-label-light mode uses exactly the preregistered probe and does not use
   non-probe labels before decision time.
6. Retrospective audit results are always claim-ineligible.
7. Result serialization validates against the versioned schema.
8. The claim guard enforces every promotion criterion independently.
9. Dry-run and full-run launcher commands route to the canonical runner.
10. The generated LaTeX table contains only values from the canonical result.
11. Existing KGA and ELARA-U test suites continue to pass.
12. The main and short paper PDFs compile and the changed pages render without
    clipping, overlap, or unreadable tables.

New behavior will be implemented test-first. Tests must be observed failing for
the intended missing behavior before production code is added.

## Scope Boundaries

This work does not:

- merge the KGA and ELARA-U packages;
- change KGA's certificate mathematics;
- tune thresholds on existing target results;
- relabel previously opened data as sealed or independent;
- promote preliminary results automatically;
- rewrite unrelated experiment runners or paper sections.

## Completion Criteria

The engineering integration is complete when the canonical command, result
schema, claim guard, generated table, tests, paper build, and visual PDF checks all
pass. The research is not described as 9+ evidence until a future preregistered
held-out run independently satisfies the promotion criteria.
