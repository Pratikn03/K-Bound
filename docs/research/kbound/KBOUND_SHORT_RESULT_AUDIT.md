# K-Bound Empirical Consistency Audit

Date: 2026-08-29

## Canonical Source

Authoritative result artifact:
`experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`

- Generator: `scripts/reconcile_result_panels.py`.
- Inputs: 106 compact source artifacts with original and released SHA-256 hashes in
  `source_manifest.json`.
- Generated consumers: `canonical_panel_table.tex`, `paper/generated/kbound_numbers.tex`, and
  `fig_decision_value_frontier.png`.
- Retrospective family sensitivity:
  `experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json`, with its
  own artifact hash, Python/NumPy runtime, analysis-script hash, live policy/certificate hashes, and
  hash of the protocol that prospectively named the six contrasts. The exact-rank replay and
  inference were retrospective. This artifact is not part of the canonical point-estimate JSON.
- Decision rule: one KGA interval rule; candidate-specific estimators and radii remain separate.
- ImageNet-C authority: 27 conditions per seed times five seeds, not historical 36-cell variants.

## Authority Boundary for Later Studies

The 106-artifact canonical panel predates two later, prospectively governed studies. They are not
silently merged into that panel:

- **CCT-20 target:** `docs/research/kbound/paper/generated/cct20_release_manifest.json`
  (SHA-256 `2ac7365df0f86eabb284fd0e0646986c4dc1f8f552b3a32f16a2072fdceaf26c`)
  and its receipt are the authority. The verdict is `SAFE_UTILITY_ONLY`: 44 FREEZE, zero ADAPT, one
  ABSTAIN across 45 checkpoint-location cells. KGA ties always-freeze and avoids the degradation
  from always-adapt, but fails the preregistered strong-success action-exposure requirement. Outcomes
  were unopened before execution, but aggregate target metadata had been inspected during dataset
  ranking; the study is therefore not described as literally label-unopened.
- **So2Sat development gate-fit:**
  `experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_candidate_selection.json`
  (SHA-256 `8db11a797d98c5f104736a5ed982a422982f9f75fc8a7d1c6e13f07a826c0b79`)
  and the adjacent candidate receipts are the authority. The locked status is
  `NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL`; therefore no gate-calibration or target stage ran.
  Target inputs are empty and target pixel/label reads are both zero. These are nine-city
  development results, not a Culture-10 target natural-shift score.

## Reconciled Panel

| Track | n | KGA | Adapt | Freeze | FA_u | Coverage | Defensible interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Office-Home primary | 35 | .0158 | .0468 | .0158 | .0000 | .3143 | descriptive tie with freeze; zero adapts; A7 open |
| Office-Home test-stream replication | 54 | .0215 | .0458 | .0217 | .0000 | .2778 | tiny point edge; test-seed CI includes zero |
| iWildCam | withheld | withheld | withheld | withheld | withheld | withheld | official-metric, population-sealed rerun required |
| ImageNet-C SAR | 135 | .0227 | .0529 | .0319 | .0000 | .2741 | point edge with two positive descriptive run-seed intervals; no independent checkpoint identities, so not CI-robust |
| ImageNet-C Tent | 135 | .0154 | .0191 | .0145 | .0222 | .0370 | trails freeze; three observed false adaptations |
| ImageNet-C EATA | 135 | .0024 | .0001 | .0342 | .0074 | .8074 | trails adapt; one observed false adaptation |
| PACS | 12 domain-seed units | .0431 | .0176 | .0446 | .0093 | .8148 | favors always-adapt; per-cell replay incomplete |
| ImageNet-R | 480 | .0150 | .0064 | .0325 | .0000 | .4042 | favors adaptation on 8/10 backbones |
| CIFAR-10-C Tent | 2160 | .0019 | .0080 | .1239 | .0009 | .6611 | lower exact-rank point regret than each fixed policy; two observed false adaptations; retrospective six-contrast Holm is non-confirmatory |
| CIFAR-10-C EATA | 2160 | .0016 | .0033 | .1313 | .0000 | .6074 | lower exact-rank point regret than each fixed policy; adapt-side family interval crosses zero; retrospective six-contrast Holm is non-confirmatory |
| CIFAR-10-C SAR | 2160 | .0018 | .0003 | .1405 | .0000 | .6546 | always-adapt has lower regret |
| Camelyon17 OOD | 18 | .0000 | .0000 | .1381 | .0000 | 1.0000 | reproduces always-adapt; one-sided |
| Camelyon17 B-v2 Tent | 108 | .0446 | .0097 | .0820 | .0000 | .2500 | comparator-favoring diagnostic |
| Camelyon17 B-v2 EATA | 108 | .0260 | .0040 | .0911 | .0278 | .4722 | comparator-favoring diagnostic |
| Camelyon17 B-v2 SAR | 108 | .0289 | .0016 | .1001 | .0093 | .4815 | comparator-favoring diagnostic |
| RxRx1 | 60 | .0000 | .2531 | .0000 | .0000 | 1.0000 | freezes throughout; endpoint no-harm |
| CIFAR-10.1 | 48 | .0017 | .0190 | .0017 | .0000 | .4583 | ties freeze; no adapts |

Protocol note: these natural-shift candidate losses come from transductive TTA. Episodic updates
read the unlabeled evaluation batch; online prediction uses evaluation-batch BatchNorm statistics
after updating on a separate auxiliary stream. The sample hashes prove that the auxiliary stream
and evaluation identities differ; they do not make the candidate evaluation inductive.

Coverage means `P(adapt or freeze)`, not adapt rate. Always-adapt and always-freeze have coverage
one. PACS displays 12 domain-seed aggregate units while its reported FA value is pooled over the
underlying cells; the paper labels that unit mismatch instead of silently treating 12 as the
false-adapt denominator.

## CIFAR-10-C Action Audit

- Tent: 1,091 adapt / 337 freeze / 732 abstain, with 2 observed false adaptations.
- EATA: 1,207 adapt / 105 freeze / 848 abstain, with 0 observed false adaptations.
- SAR: 1,414 adapt / 0 freeze / 746 abstain, with 0 observed false adaptations.
- Aggregate: 4,154 strict decisions among 6,480 candidate-cell evaluations, 64.1 percent coverage,
  and two observed false adaptations.
- Descriptive arithmetic across three separately calibrated candidates, not one policy or
  inferential estimand: aggregate point regret is 0.00178 versus 0.00386 for always-adapt and
  0.13191 for always-freeze, a 2.17x reduction relative to the better fixed policy.
- Candidate inference remains separate. The retrospective current-policy analysis averages run seeds
  and condition cells within six corruption families. For Tent, baseline-minus-KGA gaps are 0.00606
  with ordinary 95 percent interval [0.00209, 0.00954] against always-adapt and 0.12202 with interval
  [0.06282, 0.17757] against always-freeze. These confidence intervals are unadjusted. The
  within-Tent two-contrast Holm values are 0.046875 against always-adapt and 0.03125 against
  always-freeze and are explicitly post hoc. The retrospective Holm adjustment over the six
  prospectively named contrasts gives 0.140625 against always-adapt and 0.09375 against always-freeze and
  does not support a confirmatory claim.
- EATA's ordinary family interval versus always-adapt [-0.00135, 0.00438] crosses zero, and SAR loses to
  always-adapt. No candidate earns a cluster-robust or confirmatory routing-utility claim. The analysis is conditional
  on one archived checkpoint and is not independent-checkpoint, prospective, natural-shift, or
  official POEM/AETTA evidence.
- The earlier KGA-policy cluster artifact remains separately marked historical. It does not replace
  the current-policy sensitivity or regain release authority.

## Multiplicity Status — One Release Interpretation

The six candidate-by-fixed-policy contrasts for CIFAR-10-C were stated prospectively in
`research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml`. The later exact-rank policy replay,
six-corruption-family bootstrap sensitivity, sign-flip p-values, and Holm adjustment are
retrospective and non-confirmatory. Its ordinary confidence intervals are unadjusted. Under Holm
over the six contrasts, Tent is `0.140625` against always-adapt and `0.09375` against always-freeze, and no candidate rejects both
comparisons at 0.05. The within-Tent
two-contrast Holm value `0.03125` is post hoc and cannot replace the six-comparison result. The
repository-wide search census, CCT-20's separate two-comparison family, and historical POEM/AETTA
p-values are different families and must not be pooled or used to promote the CIFAR result.

## iWildCam Historical Reconciliation

- The archived H-v2 scorer used sklearn macro-F1, which does not match the official WILDS
  label-present macro-F1 contract. Its performance values, radius, and action counts are therefore
  audit-only and are not release-level numerical evidence.
- Replays of the archived records explain why the historical two-comparator flag was unstable, but a
  replay cannot repair the metric contract or seal the evaluated population retroactively.
- The release row remains withheld until a pinned official-metric rerun is completed against a
  population manifest sealed before scoring.

## Calibration Scope

- Clean calibration/test splits use the exact finite-sample order statistic, returning an infinite
  radius when the requested rank is infeasible.
- Controlled stress grids use deterministic three-way cell-outcome-disjoint cross-fit: estimator-fit,
  residual-calibration, and scored cells are disjoint, with exact-rank residual calibration.
  This removes direct and indirect score-cell outcome leakage but is not an independent validation design.
- Natural rows retain their archived development/calibration/test scope. A numerical replay does not
  create a transfer theorem that the original design did not support.
- Canonical benefit-regressor replays are generated under the release lock
  (`numpy==2.4.4`, `scikit-learn==1.8.0`). The reconciler fails closed under another numerical
  runtime because tree-ensemble version changes can alter near-boundary actions.

## Baseline Scope

POEM- and AETTA-style comparisons are protocol-matched ports, not official implementations. The
archived comparison used an earlier KGA policy and is not synchronized with the current exact-rank
actions. Its ordinary paired confidence intervals are unadjusted; Holm applies only to the archived
paired p-values. The fail-closed provenance audit confirms that complete successful native exports
are absent. These values remain historical diagnostics pending a current-policy recomputation.

## Unresolved Evidence Gaps

- No clean held-out natural single-dataset result has confidence-supported lower regret than both fixed policies.
- The completed CCT-20 camera-location result is safe-utility-only and fails the locked
  action-exposure requirement; the fresh physical-camera study remains pending.
- The promoted PACS panel lacks archived per-cell benefit estimates and residuals for full gate
  replay. A one-domain, ten-cell MPS smoke run validates the new replay schema only.
- Natural tracks need more independently trained checkpoints and prospectively locked transfer
  designs before a broad stability claim.
- So2Sat v1 declares a target action per city while its dormant target implementation constructs
  city-by-checkpoint actions; that mismatch did not affect the stopped development run. Prospective
  v2 now resolves the action unit and has a source-complete runner, but remains NO-GO until a v2
  pre-calibration seal and target authorization exist. Its 19-city calibration panel and all target
  content remain unopened.
