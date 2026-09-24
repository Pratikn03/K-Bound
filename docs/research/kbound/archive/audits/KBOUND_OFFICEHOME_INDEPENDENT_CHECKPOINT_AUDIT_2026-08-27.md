# Office-Home independent-checkpoint candidate opportunity audit (2026-08-27)

## Status and claim level

This is an **invalidated post-hoc candidate opportunity audit**, not a KGA result and not an untouched
prospective confirmation. The Office-Home target split had already been opened in earlier work. The
audit varies independently trained source checkpoints while holding the target stream seed and
evaluation recipe fixed.

## Fixed design

- Dataset: Office-Home.
- Source domain: Real World.
- Target domains: Art, Clipart, and Product test splits.
- Condition: IID composition, tiny batch regime, stream seed 0.
- Independent source checkpoints: five ResNet-50 models, training seeds 0--4.
- Candidate policies: aggressive online Tent, EATA, and SAR.
- Route: invalidated exploratory multicandidate agreement heuristic with the frozen model as anchor.
  Its identifying relation is binary-only and unsupported on 65-class Office-Home. The archived
  route outputs have no safety, routing, or certificate interpretation.
- Evaluation unit for uncertainty: independently trained checkpoint; domain results are averaged
  within checkpoint before computing the paired 95% t-interval.

Checkpoint identity passed: the five models have distinct SHA-256 hashes recorded in the result
files and in the data-quality audit. No single combined artifact seal is claimed for this directory,
because all five result files fail strict JSON.

## Result

The archived implementation printed `ABSTAIN` in all 15 cells, with a residual statistic of
0.7764--1.1765 against a threshold of 0.52. These are outputs of an invalid route and are not scored
as policy decisions. Only the frozen and candidate checkpoint accuracies are retained.

Checkpoint-level accuracy summaries:

| Policy | Mean | 95% CI |
|---|---:|---:|
| Frozen | 0.5711 | [0.5468, 0.5953] |
| Tent, fixed | 0.5711 | [0.5586, 0.5836] |
| EATA, fixed | 0.5891 | [0.5653, 0.6129] |
| SAR, fixed | 0.5914 | [0.5685, 0.6143] |
| Per-cell oracle (freeze plus three adapters) | 0.5941 | [0.5718, 0.6163] |

SAR had the highest mean among the three evaluated fixed policies. Its checkpoint-paired advantage
over freeze was 0.0203 accuracy, with a 95% CI of [0.0123, 0.0283]. Because SAR is selected
after comparing the three audit means, this is descriptive rather than a predeclared confirmatory
contrast. The best adapted candidate in each cell improved over freeze in 13 cells, tied in one,
and harmed in one. No selection performance is attributed to the invalid route.

In the regret vocabulary used by manuscript Table 4, the valid descriptive candidate audit gives:

| Policy | Mean regret to per-cell oracle | Checkpoint-level 95% CI |
|---|---:|---:|
| Freeze | 0.0230 | [0.0182, 0.0278] |
| Tent, fixed | 0.0230 | [0.0086, 0.0373] |
| EATA, fixed | 0.0049 | [0.0016, 0.0082] |
| SAR, fixed | 0.0027 | [-0.0024, 0.0077] |

## Interpretation

This audit does **not** strengthen a natural-shift routing or safety claim. The route is invalid for
seven independent reasons:

1. the identifying agreement relation is binary-only, but Office-Home has 65 classes;
2. the archived spectral estimate is sign-indeterminate and unbounded;
3. 13 of 15 anchor estimates are negative;
4. 9 of 60 stored $\widehat b$ values, across 5 cells, lie outside $[0,1]$;
5. EATA and SAR predictions are exactly identical in 10 of 15 cells, weakening effective candidate
   diversity;
6. each checkpoint provides only three LOO cells, whereas a 90% exact-rank single-candidate radius
   requires at least 10 total cells; and
7. all five result files fail strict JSON and contain 57 literal `Infinity` values in total.

The defensible evidence is narrower: five distinct checkpoints, frozen and candidate accuracies,
SAR-minus-freeze accuracy of 0.0203 with 95% CI [0.0123, 0.0283], and oracle-minus-SAR accuracy of
0.0027 with 95% CI [-0.0024, 0.0077]. A stronger confirmation requires a theory-valid route, a
genuinely unopened target environment, enough calibration units for the declared coverage level,
and both helpful and harmful adaptation opportunities under a locked protocol.

## Artifacts

The non-promotable result files are archived under:

`archive/legacy_kbound/non_promotable_results/officehome_five_checkpoint_opportunity_2026-08-27/`

Audit checks:

- five distinct checkpoint identities;
- 15 target conditions and 45 candidate records;
- 13 negative anchors;
- 9/60 out-of-range $\widehat b$ values;
- EATA=SAR predictions in 10/15 cells;
- only 3 cells per checkpoint versus 10 required for the declared exact-rank level; and
- strict-JSON failure in 5/5 files, with 57 literal `Infinity` values.
