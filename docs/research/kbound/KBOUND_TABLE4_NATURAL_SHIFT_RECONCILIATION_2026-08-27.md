# Table 4 and natural-shift reconciliation (2026-08-27)

> **SUPERSEDED POINT-IN-TIME RECONCILIATION.** This file predates the final Phase-1 consistency
> pass and retains an obsolete canonical-JSON digest and historical interpretation details. Do not
> use it as the current numerical or provenance authority. Use
> `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`,
> `KBOUND_SHORT_RESULT_AUDIT.md`, and `KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md` instead.

## Outcome

Table 4 is rendered from
`experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`, with the iWildCam
numerical row explicitly withheld at release time because its archived metric contract is invalid.
Its source manifest
SHA-256 is `391983f4fa09d65c256ab8c620fea772866b652c219dad0c36574ce38c740482`,
and the canonical result JSON SHA-256 is
`95d5d80d8eb8a91e0b7936ffb952d9c4c30e9c3bbbe51ddac1d40e52ce38a5ab`.

The prior repository-wide beats-both scan examined 25 result rows. The current data-quality audit
checked 117 aggregate score nodes across 106 source artifacts and 12,619 rows and found no internal
arithmetic or source-hash inconsistency. That result does **not** validate metric semantics,
estimator identification, calibration design, population identity, or prospective status.

## Inconsistencies found and resolved

1. **Wrong metric label in the new Office-Home audit.** The saved records use accuracy, but the
   first manuscript update called the values macro-F1. The label is now corrected to accuracy.
2. **Invalid iWildCam metric contract.** The archived scorer used sklearn macro-F1, which includes
   prediction-only classes, instead of the official WILDS label-present macro-F1. The numerical row
   and action row are now withheld. A non-promoted diagnostic official-metric recomputation gives
   KGA/adapt/freeze regret `0.005511/0.074502/0.005511`, actions `0/4/68`, and coverage `0.0556`.
   Across 864 archived candidate records, mean benefit changes by `+0.029379` and 60 signs flip.
3. **iWildCam population mismatch.** The archived selected-location population contains 14,453
   images, while 12,530 are currently present (`-13.31%`); the old records do not make overlap
   auditable. Promotion requires a pinned rerun with sample IDs and a population-manifest hash.
4. **Ambiguous Office-Home replication name.** Table 4's 54-cell result varies target-stream seeds
   and does not record independent checkpoint identities. The new 15-cell audit uses five distinct
   checkpoint hashes. The paper now names these separately and provides a two-row reconciliation
   table.
5. **Invalidated Office-Home focused route.** Five checkpoint identities are distinct, but the
   exploratory multicandidate route is not KGA evidence: its relation is binary-only on a 65-class
   task, the old spectral estimate is sign-indeterminate and unbounded, 13/15 anchors are negative,
   9/60 `b_hat` values are outside `[0,1]`, EATA=SAR in 10/15 cells, three LOO cells cannot support
   the requested 90% exact-rank radius (minimum 10), and 5/5 files fail strict JSON with 57 literal
   `Infinity` values. Only checkpoint candidate accuracies remain as a post-hoc opportunity audit.
6. **PACS denominator ambiguity.** Table 4's `n=12` is the number of equal-weight domain-seed
   inference units. Its false-adapt and coverage rates summarize 216 repeated decision evaluations,
   corresponding to 108 paired settings. The Table 4 caption now states all three quantities.
7. **New audit versus canonical snapshot.** The five-checkpoint audit postdates the locked Table 4
   manifest. It is therefore reported separately rather than silently inserted into the older
   canonical snapshot.

## Natural-shift interpretation after reconciliation

- **Office-Home primary (35 cells):** KGA ties freeze; no adapt actions; descriptive because the
  transfer-stability premise was not predeclared.
- **Office-Home 54-cell stream-seed replay:** a very small point edge with one adapt action, but the
  declared seed interval includes zero and checkpoint identities are absent.
- **Office-Home five-checkpoint candidate audit (15 cells):** route invalid; retain only freeze
  regret `0.0230`, descriptive fixed-SAR regret `0.0027`, SAR-minus-freeze accuracy `0.0203`
  (`[0.0123,0.0283]`), and oracle-minus-SAR `0.0027` (`[-0.0024,0.0077]`).
- **iWildCam:** canonical numerical and action rows withheld pending a pinned official-metric,
  population-sealed rerun. The official-metric diagnostic ties freeze but is not promoted.
- **ImageNet-R (480 cells):** KGA regret `0.0150` is worse than always-adapt `0.0064`; architecture
  diagnostic, not one deployable policy.
- **PACS:** KGA regret `0.0431` is worse than always-adapt `0.0176`; gate replay is unavailable from
  the archived per-cell payload.
- **Camelyon17 OOD:** this is an archived, already-opened diagnostic. All 18 cells are helpful, so
  KGA behaves like always-adapt and does not test harmful-update rejection.
- **RxRx1:** all-freeze no-harm behavior across the promoted row; it does not test adapt-side utility.
- **CIFAR-10.1:** negative cross-seed diagnostic tying freeze in the canonical replay.

These rows do not establish a general natural-shift win. The strongest defensible statement is that
the natural panel is mostly conservative, one-sided, or negative, while the controlled CIFAR-10-C
Tent result carries the strongest routing evidence.

## Historical artifacts

The repository intentionally retains older and superseded result files. The historical iWildCam
H-v2 route and its later cross-fitted replay both use the wrong archived metric contract, so neither
is a release-level numerical estimate. The official-metric recomputation is diagnostic only because
its numerical runtime differs from the pinned reconciliation and its current population differs from
the archived population. The manuscript promotes none of these values.
