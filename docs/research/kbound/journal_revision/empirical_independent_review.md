# Independent internal review of Task 2

Verdict: **PASS for the implemented retrospective synthesis. No actionable
correctness or scope finding identified.** This is an internal implementation
review, not external reviewer endorsement, prospective confirmation, a proof of
calibration transfer, or authentication of missing original execution records.
No implementation, frozen authority, or manuscript file was edited by this
review. Only this report was written.

## Review scope

Read the complete empirical builder and its tests, task report, JSON metric
contracts, readable synthesis and machine-readable outputs; inspected the frozen
calibration decision/outcome summaries, selected corrected ImageNet authority
selection, principal CIFAR compact records/current-policy authority and the CCT
public bundle's manifest and archived score object. No model or image inference
was executed. Protected So2Sat targets, raw images and checkpoint tensors were
not accessed.

The following checks ran in the pinned research interpreter:

```text
python -m pytest -q tests/test_kbound_empirical_synthesis.py tests/test_calibration_value.py
python docs/research/kbound/scripts/build_empirical_synthesis.py --check
```

Observed: **33 tests pass** and **all eight generated outputs reproduce
byte-for-byte**. Independently recomputed every input-manifest SHA-256 and
length: all **70 consulted inputs match**. The checker reads frozen authorities;
it does not train or recreate a historical run. Its output path policy rejects
non-empirical filenames, path traversal, output symlinks and changed existing
outputs without explicit replacement.

## Independent arithmetic checks

The checks below were computed directly from frozen records with separate
inline calculations; no builder metric, selection or aggregation helper was
called.

| Principal CIFAR policy | Frozen accuracy (%) | Always-adapt accuracy (%) | KGA accuracy (%) | KGA regret (pp) |
|---|---:|---:|---:|---:|
| EATA | 67.0931482205 | 79.8992129029 | 80.0665508901 | 0.1603009193 |
| SAR | 67.0931482205 | 81.1066666908 | 80.9740972574 | 0.1661111083 |
| Tent | 67.0931482205 | 78.6892592465 | 79.3075694696 | 0.1792592362 |

All use the 2,160 recorded cells per candidate, equal-cell weighting and
ABSTAIN/FREEZE serving the frozen predictor. SAR remains unfavorable versus
always-adapt. These are reconstructed means, not pooled image accuracy or new
raw-data execution.

For each CIFAR candidate, independently recomputed point, tuned margin,
constant-cell 90%, scaled-cell 90% and constant-group 90% weighted losses, cell
inclusion and simultaneous semantic-group inclusion. Every value matched.
The group-max-calibrated CIFAR arms contain 2,160 finite cells and 216 complete
finite semantic groups each. Simultaneous inclusion is 56.944444%, 45.833333%,
and 61.574074% for EATA/SAR/Tent. Constant-cell simultaneous inclusion is
42.129630%, 32.407407%, and 42.129630%. Thus the distinction is substantive,
not a relabeling of cell coverage. ImageNet and both mixed-checkpoint group-max
90% arms have zero finite intervals/groups and correctly retain null inclusion,
with unbounded intervals counted separately.

| Tent comparison, scaled minus comparator | Mean loss difference (pp) | Descriptive bootstrap range (pp) | Omitting JPEG, exploratory (pp) |
|---|---:|---:|---:|
| Point | -0.1438427441 | [-0.4285649762, 0.1387550934] | -0.0048890445 |
| Tuned margin, weight 5 | -0.0834953647 | [-0.4169907007, 0.1882823719] | +0.0908888744 |
| Point ranking at matched exposure | -0.0181481546 | [-0.0425000158, -0.0010648177] | -0.0065277782 |

Reconstructed the same fixed-seed 1,000 resamples of six paired corruption
means and reproduced the intervals independently. Matched selection sorts the
shared label-free prediction within each fold, breaks ties by cell ID, and
accepts exactly the gate's number of cells; no outcome is used to rank them.
JPEG stays in every primary estimate. The seemingly favorable matched-exposure
bootstrap interval is explicitly descriptive and does not become a significance
or general-superiority statement.

For CCT, selected the score bundle member by both content role and the release
manifest's original score hash (content role alone is not unique in this
bundle). Independent reconstruction gives 45 cells, 44 FREEZE, one ABSTAIN and
zero ADAPT; accuracies are 52.2936099072% frozen/KGA and 34.1033832827%
always-adapt. KGA regret is 0.0363730839 pp, versus 18.2265997083 pp for
always-adapt. One helpful cell exists and is missed. Recomputed all 135
cell/policy secondary macro-F1 values from TP/FP/FN, with zero-denominator F1=0;
the builder preserves them as secondary cell-level descriptions and does not
invent an aggregate endpoint.

## Statistical and specification assessment

- Benefit, harm-weight-5 loss, unweighted oracle regret and deployed accuracy
  are distinct, correctly scaled fraction/percentage-point quantities. Weight
  1/20 sensitivity is retrospective; the primary weight remains five.
- Negative benefit and nonpositive benefit are separately counted. A tie can
  create a directional event with zero harm magnitude. Explicit FREEZE errors
  are separate from ABSTAIN and from helpful opportunities forgone.
- Acceptance-conditional errors are null at zero ADAPT exposure, with reasons;
  unconditional zero events do not imply conditional safety. Semantic-group
  errors use exposed-group denominators, and fold identity prevents accidental
  merging of groups across evaluation designs.
- Interval inclusion is explicitly conditional on finite intervals; simultaneous
  group inclusion requires every cell in the group to have finite endpoints
  and to be included. This is not unconditional nominal conformal coverage.
- Paired comparisons reject identity, outcome, group and cluster mismatches.
  They report equal-cell and equal-cluster effects separately. Cluster
  bootstrap and leave-one-cluster-out calculations retain the shared historical
  dependence caveat. Related corruption types, stream seeds and cross-classified
  CCT checkpoints/locations are never called independent deployments.
- Coverage failures, zero-exposure arms, unfavorable ImageNet/checkpoint designs,
  SAR's unfavorable accuracy comparison and Tent's sensitivity to JPEG remain.
  The output contains all 160 arms across eight cohort/design combinations and
  12 principal policy rows. The same 108 mixed-checkpoint records appearing in
  two designs are explicitly disclosed rather than counted as new deployments.
- Published CCT cross-classified/location inference is carried with its original
  scope; the builder does not silently replace it with a 45-cell iid bootstrap.
  Retention and stronger routing success remain separate.
- Hash and arithmetic checks establish current byte lineage and reproducibility,
  not missing historical producer provenance, independent deployment validity,
  prospective outcome blindness, transport assumptions or publication readiness.

No change is requested by this review. Final release verification should bind
these reviewed sources and outputs to the final release commit and include the
required frozen evidence bundle; this review does not replace that gate.

## Reviewed byte bindings

Review recorded UTC: 2026-09-23T03:43:05.169218+00:00

- `docs/research/kbound/scripts/build_empirical_synthesis.py`: `e56c7fb324d5a1af86a4cfe568d5cd6f7df8ed6c263774b78ef570fffc5630ce`
- `tests/test_kbound_empirical_synthesis.py`: `8012078a42b545fd9bf59b2d652ecb209dacd0b9fbc9e50ec74cd1cb278f0674`
- `docs/research/kbound/journal_revision/empirical_synthesis_v1.json`: `5d7ac58bff599e3ab0bc607e5112a167c14c94c2dd1c46ba776ebdaf321507f8`
- `docs/research/kbound/journal_revision/empirical_input_manifest_v1.json`: `e675f9ff8b2955bdfebc15e3da8b612f2bf5759cb025ae0d572ef3e92694966c`
- `docs/research/kbound/journal_revision/empirical_synthesis_receipt_v1.json`: `87e8b40a4607b9a8c3592930fa732e76f18b6dc768144ba603b7f9d24c7310dd`
