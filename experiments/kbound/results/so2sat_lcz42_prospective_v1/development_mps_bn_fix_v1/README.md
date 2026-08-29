# So2Sat locked development result

Status: `NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL`.

This directory is the portable repository archive of the completed So2Sat-LCZ42
gate-fit run from runner code identity
`71d8ed2ad5ca03e139b0d33278b3e855a0b5678ed827a8d0c6575a202a2d3c38`.
The run used nine locked development cities, five independent source
checkpoints, and both preregistered adapters. Each candidate bundle contains 45
city-by-checkpoint cells. The adjacent receipts verify the exact bytes and
canonical JSON documents.

## Preregistered result

| Candidate | Helpful cities | Harmful cities | Oracle gap | LOCO sign accuracy | LOCO gain vs best fixed | Feasible |
|---|---:|---:|---:|---:|---:|---:|
| Tent | 4 | 4 | +0.8848 pp | 51.11% | -0.1544 pp | No |
| SAR | 0 | 6 | +0.5640 pp | 64.44% | -0.0842 pp | No |

| Candidate | Always freeze | Always adapt | LOCO routed | Cell oracle | Checks passed |
|---|---:|---:|---:|---:|---:|
| Tent | 54.8101% | 55.1271% | 54.9727% | 56.0119% | 7/9 |
| SAR | 54.8101% | 53.9307% | 54.7259% | 55.3741% | 6/9 |

Tent established real natural-shift heterogeneity: adaptation helped in
Hong Kong, Kyoto, Los Angeles, and Rio de Janeiro, and harmed in Cairo, Milan,
Paris, and Zurich under the locked city-mean threshold. However, its
leave-one-city-out ridge gate did not generalize: sign accuracy was below the
55% requirement and routed accuracy was below always adapting. SAR was mostly
harmful, offered no qualifying helpful city, selected ADAPT in only 5 of 45
cells, and also underperformed its best fixed policy.

The oracle gaps show that policy opportunity exists, but the locked label-free
features and ridge gate did not recover it out of city. This is an honest
negative routing result, not a target confirmation and not evidence of a
successful natural-shift router.

These are development gate-fit results on nine held-out training cities using
locked west/east spatial partitions. They are not results on the unopened
10-city culture-10 validation/testing target.

## Integrity boundary

- Both candidate bundles and the deterministic selection artifact validate.
- Frozen accuracies match exactly across Tent and SAR for every corresponding
  city-checkpoint cell.
- All stored cell benefits equal adapted accuracy minus frozen accuracy.
- Target inputs are empty and target pixel/label read counts are zero.
- Gate calibration did not start.
- The official `validation.h5` and `testing.h5` image/label containers were not
  extracted or deserialized; only their compressed `.gz` archives exist.
- The protocol forbids a pre-calibration seal or any target stage after this
  no-candidate result.

## Artifact hashes

- `so2sat_tent_adam_bn_affine_probe_transfer_v1.gate_fit.json`:
  `c91e6989f610408f0be223062620404275a1d040bcef5228743bef2366e6df2e`
- `so2sat_sar_sam_bn_affine_probe_transfer_v1.gate_fit.json`:
  `6e7ce26bf664f5f259b81885b92f554dbd70d45198ae49c078dad3d7a0320674`
- `so2sat_candidate_selection.json`:
  `8db11a797d98c5f104736a5ed982a422982f9f75fc8a7d1c6e13f07a826c0b79`

The pre-result MPS compatibility correction and prospective boundary are
recorded in
`research_lock/KBOUND_SO2SAT_DEVELOPMENT_RUNTIME_AMENDMENT_v1.json`.

## Future-use blockers

The v1 protocol declares the target action unit as one city, while the current
target implementation creates one action per city-checkpoint cell. This
literal contract mismatch did not affect the present negative stop because no
target action was created. The v1 target runner must not be used for a future
target execution until a versioned protocol resolves the mismatch by either
aggregating one action per city or declaring a city-checkpoint action unit.

All 90 gate-fit outcomes are now open. Any new adapter, feature set, ridge
penalty, or threshold chosen with knowledge of these outcomes is exploratory
and must not reuse this panel as fresh confirmatory evidence. A future
confirmatory claim requires a new versioned design and untouched evidence.
