# So2Sat natural-shift v2 result

The locked run completed all 19 calibration cities across five independently trained source checkpoints (95 cells) and returned **FAILED_GATE_AUTHORIZATION_SCREEN_NO_TARGET_ACCESS**. It produced 2 direct ADAPT, 0 direct FREEZE, and 93 ABSTAIN decisions; abstention realizes the frozen source model, so the realized actions were 2 ADAPT and 93 FREEZE. The two ADAPT cells belonged to one city, São Paulo. The pre-existing screen required at least seven cities with direct ADAPT and seven with direct FREEZE: both requirements failed. Target pixels and labels remained unopened, and the earlier v1 negative result is preserved.

The current run was locked before calibration access. Historical reserved-data nonaccess is supported by inspected records, not externally proven. Existing independently trained checkpoints were verified and reused; the run does not represent five newly trained models or 95 independent environments.

## Frozen calibration results

| Quantity | Result |
| --- | ---: |
| Distinct calibration cities | 19 |
| Independently trained checkpoints | 5 |
| City–checkpoint cells | 95 |
| Direct ADAPT / FREEZE / ABSTAIN | 2 / 0 / 93 |
| Cities with direct ADAPT / FREEZE | 1 / 0 |
| City-max residual radius | 14.223862 percentage points |
| Radius rank | 18 of 19 city maxima |
| Calibration city / cell inclusion | 18/19 and 94/95 |
| Gate mean accuracy gain over freeze | +0.098875 percentage points |
| Always-adapt mean accuracy gain | +0.193888 percentage points |
| Checkpoint-only baseline mean gain | +0.126353 percentage points |

Utilities and interval inclusion above are descriptive results on the same calibration set that determined the radius. They are not held-out target performance. The nominal 90% rank guarantee concerns all five checkpoints jointly for one exchangeable new city under the requisite assumptions; it is not simultaneous coverage of every target city or coverage conditional on passing the screen. Exchangeability with the culture-shift target cities has not been established. There is no conditional-risk certification. Even a hypothetical zero failures among 19 independent Bernoulli units with one common fixed risk would give a one-sided 95% upper risk bound of 0.145869; the actual direct ADAPT exposure was only one calibration city.

## Posthoc failure diagnosis

This section was computed after the failed screen using only its already-opened calibration JSON outputs. No refit, tuning, threshold change, raw pixel access, or target access was performed.

The radius-determining cell was **São Paulo, checkpoint 0**: prediction +14.421490 percentage points, observed benefit +0.197628 points, absolute residual **14.223862 points**. This is the 18th ordered city maximum. Berlin checkpoint 0 had the largest residual, **17.145381 points**, but did not determine the selected rank. Its prediction was +7.235370 points while its observed benefit was −9.910011 points; it abstained and was the sole cell outside its interval.

Candidate effects had both signs: **51 helpful, 43 harmful, and 1 exactly tied**. Their observed benefits ranged from −9.910011 to +10.832396 percentage points. Predictions comprised 50 positive and 45 negative values, ranging from −2.606725 to +14.421490 points. Among the 94 nonzero observed effects, 51 predictions had the correct sign and 43 had the wrong sign; these are descriptive cell counts with shared city dependence.

The 14.223862-point radius exceeded the magnitude of every negative prediction, so no interval had a negative upper endpoint and no direct FREEZE decision was possible. Only two positive predictions exceeded that radius, both in São Paulo:

| Checkpoint | Predicted benefit (points) | Observed benefit (points) | Interval lower endpoint (points) |
| --- | ---: | ---: | ---: |
| 0 | +14.421490 | +0.197628 | +0.197628 |
| 2 | +14.246258 | +9.195536 | +0.022396 |

Both accepted effects were positive, but both are calibration observations in one city. In particular, checkpoint 0 itself determined the radius; its lower endpoint equaling the observed benefit is a consequence of that construction, not independent confirmation. The 93 abstained cells included 49 helpful, 43 harmful, and 1 tied effect.

The stop therefore reflects prediction errors large enough to require broad city-cluster intervals, combined with inadequate exposure in each direct-action direction. Helpful and harmful adaptation candidates were both present. This diagnosis supports the recorded stop and does not justify removing cities, narrowing the radius, changing the controller, or accessing the targets.

## Measured execution costs and provenance

Preparation took 32.182 seconds; calibration took 2,193.110 seconds (36.55 minutes), for 2,225.292 seconds (37.09 minutes) combined. The 95 complete cells summed to 2,017.661 seconds; 175.450 seconds of calibration time were outside cell execution. Cell time was median 15.148, mean 21.239, p90 60.873, p95 63.340 seconds, with range 1.204–65.650 seconds. These measurements include source reset, inference, adaptation and feature extraction; they are not gate-only latency or stage decomposition.

Process maximum RSS was 730,546,176 bytes. The maximum observed postcell MPS driver allocation was 1,220,182,016 bytes; these samples do not establish a continuous GPU peak. Postcell current allocation was zero after model release and does not imply zero operating memory. The five historical source-training runs are not included in these execution costs.

Receipt-bound evidence is under `output/next_phase/natural_v2`: `calibration_summary.json`, `gate_screen_v2.json`, `cost_and_scope_summary.json`, and `posthoc_failure_diagnostic.json`. The scientific execution used the archived `natural_runner_locked.py`; `postrun_source_correction.json` records subsequent secure-read and prior-access guard corrections. Those corrections did not overwrite the executed snapshot, original authorities, or failed result. The diagnosis JSON binds exact input hashes and preserves the selected cell records.
