# Prospective So2Sat LCZ42 v4.2 track

This directory implements the target-label-blind preparation, source-training,
development-selection, live target execution, and separate offline scoring
boundary for a new So2Sat Culture-10 confirmation. The original immutable v1
protocol was written before a scorer existed. The later create-only
`target_boundary_amendment_v1_1.json` supersedes exactly
`/target_label_firewall/scoring_implementation_in_this_package` from `false`
to `true`; it preserves every other v1 field and binds the exact v1 file and
canonical-document hashes. The final execution seal binds both documents.

The structural protocol fixes the natural environment unit as city. City roles
are based on the complete label-free training metadata because the city sizes
are highly uneven. A gate-eligible city must have at least 256 rows and at least
two distinct 6.4-km block eastings, so its west/east split is nonempty. All nine
cities failing either condition become `source_fit_ineligible`; this includes
Orangi Town, whose 1,140 rows occupy two blocks but only one distinct block
easting. The five largest
eligible cities become `source_fit_core`. The remaining 28 eligible cities are
ranked by a fixed SHA-256 salt into 9 gate-fit and 19 residual-calibration
cities. Thus the source fit contains 14 cities in total. Source-monitor samples retain a fixed
hash split over whole 6.4-km blocks. Within every gate city, the upper median of
the sorted distinct block eastings is fixed as a geographic threshold: western
blocks are probe and eastern blocks are evaluation. No block crosses roles.

The ten official Culture-10 cities remain the target: their western
`validation` halves (24,119 rows) provide unscored label-free probe pixels and
their eastern `testing` halves (24,188 rows) provide the eventual evaluation
population.

## Firewall

`metadata_manifest.py` opens only the v4.2 geographic companions and requires
their HDF5 keys to be exactly `city`, `epsg`, and `tfw`. It refuses any extra
dataset. `label_firewall.py` provides a target loader with no arbitrary HDF5
key API; it can index only `sen1` or `sen2`, selected by the sealed modality.
The target containers are first bound by opaque whole-file SHA-256 hashes.
Hashing raw bytes does not open or deserialize any HDF5 dataset.

`target_runner.py` is the live process. It can request only the fixed `sen2`
pixel key through `LabelFreeTargetLoader`; it cannot name or receive a label
array. For each city, it computes and writes all five immutable probe-derived
action artifacts before the loader is allowed to open that city's testing
pixels. `ABSTAIN` is realized as `FREEZE`. The runner then seals all 50
city-by-checkpoint cells, replayable compressed frozen/adapted logits,
predictions, actions, and receipts before writing the complete bundle last.

`target_scorer.py` is a separate module and process; it does not import the
live runner or model-inference modules. It refuses incomplete or tampered
bundles and reserves one reveal keyed by the execution seal in a separately
supplied, receipt-bound reveal registry before it opens `testing.h5/label`
exactly once. It never opens `validation.h5` or its labels. The registry
identity is portable and publishes no absolute local path. This enforces one
reveal within that selected registry. Preventing a second reveal from a copied
registry requires external append-only storage, access control, or an
independent outcome custodian.

These controls prove nonaccess within the sealed implementation run. They do
not independently prove that a person or another process never opened the
labels historically; that stronger claim requires external access logs,
permissions, or a timestamped preregistration record.

## Preparation

Install `h5py` in the research environment. The label-free population manifest
uses the three geographic companions. Source training and both development
phases additionally use `training.h5`; neither development command accepts a
target data path. Keep the official names unchanged:

- `training.h5`, `validation.h5`, `testing.h5`
- `training_geo.h5`, `validation_geo.h5`, `testing_geo.h5`

Then build the metadata-only manifest:

```bash
python -m experiments.kbound.so2sat.metadata_manifest \
  --data-root /absolute/path/to/so2sat-v4.2 \
  --output /absolute/path/to/seals/so2sat_population_manifest_v1.json
```

The output and its receipt are create-only. The immutable v1 document is a
structural protocol whose own status remains
`STRUCTURAL_PROTOCOL_SEALED_EXECUTION_CONFIG_PENDING`; it requires the separate
execution/configuration seal before gate calibration or target-pixel access,
not before source training. Source training is therefore supported by its
fixed production command, exact code/config/data identities, per-seed receipts,
and the post-run source acceptance below. Unless an independently timestamped
pre-training configuration artifact is retained, do not describe the source
training itself as preregistered. The actual pre-calibration seal—not a later
reconstruction—must exist before any gate-calibration access.

The structural population manifest remains v1 because its city assignment and
population facts did not change. The target-boundary v1.1 amendment documents
the later separate scorer and the two-seal chronology without rewriting the
immutable v1 protocol. The pre-calibration seal binds the structural v1
protocol, the amendment, the selected gate-fit evidence, fixed gate algorithm,
source artifacts, environments, code, opaque target-container hashes, and
reveal-registry identity before gate-calibration outcomes or target pixels are
opened. The later target-execution seal must extend that exact artifact and add
the gate-calibration bundle, gate, and authorization. A final seal can never
stand in for the earlier pre-calibration artifact or its timestamp.

## Source and development phases

`train_source.py` trains five independently initialized 10-band ResNet-18
models. Only `source_train` labels update weights; only `source_monitor` labels
select the checkpoint. The source-only normalizer and every checkpoint receive
content hashes and create-only receipts.

The checkpoint-selection metric is macro recall over the 15 classes actually
supported in `source_monitor`; class IDs 0 and 6 are absent there. It must not
be described as 17-class macro recall. Development and target endpoints remain
top-1 accuracy over their complete rows. The legacy architecture-spec phrase
`independent_torchvision_kaiming_per_model_seed` is also only a shorthand:
the residual body receives torchvision ResNet initialization, while the
replacement `conv1` and `fc` layers receive their PyTorch module-default reset
initialization (`conv1` is Kaiming-uniform). The exact initial tensor hashes,
which are distinct across all five model seeds, are authoritative. The
receipt-bound source post-run acceptance artifact must record these
clarifications and the post-run `training.h5` SHA-256; the pre-calibration seal
binds its receipt. The sealed source-preflight artifact and the source-training
v1 receipt schema used by the already-started run do not expose h5py as a named
runtime field. Their artifact, code, scientific-identity, and per-seed runtime
hashes remain authoritative. The post-run acceptance records and verifies its
own exact h5py version, while explicitly marking that later observation as
non-retroactive:
it is not evidence of the h5py version used during preflight or source
training.

`development.py` has two separate commands. `select` evaluates both locked
adapter candidates on the 9 gate-fit cities and writes a selection artifact.
Immediately after selection, and before any gate-calibration access, create the
pre-calibration seal. It freezes the gate algorithm and the ridge fitted only
from the selected gate-fit bundle; raw hashing of `validation.h5` and
`testing.h5` does not deserialize HDF5. `calibrate` then verifies that seal and
may evaluate only the selected adapter on the 19 gate-calibration cities. If
neither candidate passes, the process stops before the pre-calibration seal or
gate-calibration access.

```bash
python -m experiments.kbound.so2sat.source_preflight \
  --training-h5 /absolute/path/to/training.h5 \
  --population-manifest /absolute/path/to/so2sat_population_manifest_v1.json \
  --output /absolute/path/to/source_checkpoints/so2sat_source_preflight.json

python -m experiments.kbound.so2sat.train_source \
  --population-manifest /absolute/path/to/so2sat_population_manifest_v1.json \
  --training-geo /absolute/path/to/training_geo.h5 \
  --training-data /absolute/path/to/training.h5 \
  --output-dir /absolute/path/to/source_checkpoints \
  --device mps \
  --workers 0

python -m experiments.kbound.so2sat.source_acceptance \
  --population-manifest /absolute/path/to/so2sat_population_manifest_v1.json \
  --training-data /absolute/path/to/training.h5 \
  --source-preflight /absolute/path/to/source_checkpoints/so2sat_source_preflight.json \
  --checkpoint-dir /absolute/path/to/source_checkpoints \
  --output /absolute/path/to/source_checkpoints/so2sat_source_postrun_acceptance.json

python -m experiments.kbound.so2sat.development select \
  --population-manifest /absolute/path/to/so2sat_population_manifest_v1.json \
  --source-postrun-acceptance /absolute/path/to/source_checkpoints/so2sat_source_postrun_acceptance.json \
  --source-preflight /absolute/path/to/source_checkpoints/so2sat_source_preflight.json \
  --training-geo /absolute/path/to/training_geo.h5 \
  --training-data /absolute/path/to/training.h5 \
  --checkpoint-dir /absolute/path/to/source_checkpoints \
  --output-dir /absolute/path/to/development_results \
  --device auto

SELECTED_CANDIDATE_ID="$(python -c \
  'import json; print(json.load(open("/absolute/path/to/development_results/so2sat_candidate_selection.json"))["selected_candidate_id"])')"

python -m experiments.kbound.so2sat.precalibration_seal \
  --population-manifest /absolute/path/to/so2sat_population_manifest_v1.json \
  --source-postrun-acceptance /absolute/path/to/source_checkpoints/so2sat_source_postrun_acceptance.json \
  --source-preflight /absolute/path/to/source_checkpoints/so2sat_source_preflight.json \
  --training-data /absolute/path/to/training.h5 \
  --selected-candidate /absolute/path/to/development_results/so2sat_candidate_selection.json \
  --selected-gate-fit-bundle /absolute/path/to/development_results/so2sat_${SELECTED_CANDIDATE_ID}.gate_fit.json \
  --target-boundary-amendment experiments/kbound/so2sat/target_boundary_amendment_v1_1.json \
  --checkpoint-collection /absolute/path/to/source_checkpoints/so2sat_source_checkpoint_collection.json \
  --checkpoint-dir /absolute/path/to/source_checkpoints \
  --normalizer /absolute/path/to/source_checkpoints/so2sat_sen2_source_normalizer.json \
  --validation-data /absolute/path/to/validation.h5 \
  --testing-data /absolute/path/to/testing.h5 \
  --reveal-registry-dir /absolute/path/to/outcome_registry \
  --output /absolute/path/to/seals/so2sat_precalibration_execution_seal.json \
  --calibration-device mps \
  --target-device mps

python -m experiments.kbound.so2sat.development calibrate \
  --selection /absolute/path/to/development_results/so2sat_candidate_selection.json \
  --source-postrun-acceptance /absolute/path/to/source_checkpoints/so2sat_source_postrun_acceptance.json \
  --source-preflight /absolute/path/to/source_checkpoints/so2sat_source_preflight.json \
  --precalibration-seal /absolute/path/to/seals/so2sat_precalibration_execution_seal.json \
  --target-boundary-amendment experiments/kbound/so2sat/target_boundary_amendment_v1_1.json \
  --population-manifest /absolute/path/to/so2sat_population_manifest_v1.json \
  --training-geo /absolute/path/to/training_geo.h5 \
  --training-data /absolute/path/to/training.h5 \
  --checkpoint-dir /absolute/path/to/source_checkpoints \
  --output-dir /absolute/path/to/development_results \
  --device auto
```

The candidates are Tent (Adam, learning rate `1e-3`) and SAR (SAM/SGD,
learning rate `2.5e-4`, momentum `0.9`, reliable-entropy margin
`0.4*ln(17)`). Both use batches of 128 and one update per ordered probe batch.
They reset from the source checkpoint for every city/checkpoint cell. After the
west/probe pass, BatchNorm affine parameters are frozen, source running
statistics are restored, and the model enters evaluation mode before any
east/evaluation image is processed. The pinned official commits and source-file
hashes are verified at runtime. EATA is intentionally not a third candidate:
adding its Fisher construction and source subset would introduce an unsealed
selection degree of freedom in this final prospective run.

Feasibility is fixed before results: at least two gate-fit cities must have mean
benefit at least `+0.0025`, at least two at most `-0.0025`, cell-wise oracle
routing must beat the best fixed policy by at least `0.0025`, and leave-one-city-
out ridge routing must beat it by at least `0.001` with at least 55% sign
accuracy. The cross-city replay must also allocate at least 9 of 45 cells and 2
cities to each action. Failure of any condition produces the explicit
no-candidate stop.

## Final target boundary

After calibration writes the calibrated gate and strict gate authorization,
create the final target-execution seal. It must verify and extend the earlier
pre-calibration seal; it cannot create or replace it. This final command again
hashes `validation.h5` and `testing.h5` only as opaque raw bytes and does not
deserialize an HDF5 dataset:

```bash
python -m experiments.kbound.so2sat.target_seal \
  --population-manifest /absolute/path/to/so2sat_population_manifest_v1.json \
  --source-postrun-acceptance /absolute/path/to/source_checkpoints/so2sat_source_postrun_acceptance.json \
  --selected-candidate /absolute/path/to/development_results/so2sat_candidate_selection.json \
  --selected-gate-fit-bundle /absolute/path/to/development_results/so2sat_${SELECTED_CANDIDATE_ID}.gate_fit.json \
  --selected-gate-cal-bundle /absolute/path/to/development_results/so2sat_${SELECTED_CANDIDATE_ID}.gate_cal.json \
  --precalibration-seal /absolute/path/to/seals/so2sat_precalibration_execution_seal.json \
  --gate /absolute/path/to/development_results/so2sat_ridge_gate.json \
  --gate-authorization /absolute/path/to/development_results/so2sat_gate_authorization.json \
  --target-boundary-amendment experiments/kbound/so2sat/target_boundary_amendment_v1_1.json \
  --checkpoint-collection /absolute/path/to/source_checkpoints/so2sat_source_checkpoint_collection.json \
  --checkpoint-dir /absolute/path/to/source_checkpoints \
  --reveal-registry-dir /absolute/path/to/outcome_registry \
  --normalizer /absolute/path/to/source_checkpoints/so2sat_sen2_source_normalizer.json \
  --validation-data /absolute/path/to/validation.h5 \
  --testing-data /absolute/path/to/testing.h5 \
  --output /absolute/path/to/seals/so2sat_target_execution_seal.json \
  --device mps
```

Then run the label-blind live process with `python -m
experiments.kbound.so2sat.target_runner --help`. Its production CLI constructs
the exact canonical HDF5, geographic-index, firewall-loader, and inference
types; injected factories are restricted to artifacts marked `TEST_ONLY` and
cannot emit production schemas or statuses. Only after the complete bundle and
every action, cell, logits, and master receipt exist may a distinct process
invoke `python -m experiments.kbound.so2sat.target_scorer --help`, supplying
the same source post-run acceptance, pre-calibration seal, and local root of
the receipt-bound reveal registry. Both target CLIs require
`--source-postrun-acceptance`; the exact artifact/receipt hashes must match the
selection, pre-calibration seal, final execution seal, and target bundle.

The scorer reports the sealed city-macro/equal-checkpoint estimand, paired
crossed city/checkpoint bootstrap interval, exact sign-flip test on ten city
means under the explicit joint-sign-symmetry assumption, Holm adjustment over
the two fixed-policy comparisons, direct ADAPT/FREEZE exposure, operational
ABSTAIN-to-FREEZE exposure, and every outcome direction. Positive effects are
defined as fixed-policy regret minus KGA regret, equivalently KGA accuracy
minus the fixed policy's accuracy. With ten cities and five checkpoints, the
bootstrap and exact test remain small-cluster inference and must be reported
with that limitation.

Official sources:

- <https://github.com/zhu-xlab/So2Sat-LCZ42>
- <https://github.com/zhu-xlab/So2Sat-LCZ42/blob/master/save_geotiff.py>
