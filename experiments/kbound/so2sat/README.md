# Prospective So2Sat LCZ42 v4.2 track

This directory implements the target-label-blind preparation, source-training,
and development-selection boundary for a new So2Sat Culture-10 confirmation.
It deliberately contains no target scorer and no API that can use a target
label before predictions and actions are sealed.

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

The package intentionally has no target scorer. A later offline scorer must be
implemented as a separate process only after predictions and city actions have
immutable receipts.

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

The output and its receipt are create-only. Before source training begins, a
second execution seal must fix every field listed in
`prospective_protocol_v1.json`, including preprocessing, architecture,
initialization, adapter versions, optimizer settings, checkpoint selection,
gate construction, batching, code, dataset, and environment hashes. Until that
seal exists, the protocol status correctly remains
`STRUCTURAL_PROTOCOL_SEALED_EXECUTION_CONFIG_PENDING`.

## Source and development phases

`train_source.py` trains five independently initialized 10-band ResNet-18
models. Only `source_train` labels update weights; only `source_monitor` labels
select the checkpoint. The source-only normalizer and every checkpoint receive
content hashes and create-only receipts.

`development.py` has two separate commands. `select` evaluates both locked
adapter candidates on the 9 gate-fit cities and writes a selection artifact.
`calibrate` first verifies that artifact and its exact selected gate-fit bundle;
only then may it evaluate the one selected adapter on the 19 gate-calibration
cities. If neither candidate passes, calibration fails closed without reading
gate-calibration pixels or labels.

```bash
python -m experiments.kbound.so2sat.development select \
  --population-manifest /absolute/path/to/so2sat_population_manifest_v1.json \
  --training-geo /absolute/path/to/training_geo.h5 \
  --training-data /absolute/path/to/training.h5 \
  --checkpoint-dir /absolute/path/to/source_checkpoints \
  --output-dir /absolute/path/to/development_results \
  --device auto

python -m experiments.kbound.so2sat.development calibrate \
  --selection /absolute/path/to/development_results/so2sat_candidate_selection.json \
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

Official sources:

- <https://github.com/zhu-xlab/So2Sat-LCZ42>
- <https://github.com/zhu-xlab/So2Sat-LCZ42/blob/master/save_geotiff.py>
