# Locked So2Sat execution

`natural_runner.py` supplies the missing execution bridge for the existing So2Sat prospective-v2 controller. It preserves the v1 negative candidate-selection record and the v2 requirement for at least seven directly ADAPT and seven directly FREEZE calibration cities. A failed screen completes the calibration study and forbids target access.

The current-run protocol is `docs/research/kbound/next_phase/natural_protocol.json`. The 19 calibration cities, five independent source checkpoints, and 10 target cities are not 95 or 50 independent environments. Its rank-18 radius has a nominal 90% marginal, simultaneous-over-checkpoints interpretation for one exchangeable new city. Geographic exchangeability and coverage conditional on passing the feasibility screen are not established. Conditional harmful-adaptation risk below .10 is not certified: even zero errors in 19 independent units gives a one-sided 95% binomial upper bound of about .146.

Historical receipts document zero reserved access. They are not external timestamps or universal historical nonaccess proofs. The new run binds this limitation explicitly. Reconstructed geographic metadata must match the original source acceptance byte for byte. Source checkpoint files, tensor hashes, normalizer and collection receipts are verified; no model retraining or target-guided selection occurs.

The source dependencies are the repository's pinned `external/tent_official` and `external/sar_official` submodules. For the local execution, exact source files and their licenses were copied from the immutable Desktop reference and checked against their existing SHA-256 pins. A clean checkout can initialize the pinned submodules first. The five trained weights and original data remain external dependencies.

From the repository root, with the pinned research Python:

```sh
python -B -m experiments.kbound.next_phase.natural_runner prepare --output /apfs/path/new-run --data /path/to/so2sat/v4 --source /path/to/source-checkpoints --opaque-root /bulk/path/new-opaque-containers
python -B -m experiments.kbound.next_phase.natural_runner calibrate --output /apfs/path/new-run
```

Preparation and calibration are create-only and refuse occupied outputs or repeated calibration starts. The runner does not silently resume partial scientific runs. The signed JSON authorities need a filesystem supporting atomic exclusive rename; the T9 ExFAT volume does not. Opaque compressed target containers can be decompressed and byte-hashed on T9 without HDF5 interpretation. Small authority/results files and the target prediction publication remain on APFS.

Only after a passed, independently recomputed screen:

```sh
python -B -m experiments.kbound.next_phase.natural_runner target --output /apfs/path/new-run
python -B -m experiments.kbound.next_phase.natural_runner score --output /apfs/path/new-run
```

The target runner seals all 50 actions before evaluation pixels and never opens labels. The separate scorer authenticates the prediction publication and performs one testing-label reveal, never reading validation labels. It feeds the existing fixed city-level inference implementation. No statistical choice is tuned during scoring.

Each calibration cell includes full-cell elapsed time, process maximum resident memory and current MPS allocation/driver memory. These full-cell measurements include I/O, source reset, candidate creation, inference and feature extraction; they are not a stage-by-stage latency decomposition. The deployment audit supplies that separate measurement.

The first attempt to write authority artifacts to T9 failed before calibration because ExFAT rejected atomic exclusive publication. Its residue is retained at `/Volumes/T9/kbound_next_phase_20260921/natural_v2`. The replacement APFS execution is under `output/next_phase/natural_v2`; raw opaque copies alone are in `/Volumes/T9/kbound_next_phase_20260921/natural_v2_opaque`. The locked source is preserved verbatim as `natural_runner_locked.py` alongside the run authorities.

The complete September 21 run failed the fixed feasibility screen: two direct ADAPT cells in one city, no direct FREEZE cells, and 93 ABSTAIN cells. Target pixels and labels were never opened. After that completed stop, the reader was corrected to consume the exact receipt-verified document without reopening its path, and preparation gained detection of previous current-run access markers. Regression tests reproduced both weaknesses before the fixes. `postrun_source_correction.json` records the old and corrected source hashes and the unchanged result. The corrected source does not silently reuse the original source lock; original execution bytes and all original authorities remain available.
