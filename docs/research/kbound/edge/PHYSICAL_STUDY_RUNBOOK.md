# Physical Camera Study Runbook

This runbook executes the locked edge_real_phone_v1 protocol. Do not change the
class vocabulary, object split, session assignment, alpha, window size, adapter,
or evidence schema after capture begins.

## Task

Four physical classes:

| Class | Physical state |
|---|---|
| ok | Label present, flat, aligned, and undamaged |
| missing_label | Label removed or clearly absent |
| misaligned_label | Label visibly rotated or displaced |
| damaged_label | Label torn, folded, or peeling |

Use ten distinct package instances P01-P10. Do not reuse a held-out physical
object in development.

## Locked Splits

| Sessions | Role | Phone | Objects | Physical clips | Evaluation windows |
|---|---|---|---|---:|---:|
| S01 | source train | A | P01-P06 | 120 | 120 |
| S02 | source validation | A | P07-P08 | 40 | 40 |
| S03 | calibration fit A | A | P01-P04 | 64 | 64 |
| S04 | calibration fit B | A | P01-P04 | 48 | 80 |
| S05 | conformal A | A | P01-P04 | 64 | 64 |
| S06 | conformal B | A | P01-P04 | 48 | 80 |
| S07 | held-out A | A | P09-P10 | 64 | 64 |
| S08 | held-out B | A | P09-P10 | 48 | 80 |
| S09 | replication A | B | P09-P10 | 64 | 64 |
| S10 | replication B | B | P09-P10 | 48 | 80 |

S04, S06, S08, and S10 add 32 deterministic batch-composition windows after
physical capture. Total collection is 608 physical clips and 736 evaluation
windows.

Use different recording sessions or days for source, fit, conformal, held-out,
and replication. Phone B must be a different physical device.

## Before Capture

From the repository root:

~~~bash
source .venv/bin/activate
pip install -e .
pip install opencv-python pyyaml torch torchvision scikit-learn joblib

python docs/research/kbound/edge/scripts/00_prepare_real_protocol.py
python docs/research/kbound/edge/scripts/preflight_r2.py
~~~

The prepare step writes the protocol hash and deterministic session checklists.
Commit the protocol source and hash before collecting data.

A pilot can verify framing, but pilot files are never study evidence:

~~~bash
python docs/research/kbound/edge/scripts/01_capture_real_session.py \
  --pilot --phone-id phone_a --camera 0 --max-items 4
~~~

Never use --mock for a physical-study clip.

## Capture Quality

For every clip:

- show one package state clearly;
- keep the whole label region visible;
- avoid showing laptop screens or synthetic images;
- retain the requested shift without changing the class state;
- allow the script to record all frames and write the JSON sidecar;
- do not rename, edit, transcode, or duplicate the MP4 afterward.

The sidecar records capture_mode, camera index, UTC time, frame count, and
SHA-256 hash. Strict validation rejects missing or non-physical provenance.

## Phase 1: Source Gate

The helper captures S01 and S02, validates them, trains the source model, and
requires both S02 balanced accuracy and macro-F1 to be at least 0.80.

~~~bash
EDGE_CAMERA=0 EDGE_PHONE_ID=phone_a \
  bash docs/research/kbound/edge/scripts/run_edge_source_gate.sh
~~~

Stop if the gate fails. Improve the physical setup or source data; do not use
--bypass-gate for publication.

## Phase 2: Development and Conformal Calibration

Capture S03-S06 on Phone A. Use the deterministic checklist and complete one
session at a time.

~~~bash
PY=.venv/bin/python
CAP=docs/research/kbound/edge/scripts/01_capture_real_session.py
CFG=docs/research/kbound/edge/configs/edge_real_phone_v1.yaml

$PY $CAP --config $CFG --session S03 --phone-id phone_a --camera 0
$PY $CAP --config $CFG --session S04 --phone-id phone_a --camera 0
$PY $CAP --config $CFG --session S05 --phone-id phone_a --camera 0
$PY $CAP --config $CFG --session S06 --phone-id phone_a --camera 0
~~~

Validate and seal before recording any held-out clip:

~~~bash
$PY docs/research/kbound/edge/scripts/02_validate_real_dataset.py \
  --config $CFG \
  --through calibration_conformal \
  --seal-through calibration_conformal \
  --strict
~~~

The publication gate checks capture timestamps and rejects held-out or
replication clips recorded before this seal.

## Phase 3: Held-Out Phone A

Use only P09-P10.

~~~bash
$PY $CAP --config $CFG --session S07 --phone-id phone_a --camera 0
$PY $CAP --config $CFG --session S08 --phone-id phone_a --camera 0
~~~

Do not inspect labels, tune thresholds, change evidence, or select a checkpoint
after viewing held-out outputs.

## Phase 4: Phone B Replication

Record P09-P10 on a second physical phone.

~~~bash
$PY $CAP --config $CFG --session S09 --phone-id phone_b --camera 0
$PY $CAP --config $CFG --session S10 --phone-id phone_b --camera 0
~~~

## Phase 5: Publication Pipeline

~~~bash
bash docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh
~~~

This validates all captures, rebuilds windows, trains without bypass, calibrates
KGA, replays the same streams for all policies, profiles runtime, runs
ablations, audits leakage, exports LaTeX tables, writes the report, evaluates
the publication gate, and refreshes the dashboard.

The decisive artifact is:

experiments/kbound/results/edge_real_phone_v1/publication_gate.json

A passing integrity gate does not guarantee a positive KGA result. It guarantees
that the result, including a negative or abstention-heavy result, is admissible
under the locked protocol.

## Paper Fold-In

Fold camera numbers into the paper only when publication_gate.json passes.
Report:

- balanced accuracy and macro-F1;
- regret to the per-window oracle;
- unconditional FA_u before conditional FA_c;
- adapt and abstain rates;
- Phone B replication;
- component and end-to-end latency;
- all leakage checks.

Browser preview, shadow demos, mock captures, pilots, and blank table templates
are not empirical evidence.

## Dashboard

~~~bash
bash docs/research/kbound/scripts/build_dashboard.sh
python3 -m http.server 8765 --directory docs/research/kbound
~~~

Open http://127.0.0.1:8765/kbound_dashboard.html. The Edge page shows exact
session counts and blockers.

## Privacy and Release

Keep raw video local until every frame has been reviewed for faces, addresses,
barcodes, serial numbers, screens, and private surroundings. Public release can
include privacy-reviewed clips or, at minimum, manifests, hashes, logs, metrics,
and the protocol.
