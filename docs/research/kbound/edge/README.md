# K-Bound Edge: Physical Camera Validation

This module applies KGA as a shadow controller around episodic Tent for a
four-class package/label inspection task.

The frozen model remains the official predictor. A temporary candidate is
adapted per window, label-free evidence is extracted, and KGA returns adapt,
freeze, or abstain. Freeze and abstain both retain the frozen model; their
interpretations differ.

## Current Status

- Synthetic and mock paths are software tests only.
- The real protocol edge_real_phone_v1 is locked.
- Fresh physical S01-S10 captures are pending.
- No camera result is a paper result until publication_gate.json reports
  passed: true.

The code is camera-ready; the empirical study is not yet complete.

## Architecture

1. Capture a 32-frame physical window.
2. Run frozen inference.
3. Adapt a temporary model copy with one episodic Tent step.
4. Run candidate inference.
5. Extract the fixed 14-dimensional label-free evidence vector.
6. Predict benefit and apply the calibrated interval.
7. Commit or roll back the update.
8. Log the action without live labels.
9. Reveal labels only in offline scoring.

Abstention applies to the update, not to prediction.

## Software Check

From the repository root:

~~~bash
python -m pytest docs/research/kbound/edge/tests -q
python docs/research/kbound/edge/scripts/preflight_r2.py
~~~

Browser and OpenCV previews are connectivity checks. Pilot and mock captures
cannot satisfy the publication gate.

## Physical Study

Follow [PHYSICAL_STUDY_RUNBOOK.md](PHYSICAL_STUDY_RUNBOOK.md). The short path is:

~~~bash
python docs/research/kbound/edge/scripts/00_prepare_real_protocol.py
bash docs/research/kbound/edge/scripts/run_edge_source_gate.sh
bash docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh
~~~

The source gate must pass before held-out capture. Development and conformal
sessions must be sealed before S07-S10 are recorded.

## Fail-Closed Publication Gate

The final gate rejects the study if any of these conditions fail:

- protocol hash consistency;
- complete physical clip inventory for S01-S10;
- physical rather than mock capture provenance;
- unique clip hashes;
- S02 balanced accuracy and macro-F1 at least 0.80;
- no source-gate bypass;
- held-out and replication capture after the development seal;
- complete Phone A and Phone B replay;
- all eight anti-leakage checks.

A scientifically valid negative result can pass. The gate checks protocol
integrity, not whether KGA wins.

## Research Dashboard

~~~bash
bash docs/research/kbound/scripts/build_dashboard.sh
python3 -m http.server 8765 --directory docs/research/kbound
~~~

Open http://127.0.0.1:8765/kbound_dashboard.html and select Edge.

## Layout

| Path | Role |
|---|---|
| configs/edge_real_phone_v1.yaml | Locked physical protocol |
| scripts/00_prepare_real_protocol.py | Protocol hash and deterministic checklists |
| scripts/01_capture_real_session.py | Physical capture and sidecar metadata |
| scripts/02_validate_real_dataset.py | Hash, completeness, split, and mock checks |
| scripts/03_train_source_model.py | Frozen source model and quality gate |
| scripts/04-05 | Calibration pairs and KGA fit |
| scripts/06-07 | Held-out and replication replay |
| scripts/08-12 | Leakage audit, ablations, report, tables, runtime |
| scripts/13_check_publication_gate.py | Final fail-closed study gate |
| src/kbound_edge | Runtime and research-library code |
| tests | Unit, leakage, replay, reporting, and publication-gate tests |

Raw clips, checkpoints, generated windows, and local demo media are ignored by
Git. Release only privacy-reviewed manifests, hashes, logs, metrics, and tables.

