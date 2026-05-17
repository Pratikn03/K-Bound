# Privacy, Data Handling, and Deployment Notes

This file enumerates the data-handling boundaries the ELARA empirical work
relies on, and the deployment-time controls a regulated environment would
need to bring online. It complements Appendix C of the thesis chapter and
sits alongside `EXTERNAL_GATED_RUNBOOKS.md` as a checklist for moving the
prototype to a production setting.

## 1. Data sources used in the paper

| Source | Pairing role | Privacy posture |
|---|---|---|
| MVTec 3D-AD | Primary naturally paired benchmark (RGB + depth) | Industrial scans of inanimate objects, no PII. Public research licence; the dataset ships pre-anonymized. |
| ELARA-Bench-LA | Label-aligned secondary benchmark | Synthetic-by-construction label alignment over publicly-released fraud, cyber, and behaviour tables. None of those tables contain direct identifiers; we treat them as already pseudonymised. |
| GridPulse vitals (BIDMC + MIMIC-III) | Off-paper future-work scaffold (Gap 1 runbook) | Used only by the optional `prepare_healthcare_fusion_benchmark.py` script for a thesis-appendix or clinical-AI venue replay. The script reads features pre-extracted by the existing GridPulse pipeline; raw waveforms are not re-processed inside ELARA. |

The conference manuscript's evidence pipeline never touches patient-bearing
data. The thesis chapter discusses GridPulse only as deferred future work,
and that work is gated by the controls in Section 3 below.

## 2. Training-pipeline boundaries

The training pipeline (`run_breakthrough_experiment.py`) operates on
flat CSV / parquet fusion tables and never writes back to data sources.
Every artefact written during a run lands under
`experiments/fusion/` or `models/fusion/`, both of which are gitignored
for binary outputs (the manuscript-driving JSON tables are checked in
deliberately). No raw source identifiers leak into model weights:

* Sample identifiers (`sample_id`) are hashed inside the prep scripts.
* Patient identifiers in the GridPulse path are not used as features;
  they are split keys only.
* Validation, training, and test rows live in the same CSV. The runner
  honours a configured `split_column` so patient identities cannot leak
  across splits even if a future contributor inadvertently re-shuffles
  the rows.

## 3. Deployment-time controls required for regulated environments

A production deployment would need three additional controls beyond what
the prototype provides:

1. **Credentialed data-access review.** Each benchmark would need a
   documented data-use agreement specifying who can run the pipeline,
   what derived data may be stored, and the retention horizon. MVTec
   3D-AD and ELARA-Bench-LA do not require any such review; GridPulse
   would.

2. **Calibration monitoring.** The
   `src/scripts/monitor_calibration.py` script implements the
   observe-only deployment monitor described in
   Appendix~B of the thesis chapter. It emits a JSON-lines stream of
   per-window mean reliability, per-domain KS-drift, and batch ECE so an
   on-call engineer can reconstruct exactly when the gate would have
   fired during any time interval, without granting the monitor write
   access to the inference path.

3. **Auditable inference path.** The fusion runner already logs
   per-seed gate activation rates and reliability statistics into the
   per-experiment JSON. A production deployment would extend this by
   logging one JSON event per inference call, with the gate decision,
   the realised reliability vector, and the decisive counterfactual
   domain-attribution scores. Together with item 2, this satisfies the
   "explainability requirement" typical of regulated AI systems.

## 4. Out-of-scope items

* Differential privacy. The training pipeline does not currently apply
  DP-SGD or any other privacy-amplification technique. None of the
  benchmarks used in the paper require it.
* Model weights. The released model checkpoints are score-fusion
  attention modules of < 1 MB; they do not encode any per-patient state
  and cannot be inverted to recover training data. A future deployment
  on patient-bearing data would need a model-card-style disclosure of
  this property.
* Federation. ELARA fuses scores from independent domain detectors but
  does not implement federated training; if a domain detector itself
  needs to train on regulated data, the training of that detector sits
  outside the ELARA prototype boundary.

## 5. Checklist for a future regulated deployment

* [ ] Data-use agreement filed for each benchmark in production.
* [ ] Calibration monitor scheduled with an on-call escalation policy.
* [ ] Per-inference JSON event log retained for the agreement-specified
      window.
* [ ] Model card released with the score-fusion module's interpretability
      guarantees and known failure modes (the cross-benchmark contrast).
* [ ] A "kill-switch" config switch that pins the static-attention path
      when the calibration monitor flags a sustained gate-fire condition.
