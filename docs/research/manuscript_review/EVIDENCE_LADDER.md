# RGA Evidence Ladder

**Date:** 2026-05-07

This note converts the submission ladder into concrete evidence requirements.
It should be read as a research execution map, not as current manuscript
results.

## Workshop-Submittable: Mechanism Isolation

Minimum claim: RGA helps specifically when reliability evidence indicates
domain degradation, and the effect is not an artifact of a saturated clean
benchmark.

Required evidence:

- Tau sweep: report ROC-AUC delta and adaptation rate for
  `tau in {0.4, 0.5, 0.6, 0.66, 0.7, 0.8, 0.9}`.
- Component ablation: compare full reliability against `no_ece`, `no_ks`,
  `no_sharpness`, and `no_gate`.
- Tent/TTT baselines: add at least one entropy-minimization test-time
  adaptation baseline and one self-supervised test-time training baseline.
- Harder benchmark: rerun with intentionally weakened domain scorers so clean
  ROC-AUC is not saturated.
- Learned gate: compare the fixed threshold against a small validation-trained
  switching classifier.

Current repo status:

- Implemented: tau-sweep result hooks in `run_breakthrough_experiment.py`.
- Implemented: component-ablation result hooks in `run_breakthrough_experiment.py`.
- Implemented: harder-benchmark scorer cap via
  `prepare_real_fusion_benchmark.py --scorer-train-fraction`.
- Not yet implemented: Tent/TTT baselines.
- Not yet implemented: learned gate.

## Mid-Tier Conference: Empirical Depth

Minimum claim: the fusion mechanism improves robustness on naturally paired or
otherwise co-observed multimodal events, not only label-aligned composites.

Required evidence:

- Real co-observed dataset with shared entity, time, object, or incident keys.
- Gradient-aligned per-domain attacks over subsets of domains, not only
  zero/max/Gaussian score perturbations.
- Bootstrap confidence intervals for every major table cell and delta.
- Multiple-comparison correction for adversarial or stress-test tables.
- Failure-case visualization where RGA and static attention disagree.

Current repo status:

- Partially implemented: MVTec 3D-AD paired RGB/depth preparation path.
- Partially implemented: multi-seed stress aggregation and bootstrap CIs.
- Not yet implemented: gradient-aligned per-domain attacks.
- Not yet complete: full paired-data benchmark results.

## Top-Venue Polish: Theory

Minimum claim: the reliability gate has a formal role as a switching rule under
bounded domain shift.

Required evidence:

- Formalize the gate as a switching rule between static and reliability-aware
  predictors.
- State assumptions on observable reliability, bounded score shift, and loss.
- Prove a regret or excess-risk bound against the better fixed mode in
  hindsight.
- Connect the result to classical combiner rules and selective prediction.

Current repo status:

- Not yet implemented in the manuscript.
- Best next step: add a one-proposition theory subsection after mechanism
  isolation results exist, so the proof explains measured behavior rather than
  compensating for missing evidence.
