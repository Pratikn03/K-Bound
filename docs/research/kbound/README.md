# K-Bound — The Knowability Boundary of Label-Free Adaptation

Standalone workspace for the new paper. **Self-contained: does not modify any
existing ELARA file.** It only *reads* from `../experiments/elara_u/`.

Paper question: *Can observable test-time evidence separate helpful, harmful, and
unknowable regimes of label-free adaptation, yielding a computable rule for when a
system should adapt, freeze, or abstain?*

Names: theory = **K-Bound**, algorithm = **KGA (Knowability-Guided Adaptation)**.

## Folder layout
- `scripts/` — runnable experiments (every number is produced by a real run; nothing fabricated)
- `results/` — JSON outputs of those runs
- `figures/` — generated PNGs

## What has been RUN and what it shows (honest status)

1. `scripts/knowability_experiment.py` → `results/knowability_results.json`
   - Trichotomy on the 123-task clean score archive. f0 = best-val detector,
     fa = logistic stack. Decision from label-free evidence Z via leave-one-out
     estimator + conformal radius.
   - Result: certificate is SAFE (adapt precision 0.90; abstains where true
     benefit |B| is small: 0.021 vs 0.132). BUT this suite is helpful-dominated,
     so always-adapt (0.777) beats the safe policy (0.766). Coverage only 17%.

2. `scripts/kbound_harmful_regime.py` → `results/kbound_harmful_results.json`
   - Same certificate, harmful regime: fa = elara_fuse (hurts on ~80% of tasks).
   - Result: K-Bound correctly refuses the harmful path (matches freeze 0.748;
     always-adapt only 0.728; regret cut ~11x). Safety story works.

## The decisive open experiment (make-or-break for the paper)
In each single-regime suite, a trivial policy (always-adapt OR always-freeze) is
already near-optimal, so K-Bound only *matches* the better trivial baseline.
The trichotomy's real value — beating BOTH — appears only in a **mixed regime**
(some instances helpful, some harmful, per-instance discrimination required).
The strongest evidence a mixed regime exists is MVTec-3D modality failure
(`../experiments/elara_u/multimodal_reliability_results_mvtec3d.json`): clean = no
harm (0.882=0.882), failure = +0.21 AUROC, all CIs exclude zero.

NEXT: build a per-instance mixed clean/failure benchmark and show K-Bound beats
always-adapt, always-freeze, and approaches oracle. That run decides the paper.

## Status note
- Proven/run: the two experiments above (real numbers).
- NOT yet done: the theory proofs (T1-T4), the positioning table vs AETTA /
  Protected-TTA(betting) / Steinhardt-Liang, and the decisive mixed-regime run.
- No claim of "solved" or "field-shaping" — those are not self-assignable.
