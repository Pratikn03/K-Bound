# PRE-REGISTRATION — iWildCam KGA multicandidate router (empirical ceiling-break)

**Status:** pre-registered BEFORE the TEST split is scored. Calibration (DEV) and
scoring (TEST) are disjoint by camera. This file is written by hand and committed
before the confirmatory TEST run; the runner reads the same split rule from code.

**Date drafted:** 2026-06-30
**Runner:** `experiments/kbound/wilds/run_iwildcam_kga_router.py`
**Source model (frozen f0):** `experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt`
(ResNet-50 ERM, the SAME checkpoint used by the confirmed collapse pilot
`run_iwildcam_streaming_pilot.py`).

---

## 1. Question

The collapse pilot already PROVED (locked result) that on iWildCam OOD test, in
native temporal/location order with a small online batch, online Tent COLLAPSES
below the frozen source: frozen macro-F1 = 0.2554 vs online-Tent macro-F1 = 0.0219,
paired bootstrap CI excludes 0, Tent ends predicting 1 class. So "KGA beats
always-adapt" is not in question.

The OPEN question pre-registered here: **can a label-free certificate ROUTER beat
BOTH always-online-Tent AND always-freeze**, by harvesting benefit only in windows
where a label-free certified lower bound on benefit is positive, and freezing
otherwise?

## 2. Policies compared (all over the SAME native-order OOD stream)

1. **FROZEN** — source f0 in eval mode, no adaptation. The always-freeze baseline.
2. **ONLINE-TENT** — one model adapted across the whole stream, state carried,
   1 entropy step/batch, predict-before-adapt. The always-adapt baseline (collapses).
3. **EPISODIC-TENT** — reset to f0 at every batch, adapt transductively on that
   batch (N steps), predict that batch. No accumulation. Reference candidate.
4. **LAME** (Boudiaf et al., CVPR 2022) — output-only, frozen backbone. Refine f0's
   softmax within each batch by Laplacian-regularised likelihood with a kNN affinity
   in feature space. NO weight updates. Reference candidate documented to help under
   prior/label shift.
5. **KGA-ROUTER** — the contender. Per window, choose among
   {freeze, episodic-Tent, LAME} the action whose **certified benefit lower bound is
   > 0 at the family-wise false-adapt level alpha**; else freeze.

## 3. Native order (mandatory)

Within each camera the stream is served in native order, sorted by
`(location, year, month, day, hour, minute, second, sequence)` — identical to the
pilot's `build_native_stream`. No adversarial shuffle. The stream is the
concatenation of cameras; windows never cross a camera boundary (each camera is
streamed contiguously and split into fixed-size windows).

## 4. DEV / TEST camera split (leakage protocol) — FROZEN RULE

The 48 OOD-test cameras are split disjointly by camera id:

* Order the 48 cameras by **descending sample count** (deterministic, label-free).
* **TEST** = cameras at ranks 0,2,4,… (even ranks). **DEV** = ranks 1,3,5,… (odd).

This size-alternation makes DEV and TEST comparable in total volume without ever
looking at labels or benefit. The resulting split is fixed and recorded here:

* **DEV cameras (24):** 7, 24, 69, 76, 86, 104, 115, 125, 127, 156, 163, 176, 187,
  188, 191, 207, 237, 240, 241, 245, 270, 282, 287, 311  (N≈16,931; 82 classes)
* **TEST cameras (24):** 21, 29, 49, 56, 58, 59, 62, 73, 78, 95, 101, 120, 146, 169,
  184, 193, 263, 268, 280, 288, 289, 301, 302, 315  (N≈18,439; 84 classes)

The runner derives this split from the same descending-count alternation, so the
camera lists above are reproduced by code, not hand-entered into the run.

## 5. Calibration vs scoring

* **Calibrate on DEV ONLY.** Every hyperparameter the router needs — the per-candidate
  benefit estimator GBR(Z→B), the split-conformal radius, alpha, the window size, the
  episodic step count, the LAME kNN/affinity settings — is fit or fixed using DEV
  windows. DEV labels are used ONLY to form the per-window benefit target B for the
  estimator. They never touch a TEST route decision.
* **Score TEST once.** On TEST, per window the router sees only the label-free
  evidence Z, predicts each candidate's benefit lower bound, and routes. TEST labels
  are used only to compute the final macro-F1 of each policy AFTER all routing.

## 6. The route signal (declared honestly)

For each candidate c ∈ {episodic-Tent, LAME} on a window, the router computes a
**label-free evidence vector Z** from frozen-vs-candidate predictions on that window:
pre/post softmax entropy, predicted-class diversity (histogram entropy + #unique),
mean confidence and confidence drop, frozen↔candidate disagreement rate, marginal
class-distribution KL, and (for episodic-Tent) the adaptation gradient L2-norm.
ALL of these are computable without labels.

The certificate is the project's `analysis.decide_kga`: a leave-one-out
gradient-boosted regressor B̂(Z) trained on DEV (Z→true benefit) plus a
split-conformal radius ε. The one-sided lower bound is L(δ) = B̂(Z) − ε(δ).

**Multicandidate correction (Theorem `thm:multicand`, Bonferroni):** with K=2
adaptive candidates, each per-candidate certificate is run at the corrected level
δ_K = alpha/K. The certified-helpful set is S = {c : L_c(alpha/K) > 0}. The router
**commits the candidate in S with the largest lower bound**; if S is empty it
**freezes**. This controls the family-wise false-adapt probability at ≤ alpha for an
arbitrary selector. Freeze is always an available zero-benefit action and is the
default.

**alpha = 0.10.** K = 2 (episodic-Tent, LAME). δ_K = 0.05.

The route decision on a TEST window is a deterministic function of (Z on that TEST
window) and (the DEV-fit estimator + DEV conformal radius). It does not read TEST
labels. This is asserted in the code at the point of decision.

## 7. Metric and uncertainty

* **Metric:** official iWildCam **macro-F1** (`sklearn f1_score(average="macro")`),
  computed over the full set of predictions emitted by each policy.
* **Bootstrap:** 95% CIs by resampling **TEST cameras** with replacement (cluster
  bootstrap over the 24 held-out TEST cameras), recomputing each policy's macro-F1
  and the pairwise deltas on each resample. n_boot = 1000. Camera-level resampling
  (not sample-level) is the honest unit because cameras are the correlated blocks.

## 8. Verdict rule (verbatim, pre-registered)

Let Δ₁ = macro-F1(KGA-router) − macro-F1(always-online-Tent) and
Δ₂ = macro-F1(KGA-router) − macro-F1(always-freeze), with 95% camera-bootstrap CIs.

* **BEATS-BOTH** iff CI(Δ₁) > 0 **AND** CI(Δ₂) > 0 — i.e. both CIs exclude 0 on the
  positive side.
* **NO-HARM** iff CI(Δ₂) includes 0 (KGA ties freeze) **but** CI(Δ₁) > 0 (KGA still
  beats online-Tent).
* **HARM** iff CI(Δ₂) < 0 (KGA is worse than freeze).

We report whatever is measured, including the leading indicator below.

## 9. Leading indicator (reported regardless of verdict)

On DEV and on TEST we report, honestly: **does episodic-Tent or LAME EVER beat
frozen on any window** (per-window benefit B > 0), and the fraction of windows where
each does. If NEITHER candidate ever beats frozen on any window, then beats-both is
unreachable by construction (the router can only ever tie freeze), and the expected
verdict is NO-HARM. This is stated plainly in the report.

## 10. Integrity guardrails

* Native order mandatory; no shuffling of the within-camera stream.
* The frozen, online-Tent, and episodic-Tent updates reuse the project's faithful
  Tent primitives (`tta_methods._clone_for_tta`, `_entropy`); every adapted
  prediction is a real update.
* No committed file is modified; this is a new runner + new results dir only.
* No fabricated benefit: the benefit estimator is fit on DEV measured benefits, and
  the TEST route uses only label-free Z. If the certificate cannot certify benefit,
  the router freezes.
