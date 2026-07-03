# MIXED HARMFUL+HELPFUL HEAD-TO-HEAD PROTOCOL — KGA vs POEM vs AETTA
# Pre-registration (research_lock style). REGISTERED: 2026-06-19, BEFORE any
# KGA-vs-POEM and KGA-vs-AETTA number is computed on the real cached scores.

STATUS AT REGISTRATION: NO head-to-head result exists yet. The baseline decision
rules (`poem_decision`, `aetta_decision`) and the harness were written, but the
analysis (`run_mixed_headtohead.py` on the real CIFAR-10-C records) has NOT been
executed. Only the **synthetic-labeled** verification (`verify_headtohead.py`) has
been run, and it decides nothing about the real winner (it tests the apparatus).
This document fixes the benchmark, the metric, and the win criterion in advance so
the outcome of the eventual GPU/Mac run cannot be reverse-engineered to favor KGA.

------------------------------------------------------------------------------
## 0. WHY THIS BENCHMARK EXISTS (the honest framing)

KGA already wins ("beats both" always-adapt and always-freeze, Holm-corrected) in
MIXED harmful+helpful regimes — see the locked CIFAR-10-C 432-condition stress grid
(`experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json`,
5 seeds) and the SAR-collapse online streams. It does NOT beat both on homogeneous
(all-helpful or all-harmful) panels — by design, because on a homogeneous panel one
trivial policy is optimal and nothing can beat it.

The open question this protocol tests is NARROWER and harder: **on a mixed stream,
does KGA's gated-adaptation rule dominate the no-harm SOTA — POEM and AETTA — which
were built for exactly the "don't let TTA hurt you" job?** POEM (a betting
test-martingale that protects entropy-min TTA) and AETTA (a label-free accuracy
estimate used to gate/recover adaptation) are the strongest faithful competitors to
KGA's decision core. If KGA cannot beat them, the contribution is weaker and the
paper must say so. **Whether KGA wins is an empirical question the run answers. All
three outcomes (KGA wins / ties / loses) are publishable and pre-committed below.**

------------------------------------------------------------------------------
## 1. THE MIXED BENCHMARK (data; reuses real cached scores)

### 1.1 Source of per-condition signals (NO new training for the cached arm)
Every policy consumes the SAME logged label-free signal vector `Z` (11 dims) and the
SAME ground-truth benefit `B = a_adapted - a0` per condition that KGA consumes. These
already exist on disk:

  PRIMARY (cached, reused):  experiments/kbound/results/stress_grid_multiseed_v1/
    seed{0..4}/per_condition_cifar10c_{tent,eata,sar}_seed{S}.json   (432 conds/seed)

  Z_names (frozen, identical across methods):
    [pre_entropy, pre_conf, pre_pbal, post_entropy, post_conf, post_pbal,
     pbal_drop, entropy_drop, frac_highconf, marginal_KL, update_norm]

  Per-condition fields used: B, a0, a_adapted, oracle_action, Z, Z_names, condition,
    kga_decision, b_hat, eps_conformal.

This is the integrity anchor: POEM and AETTA are NOT given any signal KGA is denied,
and KGA is NOT given any signal they are denied. The ONLY difference between policies
is the DECISION RULE applied to the shared signals.

### 1.2 The "mixed" requirement (composition is a registered axis, not a knob)
A condition is labeled by its true regime (label used for EVALUATION ONLY, never seen
by any decision rule):
    helpful  if B >  +tau_regime
    harmful  if B <  -tau_regime
    marginal otherwise          (tau_regime = 0.02, same as label_regime() in analysis.py)

The benchmark is the union of conditions whose harmful fraction is non-trivial. The
deployed-adapter mix is FIXED in advance:

  MIXED-PRIMARY  = TENT records, all 432 conditions, 5 seeds.
    Rationale: TENT is the harmful-prone adapter here (harmful fraction ~0.176,
    B in [-0.092, +0.526]); it is the regime where a no-harm method must prove
    itself. This is the headline comparison.

  MIXED-POOLED   = TENT + EATA records pooled (864 conditions/seed, 5 seeds).
    EATA harmful fraction ~0.058. Pooling lowers the harmful base rate and tests
    robustness of the verdict to composition. Reported as a SECONDARY row.

  SAR-COLLAPSE   = the aggressive online SAR-collapse streams (Protocol-E schedule),
    when their per-condition records are present on disk under the agreed run dir.
    SAR on the cached fixed/mild grid is benign (0% harmful); the collapse regime
    is where SAR genuinely goes harmful online. Reported as a SECONDARY row,
    INCLUDED ONLY IF the records exist at run time (else explicitly marked ABSENT;
    not silently dropped). See section 6.

Conditions are NOT cherry-picked: each MIXED set is the FULL condition list of its
adapter(s). Dropping or reweighting individual conditions post hoc is forbidden (sec 5).

### 1.3 Optional fresh-collected arm (camera-ready only; not required for the claim)
For camera-ready, the SAME harness can ingest freshly-collected POEM/AETTA online
streams produced by the OFFICIAL repos (yarinbar/poem, taeckyung/AETTA) on the same
checkpoint. That arm is a stronger faithfulness check but is NOT part of the
pre-registered cached-arm claim and does not gate it. See RUN_ON_MAC_POEM_AETTA.md.

------------------------------------------------------------------------------
## 2. THE POLICIES (six; all on the shared signals)

  always-adapt   : decision = ADAPT on every condition.            (trivial floor)
  always-freeze  : decision = FREEZE on every condition.           (trivial floor)
  oracle         : per-condition argmax(a0, a_adapted).            (upper bound; uses B)
  KGA            : decide_kga (LOO gradient-boosted B_hat(Z) + split-conformal eps;
                   ADAPT/FREEZE/ABSTAIN; anytime false-adapt <= alpha certificate).
                   alpha = 0.10. (experiments/kbound/wilds/analysis.py)
  POEM           : poem_decision — faithful port of the betting-martingale protector
                   (yarinbar/poem protector.py + cdf.py). See sec 2.1.
  AETTA          : aetta_decision — faithful port of Eq. 13 accuracy estimate used as
                   an adaptation gate (taeckyung/AETTA, CVPR'24). See sec 2.2.

ABSTAIN is scored as FREEZE for accuracy/regret (the safe fallback): a policy that
abstains keeps the source model. This is identical to the locked stress-grid analysis
and applies equally to every policy that can abstain.

### 2.1 POEM decision rule (faithful; documented divergences explicit)
POEM's protector (Algorithm 1 of arXiv:2408.07511) runs a test-martingale on the
source-entropy PIT u_t = CDF_source(entropy_t). Under no shift u_t ~ Uniform[0,1] and
the martingale C stays ~1 (PROTECT: suppress the entropy-min update). When the test
entropy distribution departs from source, C grows (shift detected: ADAPT). We port the
EXACT update from protector.py (b = 1 + eps*(u-0.5); SF-OGD on eps; C *= b) operating
on the logged `pre_entropy` stream, with CDF_source estimated from the in-distribution
/ low-severity entropies present in the records (the no-shift reference). Decision per
condition: ADAPT iff the protector's accumulated log-martingale exceeds its detection
threshold (shift certified at level delta) AND the protected entropy direction is
consistent with helpful self-training; else PROTECT->FREEZE. SIMPLIFICATIONS, stated:
  (S1) We drive the martingale with the logged batch-summary `pre_entropy` per
       condition rather than per-sample entropies (the records do not store the raw
       per-sample stream). This is a faithful reduction (POEM aggregates per batch),
       but coarser than the official per-sample loop. SWAP for camera-ready: feed the
       OFFICIAL protector the per-sample entropy stream from the fresh-collected arm.
  (S2) The "consistent with helpful self-training" check uses the logged
       `entropy_drop` sign as the post-adaptation entropy movement; POEM's online loop
       observes this directly. Documented, not weakened: POEM is GRANTED its protection
       signal; the gate only fires to ADAPT when its own martingale says shift.
  We DO NOT alter POEM's constants (gamma=2/sqrt(3), eps_clip=1.8) from the repo
  defaults. We DO NOT add any KGA-specific information to POEM.

### 2.2 AETTA decision rule (faithful; documented divergences explicit)
AETTA (Eq. 13) estimates error as Err ~ (E_avg/E_max)^(-alpha) * PDD, with alpha=3,
PDD = prediction-disagreement vs dropout, E_avg = entropy of dropout-averaged batch
softmax, E_max = log K. Acc_est = 1 - Err. The paper's case study RESETS/freezes the
model when AETTA-estimated accuracy shows a degradation trend. Per-condition faithful
analog: estimate post-adaptation accuracy Acc_est_post and pre-adaptation accuracy
Acc_est_pre from the logged signals; ADAPT iff Acc_est_post >= Acc_est_pre - margin
(no predicted degradation), else FREEZE. SIMPLIFICATIONS, stated:
  (A1) The records do not store dropout-inference disagreement (PDD) or the
       dropout-averaged entropy E_avg directly. We map AETTA's two ingredients onto the
       logged label-free proxies that carry the SAME information AETTA extracts:
         PDD proxy        := 1 - frac_highconf       (disagreement rises as fewer
                             samples are confidently/stably predicted)
         (E_avg/E_max)    := post_entropy / log(K)   (batch-aggregate entropy ratio;
                             AETTA's skew/over-confidence indicator)
       so Err_post = (post_entropy/logK)^(-alpha) * (1 - frac_highconf), and the pre
       counterpart uses pre_entropy and the pre-adaptation high-confidence fraction.
       This is the documented simplification; for camera-ready, swap in the OFFICIAL
       AETTA estimator run with real dropout passes (taeckyung/AETTA) on the fresh arm.
  (A2) alpha=3 and EMA smoothing are taken from the paper; not tuned here.
  We DO NOT give AETTA the true B or a_adapted. Its gate sees only label-free signals.

NOTE on faithfulness vs the official code: both ports are written to the published
algorithm and constants; the ONLY deviations are the input-granularity mappings (S1,
A1) forced by what the cached records store. Each is flagged in code and here, and
each is a place the user MUST swap in official-repo outputs for camera-ready. Neither
mapping injects label information or KGA-specific signal, and neither weakens the
baseline's protection logic — POEM still gets its martingale, AETTA still gets its
skew-corrected disagreement estimate.

------------------------------------------------------------------------------
## 3. PRIMARY METRIC (chosen and justified, in advance)

PRIMARY: **mean regret-to-oracle across the mixed stream**, per policy.
    regret(policy) = mean_conditions [ oracle_acc - policy_acc ],
    oracle_acc = max(a0, a_adapted);  policy_acc = a_adapted if policy ADAPTs that
    condition else a0.  (Identical regret convention to the locked stress-grid
    analysis and to multiseed_paired_ci.py.)

WHY regret-to-oracle is primary (not raw accuracy): on a mixed stream the achievable
accuracy ceiling differs by composition; regret-to-oracle normalizes against the
per-condition best action and so directly measures "how much did this policy leave on
the table by adapting/freezing wrongly." It is the quantity the no-harm methods and
KGA are all trying to minimize, and it is symmetric to both failure modes (adapting
when harmful AND freezing when helpful). It is the metric on which KGA's existing
"beats both" claim is stated, so the head-to-head is apples-to-apples.

SECONDARY (reported alongside, NOT the win gate): **false-adapt rate at matched
coverage.** false_adapt_rate = Pr[ ADAPT and B < 0 ]. KGA carries an anytime
false-adapt <= alpha certificate; POEM/AETTA do not. We report each policy's
false-adapt rate AND its coverage (fraction of conditions where it commits a decisive
ADAPT/FREEZE rather than abstaining), so a low regret bought by reckless adaptation is
visible. "Matched coverage": when comparing false-adapt rates we note each policy's
coverage; a policy is not credited for a low false-adapt rate achieved by abstaining
on everything. This is a diagnostic, not the primary gate, because regret already
penalizes both error directions; false-adapt is the safety lens the paper also cares
about.

------------------------------------------------------------------------------
## 4. WIN CRITERION (explicit, pre-committed, Holm-corrected)

Define on each MIXED set, per the paired bootstrap in
experiments/kbound/wilds/multiseed_paired_ci.py (paired over the per-condition mean
regret across the 5 seeds, nboot = 1e4, seed 20260619 for the head-to-head):

  diff(KGA, X) = mean_cond [ regret_KGA - regret_X ]   for X in {POEM, AETTA}.
  KGA "beats X" iff the 95% paired-bootstrap CI of diff(KGA, X) lies ENTIRELY BELOW 0
  (CI upper bound < 0) AND survives Holm correction over the comparison family
  {KGA vs POEM, KGA vs AETTA, KGA vs always-adapt, KGA vs always-freeze} at
  family-wise alpha = 0.05.

PRIMARY HEADLINE CLAIM (pre-registered three-way verdict on MIXED-PRIMARY = TENT):

  WIN ("beats the no-harm SOTA"):
      KGA beats POEM AND KGA beats AETTA (both CIs entirely below 0, both survive Holm)
      AND KGA's pooled false-adapt rate <= alpha = 0.10.
      -> paper may claim KGA dominates the no-harm SOTA on mixed shift.

  TIE ("matches the no-harm SOTA"):
      For at least one of {POEM, AETTA}, the Holm-corrected CI of diff includes 0
      (no significant difference), and KGA does not LOSE to either (no CI entirely
      above 0).  AND KGA's false-adapt rate <= alpha.
      -> paper claims KGA TIES the no-harm SOTA on regret while ADDITIONALLY carrying
         the anytime false-adapt certificate they lack (the certificate is the
         differentiator, stated as such; no dominance claimed).

  LOSE:
      For at least one of {POEM, AETTA}, the Holm-corrected CI of diff(KGA, X) lies
      entirely ABOVE 0 (KGA strictly worse).
      -> paper reports KGA is beaten by that method on mixed regret; the contribution
         is rescoped to the certificate / the regimes where KGA does win (homogeneous-
         harmful, SAR-collapse), and this is stated plainly. NOT hidden.

All three verdicts are publishable. The verdict is whatever the run returns. We will
NOT re-pick the primary set, the metric, alpha, the regime threshold, or the bootstrap
seed after seeing any diff. The SECONDARY sets (MIXED-POOLED, SAR-COLLAPSE) refine /
stress the headline but do not override the pre-registered TENT verdict.

POWER NOTE (honest): with 5 seeds x 432 paired conditions the paired bootstrap is
well-powered for regret differences >~ 0.002 acc; if a diff is smaller than the CI
half-width the verdict is TIE by construction, which is the correct (not evasive)
outcome — we report the CI, not a bare p-value.

------------------------------------------------------------------------------
## 5. FORBIDDEN ACTIONS (the integrity contract)

  - Tuning ANY policy's hyperparameters on the test conditions. KGA's alpha=0.10,
    POEM's (gamma, eps_clip), AETTA's alpha=3 are FROZEN at registration to their
    paper/repo values. No grid search on the benchmark.
  - Cherry-picking conditions: each MIXED set is the FULL condition list of its
    adapter(s). No dropping, reweighting, or re-ordering conditions after a result is
    seen. (paired CIs require identical condition order across seeds; enforced.)
  - Re-picking the primary set / metric / win criterion / regime threshold / bootstrap
    seed after observing any diff.
  - Weakening a baseline to make KGA win: POEM keeps its martingale protection, AETTA
    keeps its skew-corrected disagreement estimate; any simplification is the
    granularity mappings in sec 2.1-2.2, all label-free and explicitly logged.
  - Reporting only the favorable MIXED set. ALL pre-registered sets that have records
    at run time are reported; absent sets are marked ABSENT, never silently dropped.
  - Fabricating, interpolating, or hand-editing any policy accuracy, regret, CI, or
    decision. Every number is produced by the committed scripts from the cached/fresh
    records. (No KGA-vs-POEM/AETTA number exists at registration time.)

------------------------------------------------------------------------------
## 6. SEEDS, CONDITIONS, ARTIFACTS

  seeds:        [0, 1, 2, 3, 4]   (the 5 stress-grid replicates; seed sets ALL rng)
  conditions:   MIXED-PRIMARY 432/seed (TENT); MIXED-POOLED 864/seed (TENT+EATA);
                SAR-COLLAPSE as available.
  alpha_false_adapt: 0.10   regime_threshold tau: 0.02   Holm alpha: 0.05
  bootstrap:    nboot 1e4, rng seed 20260619 (registration date) — fixed.

  harness:      experiments/kbound/poem_aetta/run_mixed_headtohead.py
  baselines:    experiments/kbound/poem_aetta/baselines.py (poem_decision, aetta_decision)
  analysis:     experiments/kbound/wilds/multiseed_paired_ci.py (extended for >2 methods)
  verify (CPU): experiments/kbound/poem_aetta/verify_headtohead.py (SYNTHETIC, decides nothing)

  serialization contract (per policy x seed, arrays not aggregates):
    per_condition_<set>_<policy>_seed<S>.json with records[*] =
      {condition, B, a0, a_adapted, oracle_action, Z, Z_names,
       kga_decision (== the policy's decision in {ADAPT,FREEZE,ABSTAIN}),
       eps_conformal, b_hat}   (field name kga_decision kept for analysis-script
       compatibility; it holds THIS policy's decision — see harness docstring.)
    plus HEADTOHEAD_RESULTS.json (the paired-CI output + the sec-4 verdict) and a
    result_manifest.json (git hash, seeds, env, wall time, input record paths).

  output_dir:   experiments/kbound/results/mixed_headtohead_v1/

------------------------------------------------------------------------------
## 7. WHAT IS VERIFIED BEFORE THE RUN vs WHAT THE RUN DECIDES

  VERIFIED HERE (torch-free, synthetic, no real-data result):
    - the six decision rules execute and return valid decisions;
    - the regret + false-adapt metrics compute for all six;
    - the paired-bootstrap + Holm + sec-4 verdict machinery runs end-to-end and
      classifies WIN/TIE/LOSE correctly on constructed synthetic cases (a synthetic
      case engineered KGA-favorable yields WIN; one engineered POEM-favorable yields
      LOSE) — proving the apparatus is not hard-wired to say WIN.
    - the serialization schema round-trips through multiseed_paired_ci.py.

  DECIDED ONLY BY THE REAL RUN (Mac/GPU, on the cached + fresh records):
    - whether KGA actually beats / ties / loses to POEM and AETTA on the real
      CIFAR-10-C mixed stream. THIS DOCUMENT DOES NOT ASSUME THE OUTCOME.

REGISTRATION HASH: this file is committed before HEADTOHEAD_RESULTS.json exists.
