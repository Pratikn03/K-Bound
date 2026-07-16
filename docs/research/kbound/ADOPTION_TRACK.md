# K-Bound Adoption Track

Honest framing: adoption is a community outcome — it cannot be "solved," only made
probable. This file is the instrument: the levers we control, sequenced by leverage per
hour. Nothing here is a claim that adoption has happened.

## Why anyone would adopt (the asset inventory, already true)

- Drop-in: wrapper around any adapter with propose/evaluate/rollback; controller adds
  0.20 ms/decision and one model copy (paper Table VI; controller-only microbenchmark).
- No-harm tax profile: ties always-adapt in benign regimes (Camelyon17), certified
  freeze in harmful ones (RxRx1, iWildCam) — insurance, not overhead.
- Integrity surface nobody else ships: pre-registration, tier accounting, Lean-checked
  certificate implication, claim-to-artifact manifest.

## The single biggest objection (honest status)

"Where does β come from?" — in K-Bound, β is a **declared** deployment-class parameter
(historical gaps, documented domain assumption, or stress envelope), not a measured
target-test quantity. That is intentional honesty in the main paper, not a second
manuscript. Mitigate in practice with sensitivity sweeps over a pre-specified β range
and clear reporting when no credible bound exists (frontier does not certify then).

## Levers, in order

1. **Package** (highest leverage/hour). `kga`/`kbound_pkg` to PyPI with a ≤20-line
   quickstart (wrap Tent on CIFAR-10-C, print decision + certificate). Acceptance test:
   a stranger goes pip-install → first adapt/freeze/abstain decision in <10 minutes.
2. **Benchmark artifact.** Publish the decision benchmark (locked streams, regret-to-
   oracle, FA_u, evidence tiers) with a submission format, so other TTA papers can
   report "under K-Bound protocol." The mixed head-to-head harness
   (`scripts/official_baselines_headtohead.py --decisions ...`) already accepts external
   decision files — that IS the submission interface; document it as such.
3. **Upstream engagement.** After runbook item 11 (official POEM/AETTA rows), open
   issues/PRs on the POEM and AETTA repos sharing the protocol-matched comparison and
   inviting corrections — the authors are the most motivated early adopters/critics.
4. **Demonstration asset.** The pre-registered two-phone physical study (App. F) as a
   video + writeup once it passes the fail-closed gate — deployment proof beats benchmarks
   for practitioner adoption. Blocked on fresh capture sessions (user hardware).
5. **Visibility.** arXiv the K-Bound paper after runbook items 11–12; short "safety layer
   for TTA" blog/README narrative; workshop talk. Do not claim adoption metrics that
   don't exist.

## Risks (named, not hidden)

- Vocabulary-only adoption: field cites adapt/freeze/abstain framing without the artifact.
  Mitigation: lever 2 (benchmark) makes the artifact the easiest way to use the vocabulary.
- Field drift: robust adapters shrink harmful regimes, shrinking demand. Mitigation:
  insurance framing is honest about being conditional on the declared class.
- Single-maintainer risk: everything above is one person deep. Mitigation: REVIEWER_REPRO
  packet + manifest keep the project forkable.
