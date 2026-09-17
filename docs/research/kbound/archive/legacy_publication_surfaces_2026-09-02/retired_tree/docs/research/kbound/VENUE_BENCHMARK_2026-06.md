# K-Bound vs. the top-venue bar (NeurIPS / ICML / ICLR 2025–2026)

*Honest benchmark of "When Is Label-Free Adaptation Knowable?" against what actually
gets accepted in its niche. Scope note: there is no canonical "top-100" list, and a
random top-100 would be mostly LLM/diffusion/RL work irrelevant to you. The meaningful
comparison is the ~15 directly comparable accepted TTA / distribution-shift papers —
those are the ones whose reviewers you'd be drawn against. Based on titles/abstracts +
venue pages, not full reads of each; acceptance also has real reviewer-luck variance.*

## 1. What actually gets into NeurIPS/ICML/ICLR in this area (the bar)

Nearly every accepted TTA / distribution-shift paper in 2025–2026 is one of:

| Pattern | Representative accepted papers (venue) |
|---|---|
| **New method + empirical wins** (often SOTA on corruptions / VLMs / tabular / time-series) | TACT *Test-Time Adaptation by Causal Trimming* (NeurIPS'25); BufferTTA (NeurIPS'25); Backprop-Free TTA via Probabilistic Gaussian Alignment (NeurIPS'25); L2C *Adapt Frozen CLIP* (ICLR'25); Noisy TTA (ICLR'25); Test-Time Selective Adaptation for Uni-Modal Shift (ICML'25); Shift-Aware TTA for time-series (ICML'25) |
| **Method + a theoretical guarantee + wins** | Test-time Correlation Alignment (ICML'25); **POEM — Protected TTA via Online Entropy Matching (betting "no-harm")** (NeurIPS'24) |
| **Safety / monitoring / conformal that works** | Monitoring Risks in TTA (2025); Conformal Uncertainty Indicator for CTTA (2025); Adapting Prediction Sets to Shift Without Labels (2025); Prinster et al. weighted-conformal monitoring+adaptation (2025) |

**The through-line:** the bar is a *positive, working contribution* — a method that beats
baselines, or a guarantee that is **demonstrated to help**. Even the theory-forward and
safety papers pair their analysis with a method that wins or a guarantee a practitioner
can deploy. I did **not** find a top-venue accept whose headline is "an impossibility
result + a certificate + honest negatives, no SOTA claim."

## 2. The two papers closest to yours — and what they have that you don't

- **POEM (betting / online entropy matching, NeurIPS'24)** — this is your **certificate's
  nearest neighbor**: anytime-valid, betting-martingale, a "no-harm" protection guarantee.
  It got in because it is a *deployable method that protects a running adapter and is shown
  to work*. Your Thm 3 (anytime e-process false-adapt ≤ α) is the same statistical family —
  so the certificate is **not** your novel hook, and a reviewer who knows POEM / Schirmer
  will say so.
- **Test-time Correlation Alignment (ICML'25)** — theory + a working alignment method that
  improves accuracy. The theory *serves a win*; the win is the contribution.

## 3. K-Bound vs. the bar, axis by axis (the way reviewers score)

| Axis | Accepted-paper norm | K-Bound (current paper) | Gap |
|---|---|---|---|
| **Novelty of contribution** | a new method or a new, surprising result | impossibility + one-bit dichotomy + certificate; framing is fresh, machinery is known | **moderate** |
| **Theory depth** | a guarantee that enables a method | correct, but a synthesis of Le Cam / e-values / conformal; certificate preempted by POEM | **moderate–weak** |
| **Empirical result** | wins / SOTA on standard benchmarks | **honest negative** on the main natural-shift panels: KGA *matches the better policy*, does **not** beat both; wins only in mixed regimes | **this is the decisive gap** |
| **Significance / "why use this"** | a usable improvement | a safety/abstention discipline + an honest map of where routing fails | **moderate** |
| **Rigor & honesty** | assumed; rarely a differentiator | **top-decile** — pre-registration, anti-leakage, reports negatives | **strength (but not what wins accepts)** |

## 4. Verdict on your aim ("easy acceptance at a top-tier venue")

**Honestly: no — the current paper is not an easy accept at NeurIPS/ICML/ICLR, and it is
closer to a borderline-reject than a borderline-accept there.** Not because it is weak —
it is correct, careful, and well-framed — but because it is missing the one thing the
accepted comparables all have: **a positive, working result.** Its headline empirical
finding is a negative ("doesn't beat both"), its certificate is preempted, and its theory
(after this session's adversarial check) is an elegant synthesis, not a complete new
theorem. Top-tier reviewers in this track reward "here is a thing that works and beats X";
a paper that leads with "here is exactly where label-free routing *cannot* help" is
admirable science but a hard sell for a clear accept.

## 5. Where it *does* land — realistic, good homes

1. **TMLR — the best fit, and a genuinely strong outcome.** TMLR explicitly judges on
   *correctness and clarity, not perceived impact or SOTA*, and welcomes honest negative
   results. This paper's strengths (rigor, honesty, a clean question, a correct certificate)
   are exactly TMLR's acceptance criteria. This is a realistic accept, not a stretch.
2. **A top-venue TTA workshop** (e.g. the ICML'25 "PUT: Putting Updates to the Test"
   workshop) — strong fit, fast, good visibility, low risk.
3. **NeurIPS/ICML/ICLR main track — only with the change in §6.**

## 6. What would turn it into a real top-tier accept (the honest lever)

The gap is empirical, not rigor. To clear the bar you need a **positive, working result**:

- **Turn the safety framing into a demonstrated win.** Build a benchmark/setting with
  *genuinely mixed* harmful+helpful regimes (you already win there — CIFAR-10-C stress
  grid, the SAR-collapse stream) and show KGA **beats every baseline including POEM and
  AETTA** on a clear metric, at scale, multi-seed. "A label-free certificate that provably
  controls false-adapt **and empirically beats the no-harm SOTA on mixed shift**" is a
  top-tier story; "matches the better trivial policy on homogeneous panels" is not.
- **Or** land a clean, *complete* theoretical result — but the adversarial check this
  session showed the knowability dichotomy is near-definitional, so the empirical path is
  the more viable one.

**Bottom line:** your *aim* (easy top-tier accept) and your *paper* (honest, correct,
negative-leaning) are mismatched as-is. The fastest honest win is **TMLR + a workshop
now**; the top-tier path requires converting the safety guarantee into a head-to-head
*win* over POEM/AETTA on a mixed-shift benchmark. Both are reachable — but they are
different goals, and it's worth deciding which one you actually want.
