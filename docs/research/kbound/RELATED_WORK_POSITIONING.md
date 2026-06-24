# Positioning K-Bound vs. concurrent label-free TTA-safety work (2026-06-19)

Readable companion to `paper/sections/related_work_positioning.tex`. Drafted from
primary abstracts (AETTA CVPR'24; Monitoring Risks in TTA, Schirmer et al. 2025;
Suitability Filter 2025) and ATC (Garg et al. ICLR'22) / agreement-on-the-line
(Baek et al. NeurIPS'22).

## The one-paragraph positioning

K-Bound is **not** the first to note that TTA can hurt, nor the first to use
anytime-valid sequential statistics for deployed models. Its defensible core is the
**decision-theoretic knowability boundary** (Thm 1 impossibility + the one-bit
dichotomy) plus a **pre-commitment adapt/freeze/abstain** rule with a controlled
false-adapt rate. Everything else in the neighborhood either estimates a number or
monitors a running model after the fact.

## Nearest neighbors and the exact delta

| Line / method | What it does | K-Bound's distinction |
|---|---|---|
| **ATC / agreement-on-the-line** (unsup. accuracy est.) | Predict accuracy on unlabeled shifted data from confidence/agreement heuristics | Thm 1 gives the *reason* these are limited (sign-of-benefit undecidable without 1 bit); K-Bound decides, not estimates |
| **AETTA** (CVPR'24) | Label-free accuracy estimate for TTA via dropout disagreement → trigger recovery | Certified decision + boundary on when *any* such estimate is trustworthy; AETTA used only as a surrogate baseline |
| **Monitoring Risks in TTA** (2025) — *closest* | Confidence-sequence risk **monitoring**: alarm when a running adapter degrades enough to retrain | Same anytime-valid family, but **post-hoc alarm** vs **pre-commitment** decision; K-Bound adds the **impossibility lower bound** monitoring lacks; the two are complementary |
| **Suitability Filter** (2025) | Hypothesis-test covariate-shift signals to gate a **fixed** model | K-Bound is about whether to *change* the model, and handles the concept-shift impossibility |

## The honest strategic takeaway

The **certificate (Thm 3) is the least novel part** — confidence-sequence TTA
monitoring already exists (2025), and the no-assumption impossibility is already known
informally in the unsupervised-accuracy-estimation line. So:

1. **Lead the paper with the knowability boundary** (Thm 1 + one-bit dichotomy), not
   the certificate. That is the part with no direct competitor.
2. **Frame the certificate as the operational instantiation**, and cite the
   confidence-sequence monitoring work as shared machinery, not a rival.
3. **Own the pre-commitment trichotomy** (adapt/freeze/abstain with false-adapt ≤ α) —
   the decision object is genuinely different from "estimate accuracy" and "alarm on
   degradation."

This is also the sharpest available answer to the "is the contribution incremental?"
reviewer: yes on the machinery, no on the boundary + decision object — and the paper
should be written so that distinction is unmissable.

## Sources
- AETTA — arXiv:2404.01351 (CVPR 2024)
- Monitoring Risks in Test-Time Adaptation — arXiv:2507.08721 (2025)
- Suitability Filter — arXiv:2505.22356 (2025)
- ATC — Garg et al., ICLR 2022
- Agreement-on-the-line — Baek et al., NeurIPS 2022
