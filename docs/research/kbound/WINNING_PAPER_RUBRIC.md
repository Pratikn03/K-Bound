# What actually makes a "winning" paper — and where K-Bound stands

Grounded in how the field's top distribution-shift / test-time-adaptation papers get accepted (and what
reviewers reward), mapped honestly onto your paper. The point: separate what you *can* win **now** from
what needs a result.

---

## Two different "wins" (don't conflate them)

| Target | What it takes | K-Bound today |
|---|---|---|
| **Accept at a top workshop** (NeurIPS Distribution Shifts, ICML/ICLR TTA & robustness workshops, UAI, TMLR) | A **sharp problem formulation**, honest evidence, reproducibility. Beats-SOTA *not* required. | **Reachable now.** |
| **~85 / spotlight / best-paper / main-track** | All of the above **plus a memorable empirical result** (beats-SOTA or a striking, broadly-useful finding). | **Needs the experiment** (the pilot → scale-up). |

The field's own criteria say it plainly: workshops want *new ideas and open problems* (lower novelty bar
than the main track), and reviewers reward *clever problem formulations and honest experiments* over
"complex models and fancy mathematics" — but penalize tweaks reported *without* ablations that show where
gains come from.

---

## The 6-axis rubric of a winning shift/TTA paper — and your score

| # | What reviewers reward | K-Bound | Notes |
|---|---|---|---|
| 1 | **Sharp problem formulation** (a question the field hasn't framed) | ✅ strong | "Should it adapt *at all*?" — adapt/freeze/abstain as a decision problem is genuinely fresh. |
| 2 | **Clear contribution vs. prior work** | ✅ strong | Positioned as the "budget-0 face" delta over ATC / agreement-on-the-line / AETTA / e-process monitoring. |
| 3 | **Evidence supports the claims** | ✅ strong | Claims are CI/Holm-gated; you *don't* over-claim (two real-shift wins, not three). |
| 4 | **Honest ablations + negatives** | ✅ strong (rare) | The negatives (ImageNet-R, RxRx1) and the "excluded wins" section are a credibility *asset*. |
| 5 | **Reproducibility** | ✅ strong | Pre-registration, committed result JSONs, 805 passing tests, anonymizable repo. |
| 6 | **A memorable headline result + clean presentation** | ⚠️ **half** | Presentation is now top-tier (teaser, algorithm, booktabs). The missing half is a **beats-SOTA headline** — the one thing editing can't supply. |

**Score: 5.5 / 6.** You are *one ingredient* short of a top-tier result — and it's the only ingredient that
isn't editorial.

---

## What this means for you

- **Submit to a top workshop now.** On axes 1–5 + presentation you're already competitive. This is a real,
  defensible win, and exceptional for an undergrad solo author. Don't wait on it.
- **The "85/spotlight" swing is a separate bet:** run the free CIFAR-100-C pilot in `spotlight_pilot/`. If
  the KGA-gate-as-meta-router beats the best single adapter, *that* is axis 6 — the headline that turns a
  strong workshop paper into a main-track contender. It runs on free Colab; it's the lever.
- **No amount of figure/table/format work moves axis 6.** That's not a limitation of effort — it's what
  "result" means.

## Honest bottom line

Your paper is a **clean, well-presented, honest ~80** that is **ready to win a top workshop** as-is. "85"
is not a polish away; it's an experiment away. The most productive path is: **anonymize + recompile +
submit to a workshop this month**, and run the pilot in parallel for the bigger swing.

---
*Sources: field-acceptance criteria and TTA exemplars below.*
