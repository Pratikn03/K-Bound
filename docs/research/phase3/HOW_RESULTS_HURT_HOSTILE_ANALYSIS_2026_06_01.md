# How Your Results Can Hurt You — Hostile-Reviewer Analysis (2026-06-01)

This is the adversarial read: every attack a hostile reviewer, thesis examiner,
or skeptical collaborator can make on the current results, ranked by how much
damage it does, with the honest defense or admission for each. Read this BEFORE
anyone else does, so nothing is a surprise.

Current standing (the facts being attacked):
- Gate D / T5: PASS (in-domain, MVTec 3D-AD supervised-paired, 30 splits,
  Δ=+0.024 vs SAR, p=2.6e-19, 30/30).
- Gate E / F: FAIL (clean external transfer not won).
- Master-C checklist 34/38; flagship NOT achieved.
- Reliability gating = stress-regime mechanism; clean-transfer superiority not
  established.

---

## TIER 1 — Attacks that hurt the MOST (could sink the headline)

### 1. "Your headline 0.978 is supervised-paired, not one-class — it's not comparable to M3DM/AST." ⚠️ HIGH
- **The hit:** the in-domain win (Gate D/T5) is on the *supervised-paired*
  protocol, where positives are visible to the fusion training fold. The
  published leaderboard (M3DM 0.945, AST 0.937) is *one-class*. A reviewer will
  say your 0.978 "beats SOTA" reading is invalid — different protocol.
- **Damage:** kills any implicit SOTA claim; forces the paper to a narrower lane.
- **Honest defense:** the paper already states this explicitly and never claims
  leaderboard superiority; the SAR/CW comparison is apples-to-apples *within*
  the protocol. **Defensible only if you never let "0.978" sit near "M3DM 0.945"
  without the protocol caveat in the same breath.**
- **Residual risk:** a skimming reviewer still misreads it. Mitigation: the
  demarcation table + the scope paragraph must be impossible to miss.

### 2. "Gate E fails, so the flagship/transfer claim is unproven — the core thesis (gating helps) doesn't generalize." ⚠️ HIGH
- **The hit:** the *point* of reliability gating is robustness/transfer. Clean
  external transfer is a tie at best; a parameter-free mean beats your method.
  A reviewer: "so it only works where it's trained — that's not a contribution."
- **Damage:** challenges whether the method matters at all.
- **Honest defense:** reframe — the contribution is the *characterization* of
  WHEN gating helps (stress regime) and WHEN it doesn't (clean/redundant). The
  in-domain win + stress-regime recovery + the honest clean-transfer negative
  together are a *measurement* result, not a failed method. This is a strength
  IF framed as "we map the operating boundary," a weakness if framed as "we
  built a better fusion method."
- **Residual risk:** real. If a reviewer wants a method paper, this disappoints.
  It is a strong *analysis/measurement* paper, a weak *method* paper.

### 3. "A one-line confidence-weighted mean beats your method on clean transfer." ⚠️ HIGH
- **The hit:** CW (0.9349) > RGA+ (0.930) on clean 3D-ADAM. Your learned/gated
  method loses to averaging.
- **Damage:** "why add complexity if averaging wins?"
- **Honest defense:** mathematically expected — when modalities are clean and
  redundant, averaging IS near-optimal; gating's value is the stress regime,
  where it beats CW by up to +0.10. State the crossover, don't hide it.
- **Residual risk:** moderate. Survives only with the stress-regime framing.

---

## TIER 2 — Attacks that wound (force concessions, not fatal)

### 4. "Only ONE genuinely-paired vision benchmark carries the claim." MED
- MVTec 3D-AD is the only naturally-paired set with a positive headline; VisA/
  LOCO are derived-view proxies, Eyecandies failed, MulSen is cross-category OOD.
- **Defense:** honestly scoped; P3 (multimodal generalization) is explicitly
  marked PARTIAL, not claimed. **Concede:** generalization breadth is thin.

### 5. "Stress-regime win doesn't replicate — MVTec replication CIs cross zero." MED
- 3D-ADAM degradation win is significant; MVTec replication is only
  "directionally supportive." One significant dataset, not two.
- **Defense:** reported honestly as inconclusive replication. **Concede:** the
  stress claim rests largely on 3D-ADAM.

### 6. "The degradation is synthetic noise you injected, then detected." MED
- The stress-regime win uses a uniform-noise blend, not natural degradation.
- **Defense:** controlled degradation isolates the mechanism; stated as such.
  **Concede:** natural-degradation evidence is future work (Level-3 plan).

### 7. "Several theorems are operationalizations of known results, not new math." MED
- T1/T2/T4 formalize known insights; T7 is textbook Massart; GDR is suggestive
  (p=0.125, not significant).
- **Defense:** the *quantitative consequences* (KS-inflation criterion, stochastic
  dilution law, sample-complexity n*, tight router bound) are the novelty, and
  GDR is honestly reported as not-significant. **Concede:** no deep new theorem;
  these are useful applied bounds.

---

## TIER 3 — Attacks that sting (answerable, but show scars)

### 8. "Your history shows fixed bugs and a mislabeled-CW 'win' that didn't survive audit." LOW-MED
- The depth-codec bug, the fixed-split fake CIs, the RGB-mislabeled-as-CW
  comparator — all real, all caught.
- **Double-edged:** a hostile reviewer says "sloppy"; a fair reviewer says
  "exemplary self-correction." **Defense:** every bug was found, fixed, and
  documented by your own audits — that's integrity, not negligence. Lead with
  that framing.

### 9. "MulSen and Eyecandies both failed — you have a pattern of failed transfer." LOW-MED
- Three transfer attempts (Eyecandies, 3D-ADAM clean, MulSen) — none confirmed.
- **Defense:** each failure is *explained* (calibration shift; redundant
  modalities; cross-category OOD), which is more valuable than unexplained
  nulls. **Concede:** transfer is genuinely unsolved here.

### 10. "Reproducibility lives on an unstable exFAT drive; a parallel process pruned your reports." LOW
- Operational risk, not scientific. **Fix:** move to APFS/clean clone; all work
  is on GitHub now anyway.

---

## The single most dangerous misframing (avoid at all costs)

Letting anyone read the work as **"we built a multimodal fusion method that beats
SOTA / generalizes."** It does not, and every Tier-1 attack lands instantly on
that framing. The work survives — and is genuinely good — ONLY as:

> *"A rigorous measurement study that maps WHEN reliability-gated fusion helps
> (in-domain + differential-reliability stress) and WHEN it cannot (clean,
> redundant modalities, where averaging is optimal), with the operating boundary
> characterized theoretically and empirically, and every negative preserved."*

In that framing, Tier-1 attacks 1–3 become *the findings*, not the weaknesses.

---

## Net assessment of how much this hurts

| If submitted as... | Outcome |
|---|---|
| SOTA method paper | **Rejected** — Tier 1 sinks it |
| Top-tier method paper (NeurIPS/CVPR main) | **Borderline-reject** — Gate E + single-benchmark |
| Measurement / analysis paper (when-does-gating-help) | **Plausible accept** — Tier-1 attacks become contributions |
| Workshop paper | **Accept** |
| PhD thesis chapter | **Strong** — the honesty + boundary characterization is thesis-grade |

**Bottom line:** the results hurt you a lot under a method-paper framing and
barely at all under a measurement-paper framing. The science is the same; the
framing decides whether Tier-1 attacks are fatal or are your headline. Choose the
measurement framing, foreground the boundary characterization, and keep every
caveat in the same sentence as every number.
