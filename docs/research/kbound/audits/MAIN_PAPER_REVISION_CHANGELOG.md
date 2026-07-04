# MAIN_PAPER_REVISION_CHANGELOG — kbound_short.tex (2026-07-03)

No numerical value was changed anywhere. No evidence strength was upgraded. All withdrawn-claim
disclosures survive. Files touched: `kbound_short.tex`, `kbound_short_appendix.tex` (4 lines).

## 1. Abstract (replaced)
Old: 5-paragraph abstract with dataset lists, pooled-stream numbers (n=359, regret 0.0183 …),
estimator-family list, citations. New: 2-paragraph, ~205-word target-style abstract. Universal-gate
result retained as one clause **without numbers** ("a single universal gate beats both fixed
policies across seven natural sources; pre-registered pooled heterogeneous deployment").
POEM/AETTA referred to as "protocol-matched POEM-style and AETTA-style baselines." Closing
sentence: "safety layer …, not a universal accuracy booster." All removed detail remains in the
experiments sections and the long paper.

## 2. Introduction (rewritten to the five jobs)
- Jobs marked in text: (1) TTA helps or hurts; (2) prior question is committing; (3) some cases
  unknowable; (4) exact frontier |M|>β + KGA operationalizes; (5) evidence regime map.
- Contributions compressed 6 → 5 bullets; dataset enumerations moved out (pointer to regime
  table); duplicate regime paragraph deleted.
- Regime-summary table restructured to Regime | Example | Expected | Result with new rows
  "Weak evidence" and "Heterogeneous deployment (pooled seven-source … universal-gate win,
  pre-reg.)".

## 3. Theory section (restructured; contents preserved, one theorem split into two)
- Added "K-Bound in one sentence" box.
- Added "Four quantities that must not be confused" table (M/γ/β/ε) + explicit "ε is not an
  estimate of β" line.
- **Theorem title changes:**
  - Lemma "Non-identifiability forces abstention" → **Theorem 1 "Matched evidence forces
    abstention"** (env lemma→theorem; label `lem:nonid` kept; statement unchanged in substance,
    "in the declared class" made explicit).
  - Theorem "K-Bound certificate and benefit-sign frontier" (consolidated) → **split** into
    **Theorem 2 "Exact benefit-sign frontier"** (label `thm:headline` kept; sign(Δ)=sign(M+γ) and
    |M|>β displayed) and **Theorem 3 "Finite-sample adapt/freeze/abstain certificate"** (new label
    `thm:certificate`; coverage display, decision rule, both marginal error bounds displayed).
- Added 3-row region interpretation table (M>β adapt / M<−β freeze / |M|≤β abstain) after Thm 2.
- Added FA_u-vs-FA_c **warning box** directly below Theorem 3 (content previously inline in the
  consolidated theorem body — moved, not weakened).
- Added per-theorem explanation format: one-sentence intuition, "why it is true" in 3 numbered
  steps, scope/assumption note (×3 theorems).
- Added worked numerical example A (M=0.18, β=0.10 → ADAPT) and B (M=0.07, β=0.10 → ABSTAIN).
- Added proof-dependency chain ("How the pieces fit": reduction → matched evidence → Thm 1 →
  split-observable decomposition → Thm 2 → coverage → Thm 3).
- "Full proofs…" paragraph consolidated into a compact "Extensions" paragraph (one-bit +
  dominance polytopes, minimax optimality, family-wise routing, anytime streaming, Wave-4
  dichotomies → appendix/long paper). The real-data anytime demonstration sentence (iWildCam
  35,370 images, FREEZE at window 6, anytime FA 0, ties oracle, pre-registered) retained
  verbatim.
- β/γ/ε disambiguation sentences from the old consolidated theorem body now live in the
  four-quantities table + Thm 3 scope note (no meaning change).

## 4. Reference retargeting (mechanical)
- Certificate-meaning uses of `\ref{thm:headline}` → `\ref{thm:certificate}`: method section
  ("L>0 implies Δ>0"), metrics paragraph ("bounds the unconditional probability").
- Frontier-meaning uses kept at `thm:headline` (intro, threshold-derived paragraph, limitations
  ×2).
- "Lemma~\ref{lem:nonid}" → "Theorem~\ref{lem:nonid}" at all sites (setup, method, limitations).
- Limitations "sign bracketing impossible" now cites Theorems 1+2.
- Appendix header: "consolidated headline theorem" → the three-theorem structure (4-line edit in
  `kbound_short_appendix.tex`).

## 5. Empirical claim wording changes (no numbers touched)
- "POEM and AETTA are faithful ports of their published protectors" → "protocol-matched
  POEM-style and AETTA-style baselines: ports … with documented simplifications" + "Official
  per-sample POEM and dropout-based AETTA reproduction remains a camera-ready faithfulness
  check." (§ head-to-head; sentence also already present in Limitations — both retained.)
- Head-to-head table caption: added "POEM/AETTA rows are protocol-matched -style ports, not
  official reproductions."
- Verdict sentence: "vs. both POEM and AETTA" → "vs. both ported baselines".
- Guarantees table row: "Mixed vs POEM/AETTA" → "Mixed vs protocol-matched POEM-style/AETTA-style
  ports".
- Conclusion: "Wins over trivial policies and POEM/AETTA" → protocol-matched wording + adds the
  pre-registered universal-gate pooled deployment clause.
- Reproducibility: added precise Lean sentence ("kernel-checked Lean 4 formalization of the
  algebraic and finite-testing core … 52 theorems, no sorry/admit/project-defined axioms;
  deployment-facing assumptions remain explicit assumptions outside the formalization") —
  claim verified by source scan 2026-07-02.

## 6. Moved / removed
- Removed from intro: duplicated dataset lists (×2), long specialization paragraph (compressed),
  "Headline empirical evidence" bullet's inline numbers (table pointer instead).
- Removed from abstract: dataset lists, all numbers, estimator-family sentence, two citations
  (retained in intro/related work).
- Nothing moved out of the experiments sections; camera-protocol subsection left in place
  (flagged as candidate to move to appendix at venue formatting time — not done here to avoid
  breaking `edge/` table inputs).
- No appendix item was deleted.

## 7. Earlier same-day fold-in (pre-restructure, same file)
Universal-gate paragraph (§ natural shifts), head-to-head replication sentence
(WIN_HUNT_v3 Arm F), anytime real-data sentence, ImageNet-R τ′ re-analysis sentence
(Limitations), single-dataset qualifier in "Real-shift scope", guarantees-table row
"Pooled 7-source beats-both, one universal gate".

## 8. Build
Compiled by the author 2026-07-03 (`pdflatex -interaction=nonstopmode kbound_short.tex` ×2,
TeX Live 2025 homebrew): **22 pages**, 1,401,328 bytes — inside the 20–24 page target.
Unresolved references: **0** (`grep -c "??"` on the log). No errors; only benign LaTeX font-shape
substitution warnings (`T1/lmr/m/scit`, `T1/lmr/bx/sc` — small-caps italic/bold not shipped by
Latin Modern; substituted automatically). Pre-revision page count not recorded (pre-revision PDF
superseded before measurement).
