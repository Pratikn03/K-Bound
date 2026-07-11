# Bring the long paper + book manuscript up to date (paste into fresh sessions)

**Goal.** Make `kbound.tex` (long, ~59pp) and `manuscript/main.tex` (216pp) consistent with the
**already-scored** v5 results and the short-paper fixes. **Do NOT re-score.** The single frozen
scoring pass already ran (short-paper fold-in); its verdicts live in
`experiments/kbound/results/.../results_source.json` and `scripts/uniform_scorer.py`. You are
*propagating* those same verdicts, not producing new ones. Score once, everywhere the same.

Run as TWO separate sessions (the manuscript is large): do `kbound.tex` first, then `manuscript/`.

## The frozen verdicts to propagate (identical in all three documents)
- **CI-robust beats-both (only these):** CIFAR-10-C Tent/EATA; ImageNet-C SAR (new this wave);
  the pooled/constructed three-source mixture. α=0.10, paired bootstrap B=5000, Holm-corrected.
- **No-harm:** Office-Home (point edge migrates to no-harm), iWildCam, RxRx1, PACS, CIFAR-10.1.
- **Withdrawn:** Camelyon17 Protocol-G beats-both (contaminated held-out; `id_val` pooled in) —
  reclassified no-harm. Do NOT reinstate it anywhere.
- **Diagnostic / no promoted win:** ImageNet-R (weak evidence, abstains).

## Fixes to apply to BOTH documents (already done in `kbound_short.tex` — mirror them)
1. **Lean count:** any "58 indexed theorems" -> **43** (verified `#check` count in `KBound/TheoremMap.lean`).
   Down-scope any "first machine-checked conformal coverage" -> "we are not aware of a prior...".
2. **Theorem stack:** Lemma 1 (disagreement reduction), **Theorem 1** (matched-evidence impossibility,
   label `lem:nonid` — now a `\begin{theorem}`), **Theorem 2** (frontier `thm:headline`),
   **Theorem 3** (certificate `thm:certificate`). Make every prose reference say "Theorem" for
   `lem:nonid` — no "Lemma~\ref{lem:nonid}" / no colliding "Theorem 2". (theory files: many share
   `paper/sections/theory_core_main.tex`, already fixed — verify the doc actually `\input`s it.)
3. **β operational note:** β is declared not estimated; β=0 recovers ATC/AGL/AETTA; larger β widens
   the abstain band monotonically.
4. **Descriptive-first protocol names:** "Camelyon17: held-out hospitals (Protocol G)", "iWildCam
   camera-trap deployment (H v2)", "Office-Home domain transfer (M v2)".
5. **New assets** (regenerate/point at, from the short paper): `figures/fig_regime_map.png`,
   `figures/fig_frontier_transition.png` (+ recovery, fa_coverage), `figures/fig_verdict_migration.png`,
   the **Assumption Audit** table, and the **primary-numeric-evidence** table.
6. **Aggressive (harsh-deployment) wave:** add it as a section/tables ALONGSIDE the benign results
   (it is a second claim — regime moves with the operating point — not a replacement).

## Reproducibility stamps (per main table)
script name + config (`research_lock/WIN_HUNT_v5_PROTOCOL_SHELL.yaml`), seed count, calibration/test
split, where ε was fit (out of fold), target labels used for offline audit only. Verdicts from
`scripts/uniform_scorer.py` + the per-condition bootstrap; scored once against the frozen bars.

## Order of work per document
1. `grep` the doc for: `58`, `beats.both.*[Cc]amelyon`, `Lemma~\ref{lem:nonid}`, stale numbers.
2. Apply fixes 1–4 (text), then fold in the frozen verdicts + wave (5–6), then repro stamps.
3. Recompile: `pdflatex ×2` (bibtex if it uses it); `grep -c '??'` must be 0.
4. Do NOT change any scored number — reuse `results_source.json` verbatim.

## Manuscript-specific note
`manuscript/main.tex` is 216 pp and predates the v5 wave entirely (Jul 1). It also likely carries the
OLD Camelyon claim, old theorem numbering, and "58" — fix those first, then fold in the wave. It is a
thesis/reference artifact, not the TMLR submission (that is `kbound_short.pdf`), so it can lag safely
if time-constrained.
