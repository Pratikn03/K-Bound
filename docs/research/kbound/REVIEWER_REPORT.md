# K-Bound — Senior / Area-Chair Review (adversarial, 4-referee panel)

**Process.** Four independent referees (theory, experiments, novelty/positioning, presentation/integrity)
read the compiled paper, the proof sources, and the result JSONs. This report is the area-chair
synthesis. Referee scores: **Theory 4/10 · Experiments 3/10 · Novelty 4/10 · Presentation 3/10.**

**One correction to the panel up front (verified by the AC):** two referees claimed the headline result
file `decisive_tta_results.json` and `cifar10c_65cells.csv` were *missing/fabricated*. **They are not** —
both exist at `experiments/kbound/results/` (85 KB and present), and fuller ImageNet-C runs
(`imagenetc_noise_full`, `imagenetc_noiseblur`) exist too. The referees searched a wrong relative path.
The real issue is **path-consistency and artifact hygiene**, not a missing/fake result. This downgrades
that charge from Critical-fabrication to Medium-housekeeping.

---

## Verdict & rejection probability (honest)

| State | Reject probability at a top venue (NeurIPS/ICML/ICLR) |
|---|---|
| **As-is, submitted today** | **~95% (effectively certain)** — two independent desk-reject triggers fire before review |
| After **mechanical** fixes only (anonymize, trim to page limit, fix paths) | **~80%** — survives desk-reject, but novelty collisions + non-standard headline remain |
| After **substantive** fixes (re-attribute/reposition novelty, run the decision baselines, standard-protocol results with proper CIs, multi-seed) | **~50–60%** — a genuine borderline at a top venue; solid accept at a second-tier venue or workshop |

This is consistent with the standing assessment: the paper is a real **~7.5–8**, not a 9–10. Nothing below
is fatal to the *ideas*; the gaps are about **what's claimed vs. shown**, **anonymity/length mechanics**,
and **collision with very recent prior work**.

---

## A. Desk-reject mechanics (fix these first — they cause rejection *before* the science is read)

| Gap | Where anyone finds it | Who catches it | Rejection risk |
|---|---|---|---|
| **Double-blind violation**: real author name + "Independent Researcher" + date "Working draft v0.4" | `kbound.tex` L29–30; fulltext L4–5 | Program chairs, automatic check | **Critical / desk reject** |
| **De-anonymizing GitHub URL** `github.com/Pratikn03/AutoML_Flagship_V8` | Reproducibility §; fulltext L909 | Anyone who reads §Repro | **Critical / desk reject** |
| **Legacy-identity leak "ELARA / RGA", `src/elara/…`, `vendored_from_elara/`, `kbound_paper/…`** in prose, captions, Appendix A | fulltext L88, L379, L991; `kbound.tex` L421; Appendix A | Any reviewer; links to your prior project | **Critical (anonymity) / High (professionalism)** |
| **36 pages, proofs in the main body (~20 pp body)**; conference limit is ~9 pp ex-refs/appendix | whole document; §5 proofs pp. 5–10 | Format check / any reviewer | **Critical / desk reject (length)** |
| **Hardware leak "Apple-silicon MPS"** in body + captions (weakly de-anonymizing, signals laptop-scale) | fulltext L660, L788, L914 | Reviewers | High |
| **Path-consistency / artifact hygiene**: paper-cited result paths don't resolve from the paper dir; `._*`/`.pyc` junk shipped | §Repro; result dirs | Reproducibility checker | Medium (was mis-flagged as "missing file") |

## B. Novelty collisions — the true scientific rejection axis (two referees independently)

| Gap | Where in paper | Prior work that causes it (VERIFY arXiv IDs before citing) | Rejection risk |
|---|---|---|---|
| **Multi-candidate identifiability (Thm 17 / Prop 4) re-derives a known result**: `2A_ij−1=b_i b_j`, rank-one agreement covariance, anchor fixes sign | §C.6 multicandidate | **Parisi, Strino, Nadler, Kluger, PNAS 2014** (arXiv 1303.3257); **Jaffe, Nadler, Kluger, AISTATS 2015** (arXiv 1407.7644); Platanios et al.; Dawid–Skene 1979 — currently **uncited** | **Critical (novelty)** |
| **The "decide whether to adapt before adapting" framing is an already-named, benchmarked problem** | Abstract; §1; contribution (1) | **"Continually Adapt or Not (CAN)?" / Adapt-or-Skip, NeurIPS 2025** — *verify the exact ref* | High |
| **Closest competitor uncited**: label-free, anytime-valid risk monitoring built for TTA = your Appendix-B e-process niche | §2.2; Appendix B (anytime certificate) | **Schirmer, Jazbec, Naesseth, Nalisnick, "Monitoring Risks in Test-Time Adaptation," NeurIPS 2025**, arXiv **2507.08721** (this one is confirmed real) | High |
| **"No fixed TTA policy is robust; rankings invert across regimes"** independently established | §7 online; Table 5 | "Tempora…" (2026) — *verify* | Medium-High |
| **Breadth reads as "survey of small results"** (20 theorems, many self-labeled "corollary / definitional / classical") | §5 + Appendix C | structural | High |
| **Disagreement-region sign vs disagreement-discrepancy** (you do differentiate ordinal-vs-cardinal — partially survives) | §5.5; §2.3 | Rosenfeld & Garg, NeurIPS 2023 (arXiv 2306.00312) | Medium |

## C. Theory gaps

| Gap | Where | Rejection risk |
|---|---|---|
| Multi-candidate binary case hides a **per-class symmetric-accuracy assumption**; the **sign anchor is as untestable as the "calibration crutch"** it claims to remove → headline "no calibration assumption" is misleading | Def 5, Thm 17 | Major |
| **Rate theorems (Thm 19–20) solve the *labeled* paired-benefit mean-estimation problem** — once you have calibrated benefit samples the label-free difficulty is gone; "rate-optimal certificate" sidesteps the actual problem | Thm 19/20, Prop 9 | Major |
| **Unification "reach" theorem is near-definitional** (paper admits it); the per-family iff converses all reduce to "interval straddles 0 ⇒ Theorem 1" — one idea reused | Thm 16; Thms 14/15/18(ii) | Major |
| **Internal contradiction**: the reach table labels the conditional-independence row "known (Steinhardt–Liang)" while §C.6 sells the same structure as the novel contribution | Prop 3 vs Thm 17 weight-note | Major |
| **"Numerically validated" used as if it discharges proofs**; the multiclass extension is *cited, not proved*, yet "the validator confirms it" | §C.6 multiclass; throughout | Major |
| **Theorem-numbering chaos / broken cross-ref**: body "(Theorems 6–9)" points at the anytime certificate; labels disagree across abstract/body/appendix | §5 vs App B; throughout | Major (cumulative trust) |

## D. Experiment gaps

| Gap | Where | Rejection risk |
|---|---|---|
| **Headline "beats both" only on a custom pre-registered stress grid**, not standard CIFAR-10-C; on the **standard per-corruption** protocol KGA only **ties**; it also **ties SAR** (the strong baseline) | §7 Tables 6 vs 7; Table 7 SAR row | High (cherry-pick / non-standard-protocol) |
| **No label-free decision baselines actually run** (AETTA, ATC, Agreement-on-the-Line as adapt/freeze rules); no CoTTA/SHOT/TTT | §2.3; Props 6–8 (argued, not run) | High |
| **Reported significance (p<0.002, |d|>1.4) comes from bootstrapping a *synthetic* 200-condition stream**, not the real per-condition data; no multiple-comparison correction | §7; `decisive_tta_cis` | High |
| **Camelyon17 single-seed, ~94% subset, B>0 so KGA = always-adapt** → the certificate is never exercised (no error bars, no freeze/abstain tested) | §7.1 Table 8 | High |
| **The one real harmful-regime ADAPT decision was a false-adapt** (adapt precision 0.0) — undercuts the "0% false-adapt" safety narrative on real data | §7 harmful; `kbound_harmful_results.json` | Medium |
| **"ImageNet scale" rests on an 8-condition smoke where KGA abstains 100%** (fuller `imagenetc_noise_full`/`noiseblur` runs exist but the paper table still shows smoke + "(pending)") → over-claim + sync gap | §7.1 Table 9 | High |
| **Table 6 numbers (65 cells, Tent 0.626) don't match the file a reviewer opens first** (`cifar_tent_results.json` = 44 cond., 0.585); the 65-cell CSV exists but isn't the obvious one | §7 Table 6 | Medium-High |

## E. Presentation / integrity (beyond the desk-reject items in A)

| Gap | Where | Rejection risk |
|---|---|---|
| **Overclaim phrasing**: §C.6 title "Removing the calibration crutch" (body concedes it *trades* for an independence condition); "beats both" not always SAR-caveated in-sentence | §C.6 title; abstract/§7 | Medium |
| **19 figures**; synthetic illustrations (phase diagram; validation plots) sit near real-data claims without a uniform "Synthetic" label | Figs 1–19 | Medium |
| **Repro statement "every number backed by a manifest" is not literally true** until paths resolve and seed counts are stated honestly (several headlines single-seed / 2-repeat) | §Repro | Medium |

---

## The 3 prior works you MUST cite + differentiate (or you will be rejected on novelty)
1. **Jaffe–Nadler–Kluger (AISTATS 2015)** + **Parisi et al. (PNAS 2014)** — your multi-candidate estimator *is* theirs. Re-attribute; reposition your contribution as the *decision use* + the *checkable overdetermination diagnostic*, not the estimator.
2. **Schirmer & Jazbec et al., "Monitoring Risks in TTA" (NeurIPS 2025, arXiv 2507.08721)** — same niche. Differentiate: theirs is reactive post-hoc monitoring; yours is proactive sign-certification with an abstain region + impossibility floor.
3. **"Adapt-or-Skip / CAN" (NeurIPS 2025)** — your framing question, already benchmarked. Cite and either beat its baselines or carve a clear theoretical delta.

## Prioritized fix roadmap
**Tier 0 — mechanical (hours, unblocks everything):** anonymize (author/date/GitHub/ELARA/MPS); move all proofs to an appendix → ~9-pp body; fix cited paths + strip `._*`/`.pyc`; one global theorem-renumbering pass.
**Tier 1 — novelty survival (days, writing):** cite + reposition Parisi/Jaffe–Nadler; add + differentiate Schirmer–Jazbec and CAN; compress Appendix C's 20 theorems to one reach-table + the 2–3 genuinely new results; remove "first" claims that don't survive.
**Tier 2 — empirical credibility (GPU):** run AETTA/ATC/AoL **as decision baselines** head-to-head; report **per-condition** CIs with correction on the **standard** CIFAR-10-C protocol (keep SAR first-class); finish the real ImageNet-C grid and sync the table; ≥3-seed Camelyon in a regime where harm actually occurs.

## Bottom line
The ideas (trichotomy + abstain region + impossibility floor + checkable diagnostic) are genuine and likable.
The paper is **not submittable today** (anonymity + length = automatic desk reject) and, even fixed, faces a
**hard novelty problem** because its most estimator-like result is a known 2014/2015 method and its framing
collides with two NeurIPS-2025 papers it doesn't cite. Address Tier 0–1 and it's a defensible borderline
submission; add Tier 2 and it's a solid paper — realistically a strong second-tier accept, ~7.5–8, with the
assumption-free theorem (Conjecture 1) remaining the only path to a true 9.
