# K-Bound: theorem, novelty, and bibliography review

Original review date: 2026-08-30. Revision follow-up: 2026-08-31.
Current disposition: **the scoped manuscript and public-runtime repairs are implemented;
component verification is separate from the still-unissued clean-source release seal**.

Revision note: Sections 1–7 retain the original review of the 35-page artifact identified below.
Section 8 records the first T1/T2 implementation. Section 9 records the subsequent
theorem, novelty, bibliography, implementation, packaging, and component checks.
Section 10 records the later measurable-foundation work and release-boundary review.
Earlier sections retain their dated findings and hashes; they are not descriptions
of subsequently rebuilt output bytes.

The original review found no counterexample to the eight numbered statements in
the then-maintained long manuscript under their intended assumptions. It identified
population/empirical, inferential, figure and bibliography repairs, subsequently
addressed below. Section 10 separately identifies an invalid sufficiency assertion
in an excluded historical extension, not in those eight compact statements.

At the original review stage, only this report was added; no manuscript, PDF, Word file, experiment, dataset, Git history, or release seal was changed. The later source and export repairs are recorded separately in Sections 8 and 9.

## 1. Reviewed artifact and limits

The reviewed artifact is the 35-page [maintained long PDF](/Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/kbound/kbound_tmlr.pdf), titled *K-Bound: When Is Label-Free Adaptation Knowable?* Page numbers below refer to this PDF.

```text
SHA-256: 7c100a60d4dc9de8c62b094d504b2008d3b80b1045b67545cfb7ef733a0d3c27
Size:    674520 bytes
```

The review combined a manual proof audit, an independent mathematical review, a primary-literature comparison, two disjoint bibliography checks covering all 60 records, PDF citation-link inspection, and visual checks of the mathematical figures. Exact-rational calculations checked the derived frontier and audit-radius formulas on 441 feasible parameter pairs. These calculations are sanity checks, not substitutes for proofs.

The following maintained sources were still marked `compressed,dataless` by macOS during this review:

- [Shared manuscript body](/Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/kbound/kbound_submission_body.tex).
- [Population-theory source](/Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/kbound/paper/sections/theory_core_main.tex).
- [Certificate source](/Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/kbound/paper/sections/theory_certificate.tex).
- [Bibliography source](/Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/kbound/paper/references_kbound_expanded.tex).

Consequently, this review does not certify the current TeX-to-PDF correspondence, an updated short PDF, a Word export, a Lean build, or an end-to-end release rebuild. Bibliographic metadata was checked for every printed entry; the proofs and empirical claims of all 60 cited papers were not independently reproduced. Novelty conclusions are judgments based on the closest literature checked, not a guarantee that no related result exists elsewhere.

## 2. Decision summary

| Area | Review result | Submission consequence |
|---|---|---|
| Eight numbered mathematical statements | No proof failure found within the fixed-model, binary, rich-class scope and the stated coverage premise | Retain the core argument; make the scope explicit |
| Population-to-empirical bridge | Coverage of a measured cell outcome is not automatically coverage of population benefit | Resolve the estimand distinction before claiming population-risk certification |
| Empirical inference language | Independent-coordinate sign invariance is not established by exchangeability alone; bootstrap coverage is nominal | Tighten assumptions and the meaning of “exact” and “95% family” |
| Novelty | Defensible narrow contribution; the general impossibility and interval-decision principles have substantial precedent | Add the closest missing comparison and sharpen the existing comparisons |
| Bibliographic records | 54 match; 2 materially wrong/mixed; 1 author/version problem; 3 typography/edition normalizations | Correct or resolve six entries |
| Citation use in compiled PDF | 60 bibliography destinations, 40 cited destinations, 20 without citation links, 0 dangling citation destinations | Review relevance; do not automatically delete records |
| Source synchronization and release | Not verified because relevant source bytes are unavailable locally | Still a separate release blocker |

The appropriate conclusion is not “the paper is wrong,” nor “the paper is now 9.5.” It has a defensible core and specific repairable gaps. A numerical quality score would not replace these checks.

## 3. Mathematical statement audit

Write `d = μ_T(D) > 0`. The population argument fixes the binary predictors, the input-observation law, the score, and the source information while allowing target label kernels to vary. Its necessity claims require the full rich class described in the manuscript.

| Statement | Location | Verdict | Reason and essential qualification |
|---|---|---|---|
| Lemma 1: label-kernel freedom | p.6 | Valid | On disagreement, allocate conditional probability `η(x)` to the candidate's binary prediction and `1−η(x)` to the frozen prediction. Inputs and fixed-model outputs are unchanged. Measurability and a fixed input-observation process are required. |
| Lemma 2: disagreement reduction | p.6 | Valid | Losses cancel outside disagreement. Exactly one of the binary predictions is correct on disagreement, giving `Δ = 2d(M+γ)`. Ordinary source calibration is not needed for this identity. |
| Theorem 1: interior impossibility | pp.6–7 | Valid for feasible margins | For `abs(M)<β`, choose `0<t<min(β−abs(M),1/2)` and constant correctness kernels `1/2±t`. Their residuals are `±t−M`; both are admissible and give opposite nonzero benefits with identical evidence. Explicitly state `M∈[−1/2,1/2]`. |
| Corollary 1: abstention lower bound | p.7 | Valid | Identical evidence gives identical action probabilities. The negative world bounds ADAPT probability by `α`; the positive world bounds FREEZE probability by `α`. Thus abstention is at least `1−2α`. This is under separate marginal-error constraints, not a universal regret-optimality result. |
| Proposition 1: boundary and closed band | p.7 | Valid under strict semantics | At `abs(M)=β`, the admissible zero-benefit world precludes a uniformly strict directional claim. At `M=β=0`, all benefits are zero. Neither conclusion says abstention has lower task risk than committing at a tie. |
| Theorem 2: exact frontier | pp.7–8 | Valid | The residual bound gives sufficiency; the rich label-kernel class gives necessity and maximality. The narrower-class caveat must remain. This is a population-information result, not a finite-batch estimator guarantee. |
| Theorem 3: audit floor | p.9 | Valid on a fixed nonempty evidence-law fibre | All admissible worlds induce the same audit distribution. A sequence of attainable absolute residuals approaching their supremum yields the lower bound by continuity of probability. The constant supremum audit is a fibre-wise oracle benchmark, not an algorithm that learns an unknown evidence law. |
| Theorem 4: interval-to-action certificate | p.10 | Valid conditional implication | A false strict action requires interval noncoverage. This proves the stated marginal bounds, but does not establish coverage, conditional error among accepted updates, simultaneous candidate protection, or time-uniform protection. |

### 3.1 Independent identified-set calculation

The definitions imply

\[
M\in[-\tfrac12,\tfrac12],\qquad
\gamma\in[-\tfrac12-M,\tfrac12-M].
\]

Intersecting the second interval with the declared bound `[-β,β]` gives the exact feasible benefit set, under the full class:

\[
\Delta(\mathcal C_\beta)
=2d\,[\max(-\tfrac12,M-\beta),\;\min(\tfrac12,M+\beta)].
\]

All values in this interval are attainable by constant correctness kernels on disagreement. Its lower endpoint is strictly positive exactly when `M>β`; its upper endpoint is strictly negative exactly when `M<−β`. This proves the strict frontier while keeping the feasible range visible. Adding this expression would make the existing argument easier to check; it is not a newly verified empirical result.

For the same fixed fibre, the absolute residual radius is

\[
\Gamma_z(\mathcal C_\beta)=\min\{\beta,\tfrac12+|M|\}.
\]

Therefore Eq. (5), `Γ_z=β`, is correct with its printed restriction `0≤β≤1/2`. Do not silently remove that restriction. Both signs of `β` need not be feasible; attaining one endpoint in absolute value is enough.

### 3.2 Other checked implications

- **Split-conformal rank, Eq. (8), p.13:** the rank and `+∞` case are correct when calibration and deployment residuals are jointly exchangeable under the fitting procedure. Independent development fitting is one standard route. The controlled leave-one-out radii are already correctly distinguished from an exact split-conformal or jackknife+ guarantee.
- **Multiclass identity, p.5:** `Δ=P_T(D)(p_a−p_0)` is correct for multiclass 0/1 risk. It does not imply `p_0=1−p_a`, extend the binary frontier automatically, or prove a population guarantee for macro-F1.
- **Joint false-commit event:** Theorem 4 also gives `P(false ADAPT or false FREEZE)≤α` for the one covered scalar, because the union lies inside the same noncoverage event. This optional strengthening does not create a simultaneous or sequential guarantee.
- **Risk alignment, Definition 2:** absence of opposite nonzero signs is weaker than uniform strict-sign identification. A fibre containing zero and positive benefit satisfies the former but not a strict positive guarantee. The boundary proof handles this correctly; terminology should make the distinction clear.
- **Formalization, Appendix K:** the PDF discloses a limited Lean scope and incomplete full-foundation coverage. This review does not convert that disclosure into a fresh formal-verification result.

## 4. Required mathematical and inferential repairs

### T1 — High: separate population benefit from a measured cell outcome

Locations: Eq. (1) and contribution 4, p.2; §6.2 and Eqs. (6)–(7), p.11; Table 2, p.12; §9.6, p.22; conclusion, p.24.

The population target is an expected risk difference. The empirical residuals are computed from finite-cell accuracy or other measured metric differences. A prediction interval covering the latter need not cover the former. The present assertion that both layers simply share the same target is not enough to bridge that gap.

**Counterexample.** Let `f_0≡0`, `f_a≡1`, with `P_T(X=a)=0.45` and `P_T(X=b)=0.55`. Let the deterministic labels be `Y(a)=1`, `Y(b)=0`. Then the fixed population benefit is `−0.10`. A one-item cell has measured benefit `+1` or `−1`, perfectly predictable from its input. A predictor equal to that cell outcome, with radius zero, has perfect cell-outcome coverage. It nevertheless selects ADAPT with probability `0.45`, and every such action is harmful relative to the fixed population risk.

This is not a counterexample to Theorem 4 for the cell scalar. It is a counterexample to replacing population coverage with cell-outcome coverage. If a benchmark explicitly defines its target law as the empirical unit distribution, that is another valid estimand—but it must not be presented as a guarantee for a wider unseen population.

**Required resolution:** distinguish `Δ_pop` and `Δ_cell` and choose the interpretation explicitly. The conservative current-evidence repair is to scope empirical certification to the declared cell outcome. For a population claim, add an appropriate sampling-error bound. If both

\[
P(|\widehat\Delta-\Delta_{\rm cell}|\le\varepsilon)\ge1-\alpha,
\qquad
P(|\Delta_{\rm cell}-\Delta_{\rm pop}|\le b)\ge1-\delta
\]

hold on the same probability space, then the triangle inequality and union bound give population coverage at least `1−α−δ` with radius `ε+b`. No independence between those two events is needed. However, proving the second bound requires the actual sampling, clustering, candidate-selection, and metric assumptions; it is not supplied by this algebra.

### T2 — High for inference: specify the sign-flip invariance assumption

Locations: inference discussion, pp.15–16; Table 7 and its explanation, p.19.

The wording “sign symmetry and exchangeability” is not sufficient if it means symmetric marginal distributions, global sign symmetry, or exchangeability alone.

**Counterexample.** Let all nine location gaps equal the same random sign `S`, where `P(S=1)=P(S=−1)=1/2`. Their joint law is exchangeable and globally sign-symmetric; each marginal has mean zero. An all-positive vector occurs with probability `1/2`, yet independent-coordinate sign enumeration gives a one-sided value of `1/512`. Calling that a level-correct exact test would be anti-conservative.

**Required resolution:** state invariance under every coordinate-wise sign change, or a sufficient condition such as independent symmetric cluster contrasts, potentially conditional on the shared trained checkpoints. Acknowledge that this assumption is not established by the number of locations. Without it, retain the numerical outputs as sign-flip sensitivity calculations, not a verified exact inferential guarantee.

The two-way product-bootstrap intervals in Table 7 should be described as **nominal** 97.5% component intervals and a **nominal** 95% Bonferroni family. Bonferroni transfers valid component coverage; it does not prove bootstrap coverage on five checkpoint rows crossed with nine locations. This finding concerns interpretation, not evidence that the stored numeric calculations are arithmetically wrong.

### T3 — Medium: call γ a residual unless “drift” is additionally justified

Locations: Assumption 1/Eq. (3), p.5; §§4–5; Table 2.

Ordinary source calibration does not imply calibration conditional on disagreement. Therefore the defined `γ=E_T[η_a−s|D]` can be nonzero even when source and target distributions are identical.

**Counterexample.** Let `X=a,b` be equiprobable, `P_T=P_S`, `f_a≡1`, `f_0(a)=0`, `f_0(b)=1`, `Y(a)=1`, `Y(b)=0`, and `s≡1/2`. The score is perfectly source-calibrated in the ordinary sense. Yet `D={a}`, `M=0`, `γ=1/2`, and `Δ=1/2`, with no distribution shift.

**Recommended repair:** use “target disagreement-conditional calibration residual” for `γ` and “declared residual bound” for `β`. The proofs only require a bounded measurable score. If a literal source-to-target label-kernel difference is intended, define an appropriate source reference; for example, `s(x)=P_S(f_a(X)=Y|X=x)` makes that interpretation explicit. Source calibration conditional on disagreement would remove the no-shift counterexample, but does not by itself settle every covariate-reweighting interpretation.

### T4 — Medium: correct Figure 2's infeasible axis

Location: p.8, visually verified.

The figure labels its horizontal axis as raw `M`, uses ticks from `−3` to `3`, and places the band boundaries at `±1`. But Eq. (3) implies `M∈[−1/2,1/2]`. Its displayed strict-commit regions are therefore infeasible on the literal scale.

**Required repair:** either plot a feasible range, for example `M∈[−0.5,0.5]` with `β=0.2`, or identify the axis as a normalized schematic `M/β` with `β>0`. In the normalized version, the displayed band has width 2 in axis units, not `2β`; update the width annotation and caption consistently. Do not merely change the axis label while leaving contradictory units.

### T5 — Medium: separate fallback serving behavior from certified FREEZE

Location: Table 14, p.30, compared with Definitions 1 and 3 and Algorithm 1.

Missing features lead to “abstain or freeze,” and too-small batches to “wait or freeze.” Under the paper's semantics, FREEZE is a supported negative-benefit conclusion. Invalid evidence cannot support that conclusion merely because the frozen predictor remains in service.

**Required repair:** record `ABSTAIN/unavailable` while retaining `f_0`, unless a valid negative interval supports certified FREEZE. The serving state and the evidential action are separate fields. Apply this distinction consistently to the paper and, once accessible, the implementation logs.

### T6 — Medium, provenance unresolved: Figure 3 needs its source and scope

Location: p.11, visually verified.

The graphic says “123 real tasks,” uses `B` rather than `Δ`, and labels a band “conformal” with radius `0.049`. Its caption does not identify the panel, residual split, or assumptions supporting that label. This is particularly important because the text correctly withholds exact-conformal status from the controlled leave-one-out construction.

**Required resolution:** recover the figure's data/generator provenance and current-panel correspondence. Name the sampled outcome and calibration design, and use consistent notation. Until that is checked, describe it as a descriptive calibration illustration rather than proof of exact conformal or population coverage. This review does not establish that the figure's points or radius are wrong; their current provenance remains unverified.

### T7 — Lower priority: precision edits that protect the valid core

- Restrict “any margin” in Theorem 1 to feasible `M∈[−1/2,1/2]` and state measurability explicitly.
- Clarify that Definition 2's no-opposite-sign condition does not resolve zero-versus-positive boundary fibres.
- Prefer “exact fibre-wise audit floor” to “exact minimax” unless the minimax objective and information available to the audit are explicitly defined.
- State explicitly that repeated deployment and changing baselines require new sequential-validity analysis. Predeclaring that policy alone does not supply exchangeability or time-uniform safety.
- Preserve the distinction between `β` and `ε`, binary theory and multiclass evaluation, marginal and conditional errors, and strict-direction evidence and task-risk utility. These distinctions are already substantially improved in the current manuscript.

## 5. Novelty review

### 5.1 Defensible conclusion

The checked literature does not establish that this manuscript's exact combination of fixed-candidate disagreement analysis, declared residual class, strict boundary semantics, and fibre-wise audit floor is identical to an earlier theorem. That is a bounded literature-review finding, not a certificate of originality.

The broader ideas are not new by themselves: unlabeled target data can leave label-dependent risk unidentified; learning whether to adapt has precedent; covered intervals support one-sided decisions; and uncertainty can motivate retaining a baseline. The defensible contribution is an explicit, task-specific characterization and separation of these issues. The current §2.5 already credits the standard interval/risk-control logic; keep that credit rather than reintroducing a first-of-its-kind claim.

The strongest technical element is the exact characterization under a stated realizable class, not just the arithmetic rule `|M|>β`. Necessity needs the label-kernel construction. However, the construction and the audit-floor proof remain short and rely on established non-identifiability reasoning. I would not describe the paper as introducing a new general theory of distribution-shift safety on this evidence.

### 5.2 Closest primary comparisons

The distinctions in the last column are review inferences from the cited works and the current K-Bound manuscript.

| Prior work | Established overlap | Distinction and required treatment |
|---|---|---|
| Ben-David, Lu, Luu and Pál, *Impossibility Theorems for Domain Adaptation*, AISTATS 2010 | Section 4 constructs indistinguishable adaptation tasks where reweighting helps in one and harms in another | **Missing closest foundational citation.** Compare its learnability/reweighting setting with K-Bound's fixed-candidate paired benefit and bounded-residual class. The general help-versus-harm indistinguishability idea cannot be claimed as new. [Primary paper](https://proceedings.mlr.press/v9/david10a/david10a.pdf) |
| Steinhardt and Liang, 2016 | Unlabeled risk estimation can become possible under conditional-independence structure, with a label-permutation caveat | Already cited. Explain that such structure narrows the compatible class and may invalidate K-Bound's richness premise; the results are compatible, not contradictory. [Primary paper](https://arxiv.org/html/1606.05313v1) |
| Mutlu et al., Informed Adaptation, CVPR Workshops 2026 | Learned topological features predict whether an update is useful | Already cited and acknowledged as routing precedent. K-Bound's claim must rest on its stated validity structure, not on inventing learned update acceptance. Publication metadata and the accessible primary summary were checked; direct full-PDF retrieval was unavailable in this review. [CVF record](https://openaccess.thecvf.com/content/CVPR2026W/ABAW/html/Mutlu_Topology-Guided_Test-Time_Adaptation_via_Persistent_Homology_From_Affective_Behavior_Analysis_CVPRW_2026_paper.html) |
| Schirmer et al., *Monitoring Risks in Test-Time Adaptation*, NeurIPS 2025 | Sequential risk monitoring includes guarantees under explicit assumptions, not merely an engineering alarm | Already cited. Compare its running adapted-risk/source-risk reference and proxy-informativeness assumptions with paired target benefit and K-Bound's single-unit marginal interval premise. “Complementary monitoring” alone understates the relevant comparison. [Primary paper](https://arxiv.org/html/2507.08721v1) |
| Bar, Shaer and Romano, POEM, NeurIPS 2024 | Entropy matching and betting protect adaptation; online betting has its own theoretical analysis | Already cited. Betting/regret guarantees are not interchangeable with a confidence interval for paired target benefit. Do not present POEM as an unprincipled baseline or claim a current-policy empirical comparison from quarantined historical results. [Primary paper](https://arxiv.org/html/2408.07511v1) |
| Ariq, *MMD-Balls as Credal Sets*, 2026 | Target-risk intervals and an adaptation criterion under covariate-shift/RKHS assumptions | Already cited. Its uncertainty is over an MMD-based input-law class under structural assumptions, not the same fixed-input label-kernel fibre or paired-benefit estimand. Retain the preprint/workshop distinction. [Primary paper](https://arxiv.org/html/2605.21783v1) |
| Angelopoulos et al., *Learn Then Test*, 2025; Barber et al., *Conformal Prediction Beyond Exchangeability*, 2023 | Calibration and risk-control theory explicitly identify validity assumptions; LTT also handles selection over configurations with multiplicity control | Already cited. Theorem 4's covered-interval implication is an application of existing logic. KGA has not obtained the coverage premise for arbitrary natural shift merely by using a residual quantile. Add citations close to the certificate/calibration statements. [LTT](https://arxiv.org/pdf/2110.01052), [Conformal beyond exchangeability](https://arxiv.org/html/2202.13415v1) |
| Lamaakal et al., *Drift-to-Action Controllers*, 2026 | Risk certificates gate adaptation, abstention, rollback, and retraining in a controller | **Missing recent comparison.** Its active certificate queries delayed labels and concerns online risk; it is not a target-label-free paired-benefit certificate. Cite the arXiv version unless the reported workshop status is separately verified. [Primary paper](https://arxiv.org/html/2603.08578v1) |
| Laroche et al., *Safe Policy Improvement with Baseline Bootstrapping*, ICML 2019 | Uncertainty-aware improvement with a baseline fallback already exists in offline reinforcement learning | Optional conceptual context, not a directly interchangeable experimental baseline. Its logged rewards/transitions and MDP uncertainty differ from target-label-free classification. Do not claim general originality for baseline fallback. [Proceedings](https://proceedings.mlr.press/v97/laroche19a.html) |
| Chen et al., TTSA, ICML 2025; Sahoo et al., GALA, AAAI 2025 | Selective adaptation can choose modalities or layers | Already discussed. Choosing where to adapt is different from certifying the sign of a fixed candidate's benefit, but this does not establish empirical superiority over those methods. [TTSA](https://proceedings.mlr.press/v267/chen25ch.html), [GALA](https://ojs.aaai.org/index.php/AAAI/article/view/34229) |

Ben-David et al. (2010) in the current bibliography is *A Theory of Learning from Different Domains*, a different paper. Ben-David and Urner (2012) is also a different work and currently has no citation link. Neither substitutes for the specific AISTATS comparison above.

### 5.3 Proposed novelty wording, not yet applied

> K-Bound characterizes when the sign of a fixed candidate's benefit is identified within a declared disagreement-conditional residual class. It separates this population question from a practical interval-based commitment rule. The contribution is the explicit class-dependent frontier, its strict boundary semantics, and the limits of auditing the required residual bound from the same label-free evidence. It does not introduce label-free impossibility, learned update routing, or interval-based risk control in general.

In the related-work revision, add the AISTATS 2010 comparison next to the impossibility theorem and the recent controller comparison next to the guarding/monitoring discussion. Expand the existing monitoring comparison to identify estimands, label access, and the scope of probability control. A brief precise comparison is more useful than adding many loosely relevant citations.

### 5.4 Academic ceiling of the current argument

The core can support a serious, carefully scoped methodological paper. It does not, by proof simplicity alone, justify either rejection or a top score: clarity and a useful characterization can matter. But the current review does not establish a major new mathematical technique or a distribution-free natural-shift safety theorem.

A stronger theoretical claim would require actual new content—such as a justified way to restrict the fibre using realistic structure, a meaningful finite-observation result, or a valid shift-aware coverage bridge. Those are possible research directions, not hidden conclusions already proved here. Better bibliography formatting or another favorable dataset would not by itself establish that theoretical novelty.

## 6. Bibliography review

All 60 printed records were checked against primary publisher, conference, author, institutional, or arXiv records. “Matches” means the supplied identity and metadata are consistent; optional missing page ranges or DOIs are not counted as errors. Abbreviated first names and a legitimate `et al.` author prefix are acceptable.

The result is **54 matching entries and six requiring attention**: two material mixed/incorrect records, one author/version issue, and three typography/edition consistency issues. The complete ledger follows. `U` marks an entry with no citation hyperlink in the compiled PDF; it is not a judgment that the work is irrelevant.

### 6.1 Corrections to apply

| Key | Current problem | Required resolution |
|---|---|---|
| `taylor2019rxrx1` | Wrong/mixed authors and title; arXiv:1907.04758 identifies SynthCity, not RxRx1 | Replace with Sypetkowski et al., *RxRx1: A Dataset for Evaluating Experimental Batch Correction Methods*, CVPR Workshops 2023, pp.4285–4294, arXiv:2301.05768. Update the author-year citation label as well as the record. [Correct record](https://openaccess.thecvf.com/content/CVPR2023W/CVMI/html/Sypetkowski_RxRx1_A_Dataset_for_Evaluating_Experimental_Batch_Correction_Methods_CVPRW_2023_paper.html), [wrong ID's actual record](https://arxiv.org/abs/1907.04758) |
| `recht2019cifar10` | Combines a CIFAR-10 title with the ICML 2019 ImageNet publication | The citation is used for CIFAR-10.1 in Table 3, p.14, so the indicated repair is Recht et al., *Do CIFAR-10 Classifiers Generalize to CIFAR-10?*, 2018 preprint, arXiv:1806.00451. If the ImageNet paper is intended elsewhere, cite it as a separate record. [CIFAR preprint](https://arxiv.org/abs/1806.00451), [ICML ImageNet paper](https://proceedings.mlr.press/v97/recht19a.html) |
| `sonoda2025lean` | Omits Kazumi Kasaura; combines an unversioned 2025 citation with a title expanded in a 2026 revision | Restore the five-author order: Sonoda, Kasaura, Mizuno, Tsukamoto, Onda. Pin either the 2025 v3 with the shorter Rademacher title, or the 2026 v5 with the Dudley title. A 2025 original-preprint year is not intrinsically wrong, but the cited version must be clear. [2025 v3](https://arxiv.org/abs/2503.19605v3), [2026 v5](https://arxiv.org/abs/2503.19605v5) |
| `li2017pacs` | CVF pages 5542–5550 are combined with an IEEE DOI whose edition uses 5543–5551 | Normalize to one edition. The title, authors, year, and work identity are correct. [CVF edition](https://openaccess.thecvf.com/content_iccv_2017/html/Li_Deeper_Broader_and_ICCV_2017_paper.html), [publisher-version metadata](https://www.pure.ed.ac.uk/ws/portalfiles/portal/41072820/li2017dg.pdf) |
| `beery2018recognition` | CVF pages 456–473 are combined with a Springer DOI whose edition uses 472–489 | Normalize to one edition. Do not treat the CVF range as fabricated or the dataset attribution as wrong. [CVF edition](https://openaccess.thecvf.com/content_ECCV_2018/html/Beery_Recognition_in_Terra_ECCV_2018_paper.html), [published-version record](https://authors.library.caltech.edu/records/m2211-qkc66) |
| `baek2022aol` | Rendered title contains the spurious space `Agreement-on- the-line` | Normalize to `Agreement-on-the-line`; all checked substantive metadata match. [Proceedings](https://papers.nips.cc/paper_files/paper/2022/hash/7a8d388b7a17df480856dff1cc079b08-Abstract-Conference.html) |

The complete RxRx1 author order is Maciej Sypetkowski, Morteza Rezanejad, Saber Saberian, Oren Kraus, John Urbanik, James Taylor, Ben Mabey, Mason Victors, Jason Yosinski, Alborz Rezazadeh Sereshkeh, Imran Haque, and Berton Earnshaw. A consistent `et al.` truncation is acceptable.

### 6.2 Complete 60-record ledger

Author identity/order, title, year, venue, and supplied identifier were checked. Multi-author groups are abbreviated in this ledger for readability; a status of “matches” is not a full-content endorsement of the cited paper.

| # | Stable key and work | Verified publication information / primary source | Result |
|---|---|---|---|
| 1 | `wang2021tent` — Wang et al., *Tent: Fully Test-Time Adaptation by Entropy Minimization* | Five authors in the printed order; ICLR 2021. The 2020 preprint date is compatible with the 2021 conference year. [arXiv](https://arxiv.org/abs/2006.10726) | Matches |
| 2 | `liang2020shot` — Liang, Hu and Feng, *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation* | ICML 2020, PMLR 119:6028–6039. [Proceedings](https://proceedings.mlr.press/v119/liang20a.html) | Matches |
| 3 | `sun2020ttt` — Sun et al., *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts* | Six authors in the printed order; ICML 2020, PMLR 119:9229–9248. [Proceedings](https://proceedings.mlr.press/v119/sun20b.html) | Matches |
| 4 | `niu2022eata` — Niu et al., *Efficient Test-Time Model Adaptation without Forgetting* | Niu, Wu, Zhang, Chen, Zheng, Zhao, Tan; ICML 2022, PMLR 162:16888–16905. [Proceedings](https://proceedings.mlr.press/v162/niu22a.html) | Matches |
| 5 | `niu2023sar` — Niu et al., *Towards Stable Test-Time Adaptation in Dynamic Wild World* | Niu, Wu, Zhang, Wen, Chen, Zhao, Tan; ICLR 2023. [Conference paper](https://openreview.net/pdf?id=g2YraF75Tj), [arXiv](https://arxiv.org/abs/2302.12400) | Matches |
| 6 | `wang2022cotta` — Wang, Fink, Van Gool and Dai, *Continual Test-Time Domain Adaptation* | CVPR 2022, pp.7201–7211. [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html) | Matches |
| 7 | `zhang2022memo` — Zhang, Levine and Finn, *MEMO: Test Time Robustness via Adaptation and Augmentation* | NeurIPS 2022, volume 35. [Proceedings](https://papers.nips.cc/paper/2022/hash/fc28053a08f59fccb48b11f2e31e81c7-Abstract-Conference.html) | Matches |
| 8 | `schirmer2025monitoring` — Schirmer, Jazbec, Naesseth and Nalisnick, *Monitoring Risks in Test-Time Adaptation* | NeurIPS 2025, volume 38; arXiv:2507.08721. [Proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/746960ad49ddb47248970a0e1404230c-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2507.08721) | Matches |
| 9 | `mutlu2026informed` — Mutlu et al., *Topology-Guided Test-Time Adaptation via Persistent Homology: From Affective Behavior Analysis to Autonomous Driving* | Mutlu, Honarmand, Azizian, Surabhi, Wall; CVPR Workshops 2026, ABAW, pp.5331–5340. Workshop status is correctly stated. [CVF](https://openaccess.thecvf.com/content/CVPR2026W/ABAW/html/Mutlu_Topology-Guided_Test-Time_Adaptation_via_Persistent_Homology_From_Affective_Behavior_Analysis_CVPRW_2026_paper.html) | Matches |
| 10 | `chen2025ttsa` — Chen et al., *Test-Time Selective Adaptation for Uni-Modal Distribution Shift in Multi-Modal Data* | Chen, Zhang, Han, Jiang, Wang, Feng, Du, Bao; ICML 2025, PMLR 267:9711–9727. [Proceedings](https://proceedings.mlr.press/v267/chen25ch.html) | Matches |
| 11 | `sahoo2025gala` — Sahoo et al., *A Layer Selection Approach to Test Time Adaptation* | Sahoo, ElAraby, Ngnawe, Pequignot, Precioso, Gagné; AAAI 39(19):20237–20245, 2025; DOI 10.1609/aaai.v39i19.34229. [Publisher](https://ojs.aaai.org/index.php/AAAI/article/view/34229) | Matches |
| 12 | `ariq2026mmd` — Ariq, *MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation* | Ahanaf Hasan Ariq; arXiv:2605.21783, 2026. Author comments report EIML@ICML workshop acceptance; this is not an ICML main-track publication claim. [arXiv](https://arxiv.org/abs/2605.21783) | Matches as preprint |
| 13 | `podkopaev2022tracking` — Podkopaev and Ramdas, *Tracking the Risk of a Deployed Model and Detecting Harmful Distribution Shifts* | ICLR 2022; arXiv:2110.06177. Preprint and publication years are compatible. [arXiv publication record](https://arxiv.org/abs/2110.06177) | Matches |
| 14 | `bar2024poem` — Bar, Shaer and Romano, *Protected Test-Time Adaptation via Online Entropy Matching: A Betting Approach* | NeurIPS 2024, volume 37; arXiv:2408.07511. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9b35a0a20d617dc68ae98a7a57df2f51-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2408.07511) | Matches |
| 15 | `garg2022atc` — Garg et al., *Leveraging Unlabeled Data to Predict Out-of-Distribution Performance* | Garg, Balakrishnan, Lipton, Neyshabur, Sedghi; ICLR 2022; arXiv:2201.04234. [arXiv publication record](https://arxiv.org/abs/2201.04234) | Matches |
| 16 | `kalai2021abstain` — Kalai and Kanade, *Towards Optimally Abstaining from Prediction with OOD Test Examples* | NeurIPS 2021, volume 34; arXiv:2105.14119. [Proceedings](https://proceedings.neurips.cc/paper/2021/hash/6a26c75d6a576c94654bfc4dda548c72-Abstract.html) | Matches; U |
| 17 | `miller2021aol` — Miller et al., *Accuracy on the Line: On the Strong Correlation between Out-of-Distribution and In-Distribution Generalization* | Nine authors in the printed order; ICML 2021, PMLR 139:7721–7735; arXiv:2107.04649. [Proceedings](https://proceedings.mlr.press/v139/miller21b.html) | Matches |
| 18 | `baek2022aol` — Baek, Jiang, Raghunathan and Kolter, *Agreement-on-the-line: Predicting the Performance of Neural Networks under Distribution Shift* | NeurIPS 2022, volume 35; arXiv:2206.13089. Remove the rendered space in `Agreement-on- the-line`. [Proceedings](https://papers.nips.cc/paper_files/paper/2022/hash/7a8d388b7a17df480856dff1cc079b08-Abstract-Conference.html) | Typography |
| 19 | `lee2024aetta` — Lee, Chottananurak, Gong and Lee, *AETTA: Label-Free Accuracy Estimation for Test-Time Adaptation* | CVPR 2024, pp.28643–28652; arXiv:2404.01351. [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_AETTA_Label-Free_Accuracy_Estimation_for_Test-Time_Adaptation_CVPR_2024_paper.html) | Matches |
| 20 | `steinhardt2016` — Steinhardt and Liang, *Unsupervised Risk Estimation Using Only Conditional Independence Structure* | NIPS/NeurIPS 2016, volume 29; arXiv:1606.05313. Modern venue naming is acceptable. [Proceedings](https://papers.nips.cc/paper_files/paper/2016/hash/f2d887e01a80e813d9080038decbbabb-Abstract.html) | Matches |
| 21 | `bendavid2010theory` — Ben-David et al., *A Theory of Learning from Different Domains* | Ben-David, Blitzer, Crammer, Kulesza, Pereira, Wortman Vaughan; Machine Learning 79(1–2):151–175, 2010; DOI 10.1007/s10994-009-5152-4. The 2009 online-first date is compatible. [Publisher](https://link.springer.com/article/10.1007/s10994-009-5152-4), [issue](https://link.springer.com/journal/10994/volumes-and-issues/79-1) | Matches |
| 22 | `rosenfeld2023dis2` — Rosenfeld and Garg, *(Almost) Provable Error Bounds under Distribution Shift via Disagreement Discrepancy* | NeurIPS 2023, volume 36; arXiv:2306.00312. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5bacb12bf81e98e2ee0eed953a23c656-Abstract-Conference.html) | Matches |
| 23 | `geifman2017selective` — Geifman and El-Yaniv, *Selective Classification for Deep Neural Networks* | NIPS/NeurIPS 2017, volume 30. [Proceedings](https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) | Matches |
| 24 | `deng2009imagenet` — Deng et al., *ImageNet: A Large-Scale Hierarchical Image Database* | Deng, Dong, Socher, Li, Li, Fei-Fei; CVPR 2009. [Official dataset paper](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf) | Matches; U |
| 25 | `hendrycks2019cifar10c` — Hendrycks and Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* | ICLR 2019. [Conference paper](https://openreview.net/pdf?id=HJz6tiCqYm) | Matches |
| 26 | `hendrycks2021manyfaces` — Hendrycks et al., *The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization* | All 13 printed authors/order match; ICCV 2021, pp.8340–8349. [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Hendrycks_The_Many_Faces_of_Robustness_A_Critical_Analysis_of_Out-of-Distribution_ICCV_2021_paper.html) | Matches |
| 27 | `koh2021wilds` — Koh et al., *WILDS: A Benchmark of In-the-Wild Distribution Shifts* | First eight printed authors/order match; the rest are legitimately abbreviated. ICML 2021, PMLR 139:5637–5664. [Proceedings](https://proceedings.mlr.press/v139/koh21a.html) | Matches |
| 28 | `bandi2019camelyon` — Bándi et al., *From Detection of Individual Metastases to Classification of Lymph Node Status at the Patient Level: The CAMELYON17 Challenge* | Printed author prefix/order match; IEEE Transactions on Medical Imaging 38(2):550–560, 2019. [Author-hosted publication](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/19-IEEE_Franke.pdf) | Matches |
| 29 | `taylor2019rxrx1` — current RxRx1 record | Replace the mixed record with Sypetkowski et al., CVPR Workshops 2023, pp.4285–4294; arXiv:2301.05768. See §6.1 for the full correction. [CVF](https://openaccess.thecvf.com/content/CVPR2023W/CVMI/html/Sypetkowski_RxRx1_A_Dataset_for_Evaluating_Experimental_Batch_Correction_Methods_CVPRW_2023_paper.html) | Material error |
| 30 | `zhu2020so2sat` — Zhu et al., *So2Sat LCZ42: A Benchmark Data Set for the Classification of Global Local Climate Zones* | All 17 authors/order match; IEEE Geoscience and Remote Sensing Magazine 8(3):76–89, 2020; DOI 10.1109/MGRS.2020.2964708. [DLR record](https://elib.dlr.de/138056/) | Matches |
| 31 | `venkateswara2017officehome` — Venkateswara et al., *Deep Hashing Network for Unsupervised Domain Adaptation* | Venkateswara, Eusebio, Chakraborty, Panchanathan; CVPR 2017. This is the Office-Home dataset paper. [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Venkateswara_Deep_Hashing_Network_CVPR_2017_paper.html) | Matches |
| 32 | `li2017pacs` — Li et al., *Deeper, Broader and Artier Domain Generalization* | Li, Yang, Song, Hospedales; ICCV 2017. CVF pages 5542–5550 versus DOI-linked IEEE pages 5543–5551. [CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Li_Deeper_Broader_and_ICCV_2017_paper.html), [publisher version](https://www.pure.ed.ac.uk/ws/portalfiles/portal/41072820/li2017dg.pdf) | Edition normalization |
| 33 | `gulrajani2021domainbed` — Gulrajani and Lopez-Paz, *In Search of Lost Domain Generalization* | ICLR 2021. [Author preprint](https://arxiv.org/abs/2007.01434), [conference listing](https://iclr.cc/virtual/2021/papers.html) | Matches; U |
| 34 | `croce2021robustbench` — Croce et al., *RobustBench: A Standardized Adversarial Robustness Benchmark* | Eight printed authors/order match; NeurIPS Datasets and Benchmarks 2021. [Proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a3c65c2974270fd093ee8a9bf8ae7d0b-Abstract-round2.html) | Matches; U |
| 35 | `recht2019cifar10` — Recht et al., *Do CIFAR-10 Classifiers Generalize to CIFAR-10?* | Recht, Roelofs, Schmidt, Shankar; the printed CIFAR title belongs to the 2018 preprint, not ICML 2019. [CIFAR record](https://arxiv.org/abs/1806.00451), [different ICML ImageNet record](https://proceedings.mlr.press/v97/recht19a.html) | Material error |
| 36 | `paszke2019pytorch` — Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library* | Printed author prefix/order match; NeurIPS 2019. [Proceedings](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) | Matches; U |
| 37 | `pedregosa2011sklearn` — Pedregosa et al., *Scikit-learn: Machine Learning in Python* | Printed author prefix/order match; JMLR 12:2825–2830, 2011. [JMLR](https://jmlr.org/papers/v12/pedregosa11a.html) | Matches; U |
| 38 | `xie2024mano` — Xie et al., *MaNo: Exploiting Matrix Norm for Unsupervised Accuracy Estimation under Distribution Shifts* | Xie, Odonnat, Feofanov, Deng, Zhang, An; NeurIPS 2024; arXiv:2405.18979. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/49abf767d606b72f74ea6009176fafeb-Abstract-Conference.html) | Matches |
| 39 | `deng2023confidence` — Deng et al., *Confidence and Dispersity Speak: Characterizing Prediction Matrix for Unsupervised Accuracy Estimation* | Deng, Suh, Gould, Zheng; ICML 2023; arXiv:2302.01094. [Proceedings](https://proceedings.mlr.press/v202/deng23e.html) | Matches |
| 40 | `liang2025ttasurvey` — Liang, He and Tan, *A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts* | International Journal of Computer Vision 133(1):31–64, 2025; arXiv:2303.15361. Online-first publication in 2024 does not invalidate the 2025 issue year. [Publisher](https://link.springer.com/article/10.1007/s11263-024-02181-w) | Matches; U |
| 41 | `tibshirani2019conformal` — Tibshirani et al., *Conformal Prediction under Covariate Shift* | Tibshirani, Barber, Candès, Ramdas; NeurIPS 2019; arXiv:1904.06019. [Proceedings](https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) | Matches |
| 42 | `barber2023beyond` — Barber et al., *Conformal Prediction beyond Exchangeability* | Barber, Candès, Ramdas, Tibshirani; Annals of Statistics 51(2):816–845, 2023; arXiv:2202.13415. [Published paper](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.pdf) | Matches |
| 43 | `gibbs2021adaptive` — Gibbs and Candès, *Adaptive Conformal Inference under Distribution Shift* | NeurIPS 2021; arXiv:2106.00170. [Proceedings](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) | Matches; U |
| 44 | `park2022pac` — Park, Dobriban, Lee and Bastani, *PAC Prediction Sets under Covariate Shift* | ICLR 2022; arXiv:2106.09848. [Conference paper](https://openreview.net/pdf?id=DhP9L8vIyLc) | Matches; U |
| 45 | `bates2021rcps` — Bates et al., *Distribution-Free, Risk-Controlling Prediction Sets* | Bates, Angelopoulos, Lei, Malik, Jordan; Journal of the ACM 68(6), 2021; arXiv:2101.02703. ACM fetching was blocked; the journal record was cross-checked through the coauthor's publication list. [Author record](https://lihualei71.github.io/research.html), [arXiv](https://arxiv.org/abs/2101.02703) | Matches |
| 46 | `angelopoulos2025ltt` — Angelopoulos et al., *Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control* | Angelopoulos, Bates, Candès, Jordan, Lei; Annals of Applied Statistics 19(2):1641–1662, 2025; DOI 10.1214/24-AOAS1998; arXiv:2110.01052. [Stanford publication record](https://www.gsb.stanford.edu/faculty-research/publications/learn-then-test-calibrating-predictive-algorithms-achieve-risk) | Matches |
| 47 | `hoang2024petta` — Hoang, Vo and Do, *Persistent Test-Time Adaptation in Recurring Testing Scenarios* | NeurIPS 2024; arXiv:2311.18193. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/df29d63af05cb91d705cf06ba5945b9d-Abstract-Conference.html) | Matches; U |
| 48 | `lee2024deyo` — Lee et al., *Entropy Is Not Enough for Test-Time Adaptation: From Perspective of Disentangled Factors* | Lee, Jung, Lee, Park, Shin, Hwang, Yoon; ICLR 2024; arXiv:2403.07366. [Conference paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/cd0665986c1e9c15f6569ac944bcf88a-Paper-Conference.pdf) | Matches; U |
| 49 | `lim2026reset` — Lim, Hwang and Lee, *When and Where to Reset Matters for Long-Term Test-Time Adaptation* | Taejun Lim, Joong-Won Hwang, Kibok Lee; ICLR 2026; arXiv:2603.03796. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/79bdd6fe3f012befcc459ad13de65d13-Abstract-Conference.html) | Matches; U |
| 50 | `sonoda2025lean` — Sonoda et al., *Lean Formalization of Generalization Error Bound…* | Restore Kasaura as the second author. Choose the 2025 shorter-title version or the 2026 Dudley-expanded version. [2025 v3](https://arxiv.org/abs/2503.19605v3), [2026 v5](https://arxiv.org/abs/2503.19605v5) | Author/version correction; U |
| 51 | `tempora2026` — Sreeram, Kwon and Mascolo, *Tempora: Characterising Time-Contingent Utility of Online Test-Time Adaptation* | arXiv:2602.06136; ICML 2026 acceptance supported by Cambridge's accepted, peer-reviewed manuscript record. Proceedings pages were not independently checked. [Cambridge](https://www.repository.cam.ac.uk/items/6bc10490-e7c3-44c8-9ea8-51676025b8bb), [arXiv](https://arxiv.org/abs/2602.06136) | Matches; U |
| 52 | `corradaemmanuel2024` — Corrada-Emmanuel, *The Logic of NTQR Evaluations of Noisy AI Agents: Complete Postulates and Logically Consistent Error Correlations* | Sole author Andrés Corrada-Emmanuel; arXiv:2312.05392, 2023. The internal citation key does not dictate the publication year. No accepted venue established. [arXiv](https://arxiv.org/abs/2312.05392) | Matches as preprint; U |
| 53 | `gupta2021toplabel` — Gupta and Ramdas, *Top-Label Calibration and Multiclass-to-Binary Reductions* | ICLR 2022; arXiv:2107.08353. The 2021 preprint key and 2022 venue year are compatible. [arXiv publication metadata](https://arxiv.org/abs/2107.08353), [conference listing](https://iclr.cc/Downloads/2022) | Matches; U |
| 54 | `bendavid2012hardness` — Ben-David and Urner, *On Hardness of Domain Adaptation and Utility of Unlabeled Target Samples* | ALT 2012, pp.139–153. [Publisher proceedings](https://link.springer.com/book/10.1007/978-3-642-34106-9) | Matches; U |
| 55 | `lipton2018bbse` — Lipton, Wang and Smola, *Detecting and Correcting for Label Shift with Black Box Predictors* | ICML 2018, PMLR 80:3122–3130. [Proceedings](https://proceedings.mlr.press/v80/lipton18a.html) | Matches; U |
| 56 | `garg2020labelshift` — Garg et al., *A Unified View of Label Shift Estimation* | Garg, Wu, Balakrishnan, Lipton; NeurIPS 2020. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/hash/219e052492f4008818b8adb6366c7ed6-Abstract.html) | Matches; U |
| 57 | `chow1970reject` — Chow, *On Optimum Recognition Error and Reject Tradeoff* | IEEE Transactions on Information Theory 16(1):41–46, 1970; DOI 10.1109/TIT.1970.1054406. [IBM author record](https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff) | Matches |
| 58 | `cortes2016boosting` — Cortes, DeSalvo and Mohri, *Learning with Rejection* | ALT 2016, pp.67–82. The internal key is not a title error. [Publisher proceedings](https://link.springer.com/book/10.1007/978-3-319-46379-7) | Matches |
| 59 | `madras2018learntodefer` — Madras, Pitassi and Zemel, *Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer* | NeurIPS 2018. [Proceedings](https://proceedings.neurips.cc/paper/2018/hash/09d37c08f7b129e96277388757530c72-Abstract.html) | Matches; U |
| 60 | `beery2018recognition` — Beery, Van Horn and Perona, *Recognition in Terra Incognita* | ECCV 2018. CVF pp.456–473 versus Springer DOI edition pp.472–489; DOI 10.1007/978-3-030-01270-0_28. [CVF](https://openaccess.thecvf.com/content_ECCV_2018/html/Beery_Recognition_in_Terra_ECCV_2018_paper.html), [published-version record](https://authors.library.caltech.edu/records/m2211-qkc66) | Edition normalization |

### 6.3 Citation-use findings

The PDF contains 60 named bibliography destinations and links to 40 of them from its content. No citation hyperlink points to a missing bibliography destination. Twenty bibliography entries have no such links:

```text
bendavid2012hardness     corradaemmanuel2024   croce2021robustbench
deng2009imagenet        garg2020labelshift    gibbs2021adaptive
gulrajani2021domainbed  gupta2021toplabel     hoang2024petta
kalai2021abstain        lee2024deyo           liang2025ttasurvey
lim2026reset           lipton2018bbse        madras2018learntodefer
park2022pac            paszke2019pytorch     pedregosa2011sklearn
sonoda2025lean          tempora2026
```

This is an audit of rendered PDF links, not of unavailable TeX citation commands. Confirm ordinary textual mentions and any intentional bibliography-wide inclusion before removing records. Several have plausible roles: the software papers can support implementation details; the calibration and hardness papers can support precise assumptions; the continual-adaptation papers can delimit sequential scope. Add a real supporting citation where needed, or remove an unused record from the submission bibliography without destroying its archival record.

No record was deleted in this review. Missing comparison citations are a separate issue from malformed metadata or unlinked bibliography entries.

## 7. Completion and acceptance checklist

Completed in this review:

- [x] Check the eight numbered statements and their displayed proofs.
- [x] Check feasible ranges, boundary cases, audit-radius equality, split-conformal rank, multiclass identity, and error-control scope.
- [x] Construct explicit counterexamples for the cell/population substitution, no-shift calibration terminology, and insufficient sign-flip assumptions.
- [x] Compare the closest checked primary literature and identify missing/underdeveloped comparisons.
- [x] Check every printed bibliography record and audit citation links.
- [x] Identify precise corrections without changing the reported experimental outcomes.

Required at the original review stage before calling these sections submission-ready (see Section 8 for the subsequent repair status):

- [ ] Make the maintained TeX/figure sources locally readable and reconcile them with the reviewed PDF.
- [ ] Resolve T1's estimand distinction throughout the manuscript and captions.
- [ ] Correct T2's inference language, retaining numerical outputs only at their justified scope.
- [ ] Fix residual terminology, Figure 2's units, and fallback semantics; resolve Figure 3's provenance.
- [ ] Add the closest missing novelty comparisons and keep the narrow claim of contribution.
- [ ] Correct/normalize the six bibliography entries and resolve intentional versus unused records.
- [ ] Rebuild and visually verify the existing long PDF, short PDF, and Word export; recheck citations, cross-references, and release hashes.

These review findings do not require a new GPU run merely to correct the paper's wording and bibliography. A stronger population guarantee would require a valid additional argument and its assumptions, not just rewriting. The So2Sat conclusion is unchanged: no feasible candidate, no target access, and no target natural-shift score. A negative feasibility result must remain visible.

**Final review disposition:** core proofs pass this manual review within scope; empirical safety interpretation, novelty positioning, bibliography, and source-level release verification need the listed revisions. The review is complete; the revision and release are not.

## 8. Implementation follow-up: measured outcomes and statistical inference

Follow-up date: 2026-08-30. The user selected T1 and T2 for correction. The maintained
shared TeX sources were restored through targeted native iCloud downloads, read, and edited;
no unavailable working-tree source was replaced with an assumed Git version.

### 8.1 Repairs applied

- **T1, claim scope:** population benefit remains the fresh-draw risk difference for the
  realized model pair. The empirical target is now explicitly the measured evaluation-cell
  score difference, including the distinction between accuracy and nonlinear macro-F1.
  The shared certificate is stated for a predeclared generic scalar target and evaluation
  unit. Its marginal coverage premise does not become population, conditional, simultaneous,
  or sequential coverage. The bridge gives the additional sampling-error premise needed
  for a population application and the resulting radius/error-budget union bound; the
  paper states that this extra premise is not established by the reported experiments.
  The two-point counterexample is included in the reviewer-guide appendix.
- **T2, inference interpretation:** the reference null is joint invariance under every
  coordinatewise cluster sign flip. Independent zero-symmetric cluster gaps, possibly
  conditional on the evaluated checkpoints, are identified as sufficient; exchangeability,
  symmetric marginals, and one global sign symmetry alone are not sufficient. The
  dependent-nine-sign counterexample is included. Bootstrap confidence levels are described
  as nominal, and Bonferroni/Holm interpretations retain their valid-input requirements.
  A supporting primary reference was added: [Hemerik and Goeman, *Exact testing with random
  permutations*, TEST 27, 811–825 (2018)](https://link.springer.com/article/10.1007/s11749-017-0571-1).
- **CCT-20 presentation, not resealing:** a reproducible display-only table checks the
  sealed table's byte count and SHA-256 before changing its column headings. Every byte
  after its result-body delimiter is preserved. Stored gaps, interval endpoints,
  sign-flip values, Holm flags, and the SAFE_UTILITY_ONLY verdict are unchanged.
  The original CCTManuscriptClaim macro remains in sealed generated metadata but is
  not expanded as current prose. The prose guard still requires CCTVerdict and rejects
  unsupported exposure/comparator claims.
- **Related T4/T5/T6 presentation repairs:** Figure 2 now uses feasible raw margin values
  and a labeled illustrative beta of 0.1. Invalid features/small batches no longer imply
  a certified FREEZE action. Figure 3's old generator was traced to a legacy pilot,
  not the canonical panel; its plotted point arrays were not saved in that pilot's
  summary. It was replaced with a clearly labeled, data-free rendering of the manuscript's
  three existing interval examples. No pilot or benchmark was rerun. The prior image
  is preserved by Git blob a3474adfd810281eb892546b5ce082d6cbbe599c and by the
  task-local backup, rather than silently relabeled as current evidence.
- **Export repairs:** Word now counts all numbered headings, including unlabeled ones,
  and resets appendix numbering to letters. Its visible headings and references share
  one mapping; all 27 labeled sections match the compiled article's auxiliary file.
  Tables are allowed to break between body rows, not inside short rows, and table
  captions/headers stay with the first complete row. These are materialized export
  numbers, not automatically updating Word fields.

### 8.2 Validation and artifact identity

The targeted regression suites pass **67 tests**: 38 estimand/inference/display tests
and 29 Word-export tests. They cover unchanged sealed result rows, fail-closed display
generation, valid versus overstated verdict wording, the two estimands, the sign-flip
null, nominal bootstrap language, illustrative-figure parameters, section numbering,
and table pagination. Python syntax checks and the shell build-driver syntax check pass.
The document package passes its ZIP integrity check.

The maintained short and long PDFs compile successfully with no unresolved references,
missing-character diagnostics, or overfull boxes in the final build logs. All 33 short
pages and all 37 long pages were visually checked at original resolution, with subsequent
reflow pages checked again. All 32 rendered Word pages were visually checked at original
resolution; the final changed pages were inspected again, and unchanged pages were
confirmed by image hashes against the previously inspected proof.

The existing short PDF, long PDF, and Word filenames were updated in place. Final artifact identities are:

| Maintained artifact | SHA-256 |
|---|---|
| kbound_short_final_draft.pdf | 522988d746030850a1ba22726d23e909bce86e446c1d24dfda71503e4e5f7af3 |
| kbound_tmlr.pdf | 415827814fef8c8b07817a8e2b15b49c8fd75ab95c94f31eb44b0154b4cbdaba |
| kbound_short_final_draft.docx | 5b2462c442feeb21711d4e7278c07baf09c0b36ce3f1aa9af1af15099e840716 |

The following scientific authorities remain byte-for-byte unchanged from the
pre-revision snapshot:

    1f3904f71d22539498ed68e2c3c44820e5aaed4be1fe6e08e7c0d1695f636c35  paper/generated/kbound_numbers.tex
    7e63bfad43c2374d02ec9995ccdd4bf94335fc6e4a529d506b7546c722f1b77c  paper/generated/cct20_numbers.tex
    a296bbfed473e393bed064eed41ccb56c53a551ce1fbf929c6f2eccb96d48953  paper/generated/so2sat_numbers.tex
    de0ed2601497894a26c9497af7ccef853d2fd499155bda9bd8ee6906add1e9a6  paper/generated/current_policy_family_sensitivity.tex
    e3ce875900bead95095db20506dcf2cbc6a03cb51ef750549e89d36a2c845b7f  paper/generated/cct20_primary_table.tex
    722d2ebbe2d883c7eb173d72af9e4aa4c0a99b1ec320d913bf668f07d28eff48  paper/generated/cct20_release_manifest.json

Build logs, original-output backups, and page proofs are retained under
/tmp/kbound-estimand-inference.pbFAo7. These artifact hashes are a presentation
audit, not a new end-to-end scientific release seal.

### 8.3 Remaining limits

T1 and T2 are repaired as manuscript interpretation/assumption issues; population
coverage and the actual benchmark sign-invariance premise have not thereby been proved.
T5 is corrected in the paper, not a fresh audit of all deployment implementations.
T3's residual-versus-drift terminology, the remaining novelty comparisons, the six
previously flagged bibliography records and citation-use decisions, and remaining T7
precision items are not closed by this follow-up. The new sign-flip citation makes
61 bibliography records; it does not substitute for those six repairs.

No full evidence-provenance gate, new Lean build, Git integrity check, or release
reseal is claimed. A bounded Git diff check stalled and was stopped; no Git state was
reset, committed, pushed, or otherwise maintained. Minor existing typography also
remains for a later final-layout pass. No training, target access, or dataset deletion
occurred. The So2Sat result remains **no feasible candidate, no target access, and no
target natural-shift score**.

## 9. Final revision and release verification

Follow-up date: 2026-08-31. This section records the five revision packages requested
after Section 8. It is not a new experiment, a publication decision, or a numerical
quality score. Current source and document identities are recorded separately in
`audits/revision_verification_2026_08_31.json`, explicitly as an uncommitted
working-tree snapshot, not a clean-source release seal.

### 9.1 Theorem terminology and novelty

The maintained compact and TMLR drivers share the revised body and theorem inputs.
All eight numbered mathematical statements remain; no empirical result was changed
to make the theory appear stronger.

- Gamma is the target disagreement-conditional **calibration residual**. It need
  not arise from distribution drift. The identical-source/target-law example in
  Appendix J shows a nonzero residual caused by conditioning on disagreement.
- The binary predictors, measurable score in `[0,1]`, and positive disagreement
  mass are explicit. Feasible margins satisfy `M in [-1/2,1/2]`. The exact benefit
  set is clipped to the feasible correctness range, rather than treating all
  residual values as automatically attainable.
- Necessity and maximality require the declared rich evidence fibre. A narrower
  class inherits sufficiency, not an automatic converse. Risk alignment excludes
  opposite nonzero signs; it does not itself establish a uniformly strict sign
  when zero benefit is also compatible with the evidence.
- The audit-floor equality is a **fixed-fibre oracle benchmark**. It is not an
  algorithm that learns the unknown observable law from a finite batch, nor a
  blanket deployment minimax-rate result.
- Marginal coverage concerns one named scalar target and unit. Shared calibration,
  repeated deployment, adaptive candidate selection, changed checkpoints, or a
  changed evidence map require their own valid transfer or sequential argument.
  The measured-cell/population distinction, coordinatewise sign-invariance null,
  and nominal bootstrap interpretation from Section 8 remain explicit.

The closest missing impossibility comparison is now Ben-David et al.,
*Impossibility theorems for domain adaptation* (AISTATS 2010). The missing
adaptation-controller comparison is Lamaakal et al., *Drift-to-Action Controllers:
Budgeted interventions with online risk certificates*, explicitly identified as
the checked arXiv v1 preprint, not an assumed peer-reviewed publication.
The text also distinguishes K-Bound from risk monitoring, POEM, rejection and
selective prediction, conformal/risk-controlling prediction, and learn-then-test.

The contribution is deliberately narrow: a disagreement-based matched-evidence
construction, a strict-commitment frontier on a declared rich class, an
evidence-fibre residual-audit floor, and an operational comparison of a fixed
candidate with its frozen baseline under a stated interval premise. It does not
claim to invent label-free impossibility, abstention, interval thresholding, or
general online risk control. Novelty remains a literature-based judgment, not a
guarantee that no related result exists.

Primary comparison anchors:
[Ben-David et al.](https://proceedings.mlr.press/v9/david10a.html),
[Drift-to-Action v1](https://arxiv.org/html/2603.08578v1), and
[Monitoring Risks in Test-Time Adaptation](https://arxiv.org/html/2507.08721v1).

### 9.2 Bibliography and citation use

The six flagged entries are corrected in the maintained shared bibliography:

| Entry | Correction |
|---|---|
| RxRx1 | Sypetkowski et al., CVPR Workshops 2023, pp. 4285–4294; arXiv:2301.05768. The old citation key is retained for compatibility. |
| CIFAR-10.1 | Recht et al., 2018 preprint arXiv:1806.00451; no longer mixed with the separate 2019 ImageNet generalization paper. |
| Lean/Rademacher paper | Sonoda, Kasaura, Mizuno, Tsukamoto, and Onda; title and author list pinned to arXiv:2503.19605v3. |
| PACS | IEEE ICCV 2017 publication edition, pp. 5543–5551; DOI:10.1109/ICCV.2017.591. |
| CCT | Springer ECCV 2018 publication edition, pp. 472–489; DOI:10.1007/978-3-030-01270-0_28. |
| Agreement-on-the-line | Correct hyphenated title retained throughout the bibliography. |

The 20 previously unlinked entries were individually reassessed. Seven now support
specific statements: scikit-learn, PyTorch, top-label calibration, black-box label
shift estimation, the unified label-shift view, adaptive conformal inference, and
the Lean paper. Thirteen remain available in
`paper/references_kbound_context_archive.tex`, with their disposition recorded;
they are not printed as unsupported padding.

The resulting inventory is **50 cited/printed records, 11 unchanged conditional
records, and 13 context-archive records**. All 72 pre-revision citation keys are
retained across these surfaces, with two added comparison keys. The conditional
block is unchanged, and the 13 archived record texts match their prior versions
apart from surrounding whitespace. Twelve bibliography regression tests pass.
The six primary metadata checks and the original per-record review remain in
Section 6; moving a record into the context archive is not deletion of its history.

### 9.3 Implementation audit and boundaries

The publication-primary implementation is the root `kga` package and CLI, with
the maintained `deploy/api` wrappers in the repository. The following defects
were repaired and behavior-tested using synthetic inputs:

- A failed evidence, estimation, certification, probe, or decision attempt clears
  the previous certificate and provenance first. An old ADAPT or FREEZE can no
  longer be reused after a failed new attempt.
- Missing or nonfinite features cannot be silently imputed by a custom estimator.
  Protocol/schema/feature identity is checked before installing a certificate.
- Insufficient finite-rank calibration produces an unavailable radius or an
  explicit validation error, never a fabricated finite negative interval.
  Batch nonfinite entries produce ABSTAIN. Zero/negative probe budgets no longer
  silently consume the full pool.
- Unavailable candidates cannot win multicandidate routing and remain in the
  declared multiplicity family. Anytime updates validate the complete bounded
  vector before changing any process or step counter.
- The HTTP service no longer manufactures a zero-benefit certificate from
  score-only evidence or residuals without a point estimate. Score-only mode
  returns ABSTAIN/unavailable. Full mode requires paired benefits or an explicit
  point estimate plus residuals, and is explicitly a benefit **audit**, not the
  schema-bound label-free deployment estimator.
- Operational unavailable responses include a reason, JSON-safe null fields,
  and `model_action=retain_frozen`. The predictor selector returns the exact
  frozen object without changing either model's parameters. Tests also verify
  that the HTTP active-model-version registry is unchanged.
- Malformed HTTP requests remain HTTP 422 validation errors, not assessed
  ABSTAIN results or certified FREEZE records. Low-level scalar/artifact errors
  remain explicit; callers must not relabel exceptions as certificates.

Valid negative intervals still yield FREEZE. That action is distinct from an
unavailable assessment, even though both retain the frozen model. A certificate
container and model-selection helper do not themselves prove statistical
coverage or physically roll back an already mutated model.

Historical code is **not silently certified by these tests**:

- `kbound_pkg/kbound` remains a reproduction-only snapshot. Its entropy/KL
  heuristic and permissive numerical paths do not implement the current interval
  contract. Its optimizer can update on nonzero `abstain_scale`; zero gradients
  on FREEZE do not neutralize momentum or weight decay. The snapshot was not
  rewritten, and the root distribution/CLI/HTTP exclusion is regression-tested.
- CCT-20's `ridge_gate.py:apply_gate` returns ABSTAIN for unavailable/nonfinite
  live features or predictions. Invalid sealed artifacts cause hard integrity
  errors, not certified FREEZE.
- The six inspected So2Sat source modules reject missing/nonfinite evidence;
  the finite FREEZE branch requires a valid negative upper bound. Its research
  runner nevertheless **aborts** an incomplete bundle rather than recording an
  operational ABSTAIN event and continuing frozen inference. An existing
  docstring overstates that fallback. The production path remains disabled by
  the city-versus-city/checkpoint action-unit mismatch. A future version needs
  both a corrected action-unit lock and an explicit unavailable-assessment path.
  No historical So2Sat source, target data, or sealed result was changed.

### 9.4 Numerical equivalence and provenance

A full current-policy replay used the same 15 already-opened compact CIFAR-10-C
files, three candidates, 20,000 bootstrap replicates, and seed 20260827. The
archived source code was recovered at its recorded Git identity and checked
against both recorded code hashes before comparison.

**All 6,480 cell actions and all 6,480 radii are identical.** All 325 scientific
JSON leaves, including 197 numeric leaves, compare exactly in both type and
value. Reported intervals, reference p-values, and verdicts are unchanged.
Only generation time, code/HEAD identities, and dependent provenance bindings
are refreshed. This is a reproduction check, not new training or new empirical
evidence.

The exact pre-revision current-policy JSON is retained at
`archive/superseded/current_policy_cluster_inference_2026-08-29.json`, SHA-256
`5b1887fc7848ca0a23940806643416c231a04abb62bd141eb318a5e43a36fbdb`.
Its adjacent README explains supersession. The current replay SHA-256 is
`984330fa0d67c9834d8a3327d5cb52033193a69544cf8f10dd1286f82dc60615`.
The canonical panel, its source manifest, receipt-bound CCT-20 authorities,
and So2Sat selection/gate-fit authorities are preserved. The protected-set
comparison covers 196 files with no changed bytes.

The canonical-data gate, manuscript validator, 42-claim/16-entry authority check,
and storage-manifest check pass after the minimal provenance refresh. The three
downstream mirrors were independently compared with their staged updates: 18
provenance-only leaves changed, and the three historical audit CSVs are unchanged.
The dated audit retains its original expected hash and mismatch verdict; updating
the observed current-file hash does not retroactively turn that audit into a PASS.
These checks are recorded in the machine-readable revision snapshot and are not
inferred merely from a successful PDF build.

After that provenance-only step, exactly three dashboard presentation leaves
were refreshed: the actual compact PDF length is 34 pages, and the theory strip
reports four theorems and eight numbered statements under their stated
assumptions. Every other field, including empirical values, edge validation,
and the canonical generation date, is unchanged. A fail-closed PDF metadata
check and post-build metadata-only hook prevent the old page count from
returning. Nineteen focused metadata/hook tests pass.

The six numerical authorities listed in Section 8.2 remain byte-identical,
including the sealed CCT-20 table body and manifest. In particular, CCT-20 remains
`SAFE_UTILITY_ONLY`, with 44 FREEZE, 0 ADAPT, and 1 ABSTAIN. So2Sat remains
**no feasible candidate, no target access, and no target natural-shift score**.

### 9.5 Software, formalization, packaging, and Git checks

The final combined run passes **847 tests across 20 modules**, with zero skips,
failures, or errors, under the normal root pytest configuration and conftest.
It covers theory-scope/rational checks, estimand and inference wording,
bibliography, Word export, public KGA runtime, HTTP behavior, provenance/release
guards, and dashboard metadata. Strict markers and importlib mode are retained;
third-party plugin autoload and the cache provider are disabled. This is not a
claim that every repository test passed. The two warnings are dependency
deprecations from Starlette/httpx and passlib/crypt. The JUnit record is retained
and its identity is included in the revision snapshot.

This merged run includes the previously reported 750 revision checks and 97
release/provenance/metadata checks; those earlier counts are not additional
unique tests. The runbook and CI now execute the new regressions; collection
alone is not treated as execution. All 13 focused execution/dependency guards
pass, both workflow YAML files parse, and 16 affected CI shell blocks plus the
release runbook pass shell syntax checks.

The 847-test run uses the documented mixed isolated dependency environment.
A **separate clean environment** then installed the unchanged
`requirements-api.txt` and test dependencies from official PyPI binary wheels.
All 41 declared requirements are satisfied, `pip check` passes, and the 224
public runtime/API checks pass again there with the normal repository
configuration and conftest, including strict markers and importlib mode.
FastAPI 0.136.3, Pydantic 2.13.4,
Starlette 1.2.0, and AnyIO 4.13.0 match their exact pins. Allowed ranges resolve
to NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0, httpx 0.28.1, httpcore 1.0.9,
and Redis 5.3.1; ranges are not falsely described as exact locks. This verifies
the declared profile on Python 3.12.13/macOS 26.6.2 arm64, not production load,
every supported platform, or every optional service. No repository or global
Python environment was modified.

The finite-valid-input synthetic parity check also retains SHA-256
`5a852a73a25e9d1d83c3e9a4553e8c95db8bfff2ea39262f6ca01ac8b8a41c35`:
6,006 scalar decisions, rank radii at multiple sizes/levels, 50 candidate panels,
three certificate estimators, and 100 anytime steps. These fixtures support
nonregression, not a universal equivalence theorem.

Lean 4.29.1 rebuilt the current `KBound` target successfully. The **65-name
strict-core audit passes** and the proof-hole scan passes. The deliberately
stronger full-foundations gate still fails for six explicitly disclosed layers:

1. measure-theoretic split-conformal exchangeability;
2. filtered e-process/Ville optional-stopping foundations;
3. general KL/TV product-experiment Le Cam bounds;
4. concentration and martingale-rate foundations;
5. general measurable class richness;
6. the full one-bit-channel dichotomy over the declared manuscript class.

Appendix K now lists these limits. A successful Lean build checks encoded
statements under encoded assumptions; it does not establish those unencoded
foundations, benchmark transfer, or correctness of data preprocessing.

Root distribution packaging was built and tested offline in an isolated
directory. `MANIFEST.in` prevents unrelated repository tests, historical code,
data, caches, and AppleDouble files from entering the source archive; it does
not delete those working files. The wheel contains 20 KGA files plus six
metadata files, and the source archive contains 20 KGA files plus 13 metadata
files. Canary exclusions, 25 RECORD hashes, and source-archive-to-wheel content
equivalence pass. Offline installation, 17 module imports, 44 public exports,
and the module/installed CLI help commands pass. No host absolute paths or
high-confidence secret-pattern matches were found in those inspected packages;
this is not a comprehensive security audit. Python 3.11, minimum dependency
versions, and other operating systems were not tested.

Git HEAD is `38724b5985116c9f6f4e724e62b3bb7c71f79999` on
`codex/so2sat-final-run`; the required `9130a46553880bedec6b38ab8e5de0fa72e7f221`
baseline is an ancestor. Shallow commit-graph verification passes. The apparent
POEM pack corruption was an iCloud placeholder: exact pack/index/reverse-index
restoration was followed by successful pack checksum, `verify-pack`, and
`fsck --no-reflogs` checks in that repository. One specifically identified root
Git placeholder was also restored without replacing bytes: the 78-byte loose
object decompresses to a valid 66-byte tree payload and matches object SHA-1
`a918b39201b0e947f2d4c34a92c08c5427e5f575`. A bounded root connectivity check
still timed out after 30 seconds without diagnostic output. Full root object
integrity is therefore **unverified**, not asserted corrupt or clean. No bulk
object hydration or repeated unbounded integrity scan was used.

The current working tree is not clean. Source edits and existing user changes
were preserved; no reset, sweep commit, push, garbage collection, or detached
worktree removal was performed. Historical source/outer seals were not overwritten
to make the new checkout appear certified. The current hash snapshot does not
replace an authorized clean source freeze followed by the complete release gate.

### 9.6 Maintained document outputs

The existing three filenames were rebuilt and updated in place. Final PDF logs
contain no unresolved references, missing-character diagnostics, or overfull
boxes. Every page was visually inspected: **34 compact PDF pages, 38 long PDF
pages, and 32 rendered Word pages**. The final Word typography repair changes
only pages 6–8; all three were reinspected and the other 29 page images are
byte-identical to the inspected proof. Overbars render as bars, and Theorem 2's
Roman clause labels agree with its proof.

| Maintained artifact | SHA-256 |
|---|---|
| `kbound_short_final_draft.pdf` | `c93af00197e60a115b7ff42b94c448067f2dd07fd28db69c7b3e273497cf0c96` |
| `kbound_tmlr.pdf` | `a57e811ae0de62ee1b1782a6843b118dc8b763c7b557f6a30eb6194ad4dc1fe3` |
| `kbound_short_final_draft.docx` | `306a9bcb384e37fcbdc8ced507807cf4c9edc1cf20baf81c1d97f192830a84da` |

The TMLR PDF remains anonymous; its author metadata is empty and a text check of
all 38 pages finds no named-author, author-repository, host-path, TODO, or FIXME
pattern. This complements, rather than replaces, the manual page review. The
compact PDF and Word file retain their named author mode. These are not three
different scientific versions. Build logs,
previous-output backups, page proofs, the replay comparison, and package-test
artifacts are retained under `/private/tmp/kbound-release-closure.jvR7Tp`.

### 9.7 What is and is not closed

The requested theorem terminology, novelty comparisons, six bibliography repairs,
uncited-reference dispositions, maintained public-runtime fixes, and document
rebuild/visual review are implemented. Numerical findings were preserved, not
optimized or selectively replaced. Validation and PDF/Word QA workflows influenced
the separation of scientific authority from presentation and release status.

A **clean-source publication release is not yet issued**. It requires review and
authorization of the source commit, completion of the whole release gate from
that clean checkout, and an independently verified new source/outer seal. The
root integrity check and the full deployment/test-platform matrix are not silently
claimed complete. Historical prototype limitations and the disabled So2Sat
future-target path remain explicit rather than being disguised as certified
operational behavior.

None of these repairs establishes a new population-risk guarantee or a new
natural-shift routing win. No training, target access, or dataset deletion occurred.
Reproducibility scripts and historical evidence were retained; this section does
not claim that every unnecessary local file or T9 sidecar has been deleted.

## 10. Measurable foundations and final-release boundary review

This follow-up is dated 2026-08-31. It preserves the Section 9 working-tree
snapshot as historical evidence, not as the final clean-source seal. The user has
now authorized a reviewed source-freeze commit and complete release run;
authorization and component checks alone are not release completion.

### 10.1 Five probability layers now have genuine Lean proofs

The integrated pinned Lean 4.29.1/Mathlib build and transitive kernel-axiom audit
pass for **142 registered declarations**: 65 legacy names plus 77 probability
capstones and counterexample results. Supporting lemmas compile as dependencies.
Only `propext`, `Classical.choice` and `Quot.sound` occur in inspected dependency
sets. The registry count is a scope inventory, not a novelty or quality score.

The five layers, detailed in `formal/README.md`, are:

1. Exchangeable measurable score laws, strict ranks with ties, calibration
   thresholds and one-shot residual coverage/directional-error bounds.
2. Filtered nonnegative supermartingales, bounded optional stopping, countable-time
   Ville bounds, domination and a constructed predictable bounded betting product.
3. Arbitrary probability measures and randomized measurable tests, the exact TV
   testing identity, KL/Bretagnolle–Huber lower bound and finite iid products,
   including infinite KL and the zero-observation product.
4. Independent bounded Hoeffding and adapted martingale-difference concentration.
   Paired benefits in `[-1,1]` require twice the unit-interval radius. Nonlinear
   evidence-ratio and empirical-Bernstein rate extensions are not claimed.
5. Measurable label kernels and actual population loss integrals, preserved input
   and evidence laws, unchanged labels off disagreement, and exact clipped strict
   frontiers without assuming `RichAt`. This is the full measurable correctness-field
   class supported on the two predictions on disagreement, with feasible margins,
   nonnegative residual budgets and positive disagreement mass. It does not assert
   richness of arbitrary restricted deployment classes or cover unrestricted
   multiclass kernels automatically.

Benchmark exchangeability, independence, conditional nulls, preprocessing and
calibration transfer remain unencoded deployment assumptions. Batch coverage does
not become population-risk coverage, and reused marginal intervals are not
automatically anytime-valid.

The audit now inspects actual transitive axioms after building. Static-only
inventory checks report zero kernel-verified names. Strict-core without a build,
missing/duplicate dependency output, unexpected axioms, proof holes and compilation
failures fail closed. The source guard handles nested comments, character literals
and interpolated Lean code; Lean's own proof-hole warnings are also rejected.
Builds no longer delete sidecars or silently update dependencies.

### 10.2 The sixth historical extension cannot be marked complete as written

`MeasureSwap.lean` lifts the label-swap involution and opposite-population-risk
obstruction to general measurable label-free channels. However, selecting one
representative per swap orbit does not suffice to identify a sign. Four worlds
`P+`, `P-`, `Q+`, `Q-` can have two swap orbits and one evidence law. The class
`{P+, Q-}` selects one representative per orbit yet retains opposite signs with
identical evidence. `ChannelCounterexample.lean` verifies this counterexample and
the correct set-theoretic replacement: consistency on the entire evidence fibre.
A measurable decoder still requires the corresponding measurable structure.

The excluded historical `onebit_audit_rate.tex`, `main_theory_5.tex` and
`knowability_dichotomy.tex` also require more than terminology edits:

- On disagreement, binary correctness indicators satisfy `C0 + Ca = 1`;
  conditional independence forces degenerate correctness probabilities. With the
  stated class symmetry and an interior class prior, the advertised broad H/ratio
  model collapses to an extreme subcase.
- Some class-gap/rank-one equivalences omit an interior-prior condition or fail
  to allow the case of only one nonzero class gap.
- With fixed margin M, a swap sends the residual to `-gamma - 2M`, not in general
  to `-gamma`.
- Blind parameter regimes prevent describing every assumption except an
  orientation bit as empirically testable.

The compact submission excludes these stronger historical extensions. The
integrated `--build --full-foundations` run has been performed: build and all 142
axiom checks pass, but the overall stronger gate **fails for the one disclosed
historical one-bit/H/ratio-rate extension**. This is not six-layer closure.

### 10.3 Runtime fixes and source provenance

Independent review reproduced masked NumPy calibration/features being stripped
to ordinary arrays and nonfinite predeclared e-value bounds passing validation.
Both are fixed. Missing entries remain nonfinite without row deletion or
imputation; scalar validators reject them, and service/batch/routing paths apply
their unavailable-ABSTAIN policy while retaining the frozen predictor and original
candidate-family size. Invalid e-value support is rejected before certification.
`KGA.explain()` uses JSON null for nonfinite fields without changing its certificate.

The selected runtime/API suite passes **322 cases in each of two independent
dependency environments**, including 98 added boundary cases. A further 42
synthetic provenance tests pass in both environments. The current-policy inference
producer and both consumers now require exact hashes and paths for the policy,
certificate, new numeric-validation helper and protocol. Empty, omitted or
substituted bindings fail. The standalone canonical panel's generator hash is
not described as a complete transitive source seal; the final outer/source seal
also binds its runtime dependencies.

Before canonical regeneration, all **196 protected evidence/authority hashes**
match the previous baseline. No dataset, checkpoint, CCT-20 target evidence or
So2Sat development evidence was changed. An independent comparison of the refreshed
current-policy inference confirms exact equality of all 197 numerical, 30 Boolean
and 86 textual scientific leaves, including actions, intervals and p-values.
Only the timestamp, analyzer/policy/certificate identities and added validation-helper
binding changed. The four declared source bindings were independently rehashed.
The full canonical panel comparison then passed independently: all nine panel
payloads are unchanged, including 8,514 numerical, 365 Boolean, 3,586 textual and
101 null leaves. All 106 compact-input lineage rows are unchanged. Exactly three
provenance fields changed: the generator hash in each of the canonical panel and
source manifest, and the panel's source-manifest hash.

For this comparison, restoring only those known old metadata fields yielded files
whose whole-file SHA-256, byte count and indexed Git blob identity exactly matched
the independently recorded pre-edit authorities. The resident old generator blob
also matched its earlier identity. No numerical value was adjusted to obtain the
match. This establishes source-reseal invariance, not a new experiment; it does
not replace the final clean-commit pipeline.

The current root distribution was also rebuilt offline in an isolated directory.
The wheel contains 21 package files, including the new validation helper, plus
six metadata files; all 26 hashed RECORD entries were verified. Rebuilding from
the source archive yields identical member contents, including RECORD, but not
identical ZIP bytes. All 220 installed-package synthetic tests pass, and all 27
build inputs still match the working tree. These are package/component checks,
not a clean-source release seal or a general production/security certification.

The release/cleanup guard suite passes 98 cases, including tests of real Bash
source-freeze enforcement, all 52 required checksum entries and preservation of
metadata during ordinary reconciliation. The new runtime and formal-audit
regressions are explicitly executed by the release runbook and relevant CI jobs.

### 10.4 Git and final-release checkpoint

The complete non-shallow commit graph verifies 275 entries. Five resident packs
and 2,144 resident loose objects passed independent local checks. One loose blob
was restored from unchanged resident source bytes matching its exact Git hash;
its original cloud placeholder is retained recoverably. These component results
are not a successful full fsck.

The sole full strict fsck timed out after its 30-minute bound while waiting on
cloud-only objects; an unavailable pack reports an iCloud account/transfer error.
This is **unverified full integrity,
not demonstrated corruption**. No historical object was invented from different
current source bytes. Permission review rejected an exact-object GitHub recovery
request before execution; user authorization has been requested. No push or
branch change has occurred at this checkpoint.

The current HEAD tree is now independently verified. Reconstructing its directory
objects from the index produced exactly the tree identity in the verified HEAD
commit. Nineteen unavailable tree objects and the original canonical-results blob
were restored from exact-hash bytes; 11 other approved objects had become available
and were verified without replacement. All 20 original placeholders are preserved.
Git now enumerates all 4,244 HEAD paths, modes and object IDs, matching the index.
This closes the current-tree structure check, not full historical object integrity
or availability of every working file.

An independent read-only recovery inventory contains **125 exact-HEAD file
copies**: 123 of 124 required source/test files plus two audit inputs. Each saved
copy matches the recorded blob ID, SHA-256 and mode. Bulk restoration into the
working tree was rejected by permission review before execution: **none of these
125 working files was restored or overwritten**. Explicit user approval is needed
to preserve their cloud-only placeholders in backup and recreate only the exact
verified files. The remaining unavailable source is
`docs/research/kbound/edge/tests/test_protocol_inventory_reporting.py`, blob
`163ca8d2761e2945245c8544758bb434e730205f` (2,382 bytes). No available local
worktree, resident Git object or exact T9 path supplied that file.

The bounded recovery inventory is
`/private/tmp/kbound-required-object-lookup.d6x3QB/combined_125_copy_verification.json`,
SHA-256 `1206082f141e1825094eebcb31a5d00663eaac90d8713969fed0700ff16242ec`.
The durable scoped Git closeout receipt is retained under
`.git/codex-object-recovery.FnIOeE/head-tree-json-recovery.cwt09u/scoped-evidence/git-integrity-scoped-closeout.json`,
SHA-256 `f8b09d1c5843df70c288edbbde26178bf71270262b379a4bba6a9e1569edc432`.
Its 50 evidence files were independently rehashed. These are recovery/component
receipts, not a clean-source release seal.

There is not yet a final source-freeze commit, successful clean-commit release
run or newly issued final seal. The strengthened release workflow has component
tests for capturing a clean source HEAD before work, rejecting source/HEAD changes
between phases, requiring the complete output-checksum inventory and including
the formal/validation sources. Final commit and artifact identities must be
recorded only after these gates execute.

Pre-release generation refreshed the canonical data and several dependent
artifacts, but did not complete: cloud-backed inputs blocked the official-baseline
audit and dashboard generation. Both stalled processes were stopped without
substituting missing inputs. The latest canonical-data validator reports one
failure: the dashboard snapshot has a stale current-policy identity. The storage
manifest refresh passed for its four mutable authorities, with historical seals
unchanged. The complete release suite and PDF/Word rebuild have not run for the
Section 10 sources. The three Section 9 exports and the 368-file snapshot remain
historical; they must not be presented as incorporating the new Lean scope or as
the final source seal.

None of this creates a natural-shift result: CCT-20 remains `SAFE_UTILITY_ONLY`;
So2Sat has no feasible candidate, no target access and no target natural-shift score.

## 11. Scientific-narrative revision and maintained exports

This follow-up, dated 2026-08-31, implements the subsequent writing review. It
supersedes the Section 10 statements that the current paper dashboard and exports
have not been rebuilt. It does not supersede the unresolved Git, cloud-source,
clean-commit, or stronger empirical-evidence boundaries. No training, tuning, new
target access, dataset deletion, source recovery, Git commit, or push occurred in
this narrative revision.

### 11.1 Review-to-revision map

| Review issue | Implemented change |
| --- | --- |
| Abstract and main contribution obscured by audit details | The shared abstract is 206 whitespace-delimited words. The main text has one explicit research question, three scientific contributions, and the requested ten-section structure. |
| Population theory and empirical controller conflated | Section 6 identifies KGA as an empirical companion and distinguishes population benefit, observed cell benefit, and the fitted benefit prediction. KGA does not estimate or numerically apply the population residual frontier. |
| Deployment label access unclear | Development and calibration use historical labeled outcomes; the new deployment decision does not read its evaluation labels. |
| FREEZE and ABSTAIN conflated | Supported negative intervals produce FREEZE; unavailable or unresolved assessment produces ABSTAIN while retaining the frozen predictor. Additional human review or label acquisition is an optional operational response, not an experimentally validated component. |
| Elementary interval implication overstated | The former certificate theorem is an operational coverage-to-action proposition, with stable cross-reference identifiers. Coverage of the declared scalar target remains a premise, not a new calibration result. |
| Notation, metric and protocol ambiguity | Notation and claim levels precede the population theory. Commitment rate is distinct from interval inclusion. Safe utility, its estimand and strict thresholds precede results. Ordinary-accuracy and balanced-accuracy diagnostics use separate tables. |
| Favorable and invalid evidence interleaved | The primary controlled table includes Tent, EATA and the adverse SAR result. CCT-20 retains its limited conservative-retention interpretation. Auxiliary, withheld, invalid and historical studies remain in the supplement. |
| Closest work missing | The related-work comparison now discusses Kim et al. (NeurIPS 2024), Jin and Ren (JRSSB 2025), Gibbs, Cherian and Candès (JRSSB 2025), and Christensen, Moon and Schorfheide (2026 corrected advance proof), rather than merely adding citations. |
| Audit material dominated the main paper | Hashes, commands, historical census discrepancies, invalid records, detailed contracts and Lean scope are in the separately maintained supplement. Both PDF drivers place the bibliography before that supplement. |

The four added primary records are the published
[TTA/agreement-on-the-line paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d96fcc07d623a9eba68616629911143a-Abstract-Conference.html),
[selection-conditional coverage paper](https://academic.oup.com/jrsssb/article/87/4/1239/8113856),
[conditional-guarantees paper](https://academic.oup.com/jrsssb/article/87/4/1100/8058684),
and [partially identified payoffs paper](https://doi.org/10.1093/restud/rdag017).
There are 54 printed references. The thirteen contextual archive entries remain
preserved, not silently deleted. The comparison identifies differences in target,
decision rule and assumptions; it does not claim an official empirical
head-to-head comparison that has not been run.

The statistical wording retains joint coordinatewise sign invariance as the
sign-flip reference model. Exchangeability, marginal symmetry, or one global
sign symmetry alone is insufficient. Bootstrap confidence levels are nominal.
The controlled calibration is described as leave-one-cell-out empirical
order-statistic calibration, not exact split-conformal calibration.

### 11.2 New descriptive interval diagnostic, not new confirmation

`scripts/build_current_policy_interval_diagnostics.py` replays the current
`kga.policy.decide_kga` rule from the fifteen resident CIFAR-10-C compact inputs.
It ignores historical stored actions and radii, uses each original candidate/seed
residual pool, and excludes the scored residual. Its 176 input/provenance checks
and 267 exact scientific-equality checks passed. The `--check` mode independently
recomputes and compares the generated outputs without overwriting them.

| Candidate | Cells | Nominal target | Observed inclusion | Mean full width | Commitment rate | False FREEZE / FREEZE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tent | 2,160 | 0.90 | 1,945/2,160 = 0.900463 | 0.042559 | 0.678704 | 1/359 |
| EATA | 2,160 | 0.90 | 1,945/2,160 = 0.900463 | 0.034044 | 0.635648 | 0/132 |
| SAR | 2,160 | 0.90 | 1,945/2,160 = 0.900463 | 0.026031 | 0.669444 | undefined: no FREEZE |

All three candidates have zero observed false adaptations. Tent's one false
FREEZE is in the Gaussian-noise family, which contains only two FREEZE decisions.
The supplement includes all eighteen candidate/family rows, full-width summaries,
action denominators, and defined/undefined error rates. The width units are
accuracy proportions, not percentage points.

The pooled inclusion rate is strongly constrained by reuse of the same scored
collection in overlapping leave-one-out residual pools. Near-90% inclusion is
therefore not independent validation of calibration, next-family coverage,
natural-environment transfer, or population-risk protection. Family breakdowns
describe these opened data; they do not confer group-conditional guarantees.

The diagnostic JSON has SHA-256
`3fed622171168a5a3aa71d3c36b7ba823d040ebddc52fc1abf56103aea340338`.
Its main and group TeX displays have SHA-256
`3ddddf248e6d45f22f6359622101eaa5358c0df8fda68f69583233d11c1b212f`
and `7dcd64ac7690ced476c5e932be90a764abcf20f53699644cd9ef49e293ce927e`.

### 11.3 Numerical authorities and empirical limits preserved

Independent final rehashing confirms that these protected authorities are
byte-identical to their starting identities for this revision:

| Authority | SHA-256 |
| --- | --- |
| Canonical panel | `4239a7b38b087fe8d4c81eb935d9da5104545a228fffb7c956b29c6e00d97062` |
| Canonical source manifest | `266ce7cd87ae795e0742d43c40afefef82e82b555baa7dda8c6d270f73e2c031` |
| CCT-20 release manifest | `722d2ebbe2d883c7eb173d72af9e4aa4c0a99b1ec320d913bf668f07d28eff48` |
| CCT-20 sealed primary table | `e3ce875900bead95095db20506dcf2cbc6a03cb51ef750549e89d36a2c845b7f` |

CCT-20 has 45 checkpoint/location cells: 0 ADAPT, 44 FREEZE and 1 ABSTAIN.
The candidate helps in one cell, ties in zero, and harms in 44. KGA retains the
frozen predictions in every cell. The nominal pointwise 95% safe-utility intervals
are `[0.0937477621, 0.2913862611]` for always-adapt regret minus KGA regret and
`[0, 0]` for always-freeze regret minus KGA regret. The strict lower-bound
requirements are respectively greater than zero and greater than `-0.005`.
These satisfy the recorded limited endpoint, not the stronger routing criterion.
The stronger nominal 97.5% inference and full location ledger remain disclosed.

So2Sat remains a negative development-gate stop: no feasible candidate, no gate
calibration, no target input or label access, and no target natural-shift score.
iWildCam remains withheld; PACS remains aggregate-only for the missing replay
fields; historical POEM/AETTA ports remain non-official and out of sync with the
current policy. No adverse result was converted into a win by changing its name,
metric, sign convention, denominator, or inclusion status.

The remaining empirical requirements are unchanged: official current-policy
comparisons (including simple benefit-estimation baselines), an untouched natural
shift with both helpful/harmful updates and nonzero ADAPT/FREEZE exposure,
independent group-coverage evaluation, and independent-checkpoint/held-out-
environment inference. Reformatting and this retrospective diagnostic do not
satisfy those requirements.

### 11.4 Export and dashboard repairs

The Word converter now derives the algorithm from the maintained source rather
than a hard-coded older version. It preserves the early unavailable-assessment
ABSTAIN branch, all strict interval branches, and equation references; unsupported
algorithm commands or malformed branch structure fail explicitly. Bibliography
formatting stops at the next top-level heading, so appendices are neither styled
nor counted as references. Numbered multirow displays preserve their numbering;
the main bridge uses separately readable displays in both formats.

The PDF builder initially stalled while opening an old cloud-only LaTeX log.
The revised builder uses fresh local temporary intermediates and seeds empty
auxiliary files there so first-pass Hyperref cannot fall back to old checkout
sidecars. It validates the newly produced PDF and log, then atomically publishes
only allowlisted derived artifacts. Failed builds preserve previous deliverables
and retain temporary diagnostics. This does not restore unavailable source/Git
objects or authorize broad deletion. Two real temporary TeX/BibTeX builds and
failure-preservation regressions verify the behavior. Final logs contain no
overfull boxes, unresolved references, or citation warnings. Underfull-box
notices remain ordinary layout diagnostics.

The short PDF's workflow and frontier figures now use full width for readable
labels. A long reference URL was replaced in the printed bibliography by its
linked DOI. Supplementary tables now follow Appendix A and have explanatory text
under their own headings instead of leaving empty headings.

The new dashboard `--paper-only` mode refreshes verified resident manuscript
bindings while preserving the entire physical-edge payload exactly. Independent
JSON comparison confirms `edge_validation` is unchanged. Provenance explicitly
says `preserved_not_rechecked`; neither physical clips nor edge session gates were
revalidated. The subsequent PDF-metadata refresh preserves this distinction.
The final strict canonical-data validator now passes. The separate complete
official-baseline/physical-edge generation remains unfinished.

### 11.5 Component verification and current file identities

The final selected tests passed in separate processes, with zero failed, errored,
or skipped cases:

| Test group | Cases | Temporary JUnit receipt |
| --- | ---: | --- |
| Manuscript, Word, bibliography, diagnostics, metric displays and dashboard | 770 | `final-manuscript-component-tests.xml` |
| CCT-20 release builder | 22 | `final-cct20-builder-tests.xml` |
| Current-policy source bindings | 42 | `final-current-policy-tests.xml` |
| PDF isolation and source-seal guards | 34 | `final-build-seal-tests.xml` |

These are 868 distinct component cases, not a successful complete release-suite
run. Receipts are under `/private/tmp/kbound-narrative-revision.6iHvYB/`.
The final manuscript validator passes over 23 maintained LaTeX sources plus the
long driver, 13 consistency surfaces, 11 direct storage hashes and 71 sealed
evidence hashes. The strict canonical validator and independent interval
diagnostic check also pass.

A combined test collection aborted during Torch import. Fresh import-only
children reproduced `import sklearn; import torch` exiting with SIGABRT and
`OMP: Error #179: Function Can't open SHM failed` (system error 0). Controls using
`torch` before `sklearn`, or `scipy.stats` before `torch`, exited successfully.
This establishes an import-order-sensitive native OpenMP startup problem in the
local environment, not a failed scientific assertion and not, by itself, proof
of an OS permission denial. The two affected component modules passed separately
without an OpenMP override, dependency mutation, or suppressed assertion. The
combined-process environment issue remains to be resolved for the full release.

The maintained files, not additional delivery variants, were rebuilt:

| Artifact | Rendered pages | SHA-256 |
| --- | ---: | --- |
| `kbound_short_final_draft.pdf` | 34 | `960908c9d89527e2a55d5d112d2750559ba6429629c1f58afef6ef3a3bb19246` |
| `kbound_tmlr.pdf` | 36 | `b26671e581b70f24e6d1b618db815e48240858ba7fde8dd411427542ca058792` |
| `kbound_short_final_draft.docx` | 31-page Word-rendered proof | `674d1e98fbcc21c30c9e270ac74efdde1a968c1d7a5c3be3ea9d4f9c1e789cdf` |

The PDFs were rendered at 144 dpi. The long PDF's metadata author is empty, and
its extracted text and link targets contain no author, institution, email,
personal checkout or volume path. This PDF check is not an anonymity audit of an
unbuilt source/archive bundle. The short PDF and Word companion intentionally
remain named. References precede appendices in all three formats.

All 31 Word proof pages have been visually checked; after the last bibliography
and supplementary-text change, the 29 unchanged pages were confirmed pixel-equal
and the two changed pages were reinspected. All 36 long-PDF pages have been
visually checked, with the seventeen changed supplementary pages reinspected.
All 34 short-PDF pages have been visually checked, including individual review
of all seventeen supplementary pages. There are no blocking rendering defects.
The long PDF's reproduction command has ordinary soft wrapping in the appendix;
the repository runbook remains the executable source of that command.

These hashes identify the current working-tree exports only. They are not a new
`KBOUND_RELEASE_SHA256SUMS.txt`, clean-commit source seal, artifact-only commit,
full Git-integrity certification, or completed publication submission. The
Section 10 source-recovery and historical-object availability limits remain
unresolved. The five positive Lean foundation layers and the sixth layer's
counterexample retain their declared scope; this writing revision ran no new
Lean build and does not claim six-layer closure.
