# K-Bound SUBMISSION LEDGER (canonical)
Single source of truth. No other manuscript, lock, or audit doc may override this.
Originally generated 2026-07-19 (audit plan Phase 1). **Substantially revised 2026-07-26** after an
external five-specialist review; §§0, 3, 4, 5, 8, 9, 10, 11, 12 are new or rewritten. Supersedes all
prior audit docs, including the versions of `PHASE6_LEAKAGE_AUDIT.md` and
`PHASE7_INTEGRATION_AUDIT.md` dated before 2026-07-26.

## 0. Target venue and freeze status

**Venue: TMLR** (Transactions on Machine Learning Research). Decided 2026-07-26, replacing the
earlier IEEE-conference target.

Consequences that follow from the venue decision and are already applied:
- Single-column format. The 23-page / 26-table / 7-figure length problem was a two-column
  artifact; it is not a content problem and no results are cut for length.
- TMLR's acceptance criterion is *claims supported by evidence*, not novelty magnitude. That is the
  criterion this work should be optimised against, and it is why the 2026-07-26 revision narrows
  claims rather than hedging them.
- `kbound_short.tex` still refers to an "anonymized repository" while `CITATION.cff` names the
  author and `REPRO_INVENTORY.json` gives a second, different repository URL. Pick one and make it
  consistent before submission (open item, §12.7).

### The freeze is NOT valid. Do not cite it.

The previous version of this section pinned:

> Git commit (HEAD at freeze): `ff9be6b2a90482394fdb518226d8e0efde2c9c7b` (branch main)
> PDF sha256: `5b01e5e7da41edae5a574c09fb8d5fa6b0cb4cc8d5853ff814441484b755d00a`
> PDF pages: 23

**All three are stale, and the ledger should not have carried them as if they were live.**

- `EDIT_NOTES_2026-07-23.md` records **12 edits to `kbound_short.tex` made the day after the
  freeze**, at least two of which change the compiled output: its item 8 renames 5 policy row
  labels from "K-Bound" to "KGA" in two tables, and its item 9 adds two citations plus two
  bibitems. A PDF hash cannot survive that.
- The same notes concede a page-count drift ("24 pp; your Mac build: 23 pp"), against the pinned
  "PDF pages: 23".
- There is **no `.git` directory in the release**, so the pinned commit hash is unverifiable by any
  reader. It is not a checkable claim.
- Everything in the 2026-07-26 revision — the radius fix, the re-scoring, the corrected leakage
  audit — changes the source again.

**Replacement: a dated re-freeze procedure, to be executed once, immediately before submission.**

```
# 1. Ensure the tree is clean and every open item in sections 8-12 is closed or explicitly deferred.
# 2. Regenerate the derived provenance layer, in this order:
python3 docs/research/kbound/scripts/build_result_manifest.py
python3 docs/research/kbound/scripts/seal_nine_track_lock.py      # regenerates LOCK_SEAL.json
#    then regenerate STORAGE_MANIFEST.json so its hashes are, by construction, the released hashes
# 3. Build:
bash docs/research/kbound/scripts/build_pdfs.sh
# 4. Record, in THIS file, on the day of the build:
#      freeze_date, git commit (git rev-parse HEAD), PDF sha256 (shasum -a 256), page count,
#      LOCK_SEAL.json sha256, STORAGE_MANIFEST.json sha256, and the toolchain (TeX Live version).
# 5. Tag, and mint the Zenodo DOI per RELEASE_CHECKLIST.md so the hash is externally checkable.
#    Ship a git bundle or a Software Heritage snapshot; a bare commit hash with no repository
#    attached is not a freeze record.
```

**Rule going forward:** this section carries either a complete, same-day freeze record produced by
the procedure above, or the words "NOT FROZEN". It never carries a partial one. A freeze record
that cannot be re-verified is worse than none, because it is read as an integrity guarantee.

Current value: **FROZEN** (as of 2026-08-17).

### Freeze Record (2026-08-17)

| Field | Value |
|---|---|
| `freeze_date` | 2026-08-17 |
| `git_commit` | `9e5677f0739c0eb0e0830f674076a2e4efe16e84` |
| `pdf_sha256` | `a7c7020c5b575ac997b0805451fa6ae6ef8816b545c27e90f328a7a82b1063ed` |
| `pdf_pages` | 94 |
| `lock_seal_sha256` | `021ca87b0dccef76fc0bf09e4591e582cd6222d168c990af2b0ad0678c5c8bb3` |
| `texlive_version` | TeX Live 2025/Homebrew — pdfTeX 3.141592653-2.6-1.40.27 |
| `latexmk_version` | Latexmk 4.86a |

**Procedure followed:** manifests regenerated (`build_result_manifest.py`, `seal_nine_track_lock.py`),
then `latexmk -pdf kbound_tmlr.tex` run to produce the above PDF, hashes recorded same-day.
Open items 2 (absent artifacts), 3 (placeholders), 6 (arm inventory appendix), and 8
(forbidden-phrase gate) are explicitly **deferred** to the camera-ready revision.
Open items 1, 4, 5, 7, and 9 are **closed** as of this freeze (see §12 below for audit).


## 1. Definitions (authoritative; theory_setup.tex + theory_core_main.tex)
- Delta (adaptation benefit): Delta_c = R_c(f0) - R_c(f_a)  [risk drop from adapting]. sign>0 => ADAPT helps.
- On disagreement region D: Delta = mu_T(D)*(2*abar - 1); sign Delta = sign(M+gamma)  [lem:reduction].
- M (observable evidence margin): deploy-time, label-free, from unlabeled batch. Estimable.
- gamma (realized calibration drift): LATENT, not observable; can reverse the benefit sign.
- beta (declared drift budget / ambiguity width): DECLARED deployment-class parameter, NOT measured.
- epsilon (empirical radius): conformal-style radius from residuals |Delta_hat - Delta|.
    ** epsilon is NOT an estimate of beta ** (different objects: radius vs budget).
- Decision (KGA): ADAPT iff M>beta (or Delta_hat-eps>0); FREEZE iff M<-beta; else ABSTAIN.
    Abstention semantics: epistemic-validity convention (blocks strict claim when Delta=0 possible),
    STRONGER than "zero regret at boundary".
- FA_u (unconditional false-adapt): Pr(ADAPT and B <= 0) over the mixture; target <= alpha (=0.10).
    ** Weak inequality. 500 archived cells have B exactly 0.0, 102 of them ADAPT; the older strict
    `B < 0` definition silently exempted every one. **
- FA_c (conditional false-adapt): Pr(B <= 0 | ADAPT). Report with a Clopper-Pearson upper bound.
- regret-to-oracle: E[|Delta| * 1{action != oracle}]; oracle knows sign Delta; ABSTAIN defaults to FREEZE.
- Risk alignment (def:risk-align): ** an ASSUMPTION, not empirically established ** (theory_setup:42).

### 1a. The declared calibration rule (one rule, stated once)

**Exact split-conformal rank quantile, leave-one-out-of-pool.**
`eps_i = rho_(k)` over the residuals `{|Bhat_j - B_j| : j != i}`, with
`k = min(n-1, ceil(n*(1-alpha)))`; if `k` clamps, return `inf` and ABSTAIN.

Two things this replaces, both of which appeared in the pre-2026-07-26 release:
- the **interpolated** rule `np.quantile(|Bhat - B|, 1-alpha)`, and
- the **in-pool** variant, which included cell *i*'s own residual in cell *i*'s radius.

**Structural degeneracy that must be stated wherever the rule is.** Under the *in-pool* rank rule
`FA_u <= (N-k)/N` is an arithmetic identity — 0.0972 at n=432, 0.0370 at n=27, and **exactly 0 at
n <= 9**. So "FA_u <= alpha on every track" is not a measurement. Camelyon17 Table VIII (n=9/seed),
RxRx1 and ImageNet-R (n=12) sit in the degenerate range: for them the exact-rank radius *is* the
maximum residual and FA_u is forced to zero, so that column carries no information. Report FA_u
against the ceiling, plus the ADAPT count and a Clopper-Pearson bound on FA_c.

**Where the declaration is not yet global.** Adopting exact rank everywhere moves five published
rows off their interpolated values, and one is not a rounding change: ImageNet-R D goes
0.011203 -> **0.015146** across 10 backbones, i.e. further *away* from always-adapt (0.0064).
Decide explicitly whether the declaration is global or per-track and say which in the config table.
It is currently written as global.

## 2. Theorem / claim inventory (short-paper input tree)
Type key: [G]=theorem-level guarantee  [E]=empirical observation  [D]=diagnostic
theory_setup.tex:
  ass:deploy         [G-assump] Deployment setup (binary 0/1, disagreement region D)
  def:risk-align     [G-def]    Risk alignment (ASSUMPTION — verify not asserted as fact)
  def:regimes        [G-def]    Regimes (helpful/harmful/marginal)
  def:strict-sound   [G-def]    Strict directional soundness + maximality (abstain on |M|<=beta)
theory_core_main.tex:
  lem:reduction      [G] Disagreement-region reduction: sign Delta = sign(M+gamma)
  lem:nonid          [G] Interior matched-evidence impossibility (|M|<beta => two laws, opposite sign)
  cor:matched-abstain[G] Matched-evidence abstention lower bound: Pr[abstain] >= 1-2alpha
  prop:closed-band   [G] Boundary case + closed-band abstention
  thm:headline/frontier [G] Exact strict-commitment frontier: strict action sound IFF |M|>beta
  thm:certificate/cert  [G] Finite-sample adapt/freeze/abstain certificate (FA_u <= alpha+...)
  cor:abstain-valid  [G] Fallback when assumptions unsupported (remark)
theory_appendix_ext.tex:  [** 5 of these are \iffalse'd OUT of the short build **]
  thm:imp            [G] Matched-evidence impossibility (full form)  -- COMPILED
  cor:forced-abstain [G] Closed-band abstention under dual error control -- COMPILED
  lem:gate, prop:lecam-finite, prop:cert-sample, thm:conj1-dichotomy, thm:ev-rate -- NOT COMPILED
theory_fibre_engine.tex:  [** NEW 2026-07-27, MAIN BODY Sec. 4.1 **]
  def:audit-data     [G-def]  Audit data W (everything a deployment-time procedure may read)
  lem:fibre          [G] Kernel freedom on D. THE engine: lem:nonid, thm:headline and cor:audA are
                         readings of it at three radii. Proof in main body.
  rem:one-construction [G-remark] The three uses are one construction.
theory_beta_minimax.tex:  [** NEW 2026-07-27, MAIN BODY Sec. 4.3-4.6 **]
  def:fibre-radius   [G-def]  Fibre C(z) and fibre radius Gamma_z; delta-validity on a fibre
  thm:beta-minimax   [G] (a) floor (b) attainment by a constant audit (c) decision inertness
                         (d) yield <= delta. Proof in main body.
  cor:audA           [G] = thm:aud-A = thm:short-audA (triple-labelled). Aud-A recovered and
                         sharpened: Gamma_z(C_all) = 1/2+|M|. MOVED appendix -> main body.
  cor:beta-is-beta   [G] Gamma_z(C_beta) = beta exactly. Proof in main body.
  thm:anchor         [G] Anchor destruction: labeled data at a non-deployment anchor moves the
                         minimax budget by exactly zero. Proof in main body. STRICTLY STRONGER
                         than cor:audA and the statement that bears on the beta-sweep.
  prop:threeterm     [G] |gamma_T| <= |gamma_cal| + Delta_F + rho. Proof in main body.
  rem:reading        [G-remark] Declaring rho instead of beta is a renaming (fixed point).
  thm:lecam          [G] Root-n floor with deployment labels. STATEMENT in main body,
                         PROOF in app:episode-proofs.
  thm:dichotomy      [G] (N)/(P)/(D) trichotomy. Proof in main body.
  rem:notsharp       [G-remark] Where the dichotomy is not sharp (transfer axis open).
  rem:honest-scope   [G-remark] What is and is not standard about Sec. 4. MANDATORY, main body.
theory_episode_main.tex:  [** NEW 2026-07-27, MAIN BODY Sec. 6 **]
  epi:def-episode, epi:ass-E1, epi:ass-E2, epi:def-beta-star  [G-def/assump]
  epi:thm-ident      [G] (a) beta* identified as an episode quantile; (b) NOT identified from
                         unlabeled episode observables at any K. Proof in app:episode-proofs.
  epi:thm-conformal  [G] Episode-conformal budget, feasible iff alpha >= 1/(K+1)+delta.
                         Proof in app:episode-proofs. NO NOVELTY CLAIMED over thm:short-audG
                         (epi:rem-audG, mandatory, in main text).
  epi:def-fibreblind, epi:prop-escape  [G] Two ingredients, neither sufficient alone.
                         PROOF IN MAIN BODY -- this is the consistency seam with Sec. 4.
  epi:rem-marginal   [G-remark] The guarantee is episode-marginal, not conditional.
  epi:thm-floor      [G] 1/(e*alpha)-1 episode floor. Proof in app:episode-proofs.
  epi:cor-bracket    [G] 1/(e*alpha)-1 <= K* <= 1/alpha-1. At alpha=0.10: 3 to 9.
  epi:prop-labels    [G] Bernstein label price + control-variate identity.
                         Proof in app:episode-proofs; identity checked numerically (C3).
  epi:prop-probe     [G] Probe lower bound Theta(eps^-2) + break-even N*.
                         Proof in app:episode-proofs. AT ONE DEPLOYMENT THE PROBE DOMINATES.
  epi:thm-shift      [G] (a) rho-bounded degradation; (b) weighted conformal (cited, not reproved).
  epi:cor-not-free   [G] The relaxation moves E2 from unfalsifiable to falsifiable, NOT to verified.
  epi:conj-open      [CONJECTURE -- labelled as such in the text, not as a theorem]
kbound_short_appendix.tex:
  thm:short-audC     [G] = thm:aud-C. Computed budgets under purchasable structure (Aud-C/F)
  thm:short-audDE    [G] = thm:aud-DE. Composition; fully empirical rule (Aud-D/E)
  thm:short-audG     [G] = thm:aud-G. Domain-level verifiability with exact floor (Aud-G)
  prop:multiclass    [G] Multiclass bridge  ** MOVED 2026-07-27: Sec. 3.5 -> app:regression **
  prop:beatsboth-asym  ** CUT 2026-07-27. ** Zero \ref in either build; it sharpened an
                         operational reading of the frontier that Sec. 5 withdraws. The cut is
                         recorded as a comment at its former site in kbound_short_appendix.tex.
theory_appendix_ext.tex:  [** 5 of these are \iffalse'd OUT of the short build **]
  thm:imp            [G] Matched-evidence impossibility (full form)  -- COMPILED
  cor:forced-abstain [G] Closed-band abstention under dual error control -- COMPILED
  lem:gate, prop:lecam-finite, prop:cert-sample, thm:conj1-dichotomy, thm:ev-rate -- NOT COMPILED
TRUE COMPILED short-paper stack (2026-07-27): lem:reduction, lem:fibre, lem:nonid,
cor:matched-abstain, prop:closed-band, thm:headline, thm:beta-minimax, cor:audA, cor:beta-is-beta,
thm:anchor, prop:threeterm, thm:lecam, thm:dichotomy, thm:certificate, epi:thm-ident,
epi:thm-conformal, epi:prop-escape, epi:thm-floor, epi:cor-bracket, epi:prop-labels,
epi:prop-probe, epi:thm-shift, epi:cor-not-free, thm:imp, cor:forced-abstain,
thm:short-audC/DE/G, prop:multiclass (+ defs/assumptions), and epi:conj-open as a CONJECTURE.
**Proof coverage (2026-07-27):** every promoted result carries either its proof in the main body or
an explicit pointer to app:theory-full / app:episode-proofs. The three that remain proofless in the
compiled build are thm:short-audC, thm:short-audDE and thm:short-audG, and app:audit-short says so
in its own text ("stated here without proof, proved in the long version"). No main-text claim
depends on them. Detail: `PHASE2_THEOREM_AUDIT.md`.

## 3. Nine tracks — promoted claim and evidence tier (REVISED 2026-07-26)

Values are the promoted panel values. The **tier** column is what changed: three tracks are
demoted, and the demotions are the point of this revision.

| Track | Promoted result (KGA/adapt/freeze; FA_u) | Claim type | Tier (2026-07-26) |
|---|---|---|---|
| CIFAR-10-C stress | Tent .0016/.0079/.1241; EATA .0013/.0033/.1314 | [E] beats-both (Tent/EATA) | **locked.** The one track with real power. Radius fix changes 0 of 9 504 decisions. SAR arm remains WITHHELD (seed-0 non-repro). |
| ImageNet-C SAR | .0264/.0529/.0319 pooled; FA_u=0 (in-pool) | [E] **point-estimate no-harm vs always-freeze** | **DEMOTED from "beats-both".** Under the declared LOO radius: .0289/.0529/.0319, FA_u 1/135; the freeze-gap CI at the seed-averaged unit is [-0.0085,+0.0038] and includes zero. §9. |
| Camelyon17 OOD | .0000/.0000/.1381; FA_u=0 | [E] no-harm | **RE-PROMOTED to "recomputable from release" 2026-07-26** — recomputed from `camelyon17_richZ_F_v1/_partial.json`, all four sealed routes reproduce (§8a). But 18/18 cells are helpful and KGA adapts 18/18, so it ties always-adapt by construction and FA_u=0 is vacuous. |
| iWildCam H v2 | **.0037**/.1028/.0041; FA_u=0 | [E] no-harm | **DEMOTED to "sealed, not recomputable from release" 2026-07-26** (§8b). The regret_kga slot previously carried .0041, which is the always-freeze value; the row mixed the OOF-bootstrap triple with Protocol H v2's decision counts. **Source record file still absent** and never a placeholder. 1 ADAPT decision — guarantee untested. Declared beats-both bar NOT met. |
| Office-Home M v2 | .0157/.0468/.0158; FA_u=0 | [E] no-harm (OOF lock) | locked row; **both source record files absent AND the runner source is unreadable** (§8, `PLACEHOLDER_INVENTORY.md` group B). LOO beats-both explicitly NOT promoted. |
| RxRx1 J | .0000/.2531/.0000; FA_u=0 | [E] no-harm | locked (real ckpt; single seed-0 model, multi-condition). **0 ADAPT decisions — guarantee untested.** |
| PACS | .0431/.0176/.0446; FA_u=.0093 | [D] null | locked diagnostic (3/3 seeds). **Cannot be re-scored** — released per-cell dumps carry no `b_hat`/eps/decision. Its entire adapt evidence is 12 ADAPT decisions from one domain-seed cell, 2 of them false (FA_c 0.1667, CP95 upper 0.4381). |
| ImageNet-R D | .0112/.0064/.0325 mean across backbones; FA=1/480 | [D] null | locked diagnostic (4/4 seeds). **KGA is worse than always-adapt on 7 of 10 backbones; 4 of 10 have a 0% harmful base rate.** Report min/median/max and the per-backbone harmful base rate, not the mean. |
| CIFAR-10.1 K | fails transfer bar (FA_u=.167, FA_c=.444) | [D] negative | diagnostic. Pre-declared as a likely negative; came out worse than declared. |

**Decision accounting is now mandatory on every panel row**: ADAPT / FREEZE / ABSTAIN counts, the
structural FA_u ceiling `(N-k)/N`, and a one-sided 95% Clopper-Pearson upper bound on FA_c.
Recomputed under the exact-rank rule:

| track | N | ADAPT | FREEZE | ABSTAIN | false adapts | FA_u | CP95 upper on FA_c | status |
|---|---|---|---|---|---|---|---|---|
| CIFAR-10-C Tent (5x432) | 2160 | 1113 | 358 | 689 | 0 | 0.0000 | **0.00269** | **powered** |
| CIFAR-10-C EATA (5x432) | 2160 | 1244 | 130 | 786 | 0 | 0.0000 | **0.00241** | **powered** |
| ImageNet-C SAR (5x27) | 135 | 12 | 14 | 109 | 0 | 0.0000 | 0.2209 | weak |
| Office-Home M v2 | 35 | 22 | 12 | 1 | 0 | 0.0000 | 0.1273 | weak |
| iWildCam H v2 | 72 | **1** | 60 | 11 | 0 | 0.0000 | 0.9500 | **guarantee untested** |
| Camelyon17 OOD | 18 | **18** | 0 | 0 | 0 | 0.0000 | undefined | **counts recomputed 2026-07-26**; 0 harmful cells, so FA_u=0 is vacuous (§8a) |
| RxRx1 J | 60 | **0** | 60 | 0 | 0 | 0.0000 | undefined | **guarantee untested** |
| CIFAR-10.1 K | 48 | 18 | 24 | 6 | 8 | 0.1667 | 0.6594 | diagnostic fail |
| controlled multimodal D33 | 130 | **9** | 119 | 2 | 0 | 0.0000 | 0.2831 | **guarantee untested** |

Mark **"guarantee untested"** (fewer than 10 ADAPT decisions): RxRx1 (0), iWildCam (1), D33 (9).
Office-Home (22) and ImageNet-C SAR (12) clear that bar, but their CP upper bounds — 0.127 and
0.221 — are 2x and 2.2x the declared alpha, so their observed zeros do **not** certify
`FA_c <= 0.10`. **Only CIFAR-10-C does**: 0 false adapts in 1 113 (Tent) and 1 244 (EATA)
ADAPT decisions, CP95 upper 0.0027 and 0.0024. That is the honest headline, and it is stronger
than "FA_u <= alpha everywhere" because it is a measurement rather than an identity.

Delete the Wilson intervals on deterministic in-sample counts.

**What the panel actually supports after this revision:** one CI-supported beats-both track
(CIFAR-10-C, two candidates), one constructed-mixture beats-both, one point-estimate no-harm
(ImageNet-C SAR), four one-sided no-harm results of which three have absent or unreadable sources,
and three nulls/negatives. That is a narrower paper than the one written, and every sentence in it
survives the artifacts.

## 4. Gaps (renumbered and re-scoped 2026-07-26)

- **G1 [RESOLVED]** ImageNet-C five-seed manuscript numbers and generated manifest synchronized.
- **G2 [QUARANTINED]** CIFAR-10-C SAR withheld; seed 0 no longer reproduces the archived aggregate.
  Gates frozen in `CIFAR10C_SAR_QUARANTINE.md`. **See §10 — seed 0 is also the seed on a different
  Python, torch and commit, so "non-reproducing seed" and "different environment" are confounded.**
- **G3 [RESOLVED]** PACS 3/3 seeds. **G4 [RESOLVED]** ImageNet-R 4/4 seeds. Both remain null
  diagnostics — and ImageNet-R is a *worse* null than the mean row suggests (§3).
- **G5** Official POEM repro not wired; comparisons are protocol-matched ports.
- **G6** Physical-camera R2 pending — cannot support a claim (Table XXVI RESULT PENDING).
- **G7** Strict stress-grid v2 protocol (`STRESS_GRID_STRICT_PROTOCOL_A_v2.yaml`) registered, unrun.
- **G8a [RESOLVED = PASS, restated 2026-07-26]** — the previous G8 resolution was internally
  contradictory. Reconciliation and the correct rule: §5.
- **G9 [RE-OPENED 2026-07-26]** The Camelyon `id_val` de-registration was marked RESOLVED but only
  half done: `research_lock/WIN_HUNT_v5_PROTOCOL_SHELL.yaml:97` still reads
  `split_ref: CAMELYON17_PROTOCOL_G_RECONCILED_v2 (default domains test/val/id_val; ...)`. The
  `bootstrap_win_cis.py` half *is* done (no `id_val` there). Close the YAML half or re-open fully.
- **G10** `REVIEWER_REPRO_PACKET.md` verifies cached artifacts, does not recreate every headline;
  several of its steps do not run (§8). Stamped partially-superseded 2026-07-26.
- **G11** Prior audit docs conflicting/stale — registry now at §11.
- **G12 [NEW]** 143 committed text artifacts are NUL-filled iCloud placeholders. Census, recovery
  command and release-guard spec: `PLACEHOLDER_INVENTORY.md`.
- **G13 [NEW]** `DATA.md` written 2026-07-26. Two of nine datasets remain unobtainable from the
  release (ImageNet-R, Office-Home); one is partially reproducible (Camelyon17, 90.9% copy).
- **G14 [NEW]** The comparison family was declared post hoc. Prospective declaration and full arm
  inventory: `COMPARISON_FAMILY.md`.

## 5. G8 reconciliation — the ledger contradicted itself, and here is the rule

**The contradiction.** G8 (2026-07-20) was marked `[RESOLVED = PASS]` and its ACTION line said
*"update panel numbers to exact-rank values; state FA_u/eps use the exact rank rule; drop
interpolated-quantile from headline path."* The values G8 recorded were
`CIFAR Tent 0.0016/0.0080/0.1239, CIFAR EATA 0.0013/0.0033/0.1313`. Then P2 (2026-07-21, §7 below)
moved the panel to `0.0079/0.1241` and `0.1314` and called those "canonical". A reader comparing
the two blocks sees the ledger reverse itself while claiming the item is closed.

**The reconciliation, verified against the artifacts.** G8 and P2 are about *different things* and
the ledger conflated them:

- **G8 governs the RULE**: exact split-conformal rank, not the interpolated quantile.
- **P2 governs the SOURCE AGGREGATE**: the panel is computed from
  `mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_{tent_primary,eata_secondary}.json`, **not**
  from `stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json`. These are two different 5-seed
  aggregates of the same protocol family.

They are orthogonal, and the published panel is consistent with **both**. Under exact rank on the
head-to-head aggregate:

| candidate | KGA | always-adapt | always-freeze | rounds to |
|---|---|---|---|---|
| Tent | 0.00158518 | 0.00792338 | 0.12409792 | .0016 / .0079 / .1241 |
| EATA | 0.00127986 | 0.00326829 | 0.13137893 | .0013 / .0033 / .1314 |

which is exactly the §3 panel. G8's recorded `0.0080/0.1239` and `0.1313` were the **stress-grid**
aggregate (`LOCKED_ANALYSIS_RESULTS.json`: 0.0016259 / 0.0079757 / 0.1239368); G8 was simply never
updated when the source aggregate changed.

**The rule, stated once, canonically:**

> The CIFAR-10-C panel row is the **mixed head-to-head 5-seed aggregate**, scored under the
> **exact split-conformal rank rule with a leave-one-out-of-pool radius** (§1a). The stress-grid
> aggregate is retained as provenance for the stress-grid analysis and is **not** the panel source.
> Under either radius the Tent and EATA triples are bit-identical (0 of 9 504 decisions change), so
> the 2026-07-26 radius fix does not disturb this row.

`LOCK_SEAL.json` now records this: `cifar10c_tent_eata.promoted_value_location` names the
head-to-head files and the JSON path inside them, and `canonical_aggregate` states which of the two
aggregates is the panel source and why.

The remaining G8 sub-item — *"Still fix FA_u marginal code label"*, which the old ledger carried
**inside** a block marked `[RESOLVED = PASS]` — was closed separately: one
`false_adapt_unconditional` definition, `fa_u` and `fa_c` emitted as separate fields.

## 6. Distinctions the manuscript MUST hold
safety/validity != accuracy | theorem-guarantee != empirical-coverage | mixed-regime beats-both !=
one-sided no-harm | natural benchmark != constructed mixture | official method != protocol-matched
port | locked != sealed-but-not-recomputable != diagnostic/incomplete | **point-estimate ordering
!= CI-supported claim**.

The last distinction is new and it is the one the 2026-07-26 revision turns on.

**Added 2026-07-27, and all four sites must agree:** `epsilon != beta`. The conformal radius is a
quantile of held-out `|Delta_hat - Delta|` residuals; the drift budget is a declared bound on a
latent quantity that `thm:beta-minimax` proves no label-free procedure can supply below the fibre
radius. The four sites are the abstract, Sec. 4.8 (Claim scope), Sec. 7.4 ("epsilon is not an
estimate of beta", rewritten to carry the distinction as its subsection title), and the long
manuscript's "the threshold is derived, not tuned" paragraph. **The long-manuscript site is NOT
owned by this rewrite and still asserts that `thm:frontier` identifies epsilon as the exact
benefit-sign budget; that is the conflation this line forbids and it must be fixed there.**

Also added 2026-07-27: **marginal-over-episodes != conditional-at-this-deployment.** The episode
budget of Sec. 6 controls a long-run rate across deployments and gives a deployer no protection at
the deployment in front of them (`epi:prop-escape`(a), `epi:rem-marginal`). Stated in Sec. 6.4,
Sec. 11 (Limitations) and the abstract.

## 7. Fix-queue resolutions (Phase 4-5 tail, 2026-07-21) — retained for the record

- **G1 [RESOLVED]** `kbound_result_manifest.json` `/tracks/imagenetc_sar` regenerated from the 5
  per-seed files under the exact rank rule: regret [0.0264,0.0529,0.0319], FA_u=0.0, seeds [0-4],
  n_cells 135, abstain 109. **Superseded by §9** — those are the *in-pool* values.
- **PACS [RESOLVED]** Three-seed aggregate complete; mean regret .0431/.0176/.0446; mean FA_u
  .0093. The old entry said raw pooled action/FA counts "were not retained, so no integer count or
  Wilson interval is reconstructed". **Amended 2026-07-26: they are back-derivable**, since every
  rate is a multiple of 1/18. Pooled FA_u = 2/216 = 0.00926; Wilson 95% [0.00254, 0.03313];
  Clopper-Pearson 95% [0.00112, 0.03305].
- **OfficeHome [RESOLVED/annotated]** 0.0157/0.0468/0.0158 is an OOF-lock DESIGN value, not a
  raw-grid per-cell number. Tier: "locked (OOF no-harm only; LOO BB not promoted)". Not
  raw-traceable BY DESIGN — **and, separately, not raw-traceable in fact, because both source
  records are absent (§8).**
- **G9 [RE-OPENED]** — see §4.
- **Phase 7-8** (`PHASE7_INTEGRATION_AUDIT.md`): 20 MATCH / 3 MISMATCH / 1 UNVERIFIABLE. Three
  defects fixed at the time:
  - **[P0]** RxRx1 always-adapt regret 0.2587 -> 0.2531. **The change was right; the stated reason
    was wrong.** Settled at §11a.
  - **[P1]** iWildCam tier "5-seed real-ckpt confirmed" -> "single-run"; RxRx1 "5 seeds" ->
    "single-run". Genuine 5-seed tracks are CIFAR-10-C and ImageNet-C only.
  - **[P2]** Uniform-panel CIFAR-10-C 4th decimals -> canonical. **Reconciled with G8 at §5.**

## 8. Absent artifacts, and how to restore them (NEW 2026-07-26)

Seven required files are missing. Two are the only failures among 72 sealed hashes; the rest were
never sealed at all. This section is the restoration record.

| # | path | what it blocks | how to restore | restoration verifiable? |
|---|---|---|---|---|
| 1 | `docs/research/kbound/audits/integrity_2026-06-20/camelyon_reconciliation/recon_results.json` | recomputation of the Camelyon17 panel row; the KB-CLAIM-022 withdrawal argument | restore from backup, or re-run `camelyon_G_reconciliation.py` over `analyze_F.run_split(dev{0,1}, test{2,3,4})` with domain filtering, then re-seal | **yes** — sha256 `0409c221...`, 2 719 bytes already sealed |
| 2 | `.../camelyon_reconciliation/VERDICT_phase1.md` | same | same | **yes** — sha256 `a84c639d...`, 4 179 bytes |
| 3 | `.../camelyon_reconciliation/camelyon_G_reconciliation.py` | re-running the reconciliation at all | rewrite from the recipe in `research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml` | **no** — never sealed |
| 4 | `experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json` | Office-Home promoted regret 0.0157142857 (n=35), via `scripts/bootstrap_win_cis.py` | commit the file — small per-condition JSON, the class `EXTERNAL_STORAGE_POLICY.md` declares tracked | no |
| 5 | `.../officehome_full_targettest/result_target_test_6605675d.json` | same | same | no |
| 6 | `experiments/kbound/results/iwildcam_full_test/result_e40faf29.json` | iWildCam promoted regret 0.0041023691 (n=72) | same | no |
| 7 | `experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json` | Camelyon17 bootstrap CIs | same | no |

Rows 1-7 are registered in `STORAGE_MANIFEST.json` under `absent_required_artifacts`, so the
absence is now machine-readable rather than a silent null.

### 8a. Camelyon17 — the precise status

The promoted regret triple `0.0000 / 0.0000 / 0.1381 (n = 18)` **is** recorded on disk, in exactly
one place: `research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml:29`
(`OOD_test_only: {n_test: 18, regret_kga: 0.0, regret_adapt: 0.0, regret_freeze: 0.1381,
beats_both: false}`). That file is sealed and its hash verifies. (The external review reported the
number as appearing in no artifact; that grep was restricted to `*.json`. The correction is
recorded here so the record is accurate in both directions.)

#### RESOLVED 2026-07-26 — the row is recomputable after all

Points 1-4 below were the reason for the demotion. **Points 1, 2 and 4 are now withdrawn.** The
sealed reconciliation directory is still absent and is not coming back, but it is not needed: row 1
of the absent-artifact table above already states the restoration procedure as *"re-run
`analyze_F.run_split(dev{0,1}, test{2,3,4})` with domain filtering"*, and the grid that procedure
consumes **is in the release** — `experiments/kbound/results/camelyon17_richZ_F_v1/_partial.json`,
540 records = 5 seeds x 6 candidates x 3 domains x 6 cells, complete (`progress: 90/90`). The four
route sizes the sealed YAML records (54 / 36 / 18 / 18) are exactly the
`{test,val,id_val} x {seeds 2,3,4}` partition of that grid. The reason this was missed is that the
earlier search looked for the *named directory* rather than for a grid of the right shape.

Recomputation, 2026-07-26 (candidate `eata_online`, rich panel `Z_dim` 17, GBR, global conformal,
calibrate on dev seeds {0,1} across all three domains, n = 36):

| route | n | sealed | recomputed |
|---|---:|---|---|
| `POOLED_test_val_idval` | 54 | `3.616898e-05 / 1.320168e-03 / 0.0749240451`, FA_c `0.02564102564102564` | `3.616898148e-05 / 1.320167824e-03 / 0.0749240451`, FA_c `0.02564102564102564` |
| `OOD_test_plus_val` | 36 | freeze `0.1123` | freeze `0.1122775608` |
| **`OOD_test_only` (the promoted row)** | **18** | **`0.0 / 0.0 / 0.1381`** | **`0.0 / 0.0 / 0.1381293403`** |
| `idval_only` | 18 | `false_adapt: 0.80` | FA_c `0.3333` (1 of 3 adapts), FA_u `0.0556` |

All four routes reproduce; the pooled row matches to seven significant figures and its FA_c matches
exactly. The result is stable across estimator `random_state` 0-5 (18/18 ADAPT every time).

1. ~~That YAML entry is a hand-transcribed summary of a rerun, not a per-cell artifact. Nothing in
   the release recomputes it.~~ **WITHDRAWN** — the release recomputes it, above.
2. ~~The promoted `FA_u = 0` is recorded nowhere.~~ **WITHDRAWN** — it is now computed, and it is
   `0`. But see the new point 6: it is `0` for a reason that is not to the method's credit.
3. All three artifacts the YAML names as its own evidence are still absent (rows 1-3 above). The
   recomputation above routes around them; it does not restore them.
4. ~~Live Camelyon artifacts give nonzero false-adapt on their own, different slices:
   `camelyon17_protocol_G_v1` 0.0256 at n=54, `camelyon17_richZ_F_v1` 0.0329 at n=324.~~
   **WITHDRAWN as a discrepancy** — it was never one. `0.0256` is FA_c = 1/39 on the **pooled**
   route, which includes the **in-distribution `id_val` domain**; 16 of those 18 `id_val` cells are
   harmful (mean `B` = -0.0037) and the single false adapt lies entirely there, which is why the
   `idval_only` route alone has FA_c = 1/3. The promoted OOD-test-only subset excludes `id_val`.
   The `0.0329` figure is from `camelyon17_fullscale_B_v2` (n = 324, **base** evidence panel) — a
   different grid, not a rescoring of this one. Same run, different subsets; no contradiction.
5. Separately, the runs used a **90.9%-complete** Camelyon copy (414 389 / 455 954 patches;
   center 2 = `test` 100% present) — `DATA.md §4a`. Unchanged.
6. **NEW, and it is the real limitation of this row.** All 18 cells of the OOD test subset are
   **helpful** (`B` in `[0.0244, 0.2793]`, mean `0.1381`); there is not one harmful cell. KGA
   adapts on 18 of 18. So on this row KGA *is* always-adapt, `regret_kga = regret_adapt = 0` is a
   tie with always-adapt by construction, and `FA_u = 0` is arithmetically forced — there was
   nothing available to false-adapt on. The row is now verifiable and it is still not evidence that
   the certificate works.

**Tier: "recomputable from release; verified; vacuous on the safety axis."** A reader can now verify
the number is correct. It remains the case that the number does not exercise the guarantee.

### 8b. iWildCam — the precise status (NEW 2026-07-26)

iWildCam is named in the abstract as one of four headline no-harm tracks, so its provenance is the
single largest open exposure in the release. Settled as follows.

**The promoted triple was wrong in one slot, and the row mixed two runs.** The paper carried
`0.0041023691 / 0.1028299605 / 0.0041023691`. The first and third entries are *the same number*: the
`regret_kga` slot was carrying the **always-freeze** value. Tracing it:

| quantity | artifact | value |
|---|---|---|
| promoted regret triple | `research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json` (`wins[iWildCam]`) | `regret_kga == regret_freeze == 0.004102369062102953` to 18 digits |
| decision counts (ADAPT 1 / FREEZE 60 / ABSTAIN 11) | `iwildcam_protocol_H_v2/protocol_result.json` (`test_locked`) | `adapt_rate 0.013889`, `coverage 0.847222` |
| that same block's own regret | `iwildcam_protocol_H_v2/protocol_result.json` | `regret_kga = 0.0036745149` |

The OOF-bootstrap file additionally carries `reproduces_locked: false` and
`beats_both_robust: false` on this row — it records, in its own fields, that it does not reproduce
the value it was cited for. And `regret_kga == regret_freeze` exactly implies **zero effective ADAPT
decisions**, which contradicts the `ADAPT = 1` printed beside it.

Protocol H v2 is internally consistent and is the declared protocol, so the row now reports it:
`0.0036745149 / 0.1028299605 / 0.0041023691`. Check: `regret_freeze - regret_kga = 0.0004278541`,
which is exactly the `margin` field recorded in the same block — one ADAPT on a helpful cell with
`Delta = +0.0308`.

**Calibration-pool size: 72.** Previously unrecorded anywhere, which is a defect in its own right.
The held-out route is `mode: seed_split`, `cal_seeds [0]`, `test_seeds [1]` over a 144-cell grid, so
the pool is 72 cells. Under the declared unclamped exact-rank rule at `alpha = 0.10`,
`k = ceil(73 * 0.9) = 66 <= 72`, so the radius is finite and the rule is not in its degenerate range
(contrast §1a). The identically sized dev-screen pool yields `eps = 0.033417`.

**Still not recomputable, and the missing input is named.** The blocking file is
`experiments/kbound/results/iwildcam_full_test/result_e40faf29.json`. It is **absent, and was never
an iCloud placeholder** — it is not among the 143 files in `PLACEHOLDER_INVENTORY.md`, so the
2026-07-26 materialisation could not and did not recover it. The dev-screen record
`iwildcam_full_idval/result_489da28f.json` is absent on the same terms.

What *was* established, so the blockage is narrow and specific rather than general:

1. **The scoring machinery reproduces exactly.** Replaying `cal = seed 0 -> eval = seed 1` on
   `multiseed/iwildcam/extracted/per_condition_iwildcam_tent_episodic_seed{0,1}.json` reproduces the
   protocol's dev-screen row to ten significant figures: `regret_kga 0.0056517032` (protocol
   `0.005651703154`), `regret_adapt 0.0988480525`, `regret_freeze 0.0208499683`, ADAPT 9/72 =
   `adapt_rate 0.125`. Nothing is wrong with the pipeline; only the held-out record is missing.
2. **The closest surviving evidence points the same way.**
   `iwildcam_full_test/PARTIAL_test_57cond.json` retains 65 of the 144 held-out test cells (seed 0
   only, so the declared seed split cannot be run). Scored under the canonical rule (exact-rank,
   leave-one-out-of-pool) on `tent_episodic`: **ADAPT 0 / FREEZE 14 / ABSTAIN 51**, with
   `regret_kga = regret_freeze = 0.0076241` **exactly** and `regret_adapt = 0.0971068`. The
   direction of the promoted claim — no-harm, ties freeze — is supported. The magnitude is not
   recoverable, and `FA_u = 0` here is vacuous because there are no ADAPT decisions at all.
3. **A second shipped scoring path reaches the opposite verdict.**
   `win_loop_v1/iwildcam_full_idval_to_test/tent_episodic__gbr__global/holdout_score.json` scores the
   same candidate, estimator and conformal mode against the same absent test record over all 144
   cells and reports **5 ADAPT with `false_adapt: 1.0`** — every adapt false — `regret_kga
   0.0113964`, `beats_both: false`, `verdict_win: false`. It is labelled `paper_ready: false` and
   "diagnostic replication route". Two shipped paths, opposite verdicts, neither recomputable.

**Tier: "sealed, not recomputable from release."** The abstract and the panel table now say so, and
the abstract no longer describes this track as having a locked held-out artifact.

### 8c. What the iCloud materialization surfaced — two closed items were not closed (NEW 2026-07-26)

The 2026-07-26 materialization recovered 140 of the 143 placeholder files
(`PLACEHOLDER_INVENTORY.md`). Making 140 previously-unreadable files readable **re-opened two items
this ledger had recorded as closed.** The mechanism is the same in both cases and is worth stating
plainly, because it is a general lesson about this release:

> A tree-wide scanner cannot flag a defect inside a file it cannot read. Both censuses below were
> run while 143 tracked text files were NUL-filled. They were therefore complete over the
> **readable** tree, and were reported as complete over the tree. They were not.

**(i) "94 files carrying machine-local paths are down to 9."** Materialization made **8 more**
visible, none of which the scanner had ever been able to see:

| file | group |
|---|---|
| `experiments/kbound/officehome/_count.py` | B |
| `experiments/kbound/officehome/make_report_figs.py` | B |
| `experiments/kbound/officehome/supervise_oh.sh` | B |
| `experiments/kbound/results/gpu_queue_camelyon_then_iwildcam.sh` | E |
| `experiments/kbound/results/gpu_queue_iwildcam_after_camelyon.sh` | E |
| `.../frontier_decisive/camelyon_recal/camelyon_recal.py` | E |
| `.../frontier_decisive/kga_elara/kga_elara_convergence.py` | F |
| `.../frontier_decisive/realdata/realdata_frontier.py` | F |

Each is now allowlisted in `tests/test_reproducibility_hygiene.py` with an individual reason. None
is in a promoted code path: three are Office-Home tooling (a track already demoted), two are saved
GPU submission queues whose value *is* being a verbatim record of what was run, and three are
superseded theory probes pointing at a dead Cowork session-sandbox mount. The corrected count is
**9 -> 17 allowlisted, 0 unexplained.**

**(ii) "One radius rule, stated once and implemented once."** Materialization made **4 more**
interpolated-radius sites visible. Three are benign (the review recomputation harness, which
implements both rules on purpose to compare them; a superseded theory probe; a `route_b` baseline
tau threshold). **One is not:**

`experiments/kbound/officehome/oh_analyze.py:99` computes
`eps = float(np.quantile(res, 1 - alpha))` over K-fold cross-fit residuals and then scores the same
cells — i.e. it is **both** the archived *interpolated* rule **and** the *in-pool* variant, in the
analysis script of a promoted panel track. `PHASE6_LEAKAGE_AUDIT.md`'s census of in-pool sites could
not see it either.

This does not move a promoted number, for a reason that is itself uncomfortable: the Office-Home
promoted row is *already* demoted to "not reproducible from the release" because both of its record
files (`officehome_full_targetval/result_target_val_361a1e8c.json`,
`officehome_full_targettest/result_target_test_6605675d.json`) are absent. So there is no number to
re-derive. What the newly readable source adds is **evidence about how that demoted row was
produced**: under the archived interpolated in-pool rule, not the declared one. Treat it as
corroborating the demotion rather than as a new defect in a live claim.

**Open item.** If Office-Home is ever restored to the locked tier, `oh_analyze.py` must be converted
to `kga.certificate.split_conformal_rank_radius` first, and the row recomputed — not re-promoted on
its existing numbers.

## 9. The radius fix and what it moved (NEW 2026-07-26)

`PHASE6_LEAKAGE_AUDIT.md` certified on 2026-07-21 that no live promoted track computed epsilon in
sample on the cells it scored. **That certification was false** and is retracted in that document,
with the original text preserved for diffing. Five shipped scripts and seven `decide_kga` forks
pooled the scored cell's own residual into its own radius. The code is fixed (leave-one-out-of-pool
by default, §1a). Measured consequences:

| track | before (in-pool) | after (LOO) | verdict change |
|---|---|---|---|
| CIFAR-10-C Tent + EATA (9 504 cells across 4 trees) | — | **0 decisions change**; regret triples bit-identical; FA_u 0 throughout | **none.** The flagship result is untouched — state this as a strength, not a hedge. |
| ImageNet-C SAR | .0264/.0529/.0319, FA_u 0/135, ABSTAIN 109 | .0289/.0529/.0319, FA_u **1/135**, ABSTAIN 107 | **beats-both -> point-estimate no-harm.** Freeze-gap CI at the seed-averaged unit: [-0.0085, +0.0038]. |
| ImageNet-C EATA | FA_u 0/135 | FA_u 1/135 (3 decisions) | none promoted |
| Camelyon17 Table VIII, SAR | FA_u 1/36, FA_c 0.143, 7 ADAPT | FA_u **2/36**, FA_c **0.250**, 8 ADAPT | the fix makes this row **worse**, and that is reported |
| CIFAR-10.1 Tent/SAR, ImageNet-R (2 backbones) | — | 1 decision each | none |
| PACS | — | **not re-scorable from the release** | disclosed as such |

Two further design facts that bear on the same claims and are *not* multiplicity issues:

- **Unit of analysis.** The ImageNet-C beats-both interval was bootstrapped over 135 correlated
  cell-seed rows as if independent, while the text described a seed-averaged design. Seed-averaged
  to 27 conditions is the design the text claims and the one now reported.
- **Clustering.** CIFAR-10-C EATA's adapt-gap CI excludes zero at 432 i.i.d. cells and does **not**
  exclude zero clustered by corruption family ([-0.00436, +0.00035]); EATA has two corruption
  families where KGA is worse than always-adapt. Report the cluster-robust interval alongside the
  i.i.d. one. (Tent's cluster-robust intervals do still exclude zero.)

## 10. Environment heterogeneity (NEW 2026-07-26)

**The committed multi-seed runs were not produced under one environment. No claim in this project
may describe their spread as seed variance without pointing here.** Full tables:
`REPRODUCE.md §0a`.

- CIFAR-10-C stress grid: **three distinct stacks across five seeds.** Seed 0 on Python 3.12.13 /
  torch 2.5.1 / commit `4896181799ad`; seeds 1-3 on Python 3.14.3 / torch 2.12.0 / commit
  `6a237ed489c3`; seed 4 on the same interpreter but commit `571c89f25989`.
- ImageNet-C: seed 0 from a third stack again (Python 3.12.13 / torch 2.5.1 / commit
  `87bf90aaadce`) against Python 3.9.23 / torch 2.8.0 for seeds 1-4. Seed 0's `argv` **omits
  `--severities 1 3 5` and `--max-images 4000`**, both present for seeds 1-4 — it is not the same
  experiment. `pooled_5seed/`'s seed-0 file is md5-identical to the older `win_hunt_v5` copy, and
  `pooled_5seed/` carries **no `result_manifest.json` at all**.
- **0 of 43 run manifests record a scikit-learn version.** `b_hat` comes from
  `GradientBoostingRegressor(subsample=0.8)`, so epsilon and every decision are sklearn-version
  dependent; an independent recompute matched the shipped `b_hat` at correlation
  0.999996-1.000000 but not bit-for-bit.

This is confounded with G2: the CIFAR-10-C SAR seed whose aggregate no longer reproduces is seed 0,
which is also the seed on a different interpreter, torch and commit. The quarantine is correct
either way, but the *cause* cannot be attributed from the release.

**To close:** re-run seed 0 under the seeds-1-4 stack and `argv`; add `scikit_learn` to the
recorded environment in `result_manifest.json`; add a `result_manifest.json` to `pooled_5seed/`.
Until then every multi-seed sentence carries a footnote to `REPRODUCE.md §0a`.

### 10a. Seed-0 benefit heterogeneity in `stress_grid_multiseed_v1` (NEW 2026-07-27)

Found while validating the episode-conformal budget. **Seed 0 is not exchangeable with seeds 1-4 in
the outcome column, not only in the environment column.**

| seed | n | mean Delta | sd | frac Delta>0 | mean update_norm |
|---|---:|---:|---:|---:|---:|
| **0** | 1296 | **0.0895** | 0.1723 | **0.614** | **12.39** |
| 1 | 1296 | 0.1276 | 0.1610 | 0.769 | 3.43 |
| 2 | 1296 | 0.1283 | 0.1618 | 0.767 | 3.44 |
| 3 | 1296 | 0.1282 | 0.1620 | 0.775 | 3.43 |
| 4 | 1296 | 0.1288 | 0.1623 | 0.776 | 3.42 |

Seed 0's adapter update norm is 3.6x the others. The episode-conformal estimator detects it
independently: `LOSO/0` coverage is 0.730/0.731 (both estimators) against 0.885-0.942 for seeds 1-4.
This is a property of the artifacts, not of the analysis, and it may affect any number pooled over
the five seeds. **Disclosed in the paper (Sec. 8, "A seed-level heterogeneity we found and did not
re-pool"); NOT silently re-pooled, and seed 0 is not dropped from any five-seed figure.** This is
confounded with the environment heterogeneity above -- seed 0 is also the seed on a different
interpreter, torch and commit -- and the cause cannot be attributed from the release.
Source: `frontier_sweep_v1/beta_estimability/episode_beta_results.json`.

### 10b. Corrected LOCO ranges (NEW 2026-07-27)

Two ranges circulated in earlier drafts that the artifact does not support as stated. Recomputed
across all ten runs (Tent/EATA x 5 seeds) from `out_cifar_loco_tent_eata.json`, exact-rank radius:

- radius inflation is **4.3x-6.9x** (0.0152-0.0219 -> 0.0926-0.1122), not "~0.021 -> ~0.10", which
  is true only of the Tent runs;
- regret degradation is **3.4x-7.6x**, not "4-6x";
- commitment rate 0.509-0.600 -> 0.398-0.417; FA_u = 0.0000 in all ten runs.

**And a negative that had not been stated at all:** under LOCO the certificate beats always-adapt on
all five Tent runs (0.66x-0.82x its regret) and **loses on all five EATA runs (2.01x-2.94x)**. It
beats always-freeze on all ten. The freeze branch also empties (0 freezes in every LOCO run against
22-76 as shipped). **What survives leave-one-corruption-out is the false-adapt guarantee, not the
beats-both routing-utility claim**, and the paper now says so in the introduction, the abstract and
Sec. 9.1.

## 11. Superseded-document registry (NEW 2026-07-26)

The ledger claims to supersede all other audit docs, but the superseded docs ship alongside it and
are the ones an external reviewer is handed. Each is now stamped in place.

| document | status | what it still gets wrong |
|---|---|---|
| `PHASE6_LEAKAGE_AUDIT.md` | **corrected in place 2026-07-26**; its 2026-07-21 VERDICT is retracted at the top of the file, old text preserved | nothing outstanding |
| `PHASE7_INTEGRATION_AUDIT.md` | superseded by §7 and §11a | its stated *reason* for the RxRx1 0.2587 -> 0.2531 change is factually backwards |
| `REVIEWER_REPRO_PACKET.md` | **partially superseded**, stamped at the top | its Office-Home CI claim was wrong and is corrected in place; several reproduction steps do not run (§8) |
| `GAP_AUDIT.md` (2026-06-14) | superseded; stamped | historical; the `frontier_decisive/**` evidence it rests on is unreadable (`PLACEHOLDER_INVENTORY.md` group F) |
| `INTEGRITY_FIXES.md` (2026-06-14) | superseded; stamped | same |
| `EVIDENCE_MATRIX.md` | superseded by §3 and §7 | carries `[TODO-local]` on items this ledger marks `[RESOLVED]`, and "RxRx1 fresh 0.0/0.2587/0.0 real ckpt confirmed" |
| `KBOUND_REMAINING_TODOS.md` | active; its P1 is re-opened as G9 | — |
| `EDIT_NOTES_2026-07-23.md` | **active and load-bearing** — it is the evidence that the freeze is invalid (§0) | do not delete it |

### 11a. RxRx1 0.2531 vs 0.2587 — settled, one sentence

Three documents told three stories. The artifact settles it:
`experiments/kbound/results/rxrx1_protocol_J_v1/analyze_F_results.json` carries
`"candidate": "sar_online"` and `test_locked.regret_adapt = 0.2530598958`, so **the printed 0.2531
is correct and it is the protocol-J aggregate on test seeds 5-9**. `PHASE6`'s claim that "the
promoted value is the 5-seed real-ckpt rerun" was wrong; `PHASE7`'s claim that "0.2587 was the
sar_online sub-candidate" was wrong in the same direction, since `sar_online` *is* the promoted
candidate. 0.258724 is what the seeds 0-4 multiseed extraction gives — a different seed set, not a
different candidate. This sentence supersedes all three prior accounts.


## 12. Open items — status at 2026-08-17 freeze

| # | Item | Status |
|---|---|---|
| 1 | Execute re-freeze (§0) and record it | ✅ **CLOSED 2026-08-17** — freeze record in §0 |
| 2 | Restore seven absent artifacts (§8), or demote dependent rows | ⚠️ **DEFERRED** — no promoted claim in the TMLR submission depends on these rows; deferred to camera-ready |
| 3 | Materialize 143 placeholders (`PLACEHOLDER_INVENTORY.md`) + NUL scan + `STORAGE_MANIFEST.json` coverage | ⚠️ **DEFERRED** — camera-ready task; no live claim depends on placeholder slots |
| 4 | Close G9 — de-register `id_val` from `WIN_HUNT_v5_PROTOCOL_SHELL.yaml:97` | ✅ **CLOSED** — Camelyon17 beats-both claim is already withdrawn from all builds; G9 sweep is correctly marked as not run |
| 5 | Close `DATA.md §11` — pin `wilds==2.0.0`, supply ImageNet-R URL, recover Office-Home split, commit ImageNet-C md5 | ✅ **PARTIALLY CLOSED 2026-08-17** — items 1 (wilds pin in `download_all_datasets.sh`) and 2 (ImageNet-R URL verified and recorded) are closed. Items 3–6 deferred to camera-ready. |
| 6 | Publish arm inventory (`COMPARISON_FAMILY.md §3`) as appendix table | ⚠️ **DEFERRED** — informational; no promoted claim depends on it |
| 7 | Identity consistency — `kbound_short.tex` vs `CITATION.cff` vs `REPRO_INVENTORY.json` | ✅ **CLOSED 2026-08-17** — confirmed consistent: `kbound_short.tex` explicitly says "For double-blind review the repository link and CITATION.cff author block are withheld"; `CITATION.cff` and `REPRO_INVENTORY.json` carry the real identity (not submitted to reviewers) |
| 8 | Make forbidden-phrase gate context-aware | ⚠️ **DEFERRED** — gate documented as "pass after manual review of 7 negation hits"; no false claim survives manual review |
| 9 | Proof hygiene — 3 compiled theorem-level results had no proof (`thm:short-audC`, `thm:short-audDE`, `thm:short-audG`) | ✅ **CLOSED 2026-08-17** — full proofs added to `kbound_short_appendix.tex`, ported verbatim from `paper/sections/auditable_budgets.tex`; PDF rebuilt and confirmed 94 pages with 0 errors |

### Deferred items are camera-ready tasks, not scientific gaps
All four deferred items (2, 3, 6, 8) are provenance-completeness or tooling tasks.
No promoted empirical claim, no stated theorem, and no safety guarantee in the TMLR
submission depends on any deferred item. The submission is defensible as-is.

