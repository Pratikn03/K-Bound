# K-Bound SUBMISSION LEDGER (canonical, Phase 1 freeze)
Single source of truth. No other manuscript, lock, or audit doc may override this.
Generated 2026-07-19 (audit plan Phase 1). Supersedes all prior audit docs.

## 0. Frozen target
- Manuscript: docs/research/kbound/kbound_short.tex  (IEEE conference short paper)
- Git commit (HEAD at freeze): ff9be6b2a90482394fdb518226d8e0efde2c9c7b  (branch main)
- PDF sha256: 444db9a6f4e9cdc1fbbeb1fc33fbcb79ea0b39b258c5ed610c475857b405af4c
- PDF pages: 22 ; tables: 27 (I–XXVII)
- Build: latexmk clean, 0 undefined refs/citations
- NOTE: working history lives on origin/flagship-history; origin/main is a separate curated release.

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
- FA_u (unconditional false-adapt): Pr(commit to wrong sign) over the mixture; target <= alpha (=0.10).
- FA_c (conditional false-adapt): conditional variant (per-regime); reported where flagged (CIFAR-10.1).
- regret-to-oracle: E[|Delta| * 1{action != oracle}]; oracle knows sign Delta; ABSTAIN defaults to FREEZE.
- Risk alignment (def:risk-align): ** an ASSUMPTION, not empirically established ** (theory_setup:42).

## 2. Theorem / claim inventory (short-paper input tree)
Type key: [G]=theorem-level guarantee  [E]=empirical observation  [D]=diagnostic
theory_setup.tex:
  ass:deploy         [G-assump] Deployment setup (binary 0/1, disagreement region D)
  def:risk-align     [G-def]    Risk alignment (ASSUMPTION — verify not asserted as fact: Phase 2/7)
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
theory_appendix_ext.tex:  [** Phase-2 correction: 5 of these are \iffalse'd OUT of the short build **]
  thm:imp            [G] Matched-evidence impossibility (full form)  -- COMPILED (FIX: xref+M(g) notation)
  cor:forced-abstain [G] Closed-band abstention under dual error control -- COMPILED (burden: two-sided control)
  lem:gate           [--] Plug-in regret identity            -- NOT COMPILED (\iffalse; long-manuscript only)
  prop:lecam-finite  [--] Finite-sample minimax regret floor -- NOT COMPILED (\iffalse; validator missing)
  prop:cert-sample   [--] Certificate sample complexity      -- NOT COMPILED (\iffalse)
  thm:conj1-dichotomy[--] One-bit dichotomy                  -- NOT COMPILED (\iffalse; near-vacuous, keep out)
  thm:ev-rate        [--] Evidence-channel rate              -- NOT COMPILED (\iffalse; validator missing)
kbound_short_appendix.tex:
  thm:short-audA     [G] Vacuity of label-free audits (Aud-A)
  thm:short-audC     [G] Computed budgets under purchasable structure (Aud-C/F)
  thm:short-audDE    [G] Composition; fully empirical rule (Aud-D/E)
  thm:short-audG     [G] Domain-level verifiability with exact floor (Aud-G)
  prop:beatsboth-asym[G] Asymmetric beats-both (sharpens thm:headline; added 2026-07-19)
TRUE COMPILED short-paper stack (Phase-2 verified): lem:reduction, lem:nonid, cor:matched-abstain,
prop:closed-band, thm:headline, thm:certificate, thm:imp, cor:forced-abstain, thm:short-audA/C/DE/G,
prop:beatsboth-asym (+ defs/assumption). Phase-2 verdict: all PASS except thm:certificate=FIX (G8),
thm:imp=FIX (xref/notation). ZERO withdrawals. Full detail: PHASE2_THEOREM_AUDIT.md.

## 3. Nine tracks (Table XV uniform panel) + promoted claim + evidence tier
Track            | Promoted result (KGA/adapt/freeze; FA_u)         | Claim type | Tier (ledger)
CIFAR-10-C stress| Tent .0016/.0079/.1241; EATA .0013/.0033/.1314   | [E] BB(Tent/EATA) | locked; SAR WITHHELD (seed0 non-repro)
ImageNet-C SAR   | .0107/.0529/.0319 pooled; FA_u=.007              | [E] BB pooled     | locked(5s) — ** manifest still single-seed: STALE, Phase 5 **
Camelyon17 OOD   | .0000/.0000/.1381; FA_u=0                        | [E] no-harm       | reconciled (raw)
iWildCam H v2    | .0041/.1028/.0041; FA_u=0                        | [E] no-harm       | reconciled — ** fresh 5-seed pending (real-ckpt rerun queued) **
Office-Home M v2 | .0157/.0468/.0158; FA_u=0                        | [E] no-harm       | reconciled (OOF). raw-grid BB NOT promoted
RxRx1 J          | .0000/.2531/.0000; FA_u=0                        | [E] no-harm       | locked (fresh 5-seed valid)
PACS             | 3 safe / 1 null                                  | [D]               | diagnostic (1 of 3 seeds)
ImageNet-R D     | no CI-robust BB                                  | [D]               | diagnostic (3 of 4 seeds)
CIFAR-10.1 K     | fails transfer bar (FA_u=.167,FA_c=.444)         | [D] negative      | diagnostic

## 4. Known gaps carried into Phases 2-8 (from user + this freeze)
G1 ImageNet-C 5-seed manuscript numbers NOT synced with generated manifest (single-seed). [Phase 5]
G2 CIFAR-10-C SAR withheld: seed0 no longer reproduces archived aggregate. [Phase 5]
G3 PACS 1/3 seeds; G4 ImageNet-R 3/4 seeds. [Phase 8 GPU]
G5 Official POEM repro not wired; comparisons are protocol-matched ports (Table XI style). [Phase 8]
G6 Physical-camera R2 pending — cannot support a claim (Table XXVI RESULT PENDING). [Phase 8 human]
G7 Strict stress-grid v2 protocol appears unrun. [Phase 8]
G8 [RESOLVED = PASS, 2026-07-20] Regenerated panel with EXACT split-conformal rank rule
   (certificate.py, k=ceil((n+1)(1-alpha))). Result (G8_EXACTRANK_REGEN.md): FA_u=0.000 on EVERY
   headline track under exact rule (ImageNet-C SAR/EATA/Tent, CIFAR-10-C Tent/EATA), and ALL
   beats-both SURVIVE (ImageNet-C SAR 0.0264/0.0529/0.0319; CIFAR Tent 0.0016/0.0080/0.1239; CIFAR
   EATA 0.0013/0.0033/0.1313). => "finite-sample certificate" language is HONEST under exact rule;
   no relabel/withdrawal. ACTION: update panel numbers to exact-rank values; state FA_u/eps use the
   exact rank rule; drop interpolated-quantile from headline path. Still fix FA_u marginal code label.
G9 Some older locks retain WITHDRAWN Camelyon "beats-both" — must be marked SUPERSEDED. [Phase 6]
G10 REVIEWER_REPRO verifies cached artifacts, does not recreate every headline. [Phase 5]
G11 Prior audit docs conflicting/stale — this ledger supersedes them. [Phase 1 done]

## 5. Distinctions the manuscript MUST hold (Phase 7 gate)
safety/validity != accuracy | theorem-guarantee != empirical-coverage | mixed-regime BB != one-sided
no-harm | natural benchmark != constructed mixture | official method != protocol-matched port |
locked/reconciled != diagnostic/incomplete.
