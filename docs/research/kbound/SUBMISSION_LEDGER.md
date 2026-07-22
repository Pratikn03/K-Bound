# K-Bound SUBMISSION LEDGER (canonical, Phase 1 freeze)
Single source of truth. No other manuscript, lock, or audit doc may override this.
Generated 2026-07-19 (audit plan Phase 1). Supersedes all prior audit docs.

## 0. Frozen target
- Manuscript: docs/research/kbound/kbound_short.tex  (IEEE conference short paper)
- Git commit (HEAD at freeze): ff9be6b2a90482394fdb518226d8e0efde2c9c7b  (branch main)
- PDF sha256: 5b01e5e7da41edae5a574c09fb8d5fa6b0cb4cc8d5853ff814441484b755d00a
- PDF pages: 23; long manuscript: 60 pages; every page rendered and counted
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
ImageNet-C SAR   | .0264/.0529/.0319 pooled; FA_u=0                 | [E] BB pooled     | locked (5 seeds; exact-rank manifest synced)
Camelyon17 OOD   | .0000/.0000/.1381; FA_u=0                        | [E] no-harm       | reconciled (raw)
iWildCam H v2    | .0041/.1028/.0041; FA_u=0                        | [E] no-harm       | reconciled (single trained model; multi-condition)
Office-Home M v2 | .0157/.0468/.0158; FA_u=0                        | [E] no-harm       | reconciled (OOF). raw-grid BB NOT promoted
RxRx1 J          | .0000/.2531/.0000; FA_u=0                        | [E] no-harm       | locked (real ckpt; single seed-0 model, multi-condition)
PACS             | .0431/.0176/.0446; mean reported FA_u=.0093       | [D] null          | locked diagnostic (3 of 3 seeds; pooled FA count not retained)
ImageNet-R D     | .0112/.0064/.0325 mean across backbones; FA=1/480 | [D] null          | locked diagnostic (4 of 4 seeds; 0/10 CI BB)
CIFAR-10.1 K     | fails transfer bar (FA_u=.167,FA_c=.444)         | [D] negative      | diagnostic

## 4. Known gaps carried into Phases 2-8 (from user + this freeze)
G1 [RESOLVED] ImageNet-C five-seed manuscript numbers and generated manifest are synchronized.
G2 [QUARANTINED] CIFAR-10-C SAR is withheld because seed 0 no longer reproduces the archived aggregate;
   reinstatement gates are frozen in CIFAR10C_SAR_QUARANTINE.md.
G3 [RESOLVED] PACS 3/3 seeds; G4 [RESOLVED] ImageNet-R 4/4 seeds. Both remain null diagnostics. [Phase 8 GPU]
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


## 6. Fix-queue resolutions (Phase 4-5 tail, 2026-07-21)
G1 [RESOLVED] paper/generated/kbound_result_manifest.json /tracks/imagenetc_sar regenerated from the
   5 per-seed files (win_hunt_v5_imagenetc_ms/pooled_5seed) under the EXACT split-conformal rank rule:
   regret [0.0264,0.0529,0.0319], FA_u=0.0, seeds [0-4], n_cells 135, abstain 109, gap CIs
   (adapt [-0.0518,-0.0038], freeze [-0.0086,-0.0026]) => paired-bootstrap beats-both. Now synced to manuscript.
PACS [RESOLVED] The registered three-seed aggregate is complete. Mean regret across four LODO targets
   is .0431/.0176/.0446 (KGA/adapt/freeze), so the track remains a null diagnostic. The mean reported
   FA_u is .0093; raw pooled action/FA counts were not retained, so no integer count or Wilson interval
   is reconstructed. The superseded single-seed interpretation is not promoted.
OfficeHome [RESOLVED/annotated] 0.0157/0.0468/0.0158 is an OOF-lock DESIGN value (saved bootstrap lock,
   App:claim-artifact), not a raw-grid per-cell number - already stated at lines ~851,1095-1097. Final panel tier: "locked (OOF no-harm only; LOO BB not promoted)". Not raw-traceable BY DESIGN.
G9 [RESOLVED] claim_ledger.json KB-CLAIM-022 (Camelyon Protocol-G pooled beats-both) status=withdrawn,
   test_split "pooled id_val (invalid)", forbidden_wording "beats both Camelyon17". That wording is ABSENT
   from the compiled manuscript (grep rc=1); artifact archived under archive/audit_only. Manuscript only ever
   states Camelyon "reconciled no-harm". No live lock/script asserts Camelyon beats-both. Quarantine intact.
Phase 6 (leakage/timing) + Phase 7-8 (claim-by-claim integration): audited by two read-only subagents;
   see PHASE6_LEAKAGE_AUDIT.md and PHASE7_INTEGRATION_AUDIT.md.

## 7. Phase 6-8 audit verdict (2026-07-21)
Phase 6 (leakage/timing, PHASE6_LEAKAGE_AUDIT.md): PASS/clean. All 7 live tracks fit epsilon on the
   calibration split only (LOO/cross-fit), score a disjoint test partition; certificate decisions use
   (b_hat, epsilon) only, true benefit B used solely for post-hoc FA_u scoring; ImageNet-C 5-seed pools
   epsilon PER SEED (not one epsilon across pooled cells). KB-CLAIM-022 in-sample-radius quarantine confirmed.
Phase 7-8 (integration, PHASE7_INTEGRATION_AUDIT.md): 20 MATCH / 3 MISMATCH / 1 UNVERIFIABLE. All six sec-5
   distinctions hold; every withdrawn-claim forbidden phrase absent from compiled PDF. 3 defects fixed:
   [P0] RxRx1 always-adapt regret 0.2587 -> 0.2531 (canonical; 0.2587 was the sar_online sub-candidate, not
        the promoted protocol-J aggregate) at kbound_short.tex:902 and :940.
   [P1] iWildCam tier "5-seed real-ckpt confirmed" -> "single-run" (:900); RxRx1 tier "5 seeds" -> "single-run"
        (:902) and ":940" drop "(5 seeds)". Matches body text :865-866 (iWildCam/Office-Home/RxRx1 single-run).
        Genuine 5-seed tracks remain CIFAR-10-C and ImageNet-C only (real seed0-4 grids, agent-validated).
   [P2] Uniform-panel CIFAR-10-C Tent/EATA 4th-decimals 0.0080/0.1239, 0.1313 -> canonical 0.0079/0.1241,
        0.1314 (:897), matching primary numeric table :936-937.
Non-issues verified: undefined refs/citations = 0 (earlier "46" was a grep artifact); tab_multiseed_natural.tex
   is not \input and does not exist (moot); RxRx1/Office-Home/iWildCam runs are single trained model.
