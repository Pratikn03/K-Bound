# Phase 2 — Theorem audit (PASS/FIX/WITHDRAW). 2026-07-19
Two independent adversarial subagent audits (core + appendix). Anchored to SUBMISSION_LEDGER.md.
NET: theory is SOUND. No false theorems. ZERO withdrawals. Fixes are labeling/hygiene + 1 code defect.

## Core theory (theory_setup + theory_core_main) — 9 PASS / 1 FIX
ass:deploy .............. PASS (tautological decomposition; binary 0/1 exact)
def:risk-align ......... PASS (used as ASSUMPTION, never asserted as fact)
def:regimes ............ PASS
def:strict-sound ....... PASS (maximality is vs epistemic-validity convention; disclosed)
lem:reduction .......... PASS (sign Delta = sign(M+gamma), airtight)
lem:nonid .............. PASS (kernel witness valid; |gamma|<beta; opposite signs; beta=0 handled)
cor:matched-abstain .... PASS (Pr[abstain]>=1-2alpha)
prop:closed-band ....... PASS (boundary |M|=beta epistemic; disclosed)
thm:headline/frontier .. PASS (exact iff over full C_beta; necessity needs class richness; disclosed)
thm:certificate ........ **FIX** (G8 — see below). Conditional implication is CORRECT; the
                          calibration procedure does not deliver its coverage hypothesis at finite n.

## G8 verdict — thm:certificate = FIX (not WITHDRAW)
- Theorem is a valid conditional: coverage(1-alpha) ⇒ FA_u<=alpha. Proof correct (false-adapt ⊆ miscoverage).
- BUT Algorithm 1 sets eps = (1-alpha) INTERPOLATED np.quantile of leave-one-cell-out residuals.
  Simulated coverage (exchangeable best case): n=27 → 0.876; n=72 → 0.883; n=250 → 0.894 (nominal 0.90).
  Under-covers at every finite n (matches paper's own "realized 0.898 at nominal 0.90").
- Exchangeability fails two ways: (i) LOO REFITS estimator per cell = jackknife (no distdist-free
  guarantee; jackknife+ NOT implemented, already withdrawn KB-CLAIM-012); (ii) grid cells heteroskedastic.
  Real breaches in the data: SAR→Tent FA_u=0.25 (kbound_short.tex:992); CIFAR-10.1 FA_u=0.167.
- The EXACT rank rule eps=r_(k), k=ceil((n+1)(1-alpha)) EXISTS in code (kbound_pkg/kbound/certificate.py:65)
  and is used for CIFAR-10-C ablations — but the headline 9-track panel JSONs used the interpolated quantile.
- FIX actions: (1) retitle "Finite-sample certificate" → "coverage-conditional certificate", OR
  (2) regenerate panel FA_u with exact-rank split-conformal, OR (3) annotate every FA_u cell as
  interpolated-empirical evidence. Paper prose already half-concedes this (kbound_short.tex:390).

## Code defect found (Phase 3 spillover) — FIX
- FA_u mislabel: run_wilds_camelyon17.py:75 & mixed_regime_experiment.py:111 compute mean(B[adapt]<0)
  = CONDITIONAL FA_c, while theorem/marginal FA_u = mean(adapt & B<=0) (correct in gapclose_wave5/...:164).
  Conditional>=marginal so mislabel is CONSERVATIVE, but the reported "FA_u" column must be the marginal.
  Also Delta=0 boundary: code B<0 vs theorem B<=0.

## Appendix theory — mostly PASS; FIX items
thm:short-audA/C/DE/G ... PASS (delta-budgets all sum correctly; 5δ/4 bug already fixed; audG floor exact)
prop:beatsboth-asym ..... PASS (regret identities (F),(A) verified algebraically + numerically ~1e-17)
cor:forced-abstain ...... PASS but BURDEN: its >=1-2alpha needs TWO-SIDED error control; deployment
                          targets one-sided FA_u — flag this contingency in the text.
thm:imp ................. **FIX** xref: "lem:nonid is part (iii)" → part (i); (iii) is cor:matched-abstain.
                          Also M(g) notation collides with observable margin M — rename to total error.
thm:conj1-dichotomy ..... near-vacuous (content in hypothesis H); CORRECTLY \iffalse-excluded — keep out.
lem:gate,prop:lecam-finite,prop:cert-sample,thm:ev-rate ... PASS-informal but \iffalse (NOT in short build).
Dangling validators: val_thm2_lecam_finite_n.py, val_ev_rate.py cited but MISSING on disk (in \iffalse). FIX/remove.

## LEDGER SELF-CORRECTION (audit caught my Phase-1 error)
SUBMISSION_LEDGER §2 listed lem:gate/prop:lecam-finite/prop:cert-sample/thm:conj1-dichotomy/thm:ev-rate
as active [G] in the compiled short paper. They are \iffalse-commented OUT. Corrected in ledger.
TRUE compiled short-paper theorem stack: lem:reduction, lem:nonid, cor:matched-abstain, prop:closed-band,
thm:headline, thm:certificate, thm:imp, cor:forced-abstain, thm:short-audA/C/DE/G, prop:beatsboth-asym.

## Phase 2 fix queue (all LOCAL/CPU)
1. thm:certificate: reconcile finite-sample language + panel FA_u with exact-rank rule (G8).  [decision needed]
2. thm:imp: fix cross-reference + M(g) notation collision.                                    [local edit]
3. Ledger §2: mark 5 theorems long-manuscript-only; fix count.                                [done here]
4. FA_u code: standardize on marginal mean(adapt & B<=0); fix Delta=0 boundary.               [code edit]
5. Remove/restore 2 dangling validator references.                                            [local edit]
6. cor:forced-abstain: note the two-sided-control contingency in text.                         [local edit]
