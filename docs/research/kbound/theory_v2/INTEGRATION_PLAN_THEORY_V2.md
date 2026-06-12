# Theory V2 — Integration Plan (June 2026)

Response to the 5-point theory critique. Everything below is proved in
`THEORY_V2_PROOFS.md`, validated by `theory_v2_validation.py` →
`validation_results.json` (+ figures `fig_v1..v4_*.png`), written into the paper as
`paper/sections/onebit_audit_rate.tex` (input from `kbound.tex` after §rates), with
surgical corrections to `paper/sections/multicandidate.tex`. Literature positions
verified live (June 2026) against arXiv/OpenReview.

## How each critique point was resolved

**P1 (Thm 2 assumes the hard part; ε's premise untestable).** Not patched — *characterized*.
Lemma `gamma-id`: under the (falsifiable) hypothesis H, the budgeted drift γ is
point-identified up to one bit (γ = b_a/2 − M). Theorem `audit`: a level-α, bit-robust
falsification test for any budget β, with a proven finite-sample radius (sound; bootstrap
variant powerful — power 1.000 at min-flip |γ| = 0.216, level 0.000 ≤ α = 0.05).
Theorem `no-verify`(a): verification (as opposed to falsification) is impossible exactly
on the bit-ambiguous set B_β = {||M|−|b_a|/2| ≤ β < |M|+|b_a|/2} — so the certificate's
premise is now: testable everywhere except a provably irreducible set, which the system
treats as ABSTAIN. The premise is no longer epistemically equal to the crutches: crutches
are unexamined; this one is audited to its information-theoretic limit.

**P2 (budgets β, η, B, anchor all untestable; no estimation from data).** Theorem
`onebit-id` + `flip-witness` + `bit-necessity` + `bit-dictionary` (the one-bit theorem):
within H, the evidence law determines all magnitudes |b_j|, all products, all benefit
magnitudes |b_j − b_0| — everything except ONE global sign bit, at every moment order
(TV = 0 witness; max|b′+b| = 1.1e−16). Every published identifying device (γ-budget via
sign M, majority-above-chance, AoL slope sign) is proven to be exactly a selector of that
same bit. So the budgets ARE now estimable from data (up to the bit), their premises ARE
falsifiable (audit + τ), and what cannot be estimated is exactly one bit — proven
irreducible, not conjectured. Corollary `unify-budgets` restates Props 9/11/13 + anchor as
(testable structure) + (one bit).

**P3 (one proof idea, ~20 theorems; breadth padding).** Consolidation delivered by
Corollary `unify-budgets`: the recommended main-body structure is 4 pillars —
(I) Impossibility floor: Thm 1 + Le Cam (Prop 1) + irreducibility (Cor 3, now sharpened to
"the irreducible content is one bit");
(II) Certificate + audit: Thm 2 + Theorems `audit`/`no-verify` (Props 6, 9, 11, 13 become
corollaries of the audit framework — each budget premise gets the same falsifier);
(III) Identifiable regimes: Thm 5 frontier as master theorem; per-family boundary props
(label shift, smooth drift, reach table) → corollaries/table rows;
(IV) Rates: evidence-channel rate (new Thm `ev-rate`) headline + labeled-rate props as
companions. Appendix keeps full statements; main body presents 4 + the trichotomy.

**P4 (rate props solve the wrong problem; label-free rate open).** Theorem `ev-rate`:
matching m^{−1/2} upper/lower bounds for the GENUINE evidence channel (agreement rates,
no labels) within audited H — Le Cam two-point inside the model class; empirical slope
−0.510/−0.525 vs theory −0.5; KL/ε² constant (0.164–0.171). Honest qualifier kept: labels
buy a margin-dependent constant (∝1/c_min, diverges as agreement→chance: ratio 2.15 at
c_min=0.21, 4.49 at 0.067, divergent ≤0.022) and the bit — not the rate.

**P5 (Thm 6 is the anchored law, not classical AoL).** Already scoped honestly in
`benefit_sign_frontier.tex`; deepened: Theorem `bit-dictionary`(iii) proves the anchored
slope's SIGN is the same bit (flip sends w→1−w, slope→−slope), so the minimal
calibration-transfer premise for sign decisions is sign(w̄−1/2) transfer — strictly weaker
than w̄_S = w̄_T. Claim boundary vs Baek et al. (2206.13089), Kim et al. (2310.04941),
foundation-model AoL (2404.01542) stated in the positioning paragraph.

## Hypothesis correction (load-bearing; found by adversarial check)

The paper's CEI definition was INSUFFICIENT for the rank-one law 2A_ij−1 = b_i b_j.
Exact deficit (new Theorem `H-deficit`): c_ij − b_i b_j = π(1−π) δ_i δ_j, δ_j the
per-class accuracy gap. Minimal fix H = CEI + per-class symmetry (validated: identity
fails by 0.052 under plain CEI; deficit matches theory to 5.7e−4). `multicandidate.tex`
Definition updated; the M≥4 diagnostic τ is proven sound but NOT complete (explicit
strictly-interior K=4 stealth law with τ = 2e−17, recovery bias 0.229, and a witness that
flips the decision sign) — stated in the paper instead of implied.

## Verified positioning (live-web, June 2026) — must-cite, with confirmed IDs

Parisi et al. PNAS 2014 (1303.3257); Jaffe et al. AISTATS 2015 (1407.7644) and AISTATS
2016 (1510.05830, prior CI-violation detection); Platanios et al. UAI 2014 / ICML 2016 /
NeurIPS 2017 (1705.07086); Ibrahim–Fu–Sidiropoulos NeurIPS 2019 (1909.12325, pairwise
identifiability); Steinhardt–Liang NeurIPS 2016 (1606.05313); Corrada-Emmanuel NTQR
program (2312.05392 NeurIPS 2024; 2409.11052 "logical alarm"; 2412.16238) — closest
overall competitor, must be cited and distinguished (algebraic/logical vs our statistical
level-α + necessity); Rosenfeld–Garg NeurIPS 2023 (2306.00312); Schirmer et al. NeurIPS
2025 (2507.08721) — reactive monitoring foil; Vovk et al. (2102.10439) exchangeability
martingales; Baek et al. (2206.13089), Kim et al. (2310.04941), 2404.01542 for AoL;
Podkopaev–Ramdas (2110.06177). Novelty verdicts: one-bit necessity = novel as scoped
(must scope to the prediction channel — done via the all-orders witness); audit = novel
as a statistical budget test but MUST be positioned as delta over NTQR alarm + Jaffe 2016
+ Vovk (done in the positioning paragraph); evidence-channel rate = cleanest standalone
novelty; anchored AoL = delta over AoL line (done).

### Citation corrections required elsewhere in the paper
- AETTA is Lee, Chottananurak, Gong, Lee — CVPR 2024, arXiv 2404.01351 (not "Kim, Ji").
- Do NOT attribute permutation-identifiability to Dawid–Skene 1979 (they give the EM
  model); attribute to the tensor line (Anandkumar et al.) / Ibrahim et al. 2019.
  `multicandidate.tex` multiclass paragraph: adjust attribution accordingly.
- "CAN / Adapt-or-Skip" naming is taken (NeurIPS 2025 Imageomics workshop, OpenReview
  jEItIzAgHs) — avoid as branding; cite as motivating evidence.

## Honest status

- Proved + machine-validated this session: H-deficit, one-bit (id/witness/necessity/
  dictionary), audit (level+power, worst-case + bootstrap radii), no-verify (B_β exact;
  τ sound-not-complete with constructive witness), evidence-channel rate (matching
  bounds; honest constants). All numbers trace to `validation_results.json`.
- NOT yet done (next phase — training/experiments): real-data instantiation. The audit,
  τ, and the evidence-channel estimator have run only on synthetic panels. Open problems
  stated in the paper: tight audit radius (C₂ is loose — bootstrap powerful but its
  validity empirical); weakest falsifiable class beyond H; small-‖δ‖ robustness via the
  deficit identity.

## Training-phase handoff (when you say go)

1. Real panels exist already: 123-task anomaly bank (6 detectors/task = natural K=6
   panel) and CIFAR-10-C stress grid (Tent/EATA/SAR + frozen = K=4). For each: estimate
   ĉ, τ̂, run the audit, compare certified signs vs ground truth; report where H is
   rejected (expect co-trained TTA candidates to violate CEI — that is the diagnostic
   working, and the honest story).
2. Pre-register: α=0.05, the bootstrap radius, the H-rejection rule, and the abstain
   accounting BEFORE looking at outcomes (protocol yaml in research_lock/, same style as
   existing).
3. Report all three zones per condition: certified / falsified / blind (B_β) — the
   trichotomy is the deliverable, not a single accuracy number.
4. Then the decision baselines the reviewer demanded (AETTA/ATC/AoL as adapt-freeze
   rules) on the same grid.
