# K-Bound: Honest Novelty Map & the Level-9 Target
_Synthesis of a 5-agent deep literature search (label-free OOD accuracy estimation, identifiability/impossibility, agreement-on-the-line, calibration transfer, TTA safety). Every claim traces to a primary paper below._

## Bottom line
The deep search **confirms the gap is real but reclassifies the contribution**. K-Bound's genuinely novel pieces are (a) the *target* — the **sign of the adaptation benefit** Δ=R(f₀)−R(f_adapt), a comparative adapt-vs-freeze object **no prior work estimates**; (b) a **certified adapt/freeze/abstain gate with a false-adapt rate** (a *decision* certificate, not an after-the-fact alarm); and (c) the **exact disagreement-region frontier** that *formalizes and unifies* the field's calibration-style assumptions. The impossibility itself, and "calibration is the uncheckable obstruction," are **already known** and must be cited, not claimed. Net: this is a **strong, honest 8 / 8.5** once repositioned. The 9 is a different, harder theorem the search pinpoints precisely (below).

## What is PROVEN (cite these — do not claim them)
- **Impossibility of label-free accuracy is classical.** Ben-David, Luu, Lu & Pál, *Impossibility Theorems for Domain Adaptation* (AISTATS 2010) — the canonical two-point/Le Cam construction: covariate shift alone does **not** suffice; two targets indistinguishable from (labeled source, unlabeled target) force error ≥ ½ on one. K-Bound's Theorem 1 is a **re-derivation specialized to adapt/freeze** — position it as such.
- **ATC states the iff + impossibility.** Garg et al., *Leveraging Unlabeled Data to Predict OOD Performance* (ICLR 2022): target accuracy is identifiable **iff** p_t(y|x) is pinned down by the source joint + target marginal; "identifying accuracy is as hard as identifying the optimal predictor." So the identifiability *iff for accuracy* already exists.
- **"Calibration cannot be identified without labels" is explicit.** Rosenfeld & Garg, *(Almost) Provable Error Bounds via Disagreement Discrepancy* (NeurIPS 2023) say it verbatim — and replace calibration with a **disagreement-discrepancy condition that IS checkable from unlabeled data** and yields a valid error **bound**.
- **Disagreement = error under calibration.** Jiang et al., *Assessing Generalization via Disagreement* (ICLR 2022) — exact equality, but conditional on (class-aggregated) calibration; shown fragile by Kirsch & Gal (2022).
- **Agreement-on-the-Line theory is linear/Gaussian only.** Miller et al. (ICML 2021, accuracy-on-the-line); Baek et al. (NeurIPS 2022, agreement-on-the-line, *empirical*); Baek, Raghunathan & Kolter (AISTATS 2025, *proof only in linear/Gaussian + GD-as-interpolation*, with a residual coupling AGL-tightness to ACL-tightness). Failure regimes named: spurious correlation (Accuracy-on-the-Curve, ICML 2023), label noise (Accuracy-on-the-Wrong-Line, ICML 2024), CIFAR-10-C Gaussian noise.
- **Closest "certified gate" prior art (must differentiate).**
  - Schirmer & Jazbec et al., *Monitoring Risks in TTA* (NeurIPS 2025) — label-free **anytime-valid PFA control**, but it raises an **alarm after degradation**, does not gate the step.
  - Bar, Shaer & Romano, *Protected TTA via Online Entropy Matching* (NeurIPS 2024) — **betting-martingale** with anytime-valid control on **shift detection**, then *steers* (not gates) adaptation.
- **Checkable-from-unlabeled sufficient conditions EXIST** (this kills any "no checkable condition" claim): disagreement-discrepancy (Rosenfeld–Garg 2023) and **Sparse Joint Shift** (Chen, Zaharia & Zou, NeurIPS 2022 — identifiability broader than covariate shift). Also: calibration≡quantification≡accuracy-prediction equivalence under shift (Moreo et al. 2025) — supports the "calibration is the crux" unification.

## What is OPEN (this is where 9–10 lives)
1. **No exact necessary-and-sufficient frontier for the SIGN of Δ.** Every iff in the literature characterizes a *nuisance parameter* (label-shift prior; BBSE invertibility — Lipton et al. 2018) or *whether an assumption holds* (generalized label shift — Tachet des Combes et al. 2020), **never** the comparative benefit sign. The decision object is genuinely untouched.
2. **No general characterization of Agreement-on-the-Line.** Proven only in linear/Gaussian/random-features. A **structural condition on the shift that provably forces AGL for generic models** (the advisor's suggested 9) is open and hard.
3. **No per-step adapt/freeze/abstain decision certificate.** The e-process/confidence-sequence machinery exists (Monitoring-Risks, Protected-TTA) but has not been lifted onto the *action* with a false-adapt guarantee + abstention.

## K-Bound claim-by-claim (honest matrix)
| Claim in K-Bound | Verdict | Action |
|---|---|---|
| Impossibility / unknowable regime (Thm 1) | **Known** (Ben-David 2010; Garg 2022) | Re-cite as re-derivation for adapt/freeze; drop any "novel impossibility" tone |
| Calibration drift = the obstruction | **Known/folklore** (Rosenfeld–Garg 2023) | Cite; claim only the *exact frontier formalization* |
| Certified adapt/freeze/**abstain** gate w/ false-adapt rate | **Novel as a decision certificate** | Differentiate sharply from Monitoring-Risks (alarm) & Protected-TTA (detection) |
| Target = **sign of adaptation benefit** | **Novel** (nobody estimates the comparative object) | Make this the headline framing |
| Exact M+γ frontier, ATC/AETTA/AGL = β=0 face | **Novel as formalization+unification** | Cite DIS²/SJS as checkable-condition prior art; do **not** claim "no checkable condition exists" |
| Disagreement-region sign reduction (Thm 5) | Novel-ish, elementary | Keep, modest framing |

## The sharpest forward target (named)
Per the limitations analysis and Camelyon17 bias--variance diagnostic, the actionable forward path is the **Target-Label-Light Multimodal Safety Guard** (see `TARGET_LABEL_LIGHT_MULTIMODAL_PLAN.md`, Protocol D24):

1. **(a) Target-label-light probe certificate** — a micro-probe of k≈8–64 held-out target labels removes calibration-drift bias γ in B̂(Z) that caps label-free certifiability on natural shift. Prop. impossibility applies to unconditional label-free rules only; the probe is a distinct operating point (Prop. `prop:tll-escape` in `kbound.tex`).

2. **(b) Multimodal fusion router guard** — apply the certificate to reliability-fusion systems where modality failure is structurally detectable (Real-IAD-D3 FREEZE, NatDeg ADAPT; `multimodal_guard.py` + Table `tab:multimodal`).

The abstract Level-9 targets below remain open theory problems; (a)+(b) above are the **deployment-forward** path grounded in existing negative results.

### Legacy Level-9 targets (theory, still open)
Per both the advisor criteria and this search, the reachable 9 is **one** of:
- **(A) An exact, tight frontier for sign(Δ) identifiability** that is genuinely *assumption-minimal* — i.e., removes the supplied drift-budget β, or proves (beyond Ben-David/Garg) an *exact* impossibility that no unlabeled-checkable functional can certify the sign, **distinguished from DIS²/SJS** by targeting the comparative benefit sign rather than an error bound. This is the closest extension of what you already have; risk: it may collapse to "β is irreducible," which is a sharper-8.
- **(B) A general necessary-and-sufficient structural characterization of Agreement-on-the-Line** (when does a shift provably force AGL, for generic models). Highest payoff, highest risk; genuinely open; months.

Either is a real theorem, not assembly. (A) is reachable from K-Bound's existing machinery; (B) reorganizes a subfield.

## Recommended paper actions (lifts the 8 to a confident, defensible 8.5)
1. Add a **Positioning** paragraph + the matrix above; cite Ben-David 2010, Garg 2022, Rosenfeld–Garg 2023, Baek 2022/2025, Miller 2021, AETTA 2024, Monitoring-Risks 2025, Protected-TTA 2024, Chen-Zaharia-Zou 2022, Moreo 2025.
2. **Reframe the headline** around *the sign of the adaptation benefit* and the *certified gate* (the two defensibly-novel pieces), not the impossibility.
3. **Soften** the impossibility/calibration claims to "we formalize and sharpen a known obstruction."
4. Add the **exact frontier** (validated: `val_frontier.py`, all pass) as the unification result, citing the heuristics it subsumes.
5. State the **forward deployment path** (Target-Label-Light Multimodal Guard, §Forward work in `kbound.tex`) explicitly — reviewers reward a paper that names its own frontier *and* the escape hatch.
