# Reviewer Response: Full Resolution of Task T2 (Theory–Algorithm Relationship)

**Submission Targets:**
- Primary TMLR Submission: `kbound_tmlr_v2.pdf` (49 pages)
- Compact Submission: `kbound_submission_v2.pdf` (44 pages)
- Full Extended Version: `kbound_full_v2.pdf` (62 pages)

**Task Status:**
- **Previous Verdict:** 🟡 *Substantially addressed; needs 108-cell results table + non-oracle eval*
- **Updated Verdict:** 🟢 **FULLY ADDRESSED AND CLOSED**

---

## 1. Reviewer Comments and Point-by-Point Resolutions

### Comment 1: "Display the population replay's results"
> *Reviewer: "This may be mostly a reporting task: use the results you already generated... Display the population replay’s results; non-oracle validation is needed for the stronger extension, not merely the original clarification."*

**Response:**
We have prominently displayed the complete numerical results of the 108-cell ImageNet-C population replay in **both the main scientific body (§6.2, Table 3) and the supplementary material (Appendix A.7, Table 12)**:

1. **Main Scientific Body (§6.2, Table 3: `\label{tab:main-pop-replay-summary}`):**
   Right after the theoretical cell-to-population transfer proposition (Proposition 2, `thm:population-transfer`), we have embedded a concise summary table displaying:
   - Candidate breakdown ($N=36$ each for Tent, EATA, SAR; $N=108$ total)
   - Exact oracle regret for Cell Gate vs. Compound Population Interval ($0.0162/0.0468$ Tent, $0.0095/0.0386$ EATA, $0.0245/0.0267$ SAR; $0.0167 \to 0.0374$ Panel)
   - Decisions before (Cell Gate: $40$ Adapt / $11$ Freeze / $57$ Abstain) vs. after (Compound Population Interval: $1$ Adapt / $10$ Freeze / $97$ Abstain)
   - Zero false adaptations ($\mathrm{FA}_u = 0.0000$ across all 108 conditions)

2. **Supplementary Material (Appendix A.7, Table 12: `\label{tab:imagenetc-108-population-replay}`):**
   Provides the comprehensive 12-column breakdown:
   - 5-fold cell-outcome-disjoint cross-fitting ($9$--$10$ fit, $19$ cal, $7$--$8$ scored cells per fold)
   - Exact cell radii ranges: mean $\bar\varepsilon = 0.0665$ (Tent: $[0.0331, 0.0659]$; EATA: $[0.0196, 0.0393]$; SAR: $[0.0365, 0.2106]$)
   - Hoeffding concentration radius: $b = 0.0607$ ($m=2{,}000, \delta=0.05$)
   - Non-degenerate calibration: $108/108$ cells ($100\%$) yield finite radii
   - Population truth: Explicitly designated as *"Not established (development replay)"*
   - Explicit sample-size parameter clarification: *"The parameter $m=2{,}000$ is a sensitivity input to the enlarged-radius formula, not a count of independently drawn evaluation samples. All 108 cells are existing development records; this replay supplies no new independent sampling."*
   - Verified receipts: `experiments/kbound/results/task2_imagenetc_population_replay_20260911/population_panel.json`.

---

### Comment 2: "Non-oracle validation is needed for the stronger extension, not merely the original clarification"
> *Reviewer: "Table 10 uses $\widehat\Delta(Z) = \Delta(Z)$ — coverage is trivial. Need imperfect predictor fitted on dev episodes. Evaluate: point rule vs $\widehat\Delta \pm \varepsilon$ vs $\widehat\Delta \pm (\varepsilon + b)$. Report directional errors and commitment cost."*

**Response:**
We completely agree. The original synthetic simulation evaluated exact Bayes prediction, which verified algebraic consistency but left the stronger population extension unvalidated under estimation error.

To fully resolve this, we implemented and executed a comprehensive **non-oracle population diagnostic** across 1,000 independent trials per configuration (16 grid configurations; $N_{\mathrm{cal}} \in \{18, 19, 39, 99\}$, $m \in \{32, 128, 512, 2048\}$; artifact: `experiments/kbound/results/synthetic_diagnostic/nonoracle_diagnostic_summary.json`). Rather than assuming an oracle, an imperfect linear predictor is fitted on 15 independent development episodes ($m=32$) subject to estimation noise. 

The empirical findings are reported in **Table 11 (`\label{tab:nonoracle-pop-diagnostic}`) in Appendix A.7 and detailed in §6.2**:
- **Unhedged Point Rule ($\operatorname{sign}(\widehat\Delta)$):** Commits unconditionally ($0.0\%$ abstention), suffering a **$30.9\%$--$35.6\%$ directional error rate** (where $\mathrm{FA}_u$ strictly measures false adapt, whereas directional error combines false adapt and false freeze) because random estimation noise on null/tie states ($Z=0, \Delta=0$) forces spurious commitments, though errors on strictly harmful states remain zero.
- **Conformal Cell Interval ($\widehat\Delta \pm \varepsilon_{\mathrm{cell}}$):** Conformal residual calibration over the imperfect predictions guarantees valid empirical cell coverage ($93.9\%$--$96.5\%$, tightly tracking $1-\alpha_{\mathrm{cell}} = 0.95$) and eliminates wrong-direction error ($0.000$). However, at small batch sizes ($m=32$), it fails to account for sampling variance $b(32) = 0.4802$, maintaining a misleadingly high commitment rate ($\approx 65\%$).
- **Compound Population Interval ($\widehat\Delta \pm (\varepsilon_{\mathrm{cell}} + b)$):** Incorporating the Hoeffding concentration radius $b(m, \delta)$ guarantees **$100\%$ population coverage** and **strictly zero wrong-direction error ($0.000 \le \alpha_{\mathrm{pop}}=0.10$)** across all 16 configurations. At $m=32$, it enforces the *Price of Rigor* ($99.9\%$--$100\%$ abstention), while at $m \ge 128$, abstention safely contracts to $32\%$--$34\%$ (matching the $33.3\%$ null state frequency).

---

### Comment 3: Conceptual & Mathematical Bridge (Theory $\longleftrightarrow$ Algorithm)

**Response:**
To ensure the conceptual relationship between the asymptotic population theory and the empirical conformal algorithm is transparent, we added:
1. **Theory-to-Algorithm Correspondence Table (§6.1, Table 2: `\label{tab:theory-algorithm-bridge}`):**
   Directly maps each theoretical construct ($M, \gamma, \beta, \Delta$) to its counterpart in the empirical KGA gate ($\widehat\Delta = h_\theta(Z), \varepsilon, b(m,\delta)$), resolving the dimensional and category distinctions.
2. **The Implication Chain (§6.1):**
   $$\boxed{\mathcal C_\beta\ \text{declared}} \implies \boxed{\mathcal P_{\mathrm{KGA}}\subseteq\mathcal C_\beta \text{ and risk-aligned for }Z} \implies \widehat\Delta=h_\theta(Z) \implies \Pr(|\widehat\Delta-\Delta|\le\varepsilon)\ge1-\alpha \implies \boxed{\adapt/\freeze/\abstain}$$
3. **Formal Lean Kernel Verification:**
   The deterministic decision layer is kernel-checked in `docs/research/kbound/formal/KBound/Population.lean` (proving soundness of strict ADAPT and FREEZE decisions and guaranteed ABSTAIN on zero containment).
4. **Outcome-Blind Execution Interface:**
   `docs/research/kbound/scripts/run_population_panel.py` enforces fail-closed validation that rejects any label-bearing inputs at decision time.

---

## 2. Verification Summary

- **LaTeX Compilation:** Clean compilation under TeX Live 2025 (`pdflatex` / `latexmk`). Exit code: 0. Errors: 0. Undefined references: 0.
- **Active Submission PDFs:**
  - `kbound_tmlr_v2.pdf`: 49 pages
  - `kbound_submission_v2.pdf`: 44 pages
  - `kbound_full_v2.pdf`: 62 pages

**Conclusion:** Task T2 is now completely closed across theory, Lean formal verification, implementation scripts, non-oracle experimental validation, main-body and supplementary tables, and recompiled publication PDFs.
